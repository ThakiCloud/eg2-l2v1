"""eg2q — out-of-tree vLLM plugin that serves the 4-bit EG2 Hybrid W4 checkpoint unchanged.

Checkpoint: ThakiCloud/eg2-hybrid-w4-media (format eg2-quant-v1, 4-bit codes + fp16 scale/zero-point, group size 64).
No vLLM fork and no re-quantization: the plugin registers a QuantizationConfig named "eg2q" through vLLM's
`vllm.general_plugins` entry point.

How it works
  * Every Linear the checkpoint stores as `<prefix>.weight.{qcodes,scale,zp}` (247 in the vision and audio towers)
    gets EG2QLinearMethod: the 4-bit codes stay resident on the GPU, and each forward call dequantizes that one
    layer to BF16 and runs the ordinary BF16 GEMM. This is NOT a native INT4 GEMM; only weight storage changes.
  * Every other Linear (the language model and the BF16 projections) gets vLLM's UnquantizedLinearMethod.
  * Which layers are quantized is read from the checkpoint header (`__metadata__.shapes`), not guessed from layer type.

Environment
  EG2Q_CKPT  (required) local directory of the checkpoint snapshot, i.e. the same path passed as `model`.
  EG2Q_LOG   directory for the load report (state_<pid>.json); default /tmp/eg2q_log.
  EG2Q_HASH  "1" = also record sha256 of every dequantized BF16 weight (used by the self-test; slower load).
"""
import json, os, struct

import torch
import torch.nn.functional as F

__version__ = "0.2.0"


# ---------------------------------------------------------------- pure torch (same arithmetic as the checkpoint's eg2_quant.py)
# Everything above the __PURE_END__ marker is exercised by the CPU unit test without vLLM installed.

def unpack_codes(buf, bits, n):
    """Same result as eg2_quant.unpack: 8 codes occupy `bits` bytes (little-endian).
    bits == 4 uses a uint8 fast path (low nibble = even code); other widths use the int64 path of the official code."""
    if bits == 4:
        b = buf.reshape(-1)
        return torch.stack((b & 0x0F, b >> 4), dim=1).reshape(-1)[:n]
    if not 2 <= bits <= 8:
        raise ValueError(f"unsupported bit width: {bits}")
    b = buf.to(torch.int64).reshape(-1, bits)
    w = torch.zeros(b.shape[0], dtype=torch.int64, device=b.device)
    for j in range(bits):
        w |= b[:, j] << (8 * j)
    mask = (1 << bits) - 1
    return torch.stack([(w >> (bits * k)) & mask for k in range(8)], 1).reshape(-1)[:n]


def dequant_bf16(qcodes, scale, zp, rows, cols, bits, group_size):
    """`(code - zp) * scale` in fp32, then cast to bf16 (identical to eg2_quant.dequant(...).to(torch.bfloat16))."""
    c = unpack_codes(qcodes, bits, rows * cols).reshape(rows, cols).float()
    scale = scale.float(); zp = zp.float()
    full = (cols // group_size) * group_size
    parts = []
    if full:
        ng = full // group_size
        grouped = c[:, :full].reshape(rows, ng, group_size)
        parts.append(((grouped - zp[:, :ng, None]) * scale[:, :ng, None]).reshape(rows, full))
    if cols > full:                       # a short tail group reuses the last scale/zp column
        parts.append((c[:, full:] - zp[:, -1:]) * scale[:, -1:])
    return torch.cat(parts, 1).to(torch.bfloat16)

# __PURE_END__
# ---------------------------------------------------------------- vLLM-dependent part
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod

LOG = os.environ.get("EG2Q_LOG", "/tmp/eg2q_log")
STATE = {"pid": os.getpid(), "version": __version__, "routed": {}, "quantized": [], "matched": {}, "unloaded": [],
         "mode": "stream", "bits": None, "group_size": None}
QKEYS = None          # checkpoint names of quantized Linears (`<prefix>.weight` without `.weight`)


def _qkeys():
    """Read the quantized-layer names from the checkpoint header named by EG2Q_CKPT."""
    global QKEYS
    if QKEYS is None:
        p = os.environ.get("EG2Q_CKPT")
        if not p:
            raise RuntimeError("eg2q: set EG2Q_CKPT to the local checkpoint directory (the same path you pass as the model)")
        p = p if p.endswith(".safetensors") else os.path.join(p, "model.safetensors")
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        meta = h.get("__metadata__", {})
        if meta.get("format") != "eg2-quant-v1":
            raise RuntimeError(f"eg2q: {p} is not an eg2-quant-v1 checkpoint (format={meta.get('format')})")
        shapes = json.loads(meta["shapes"]) if isinstance(meta.get("shapes"), str) else meta["shapes"]
        qs = {k[: -len(".weight")] for k in shapes}
        have = {k[: -len(".qcodes")] for k in h if k.endswith(".qcodes")}
        if {q + ".weight" for q in qs} != have:
            raise RuntimeError(f"eg2q: header shapes ({len(qs)}) and qcodes tensors ({len(have)}) disagree")
        QKEYS = qs
        STATE["ckpt_quantized"] = len(qs)
    return QKEYS


def _match(prefix):
    """vLLM module prefix -> checkpoint name. Leading prefixes (`model.` etc.) can differ, so match the unique key with the same tail."""
    hits = [k for k in _qkeys() if prefix == k or prefix.endswith("." + k) or k.endswith("." + prefix)]
    if len(hits) > 1:
        raise RuntimeError(f"eg2q: {prefix} matches several checkpoint keys: {hits[:3]}")
    return hits[0] if hits else None


def _top(prefix):
    for k in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "language_model"):
        if k in prefix:
            return k
    return "other"


def _dump():
    os.makedirs(LOG, exist_ok=True)
    with open(os.path.join(LOG, f"state_{STATE['pid']}.json"), "w") as f:
        json.dump(STATE, f)


class EG2QLinearMethod(LinearMethodBase):
    """One checkpoint layer stored as qcodes/scale/zp. Keeps the 4-bit codes on the GPU and dequantizes only this layer to
    BF16 on every call before F.linear. No whole-model dequantization cache, no native INT4 GEMM."""

    supports_pre_processed_weights = False

    def __init__(self, cfg, prefix):
        self.cfg, self.prefix = cfg, prefix

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        # Tower Linears are replaced 1:1 by ReplicatedLinear (no fusion, no tensor-parallel sharding). Fail loudly otherwise.
        if len(output_partition_sizes) != 1 or input_size_per_partition != input_size or output_partition_sizes[0] != output_size:
            raise RuntimeError(f"eg2q: fused/sharded layer is not supported: {self.prefix} sizes={output_partition_sizes} "
                               f"in={input_size_per_partition}/{input_size}")
        bits, g = self.cfg.bits, self.cfg.group_size
        rows, cols = output_size, input_size
        nbytes = -(-(rows * cols) // 8) * bits
        ng = -(-cols // g)
        layer._eg2q = {"rows": rows, "cols": cols, "bits": bits, "g": g, "loaded": set(), "prefix": self.prefix}

        def mk(kind, data):
            p = torch.nn.Parameter(data, requires_grad=False)
            p.weight_loader = self._loader(layer, kind)
            layer.register_parameter(kind, p)

        mk("weight_qcodes", torch.empty(nbytes, dtype=torch.uint8))
        mk("weight_scale", torch.empty(rows, ng, dtype=torch.float16))
        mk("weight_zp", torch.empty(rows, ng, dtype=torch.float16))
        # Zero-element placeholder `weight`: the HF model code reads `input_proj.weight.dtype`. Its values are never used.
        layer.register_parameter("weight", torch.nn.Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False))

    def _loader(self, layer, kind):
        def loader(param, loaded_weight, shard_id=None):
            st = layer._eg2q
            if shard_id is not None:
                raise RuntimeError(f"eg2q: shard_id={shard_id!r} for {st['prefix']} (a stacked mapper must not match tower layers)")
            if param.numel() != loaded_weight.numel():
                raise RuntimeError(f"eg2q: {kind} numel {loaded_weight.numel()} != expected {param.numel()} for {st['prefix']} "
                                   "(bits/group_size mismatch?)")
            if kind == "weight_qcodes" and loaded_weight.dtype != torch.uint8:
                raise RuntimeError(f"eg2q: qcodes dtype {loaded_weight.dtype} (uint8 expected) for {st['prefix']}")
            param.data.copy_(loaded_weight.reshape(param.shape))
            st["loaded"].add(kind)
        return loader

    def process_weights_after_loading(self, layer):
        st = layer._eg2q
        if st["loaded"] != {"weight_qcodes", "weight_scale", "weight_zp"}:
            STATE["unloaded"].append(st["prefix"]); _dump()
            raise RuntimeError(f"eg2q: {st['prefix']} quantized tensors incomplete (loaded={sorted(st['loaded'])})")
        STATE["quantized"].append(st["prefix"])
        if os.environ.get("EG2Q_HASH") == "1":           # self-test: hash of the BF16 weight this engine reconstructs
            import hashlib
            w = dequant_bf16(layer.weight_qcodes.data, layer.weight_scale.data, layer.weight_zp.data,
                             st["rows"], st["cols"], st["bits"], st["g"])
            STATE.setdefault("sha", {})[STATE["matched"].get(st["prefix"], st["prefix"])] = \
                hashlib.sha256(w.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
            del w
        STATE.update(mode=self.cfg.mode, bits=self.cfg.bits, group_size=self.cfg.group_size)
        _dump()

    def apply(self, layer, x, bias=None):
        st = layer._eg2q
        w = dequant_bf16(layer.weight_qcodes, layer.weight_scale, layer.weight_zp, st["rows"], st["cols"], st["bits"], st["g"])
        return F.linear(x, w.to(x.dtype) if w.dtype != x.dtype else w, bias)


@register_quantization_config("eg2q")
class EG2QConfig(QuantizationConfig):
    def __init__(self, bits=4, group_size=64, mode="stream"):
        super().__init__()
        if mode != "stream":
            raise ValueError(f"eg2q mode must be 'stream' (per-layer dequantization), got {mode}")
        self.bits, self.group_size, self.mode = int(bits), int(group_size), mode

    def get_name(self):
        return "eg2q"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        # hf_overrides={"quantization_config": {"quant_method": "eg2q", "bits": 4, "group_size": 64, "mode": "stream"}}
        return cls(config.get("bits", 4), config.get("group_size", 64), config.get("mode", "stream"))

    @staticmethod
    def get_checkpoint_weight_mapper():
        from vllm.model_executor.models.utils import WeightsMapper
        # `x.weight.qcodes` would be seen as a tensor nested under the `weight` parameter; rename the suffixes instead.
        # AutoWeightsLoader merges this mapper into the model's own (suffix rules apply last). Keeps the base `.g_idx -> None` rule.
        return WeightsMapper(orig_to_new_suffix={
            ".g_idx": None,
            ".weight.qcodes": ".weight_qcodes",
            ".weight.scale": ".weight_scale",
            ".weight.zp": ".weight_zp",
        })

    def get_quant_method(self, layer, prefix):
        tag = _top(prefix)
        if not isinstance(layer, LinearBase):
            return None
        STATE["routed"][tag] = STATE["routed"].get(tag, 0) + 1
        key = None if tag == "language_model" else _match(prefix)
        if key is None:                                          # layers the checkpoint stores in BF16
            return UnquantizedLinearMethod()
        STATE["matched"][prefix] = key
        return EG2QLinearMethod(self, prefix)


def register():
    """vLLM general-plugin entry point. Importing this module registers the "eg2q" quantization config."""
    _dump()
