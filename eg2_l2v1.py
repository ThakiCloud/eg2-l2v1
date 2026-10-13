"""eg2-l2v1 runtime v0.1.0 — GPU-resident W4 loader for ThakiCloud/eg2-hybrid-w4-media.

Copyright 2026 ThakiCloud. Licensed under the Apache License, Version 2.0 (see LICENSE).

What this is: the published t16 checkpoint keeps its media Linear weights as 4-bit codes. The stock loader
(`eg2_quant.load_sentence_transformer`) dequantizes everything to bf16 at load. This runtime instead keeps the
4-bit codes on the GPU and dequantizes each Linear's weight on the fly, right before a standard bf16 GEMM.

What this is NOT: a native INT4 GEMM. The matrix multiply is the same bf16 GEMM; only weight storage changes.
Embeddings are bit-identical in weights to the stock loader's dequantized model (same fp32 formula, same order).

Measured on NVIDIA B200 ×1 (see README for the full tables and environment):
  resident GPU memory after load 0.874 GB = 0.582× bf16 · peak during load 1.797 GB (budget for this, not 0.874)
  embeddings bit-identical to the stock loader on real FSD50K audio · min cosine ≥ 0.9999998 on synthetic inputs
  throughput vs bf16 is 0.83-0.94× for audio at batch 16 on real recordings — this is a memory saving, not a speed-up.

Two weight paths (selected per tensor at load):
  * audio tower W4 Linear  → "vec1": fp32 scale/zero-point cached once at load (+≈38 MB), unpack → dequant → bf16
  * every other W4 Linear  → "vec" : same formula, scale/zero-point cast per call
Requires: CUDA GPU, the checkpoint's `eg2_quant.py` (shipped in the model repo, used unchanged for non-Linear tensors).
"""
from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

__version__ = "0.1.0"
RUNTIME = "eg2-l2v1"


def _unpack4(buf: torch.Tensor, n: int) -> torch.Tensor:
    b = buf.reshape(-1).to(torch.uint8)
    return torch.stack((b & 0xF, b >> 4), 1).reshape(-1)[:n]


def _dequant(codes, scale, zp, shape, g):
    rows, cols = shape
    full = (cols // g) * g
    c = codes.reshape(rows, cols).float(); sc = scale.float(); z = zp.float()
    if full == cols:
        return ((c.reshape(rows, cols // g, g) - z[:, :, None]) * sc[:, :, None]).reshape(rows, cols)
    head = ((c[:, :full].reshape(rows, full // g, g) - z[:, :full // g, None]) * sc[:, :full // g, None]).reshape(rows, full)
    return torch.cat((head, (c[:, full:] - z[:, -1:]) * sc[:, -1:]), 1)


class W4ResidentLinear(nn.Module):
    """Linear whose weight lives as 4-bit codes; `.weight` dequantizes on access (model code reads `.weight.dtype`)."""

    def __init__(self, bias, codes, scale, zp, shape, g, cache_fp32: bool):
        super().__init__()
        self.register_buffer("codes", codes); self.register_buffer("scale", scale); self.register_buffer("zp", zp)
        self.bias = bias
        self.shape, self.g = tuple(shape), int(g)
        self.in_features, self.out_features = self.shape[1], self.shape[0]
        self._v1 = None
        if cache_fp32 and self.shape[1] % self.g == 0:
            ng = self.shape[1] // self.g
            self._v1 = (scale.float().reshape(self.shape[0], ng, 1).contiguous(),
                        zp.float().reshape(self.shape[0], ng, 1).contiguous())

    @property
    def weight(self) -> torch.Tensor:
        rows, cols = self.shape
        if self._v1 is not None:
            sc32, z32 = self._v1
            b = self.codes.reshape(-1).to(torch.uint8)
            c = torch.stack((b & 0xF, b >> 4), 1).reshape(rows, cols // self.g, self.g).float()
            return ((c - z32) * sc32).reshape(rows, cols).to(torch.bfloat16)
        return _dequant(_unpack4(self.codes, rows * cols), self.scale, self.zp, self.shape, self.g).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.linear(x.reshape(-1, x.shape[-1]), self.weight.to(x.dtype), self.bias)
        return y.view(*x.shape[:-1], y.shape[-1])

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, g={self.g}, path={'vec1' if self._v1 is not None else 'vec'}"


def load(path: str | Path, device: str = "cuda"):
    """Return (SentenceTransformer, info). `path` = local snapshot of ThakiCloud/eg2-hybrid-w4-media."""
    from safetensors import safe_open
    from sentence_transformers import SentenceTransformer
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("eg2-l2v1 runtime is GPU-only (measured on CUDA); use eg2_quant.load_sentence_transformer on CPU")
    path = Path(path)
    sys.path.insert(0, str(path))
    import eg2_quant as E                                         # the checkpoint's own decoder, unchanged
    lv = logging.getLogger("transformers").level
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        st = SentenceTransformer(str(path), device=device, model_kwargs={"dtype": torch.bfloat16})
    finally:
        logging.getLogger("transformers").setLevel(lv)
    inner = st[0].auto_model; own = inner.state_dict()
    f = safe_open(str(path / "model.safetensors"), "pt", device=device); meta = f.metadata()
    shapes = {k: tuple(v) for k, v in json.loads(meta["shapes"]).items()}
    g, bits = int(meta["g"]), int(meta["bits"])
    if bits != 4:
        raise RuntimeError(f"eg2-l2v1 supports 4-bit checkpoints only (got bits={bits})")
    plain = [k for k in f.keys() if not k.endswith((".qcodes", ".scale", ".zp"))]
    info = {"runtime": RUNTIME, "version": __version__, "w4_linear": 0, "w4_linear_vec1": 0, "dequantized_at_load": 0}
    with torch.no_grad():
        for k in plain:
            own[k].copy_(f.get_tensor(k))
        for tname, shp in shapes.items():
            c, s, z = f.get_tensor(f"{tname}.qcodes"), f.get_tensor(f"{tname}.scale"), f.get_tensor(f"{tname}.zp")
            mod_name, leaf = tname.rsplit(".", 1)
            mod = inner.get_submodule(mod_name)
            if leaf == "weight" and isinstance(mod, nn.Linear) and tuple(mod.weight.shape) == shp:
                parent, child = mod_name.rsplit(".", 1)
                w = W4ResidentLinear(mod.bias, c, s, z, shp, g, cache_fp32="audio" in tname)
                setattr(inner.get_submodule(parent), child, w)
                info["w4_linear"] += 1; info["w4_linear_vec1"] += w._v1 is not None
            else:
                own[tname].copy_(E.dequant(E.unpack(c, bits, shp[0] * shp[1]), s, z, shp, g).to(torch.bfloat16))
                info["dequantized_at_load"] += 1
            del c, s, z
    del own; gc.collect(); torch.cuda.empty_cache()
    return st, info
