"""CPU unit test (no GPU, no vLLM): the plugin's dequantization equals the checkpoint's official eg2_quant.py bit for bit.

    pip install torch safetensors huggingface_hub pytest && pytest vllm/tests -q

The official loader is fetched from the model repository at the verified revision (a 14 KB file), or taken from
EG2_QUANT_PY if set. Only the pure-torch part of eg2q.py (above the __PURE_END__ marker) is executed.
"""
import importlib.util, os, pathlib

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO, REVISION = "ThakiCloud/eg2-hybrid-w4-media", "bc3574a285a2779ac0686cc149342f30fbbf8f5b"


def _pure_plugin():
    src = (HERE.parent / "eg2q" / "eg2q.py").read_text()
    assert "# __PURE_END__" in src
    ns = {"__name__": "eg2q_pure"}
    exec(compile(src.split("# __PURE_END__")[0], "eg2q_pure", "exec"), ns)
    return ns


def _official():
    p = os.environ.get("EG2_QUANT_PY")
    if not p:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, "eg2_quant.py", revision=REVISION)
    spec = importlib.util.spec_from_file_location("eg2_quant_official", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P, E = _pure_plugin(), _official()


def _pack(codes, bits):
    """Pack integer codes (8 codes per `bits` bytes, little-endian) the way eg2-quant-v1 stores them."""
    n = codes.numel()
    pad = (-n) % 8
    c = torch.cat([codes.to(torch.int64), torch.zeros(pad, dtype=torch.int64)]).reshape(-1, 8)
    w = torch.zeros(c.shape[0], dtype=torch.int64)
    for k in range(8):
        w |= c[:, k] << (bits * k)
    return torch.stack([(w >> (8 * j)) & 0xFF for j in range(bits)], 1).reshape(-1).to(torch.uint8)


@pytest.mark.parametrize("bits", [4, 3, 2])   # widths the official loader at this revision decodes
@pytest.mark.parametrize("rows,cols,g", [(16, 64, 64), (7, 200, 64), (32, 1152, 64), (5, 33, 64)])
def test_dequant_matches_official(bits, rows, cols, g):
    gen = torch.Generator().manual_seed(rows * 1000 + cols + bits)
    codes = torch.randint(0, 1 << bits, (rows * cols,), generator=gen)
    buf = _pack(codes, bits)
    ng = -(-cols // g)
    scale = (torch.rand(rows, ng, generator=gen) * 0.02 + 1e-4).to(torch.float16)
    zp = torch.randint(0, 1 << bits, (rows, ng), generator=gen).to(torch.float16)
    assert torch.equal(P["unpack_codes"](buf, bits, rows * cols), E.unpack(buf, bits, rows * cols))
    ours = P["dequant_bf16"](buf, scale, zp, rows, cols, bits, g)
    ref = E.dequant(E.unpack(buf, bits, rows * cols), scale, zp, (rows, cols), g).to(torch.bfloat16)
    assert ours.dtype == torch.bfloat16 and ours.shape == (rows, cols)
    assert torch.equal(ours.view(torch.int16), ref.view(torch.int16))
