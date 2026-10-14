#!/usr/bin/env bash
# eg2-l2v1 v0.2.0 — vLLM path install (Linux x86_64, Python 3.12, NVIDIA driver supporting CUDA 13.0).
# Use a fresh environment: this path pins torch 2.13.0+cu130 (pulled by vLLM), which differs from the
# sentence-transformers runtime at the repository root (torch 2.11.0).
#
# System packages needed first (Debian/Ubuntu): gcc g++ (Triton compiles its launcher), libgl1 libglib2.0-0 (OpenCV), ffmpeg.
#   sudo apt-get install -y gcc g++ libgl1 libglib2.0-0 ffmpeg
#
# 1) every Python package from requirements.lock.txt with --no-deps (vLLM is the upstream nightly wheel of one commit, no fork)
# 2) the eg2q plugin from ./eg2q (registers itself through vLLM's general-plugin entry point)
# 3) pip check, then assert the versions that were verified
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
VLLM_COMMIT="a8260cbc5c020d250b3964141cde8fd839152714"
VLLM_VER="0.31.1rc1.dev48+ga8260cbc5"
TORCH="2.13.0+cu130"
TRANSFORMERS="5.19.0"

"$PY" -m pip install --no-cache-dir --no-deps -r "$HERE/requirements.lock.txt" \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}" --extra-index-url "https://download.pytorch.org/whl/cu130"
"$PY" -m pip install --no-cache-dir --no-deps "$HERE/eg2q"
"$PY" -m pip check
"$PY" - <<EOF
import importlib.metadata as md, torch, transformers, vllm
assert vllm.__version__ == "${VLLM_VER}", vllm.__version__
assert torch.__version__ == "${TORCH}", torch.__version__
assert transformers.__version__ == "${TRANSFORMERS}", transformers.__version__
eps = [e.name for e in md.entry_points(group="vllm.general_plugins")]
assert "eg2q" in eps, eps
print("eg2-l2v1 vllm install OK: vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "transformers", transformers.__version__, "eg2q", md.version("eg2q"))
EOF
