#!/usr/bin/env bash
# eg2-l2v1 v0.1.0 — public install path (Linux x86_64, Python 3.12, NVIDIA driver supporting CUDA 13.0).
# 1) PyTorch 2.11.0 + torchvision 0.26.0 for CUDA 13.0 from the official PyTorch index
#    (torchvision is required: transformers 5.19 imports it for EmbeddingGemma 2 image/video preprocessing).
# 2) Locked Python dependencies with --no-deps, so pip cannot replace torch.
# 3) pip check, then assert the torch build did not change.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
TORCH="2.11.0+cu130"
TORCHVISION="0.26.0+cu130"
INDEX="https://download.pytorch.org/whl/cu130"

"$PY" -m pip install --no-cache-dir "torch==${TORCH}" "torchvision==${TORCHVISION}" --index-url "$INDEX"
"$PY" -m pip install --no-cache-dir --no-deps -r "$HERE/requirements.lock.txt"
"$PY" -m pip check
"$PY" - <<EOF
import torch, torchvision
assert torch.__version__ == "${TORCH}", torch.__version__
assert torchvision.__version__ == "${TORCHVISION}", torchvision.__version__
print("eg2-l2v1 install OK: torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.version.cuda)
EOF
