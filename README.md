🇰🇷 한국어: [README.ko.md](./README.ko.md)

# eg2-l2v1 — low-memory CUDA runtime for EG2 Hybrid W4 (v0.1.0, B200 only)

**L2V1 is a low-memory CUDA runtime for ThakiCloud's EmbeddingGemma 2 hybrid W4 model ([`ThakiCloud/eg2-hybrid-w4-media`](https://huggingface.co/ThakiCloud/eg2-hybrid-w4-media)). On NVIDIA B200, it reduces resident GPU memory by 41.8% compared with BF16 while preserving bit-identical embeddings relative to the published reference loader. Audio batch-16 throughput is 82.2% of BF16 in the public reproduction environment. It does not provide a speedup or native INT4 GEMM.**

This repository contains **code only** — it does not redistribute any weights. It downloads the published checkpoint [`ThakiCloud/eg2-hybrid-w4-media`](https://huggingface.co/ThakiCloud/eg2-hybrid-w4-media) at a pinned revision and uses it unchanged; the model, its card and its official loader (`eg2_quant.py`) are not modified.

## What it does differently

The official loader dequantizes every 4-bit vision/audio Linear to BF16 when the model loads. This runtime keeps the 4-bit codes resident and dequantizes each Linear's weight right before its BF16 matmul. **This is not a native INT4 GEMM** — the matmul is the same BF16 GEMM; only weight storage changes. You trade some throughput for memory.

## Measured (NVIDIA B200 ×1)

Environment: torch 2.11.0+cu130 · transformers 5.19.0 · sentence-transformers 6.1.0 · driver 580.65.06 · model revision `bc3574a285a2779ac0686cc149342f30fbbf8f5b`. Every throughput figure is the median of 3 interleaved rounds against BF16 in the same process.

| | |
|---|---|
| Resident GPU memory after load | **0.874 GB** (BF16 1.502 GB → **0.582×**) |
| Peak GPU memory while loading | **1.797 GB** — reserve this much headroom |
| Accuracy vs official loader | synthetic inputs: min cosine ≥ 0.9999998 (all modalities) · real audio: bit-identical |

Throughput relative to BF16, synthetic inputs (3 s audio clips), installed from the lock file:

| Modality | batch 16 | batch 64 |
|---|---|---|
| text | 1.014 | 1.002 |
| image | 0.956 | 0.988 |
| audio | **0.825** (rounds 0.816 · 0.843 · 0.825) | 0.875 |
| video | 0.989 | 0.998 |

Throughput relative to BF16, real recordings (FSD50K dev, 44.1 kHz decoded and resampled to 16 kHz, 64 clips per bucket):

| Clip length | batch 16 | batch 64 |
|---|---|---|
| ≤ 4 s | 0.831 | 0.908 |
| 7–12 s | 0.884 | 0.960 |
| 27–30 s | 0.944 | 0.984 |
| > 30 s (truncated to 30 s by the processor) | 0.944 | 0.984 |

Short audio clips are the slowest case. Forward-pass activation memory is the same as BF16 (e.g. +6.2 GB for 64 clips of 30 s).

### Public reproduction

Everything installed from public sources on one B200: base image `python:3.12-slim-bookworm`, PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130 from the official PyTorch index via `install.sh`, the lock file with `--no-deps`, `pip check` clean, both models downloaded from Hugging Face at pinned revisions.

| | |
|---|---|
| `selftest.py` | PASS — embeddings bit-identical to the official loader on all four modalities (max abs difference 0) |
| Resident / peak memory | 0.874 GB (0.582× BF16) / 1.797 GB |
| Throughput vs BF16, batch 16 / 64 | text 1.002 / 1.005 · image 0.954 / 0.987 · audio **0.822** / 0.865 · video 0.988 / 0.997 |
| audio @ batch 16, per round | 0.832 · 0.822 · 0.861 |
| Measurement control (BF16 vs BF16) | 1.000–1.029 on every cell |

## Verified environment

| | |
|---|---|
| GPU | NVIDIA B200 ×1 (SM 10.0) |
| NVIDIA driver | 580.65.06 (CUDA 13.0) |
| OS / arch | Linux x86_64 |
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 · torchvision 0.26.0+cu130 (official PyTorch index) |
| transformers / sentence-transformers | 5.19.0 / 6.1.0 (full list in `requirements.lock.txt`) |
| Model | `ThakiCloud/eg2-hybrid-w4-media` @ `bc3574a285a2779ac0686cc149342f30fbbf8f5b` |

## Install

```bash
git clone https://github.com/ThakiCloud/eg2-l2v1.git && cd eg2-l2v1
bash install.sh          # torch + torchvision (cu130) from download.pytorch.org, then the lock with --no-deps, then pip check
python selftest.py --out result.json     # exit 0 = PASS
```

`install.sh` installs `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130` from the official PyTorch index, then `requirements.lock.txt` with `--no-deps` so pip cannot replace torch, runs `pip check`, and asserts the torch build afterwards. torchvision is required: transformers 5.19 uses it for EmbeddingGemma 2 image and video preprocessing. The lock was resolved inside the verified container from `requirements.in` under `constraints.txt`.

## Usage

Audio must be mono float32 at 16 kHz; clips longer than 30 s are truncated to 30 s by the model's processor.

```python
import numpy as np
from huggingface_hub import snapshot_download
import eg2_l2v1

path = snapshot_download("ThakiCloud/eg2-hybrid-w4-media",
                         revision="bc3574a285a2779ac0686cc149342f30fbbf8f5b")
model, info = eg2_l2v1.load(path)      # CUDA only
print(info)                            # runtime version, number of 4-bit Linear modules kept resident

wav = np.zeros(16000 * 3, dtype=np.float32)          # replace with your 16 kHz mono audio
audio_emb = model.encode([{"audio": wav}], normalize_embeddings=True)
text_emb = model.encode(["a dog barking in the distance"], prompt_name="SearchQuery", normalize_embeddings=True)
print(audio_emb.shape, float(audio_emb @ text_emb.T))
```

## Self-test

`selftest.py` records the snapshot commit, file sha256 and environment versions, measures resident and peak memory, checks the import environment (which torch is loaded, that a CUDA GEMM runs, that installed packages match the lock), then compares embeddings with the official loader on synthetic text/image/audio/video inputs (min cosine ≥ 0.99999 after fp64 renormalization).

## Limits

- Verified GPU: **B200 (SM 10.0) only.** On H200 the median met the same criteria, but one of three rounds measured audio@16 at 0.796, so H200 is not listed as supported.
- 4-bit checkpoint only. No CPU path — use the official loader on CPU.
- Not a speed-up. If GPU memory is not your constraint, the official loader is faster for image and audio.

## License

Apache-2.0, same as the model.
