🇰🇷 한국어: [README.ko.md](./README.ko.md)

# eg2-l2v1 — low-memory runtimes for EG2 Hybrid W4 (v0.2.0, B200 only)

**v0.2.0 adds a vLLM plugin, `eg2q`, that serves ThakiCloud's EmbeddingGemma 2 hybrid W4 model ([`ThakiCloud/eg2-hybrid-w4-media`](https://huggingface.co/ThakiCloud/eg2-hybrid-w4-media)) from an unmodified upstream vLLM through its OpenAI-compatible `/v1/embeddings` API, for text, image, audio and video. The published 4-bit checkpoint is used as is: no re-quantization, no new checkpoint, no vLLM fork. On one NVIDIA B200, weight-resident GPU memory is 43.7% lower than a BF16 reference served by the same vLLM path, and throughput stays at or above 0.826× of that reference. It is not a speed-up and not a native INT4 GEMM.**

This repository contains **code only** — it does not redistribute model weights or evaluation data. Both runtimes download the published checkpoint at a pinned revision and use it unchanged; the model, its card and its official loader (`eg2_quant.py`) are not modified.

| Path | What it is | Since |
|---|---|---|
| [`vllm/`](./vllm) | `eg2q` vLLM plugin + server script + examples + self-test | **v0.2.0** |
| repository root | sentence-transformers runtime `eg2_l2v1` (unchanged) | v0.1.0 |

The two paths pin different PyTorch builds (vLLM path: 2.13.0+cu130 from the vLLM wheel; root runtime: 2.11.0+cu130). Install each in its own environment.

## vLLM plugin (`vllm/`, new in v0.2.0)

### How it works

The plugin registers a vLLM quantization config named `eg2q` through vLLM's standard `vllm.general_plugins` entry point. The checkpoint stores 247 vision/audio-tower Linear layers as 4-bit codes with fp16 scale and zero-point (group size 64). For exactly those 247 layers — read from the checkpoint header, not guessed from layer type — the 4-bit codes stay resident on the GPU and each forward call dequantizes that one layer to BF16 right before the ordinary BF16 GEMM. Every other Linear (the language model and the BF16 projections) uses vLLM's unquantized path. Nothing else in vLLM is patched.

### Measured (NVIDIA B200 ×1)

Reference arm: the same checkpoint dequantized to BF16 once with its official loader and served by the same vLLM build with the same engine settings. Plugin arm: the 4-bit checkpoint served with `eg2q`. Environment: vLLM `0.31.1rc1.dev48+ga8260cbc5` (upstream nightly wheel of commit `a8260cb`), torch 2.13.0+cu130, transformers 5.19.0, model revision `bc3574a285a2779ac0686cc149342f30fbbf8f5b`. Every criterion below was registered before the measurement and was not changed afterwards.

| Item | Result | Criterion |
|---|---|---|
| 4-bit layers loaded by checkpoint key | **247 / 247** (vision 113 · audio 134); 146 language-model Linears stay BF16 | all 247 |
| Dequantized weights vs the official loader | **bit-identical**, 247 / 247 by sha256 | bit-identical |
| Weight-resident GPU memory (vLLM "Model loading took") | **0.80 GiB** vs 1.42 GiB reference → **0.563× (−43.7%)** | ≤ 0.70× |
| GPU memory at the end of loading (whole process) | 1,842 MiB vs 2,412 MiB | recorded |
| Throughput vs reference, worst cell over 3 valid rounds | **0.826×** (audio, batch 64) | ≥ 0.80× at batch 16 and 64 |
| Accuracy vs reference | min cosine **1.0** on every plugin×reference pair (see below) | ≥ 0.99999 |
| OpenAI-compatible serving | text · image · audio · video single requests OK; **32 / 32** concurrent requests (16 in flight) OK; the server loaded all 247 layers through the plugin | all four modalities |

Throughput relative to the reference, median of the 3 valid rounds (minimum in parentheses):

| Modality | batch 16 | batch 64 |
|---|---|---|
| text | 0.993 (0.987) | 0.998 (0.995) |
| image | 0.932 (0.923) | 0.992 (0.941) |
| audio | 1.082 (0.898) | 1.040 (**0.826**) |
| video | 1.007 (0.994) | 0.989 (0.961) |

A round was valid only if a BF16-vs-BF16 control in the same round stayed within 0.90–1.10 on every cell; 3 of 5 rounds were valid and the other 2 were discarded by that pre-registered rule. Ratios above 1.0 are inside that control noise and are **not** a speed-up claim — the plugin runs the same BF16 GEMM after an extra dequantization step.

**About the memory number.** −43.7% is the weight-resident memory vLLM reports after loading. It is not the reduction of total server VRAM: vLLM reserves memory for activations and caches up to `--gpu-memory-utilization`, so the server's footprint is set by that flag, not by the weights.

### What the accuracy result shows — and what it does not

*Shown:* on the same vLLM path, serving the 4-bit weights resident meets the pre-registered accuracy criterion against the BF16-dequantized reference. 32 items per modality (128 in total) were embedded by three plugin runs and three reference runs; all nine plugin×reference pairs had min cosine 1.0 at the reported precision in every modality, against a criterion of 0.99999. Three further reference runs (negative control) also passed, and a deliberately perturbed plugin run (±2⁻⁸ relative error injected into the 4-bit layers only) was detected for image, audio and video. Text did not move under that perturbation, as expected: the language model is not quantized, so the injected error never reaches the text path.

*Not shown:* that the plugin produces exactly the same output under every execution condition. Outputs are not bit-identical across processes even for the reference itself. For 4 of the 128 items (2 text, 2 video), the plugin's deviation from the reference was slightly larger than the largest run-to-run deviation observed among six reference runs. The size is at float32 last-digit level (max absolute component difference ≤ 7.3 × 10⁻⁸, 1 − cos ≈ 10⁻¹⁵), but it is reported here rather than rounded away. Earlier diagnostic runs also saw the reference itself drift intermittently on one 3.85 s audio item (cosine down to 0.99978 between reference runs); that drift did not reproduce in this measurement, which means it was not observed, not that it is fixed.

### Install

Linux x86_64, Python 3.12, NVIDIA driver supporting CUDA 13.0. Use a fresh environment.

```bash
git clone https://github.com/ThakiCloud/eg2-l2v1.git && cd eg2-l2v1
sudo apt-get install -y gcc g++ libgl1 libglib2.0-0 ffmpeg     # Triton launcher build, OpenCV, decoders
bash vllm/install.sh          # locked deps with --no-deps (vLLM wheel of commit a8260cb), the eg2q plugin, pip check
python vllm/selftest_vllm.py --out vllm_result.json            # exit 0 = PASS (about 10 minutes on B200)
```

`vllm/requirements.lock.txt` was resolved from `vllm/requirements.in` on public indexes only (PyPI, `wheels.vllm.ai/<commit>`, `download.pytorch.org/whl/cu130`). vLLM is installed from the upstream nightly wheel index for commit `a8260cbc5c020d250b3964141cde8fd839152714`; if that index is ever pruned, the same commit can be built from source.

### Serve

```bash
bash vllm/serve.sh            # downloads the pinned revision, sets EG2Q_CKPT, starts the API server on :8000
python vllm/client_example.py --url http://127.0.0.1:8000 --image cat.png --audio clip.wav --video clip.mp4
```

`serve.sh` runs `python -m vllm.entrypoints.openai.api_server` with exactly the arguments that were verified (`--runner pooling --dtype bfloat16 --max-model-len 8192 --quantization eg2q`, Matryoshka dimensions 768/512/256/128, `audio_seq_length=750`). The plugin needs `EG2Q_CKPT` to point at the same local checkpoint directory that is passed as the model; `serve.sh` sets it.

Request formats (`/v1/embeddings`):

```python
{"model": "eg2", "input": ["task: search result | query: how do heat pumps work?"]}                      # text
{"model": "eg2", "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},                          # image
    {"type": "text", "text": "Describe this image:"}]}]}
# audio: {"type": "input_audio", "input_audio": {"data": "<base64 wav>", "format": "wav"}} + "Describe this audio:"
# video: {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,<...>"}} + "Describe this video:"
```

Audio should be 16 kHz mono; clips longer than 30 s are truncated to 30 s by the model's processor. In-process use with `vllm.LLM` is shown in [`vllm/offline_example.py`](./vllm/offline_example.py).

### Self-test

`vllm/selftest_vllm.py` downloads the checkpoint at the verified revision and checks, by computation: the checkpoint format and its 247 4-bit layers; that an offline `vllm.LLM` loads all 247 through the plugin; that every BF16 weight the engine reconstructs is bit-identical (sha256) to the checkpoint's official `eg2_quant.py`; 768-d finite unit-norm embeddings for text, image, audio and video; and that `serve.sh` serves one request per modality plus 32 concurrent requests with the plugin, recording vLLM's weight memory. A CPU unit test (`pytest vllm/tests`) checks the plugin's dequantization against the official loader without a GPU.

### Independent install check (public sources only)

A clean NVIDIA B200 container with no access to internal registries, object storage or shared directories ran exactly the steps above: public base image `python:3.12-slim-bookworm`, system packages from Debian, `requirements.in` resolved on public indexes into the committed lock (199 packages: 177 from PyPI, 15 from pypi.nvidia.com, 6 from download.pytorch.org, vLLM from wheels.vllm.ai), `vllm/install.sh` (`pip check` clean), then `vllm/selftest_vllm.py` with the checkpoint downloaded from Hugging Face.

| Check | Result |
|---|---|
| Environment | NVIDIA B200 (SM 10.0) · driver 580.65.06 · Python 3.12.15 · vLLM 0.31.1rc1.dev48+ga8260cbc5 · torch 2.13.0+cu130 · transformers 5.19.0 · eg2q 0.2.0 |
| Checkpoint | revision `bc3574a2…` resolved, eg2-quant-v1, 247 4-bit layers |
| Offline load | 247 / 247 through the plugin (vision 113 · audio 134); 146 language-model Linears unquantized |
| Bit identity vs official loader | 247 / 247 identical (sha256) |
| Offline embeddings | text 4 · image 4 · audio 4 (3–36 s) · video 2 — all 768-d, finite, unit-norm |
| Server (`serve.sh`) | healthy in 70 s; one request per modality OK; 32 / 32 concurrent requests OK (16 in flight, 0.98 s); server loaded 247 / 247 through the plugin |
| Weight memory reported by vLLM | 0.80 GiB (same as the measurement above) |
| Verdict | **PASS** |

The same check was repeated on the exact release tree, installing the committed `requirements.lock.txt` without re-resolving (`pip freeze` matched the lock on all 199 packages): **PASS** on every item, with the same results (32 / 32 concurrent requests in 1.15 s, weight memory 0.80 GiB).

## sentence-transformers runtime (repository root, unchanged since v0.1.0)

**L2V1 keeps the 4-bit vision/audio weights resident inside sentence-transformers. On NVIDIA B200 it reduces resident GPU memory by 41.8% compared with BF16 while preserving bit-identical embeddings relative to the published reference loader. Audio batch-16 throughput is 82.2% of BF16 in the public reproduction environment. It does not provide a speedup or native INT4 GEMM.** The official loader dequantizes every 4-bit vision/audio Linear to BF16 when the model loads; this runtime keeps the 4-bit codes resident and dequantizes each Linear's weight right before its BF16 matmul.

### Measured (NVIDIA B200 ×1)

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

#### Public reproduction

Everything installed from public sources on one B200: base image `python:3.12-slim-bookworm`, PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130 from the official PyTorch index via `install.sh`, the lock file with `--no-deps`, `pip check` clean, both models downloaded from Hugging Face at pinned revisions.

| | |
|---|---|
| `selftest.py` | PASS — embeddings bit-identical to the official loader on all four modalities (max abs difference 0) |
| Resident / peak memory | 0.874 GB (0.582× BF16) / 1.797 GB |
| Throughput vs BF16, batch 16 / 64 | text 1.002 / 1.005 · image 0.954 / 0.987 · audio **0.822** / 0.865 · video 0.988 / 0.997 |
| audio @ batch 16, per round | 0.832 · 0.822 · 0.861 |
| Measurement control (BF16 vs BF16) | 1.000–1.029 on every cell |

### Verified environment

| | |
|---|---|
| GPU | NVIDIA B200 ×1 (SM 10.0) |
| NVIDIA driver | 580.65.06 (CUDA 13.0) |
| OS / arch | Linux x86_64 |
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 · torchvision 0.26.0+cu130 (official PyTorch index) |
| transformers / sentence-transformers | 5.19.0 / 6.1.0 (full list in `requirements.lock.txt`) |
| Model | `ThakiCloud/eg2-hybrid-w4-media` @ `bc3574a285a2779ac0686cc149342f30fbbf8f5b` |

### Install

```bash
git clone https://github.com/ThakiCloud/eg2-l2v1.git && cd eg2-l2v1
bash install.sh          # torch + torchvision (cu130) from download.pytorch.org, then the lock with --no-deps, then pip check
python selftest.py --out result.json     # exit 0 = PASS
```

`install.sh` installs `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130` from the official PyTorch index, then `requirements.lock.txt` with `--no-deps` so pip cannot replace torch, runs `pip check`, and asserts the torch build afterwards. torchvision is required: transformers 5.19 uses it for EmbeddingGemma 2 image and video preprocessing. The lock was resolved inside the verified container from `requirements.in` under `constraints.txt`.

### Usage

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

### Self-test

`selftest.py` records the snapshot commit, file sha256 and environment versions, measures resident and peak memory, checks the import environment (which torch is loaded, that a CUDA GEMM runs, that installed packages match the lock), then compares embeddings with the official loader on synthetic text/image/audio/video inputs (min cosine ≥ 0.99999 after fp64 renormalization).

## Limits (both paths)

- **Verified GPU: NVIDIA B200 (SM 10.0) only.** Support for other GPUs is not claimed. (For the root runtime, H200 met the median criteria but one of three rounds measured audio@16 at 0.796, so H200 is not listed.)
- **The memory figure is weight-resident memory**, not total server VRAM (see "About the memory number").
- **Not a speed-up.** Both paths run the same BF16 GEMM after dequantizing; there is no native INT4 kernel. If GPU memory is not your constraint, BF16 weights are as fast or faster.
- **The accuracy reference is the same vLLM path with the 4-bit weights dequantized to BF16** (vLLM path) or the official loader (root runtime). It is not a comparison with another serving stack.
- **Open issue (G1):** embeddings from vLLM and from the sentence-transformers pipeline are not yet shown to be equivalent for image and audio inputs, even for the BF16 model. This release does not resolve it; do not mix vectors from the two stacks in one index without checking.
- 4-bit checkpoint `ThakiCloud/eg2-hybrid-w4-media` at the pinned revision only. No CPU path.
- No model weights or evaluation data are distributed in this repository.

## License

Apache-2.0, same as the model. See [NOTICE](./NOTICE).
