# Changelog

## v0.2.0

- New: `vllm/` — `eg2q`, an out-of-tree vLLM plugin that serves `ThakiCloud/eg2-hybrid-w4-media` (revision `bc3574a2`) unchanged on upstream vLLM `0.31.1rc1.dev48+ga8260cbc5` (no fork, no re-quantization). Text, image, audio and video through the OpenAI-compatible `/v1/embeddings` API.
- New: `vllm/install.sh` + `vllm/requirements.lock.txt` (resolved on public indexes), `vllm/serve.sh`, `vllm/client_example.py`, `vllm/offline_example.py`, `vllm/selftest_vllm.py`, CPU unit test `vllm/tests/`.
- Measured on one NVIDIA B200 against a BF16-dequantized reference on the same vLLM path: weight-resident memory 0.80 vs 1.42 GiB (−43.7%), worst throughput cell 0.826×, accuracy min cosine 1.0 on all plugin×reference pairs (criterion ≥ 0.99999). Details and limits in README.
- Unchanged: the sentence-transformers runtime at the repository root (`eg2_l2v1.py`, `install.sh`, `selftest.py`, lock files) is byte-identical to v0.1.0.

## v0.1.0

- Low-memory CUDA runtime for EG2 Hybrid W4 inside sentence-transformers (B200 only).
