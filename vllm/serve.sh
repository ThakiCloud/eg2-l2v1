#!/usr/bin/env bash
# Start an OpenAI-compatible embedding server for ThakiCloud/eg2-hybrid-w4-media with the eg2q plugin.
# Downloads the checkpoint at the verified revision (unchanged), points the plugin at it, and runs vLLM's api_server
# with the exact arguments used in the verification run.
#
#   bash vllm/serve.sh                    # port 8000, served model name "eg2"
#   PORT=8012 GPU_MEM_UTIL=0.4 bash vllm/serve.sh
#   MODEL_DIR=/path/to/snapshot bash vllm/serve.sh   # skip the download
set -euo pipefail
PY="${PYTHON:-python3}"
REPO="ThakiCloud/eg2-hybrid-w4-media"
REVISION="${REVISION:-bc3574a285a2779ac0686cc149342f30fbbf8f5b}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
SERVED_NAME="${SERVED_NAME:-eg2}"

if [ -z "${MODEL_DIR:-}" ]; then
  MODEL_DIR="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', revision='$REVISION'))")"
fi
export EG2Q_CKPT="$MODEL_DIR"                 # the plugin reads the quantized-layer list from this checkpoint header
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

HF_OVERRIDES='{"matryoshka_dimensions": [768, 512, 256, 128], "embedding_size": 768, "quantization_config": {"quant_method": "eg2q", "bits": 4, "group_size": 64, "mode": "stream"}}'
MM_KWARGS='{"audio_seq_length": 750}'        # 750 frames = 30 s, the length the official processor uses

set -x
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
  --runner pooling --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS" --port "$PORT" \
  --quantization eg2q --hf-overrides "$HF_OVERRIDES" --mm-processor-kwargs "$MM_KWARGS" "$@"
