#!/usr/bin/env python3
"""Offline (in-process) embedding with vllm.LLM and the eg2q plugin.

    python vllm/offline_example.py
"""
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import numpy as np
from huggingface_hub import snapshot_download

REPO, REVISION = "ThakiCloud/eg2-hybrid-w4-media", "bc3574a285a2779ac0686cc149342f30fbbf8f5b"
SR = 16000


def llm_kwargs(model_dir, **extra):
    """The engine arguments used in the verification run. EG2Q_CKPT must name the same directory as `model`."""
    os.environ["EG2Q_CKPT"] = model_dir
    hf = {"matryoshka_dimensions": [768, 512, 256, 128], "embedding_size": 768,
          "quantization_config": {"quant_method": "eg2q", "bits": 4, "group_size": 64, "mode": "stream"}}
    kw = dict(model=model_dir, runner="pooling", max_model_len=8192, dtype="bfloat16", gpu_memory_utilization=0.5, seed=0,
              quantization="eg2q", hf_overrides=hf, mm_processor_kwargs={"audio_seq_length": 750})
    kw.update(extra)
    return kw


def main():
    from PIL import Image
    from vllm import LLM
    path = snapshot_download(REPO, revision=REVISION)
    llm = LLM(**llm_kwargs(path))
    query = "task: search result | query: a dog barking in the distance"
    image = Image.fromarray((np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8))
    audio = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * SR) / SR)).astype(np.float32)     # replace with 16 kHz mono audio
    outs = llm.embed([query,
                      {"prompt": "Describe this image: <|image|>", "multi_modal_data": {"image": image}},
                      {"prompt": "Describe this audio: <|audio|>", "multi_modal_data": {"audio": (audio, SR)}}])
    e = np.asarray([o.outputs.embedding for o in outs])
    print("shape", e.shape, "norms", np.linalg.norm(e, axis=1).round(4))
    print("cos(query, image) =", float(e[0] @ e[1]), " cos(query, audio) =", float(e[0] @ e[2]))
    # Video: {"prompt": "Describe this video: <|video|>", "multi_modal_data": {"video": transformers.video_utils.load_video(p, backend="opencv")}}


if __name__ == "__main__":
    main()
