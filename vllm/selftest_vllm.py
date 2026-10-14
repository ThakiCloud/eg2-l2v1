#!/usr/bin/env python3
"""Self-test for the vLLM path. Exit 0 = PASS.

    python vllm/selftest_vllm.py --out vllm_result.json

What it checks (every check is computed, not asserted by hand):
  1. checkpoint   ThakiCloud/eg2-hybrid-w4-media is downloaded at the verified revision and is eg2-quant-v1 with 247 4-bit layers.
  2. offline load vllm.LLM with the eg2q plugin loads all 247 4-bit layers by checkpoint key; nothing else is routed to the plugin.
  3. bit identity every BF16 weight the engine reconstructs equals the checkpoint's own official loader (eg2_quant.py), by sha256.
  4. offline forward  text, image, audio and video embeddings are 768-d, finite and unit-norm.
  5. serving      serve.sh starts the OpenAI-compatible server; one request per modality; 32 concurrent requests (16 in flight);
                  the server process also loaded all 247 layers through the plugin; vLLM's reported weight memory is recorded.
GPU support: verified on NVIDIA B200 only. On other GPUs the test still runs and reports the device, but PASS there is not a
support claim.
"""
import argparse, hashlib, json, math, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO, REVISION = "ThakiCloud/eg2-hybrid-w4-media", "bc3574a285a2779ac0686cc149342f30fbbf8f5b"
N_QUANT = 247
SR = 16000
WEIGHT_GIB_MAX = 0.85            # W4 resident weights measured 0.80 GiB (BF16 reference 1.42 GiB)
log = lambda *a: print(*a, flush=True)


# ------------------------------------------------------------------ inputs (synthetic, generated locally)
def make_inputs(d):
    import numpy as np
    from PIL import Image
    import soundfile as sf
    import cv2
    rng = np.random.default_rng(20261014)
    texts = ["task: search result | query: how do heat pumps work in cold climates?",
             "title: none | text: District heating networks move waste heat from power plants through insulated pipes. " * 20,
             "task: search result | query: 김치는 왜 시간이 지나면 신맛이 강해지나요?",
             "title: none | text: 판소리는 소리꾼 한 명과 고수 한 명이 긴 이야기를 노래와 말로 엮는다. " * 15]
    imgs, auds, vids = [], [], []
    for k, (w, h) in enumerate([(256, 256), (640, 360), (360, 640), (1024, 768)]):
        yy, xx = np.mgrid[0:h, 0:w]
        a = np.zeros((h, w, 3), np.uint8)
        a[..., 0] = (xx * (k + 2)) % 256; a[..., 1] = (yy * (k + 5)) % 256; a[..., 2] = int(rng.integers(0, 255))
        p = os.path.join(d, f"img{k}.png"); Image.fromarray(a).save(p); imgs.append(p)
    for k, (f0, sec) in enumerate([(440, 3), (880, 12), (220, 29), (330, 36)]):
        t = np.arange(int(sec * SR)) / SR
        a = (0.3 * np.sin(2 * np.pi * f0 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
        p = os.path.join(d, f"aud{k}.wav"); sf.write(p, a, SR, subtype="PCM_16"); auds.append(p)
    for k, (sec, fps, w, h) in enumerate([(3, 10, 320, 240), (8, 12, 640, 360)]):
        p = os.path.join(d, f"vid{k}.mp4")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        yy, xx = np.mgrid[0:h, 0:w]
        for i in range(int(sec * fps)):
            f = np.zeros((h, w, 3), np.uint8)
            f[..., 0] = (xx * 2 + i * 7) % 256; f[..., 1] = (yy + i * 3) % 256; f[..., 2] = (xx + yy + i * 5) % 256
            vw.write(f)
        vw.release()
        if not os.path.getsize(p):
            raise RuntimeError(f"video writer produced an empty file: {p}")
        vids.append(p)
    return {"text": texts, "image": imgs, "audio": auds, "video": vids}


def emb_ok(e, dim=768):
    return len(e) == dim and all(math.isfinite(x) for x in e) and abs(math.sqrt(sum(x * x for x in e)) - 1.0) < 1e-3


# ------------------------------------------------------------------ load report (written by the plugin inside the engine process)
def read_ckpt_header(path):
    import struct
    with open(os.path.join(path, "model.safetensors"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h.pop("__metadata__", {}), h


def load_report(log_dir, shapes):
    want = set(shapes)
    got, unloaded, routed, sha = set(), [], {}, {}
    for fn in sorted(os.listdir(log_dir)):
        if not fn.startswith("state_"):
            continue
        s = json.load(open(os.path.join(log_dir, fn)))
        got |= {s["matched"][p] + ".weight" for p in s["quantized"] if p in s.get("matched", {})}
        unloaded += s["unloaded"]
        for k, v in s["routed"].items():
            routed[k] = max(routed.get(k, 0), v)
        sha.update({k + ".weight": v for k, v in (s.get("sha") or {}).items()})
    return {"ckpt_quantized": len(want), "loaded_quantized": len(got), "missing": sorted(want - got)[:10],
            "extra": sorted(got - want)[:10], "unloaded": unloaded[:10], "routed": routed,
            "ok": got == want and not unloaded and len(got) == N_QUANT}, sha


# ------------------------------------------------------------------ phase: offline (runs in its own process so the GPU is released)
def phase_offline(model_dir, inputs_json, out_json):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import numpy as np
    from PIL import Image
    import soundfile as sf
    from transformers.video_utils import load_video
    from vllm import LLM
    from offline_example import llm_kwargs
    I = json.load(open(inputs_json))
    t = time.time()
    llm = LLM(**llm_kwargs(model_dir))
    load_sec = round(time.time() - t, 1)

    def audio(p):
        a, sr = sf.read(p, dtype="float32")
        assert sr == SR, sr
        return a
    reqs = {"text": I["text"],
            "image": [{"prompt": "Describe this image: <|image|>", "multi_modal_data": {"image": Image.open(p).convert("RGB")}} for p in I["image"]],
            "audio": [{"prompt": "Describe this audio: <|audio|>", "multi_modal_data": {"audio": (audio(p), SR)}} for p in I["audio"]],
            "video": [{"prompt": "Describe this video: <|video|>", "multi_modal_data": {"video": load_video(p, backend="opencv")}} for p in I["video"]]}
    res = {"load_sec": load_sec}
    for m, x in reqs.items():
        try:
            e = [list(map(float, o.outputs.embedding)) for o in llm.embed(x)]
            res[m] = {"n": len(e), "dim": sorted({len(v) for v in e}), "ok": all(emb_ok(v) for v in e)}
        except Exception as ex:
            res[m] = {"ok": False, "error": repr(ex)[:500]}
    json.dump(res, open(out_json, "w"))


# ------------------------------------------------------------------ phase: serving
def phase_serve(model_dir, inputs, work, port):
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    import client_example as C
    logd = os.path.join(work, "eg2q_srv"); os.makedirs(logd, exist_ok=True)
    env = dict(os.environ, MODEL_DIR=model_dir, PORT=str(port), EG2Q_LOG=logd, PYTHON=sys.executable)
    env.pop("EG2Q_HASH", None)
    slog = os.path.join(work, "server.log")
    srv = subprocess.Popen(["bash", os.path.join(HERE, "serve.sh")], env=env, stdout=open(slog, "w"), stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    res = {"health": False}
    t0 = time.time()
    try:
        while time.time() - t0 < 900:
            try:
                urllib.request.urlopen(url + "/health", timeout=2); res["health"] = True; break
            except Exception:
                if srv.poll() is not None:
                    break
                time.sleep(2)
        res["startup_sec"] = round(time.time() - t0, 1)
        if not res["health"]:
            res["server_log_tail"] = open(slog).read()[-3000:]
            return res, logd
        bodies = {"text": lambda k: C.text_body(inputs["text"][k % len(inputs["text"])]),
                  "image": lambda k: C.image_body(inputs["image"][k % len(inputs["image"])]),
                  "audio": lambda k: C.audio_body(inputs["audio"][k % len(inputs["audio"])]),
                  "video": lambda k: C.video_body(inputs["video"][k % len(inputs["video"])])}
        single = {}
        for m, f in bodies.items():
            try:
                e = C.post(url, f(0)); single[m] = {"dim": len(e), "ok": emb_ok(e)}
            except Exception as ex:
                single[m] = {"ok": False, "error": repr(ex)[:500]}
        res["single"] = single
        jobs = [(m, k) for m in bodies for k in range(8)]

        def one(mk):
            m, k = mk
            try:
                return m, emb_ok(C.post(url, bodies[m](k)))
            except Exception as ex:
                return m, repr(ex)[:300]
        t = time.time()
        with ThreadPoolExecutor(16) as ex:
            rr = list(ex.map(one, jobs))
        res["concurrent"] = {"requests": len(rr), "in_flight": 16, "sec": round(time.time() - t, 2),
                             "ok": sum(1 for _, v in rr if v is True), "fail": [v for _, v in rr if v is not True][:5],
                             "by_modality": {m: sum(1 for mm, v in rr if mm == m and v is True) for m in bodies}}
    finally:
        srv.terminate()
        try:
            srv.wait(90)
        except Exception:
            srv.kill()
    txt = open(slog).read()
    m = re.findall(r"Model loading took ([0-9.]+) ?GiB", txt)
    res["weight_memory_gib"] = float(m[-1]) if m else None
    sc = re.findall(r"\+ exec .*", txt)
    res["serve_command"] = sc[-1][7:] if sc else None
    return res, logd


# ------------------------------------------------------------------ main
def env_info():
    import torch, transformers, vllm
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__, "transformers": transformers.__version__}
    try:
        info["eg2q"] = __import__("importlib.metadata").metadata.version("eg2q")
    except Exception as ex:
        info["eg2q"] = f"not installed ({ex!r})"
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    try:
        info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        info["driver"] = None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vllm_selftest.json")
    ap.add_argument("--model-dir", help="local snapshot of the checkpoint (default: download the verified revision)")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--skip-serve", action="store_true")
    ap.add_argument("--phase"); ap.add_argument("--inputs"); ap.add_argument("--phase-out")
    a = ap.parse_args()
    if a.phase == "offline":
        return phase_offline(a.model_dir, a.inputs, a.phase_out)

    R = {"selftest": "eg2-l2v1 vllm 0.2.0", "env": env_info(), "checks": {}}
    work = tempfile.mkdtemp(prefix="eg2q_selftest_")
    try:
        # 1. checkpoint
        from huggingface_hub import HfApi, snapshot_download
        path = a.model_dir or snapshot_download(REPO, revision=REVISION)
        if not a.model_dir:
            R["model"] = {"repo": REPO, "revision": REVISION, "resolved": HfApi().model_info(REPO, revision=REVISION).sha}
        meta, hdr = read_ckpt_header(path)
        shapes = json.loads(meta["shapes"])
        h = hashlib.sha256()
        with open(os.path.join(path, "model.safetensors"), "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        R["model"] = dict(R.get("model", {}), path=path, format=meta.get("format"), bits=meta.get("bits"), group_size=meta.get("g"),
                          quantized_layers=len(shapes), safetensors_sha256=h.hexdigest())
        R["checks"]["checkpoint"] = meta.get("format") == "eg2-quant-v1" and len(shapes) == N_QUANT and \
            (a.model_dir or R["model"]["resolved"] == REVISION)
        log("[1] checkpoint", R["checks"]["checkpoint"], R["model"]["safetensors_sha256"])

        # 2-4. offline load, bit identity, forward
        inputs = make_inputs(work); ij = os.path.join(work, "inputs.json"); json.dump(inputs, open(ij, "w"))
        offd = os.path.join(work, "eg2q_offline"); po = os.path.join(work, "offline.json")
        env = dict(os.environ, EG2Q_LOG=offd, EG2Q_HASH="1", EG2Q_CKPT=path)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", "offline", "--model-dir", path, "--inputs", ij,
                            "--phase-out", po], env=env, capture_output=True, text=True, timeout=3600)
        if r.returncode or not os.path.exists(po):
            R["offline"] = {"rc": r.returncode, "log_tail": (r.stdout + r.stderr)[-3000:]}
            raise SystemExit("offline phase failed")
        R["offline"] = json.load(open(po))
        rep, eng_sha = load_report(offd, shapes)
        R["offline"]["load"] = rep
        R["checks"]["offline_load_247"] = rep["ok"]
        R["checks"]["offline_forward_4_modalities"] = all(R["offline"][m].get("ok") for m in ("text", "image", "audio", "video"))
        log("[2] offline load", rep["ok"], rep["routed"]); log("[4] offline forward", {m: R["offline"][m] for m in ("text", "image", "audio", "video")})

        import torch
        sys.path.insert(0, path)
        import eg2_quant                                     # the checkpoint's own official loader, unchanged
        sd = eg2_quant.load_state_dict(path, device="cpu")
        ref = {k: hashlib.sha256(sd[k].view(torch.int16).numpy().tobytes()).hexdigest() for k in shapes}
        del sd
        same = sum(1 for k in shapes if eng_sha.get(k) == ref[k])
        R["bit_identity"] = {"layers": len(shapes), "engine_hashed": len(eng_sha), "identical": same,
                             "first_diff": [k for k in shapes if eng_sha.get(k) != ref[k]][:5]}
        R["checks"]["dequant_bit_identical_247"] = same == len(shapes) == N_QUANT
        log("[3] bit identity", R["bit_identity"])

        # 5. serving
        if not a.skip_serve:
            res, logd = phase_serve(path, inputs, work, a.port)
            if res.get("health"):
                res["load"], _ = load_report(logd, shapes)
            R["serving"] = res
            R["checks"]["serving"] = bool(res.get("health")) and all(res["single"][m]["ok"] for m in res["single"]) and \
                len(res["single"]) == 4 and res["concurrent"]["ok"] == res["concurrent"]["requests"] == 32 and res["load"]["ok"]
            R["checks"]["serving_weight_memory_w4"] = res.get("weight_memory_gib") is not None and res["weight_memory_gib"] <= WEIGHT_GIB_MAX
            log("[5] serving", {k: res.get(k) for k in ("health", "single", "concurrent", "weight_memory_gib")})
    except SystemExit as ex:
        R["error"] = str(ex)
    except Exception as ex:
        import traceback
        R["error"] = traceback.format_exc()[-3000:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    R["pass"] = bool(R["checks"]) and all(R["checks"].values()) and "error" not in R
    R["verdict"] = "PASS" if R["pass"] else "FAIL"
    json.dump(R, open(a.out, "w"), indent=1)
    log("SELFTEST_JSON " + json.dumps(R))
    log(f"VERDICT: {R['verdict']} {R['checks']}")
    if R["env"].get("gpu") and "B200" not in R["env"]["gpu"]:
        log(f"note: verified GPU is NVIDIA B200; this run used {R['env']['gpu']}")
    sys.exit(0 if R["pass"] else 1)


if __name__ == "__main__":
    main()
