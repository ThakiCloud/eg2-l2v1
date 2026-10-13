#!/usr/bin/env python3
"""eg2-l2v1 self-test. Exit 0 = PASS, 1 = FAIL.

1. resolve the model snapshot (HF repo id or local dir) and record its commit + sha256 of every file we execute/load
2. record the environment (python / torch / CUDA / GPU / transformers / sentence-transformers / safetensors)
3. load the runtime FIRST in a fresh process state → resident memory after load and peak during load
4. encode deterministic synthetic text / image / audio / video examples
5. load the stock loader (eg2_quant.load_sentence_transformer) on the same snapshot → min cosine per modality
Gates: min cos ≥ 0.99999 on every modality; runtime loads with 0 errors; examples finite.
Embedding fingerprints are written so two environments can be compared byte-for-byte later.
"""
import argparse, hashlib, json, math, os, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def synth(n_img=16, n_aud=16, n_vid=4, n_txt=32):
    import numpy as np
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0); imgs = []
    for i in range(n_img):
        im = Image.new("RGB", (384, 384), tuple(int(x) for x in rng.integers(0, 255, 3))); d = ImageDraw.Draw(im)
        for _ in range(3):
            x0, y0 = (int(v) for v in rng.integers(0, 300, 2)); s = int(rng.integers(30, 80))
            d.ellipse((x0, y0, x0 + s, y0 + s), fill=tuple(int(x) for x in rng.integers(0, 255, 3)))
        imgs.append(im)
    t = np.arange(3 * 16000) / 16000
    auds = [(0.3 * np.sin(2 * math.pi * (200 + 40 * i) * t)).astype(np.float32) for i in range(n_aud)]
    vids = [[imgs[(k * 8 + j) % n_img] for j in range(8)] for k in range(n_vid)]
    txt = [f"synthetic query number {i} about a red circle and a blue square" for i in range(n_txt)]
    return {"text": txt, "image": [{"image": i} for i in imgs], "audio": [{"audio": a} for a in auds],
            "video": [{"video": v} for v in vids]}


def encode(st, X, torch):
    out = {}
    for m, xs in X.items():
        kw = {"prompt_name": "SearchQuery"} if m == "text" else {}
        with torch.inference_mode():
            # The model's final Normalize module runs in bf16. Do not take dot products of those vectors directly:
            # a bf16-normalized vector has self-dot ≈ 0.996, so identical embeddings would score cos ≈ 0.9964.
            # Take the raw outputs and renormalize in fp64 instead.
            out[m] = st.encode(xs, batch_size=16, convert_to_tensor=True, normalize_embeddings=False,
                               show_progress_bar=False, **kw).float().cpu()
    return out


def cos_min(a, b, torch):
    import torch.nn.functional as F
    return float((F.normalize(a.double(), dim=-1) * F.normalize(b.double(), dim=-1)).sum(-1).min())


def isolation(a, torch):
    """Import-environment isolation: no stray shared package dir on sys.path, where torch was loaded from, and a real CUDA GEMM."""
    site = str(Path(a.site).resolve()) if a.site else None
    tf = str(Path(torch.__file__).resolve())
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16); w = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    y = (x @ w).float(); ref = (x.float() @ w.float())
    gemm_rel = float((y - ref).norm() / ref.norm())
    out = {"sys_path": list(sys.path), "torch_file": tf, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
           "pylib_on_path": any("/tmp/pylib" in p for p in sys.path), "torch_from_site": bool(site and tf.startswith(site)),
           "torch_is_image_torch": (str(Path(a.image_torch).resolve()) == tf) if a.image_torch else None,
           "cuda_gemm_rel_err": gemm_rel}
    out["ok"] = (not out["pylib_on_path"]) and (not out["torch_from_site"]) and out["torch_is_image_torch"] is not False and gemm_rel < 1e-2 \
        and torch.__version__.startswith("2.11.0") and torch.version.cuda is not None
    return out


def dep_check(a):
    """Every installed distribution's declared requirements (markers applied) are met by the installed versions, torch included."""
    import importlib.metadata as md
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name as cn
    inst, where = {}, {}
    for d in md.distributions():                      # sys.path order: the first one found is the one that gets imported
        n = d.metadata["Name"]
        if n and cn(n) not in inst:
            inst[cn(n)] = d.version; where[cn(n)] = str(getattr(d, "_path", ""))
    lock = {}
    if a.lock and Path(a.lock).exists():
        for ln in Path(a.lock).read_text().splitlines():
            if "==" in ln and not ln.startswith("#"):
                n, v = ln.split("=="); lock[cn(n)] = v.strip()
    bad, missing = [], []
    for n in lock or inst:
        d = md.distribution(n) if n in inst else None
        if d is None:
            missing.append(n); continue
        for rq in d.requires or []:
            r = Requirement(rq)
            if r.marker and not r.marker.evaluate({"extra": ""}):
                continue
            have = inst.get(cn(r.name))
            if have is None:
                missing.append(f"{n} -> {r}")
            elif r.specifier and not r.specifier.contains(have, prereleases=True):
                bad.append(f"{n} needs {r} but has {have}")
    lock_mismatch = [f"{n}: lock {v} installed {inst.get(n)}" for n, v in lock.items() if inst.get(n) != v]
    # Whole import environment: transitive requirement closure of the runtime's top-level packages and torch
    seen, stack, closure_bad = set(), ["sentence-transformers", "transformers", "safetensors", "huggingface-hub", "pillow", "torch", "torchvision"], []
    while stack:
        n = cn(stack.pop())
        if n in seen or n not in inst:
            continue
        seen.add(n)
        for rq in md.distribution(n).requires or []:
            r = Requirement(rq)
            if r.marker and not r.marker.evaluate({"extra": ""}):
                continue
            have = inst.get(cn(r.name))
            if have is None:
                closure_bad.append(f"{n} -> {r} (not installed)")
            elif r.specifier and not r.specifier.contains(have, prereleases=True):
                closure_bad.append(f"{n} needs {r} but has {have} ({where.get(cn(r.name), '')})")
            stack.append(r.name)
    return {"checked": len(lock or inst), "unsatisfied": bad, "missing": missing, "lock_mismatch": lock_mismatch,
            "torch_requirers": sorted(f"{n}: {rq}" for n in (lock or inst) if n in inst
                                      for rq in (md.distribution(n).requires or []) if Requirement(rq).name.lower() == "torch"),
            "closure_size": len(seen), "closure_unsatisfied": closure_bad,
            "torch_dists_on_path": sorted(f"{d.metadata['Name']}=={d.version} @ {getattr(d, '_path', '')}" for d in md.distributions()
                                          if (d.metadata["Name"] or "").lower() == "torch"),
            "ok": not bad and not missing and not lock_mismatch and not closure_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ThakiCloud/eg2-hybrid-w4-media")
    ap.add_argument("--revision", default="bc3574a285a2779ac0686cc149342f30fbbf8f5b", help="verified model revision (v0.1.0)")
    ap.add_argument("--local", default=None, help="use an existing local snapshot instead of downloading")
    ap.add_argument("--out", default="selftest_result.json")
    ap.add_argument("--site", default=None, help="isolated dependency dir (clean-env job) — checked for isolation")
    ap.add_argument("--lock", default=str(HERE / "requirements.lock.txt") if (HERE / "requirements.lock.txt").exists() else None,
                    help="requirements.lock.txt — installed versions must match it (default: the one next to this file)")
    ap.add_argument("--image-torch", default=None, help="expected torch.__file__ from the base image (clean-env job)")
    a = ap.parse_args()
    import torch, transformers, sentence_transformers, safetensors
    R = {"runtime_files": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}, "errors": []}
    if a.local:
        snap = Path(a.local); R["snapshot"] = {"local": str(snap)}
    else:
        from huggingface_hub import snapshot_download, HfApi
        info = HfApi().model_info(a.repo, revision=a.revision)
        snap = Path(snapshot_download(a.repo, revision=info.sha))
        R["snapshot"] = {"repo": a.repo, "commit": info.sha}
    R["snapshot"]["files_sha256"] = {p.name: sha(p) for p in sorted(snap.iterdir()) if p.is_file() and p.suffix in (".safetensors", ".py", ".json")}
    R["env"] = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
                "safetensors": safetensors.__version__, "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0))}
    try:
        import importlib.metadata as md                      # full installed package list (name==version), for reproduction
        R["env"]["packages"] = sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"]})
        R["env"]["torch_file"] = torch.__file__
    except Exception as ex:
        R["errors"].append(f"package list: {ex!r}"[:200])
    try:
        import subprocess
        R["env"]["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    R["isolation"] = isolation(a, torch)
    R["deps"] = dep_check(a)
    X = synth()
    import eg2_l2v1 as L
    torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated(); t0 = time.time()
    st, info = L.load(snap)
    torch.cuda.synchronize()
    R["runtime"] = {**info, "load_sec": round(time.time() - t0, 2), "peak_load_bytes": torch.cuda.max_memory_allocated() - m0,
                    "after_load_bytes": torch.cuda.memory_allocated() - m0}
    e_rt = encode(st, X, torch)
    del st; import gc; gc.collect(); torch.cuda.empty_cache()
    import eg2_quant as E
    torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    ref = E.load_sentence_transformer(snap, device="cuda")
    torch.cuda.synchronize()
    R["stock"] = {"after_load_bytes": torch.cuda.memory_allocated() - m0, "peak_load_bytes": torch.cuda.max_memory_allocated() - m0}
    e_ref = encode(ref, X, torch)
    R["resident_ratio_vs_stock_bf16"] = round(R["runtime"]["after_load_bytes"] / R["stock"]["after_load_bytes"], 4)
    R["min_cos"] = {m: cos_min(e_rt[m], e_ref[m], torch) for m in X}
    R["metric_control"] = {"self_cos_min": {m: cos_min(e_rt[m], e_rt[m], torch) for m in X},
                           "bitwise_equal": {m: bool(torch.equal(e_rt[m], e_ref[m])) for m in X},
                           "max_abs": {m: float((e_rt[m] - e_ref[m]).abs().max()) for m in X}}
    metric_ok = all(abs(v - 1.0) < 1e-9 for v in R["metric_control"]["self_cos_min"].values())
    R["finite"] = all(bool(torch.isfinite(e_rt[m]).all()) for m in X)
    R["fingerprint"] = {m: hashlib.sha256(e_rt[m].numpy().round(5).tobytes()).hexdigest()[:16] for m in X}
    R["gates"] = {"metric_valid": metric_ok, "cos_ok": all(v >= 0.99999 for v in R["min_cos"].values()), "finite": R["finite"],
                  "isolation_ok": R["isolation"]["ok"], "deps_ok": R["deps"]["ok"],
                  "loaded_all_w4": info["w4_linear"] > 0 and info["w4_linear_vec1"] > 0}
    R["verdict"] = "PASS" if all(R["gates"].values()) else "FAIL"
    Path(a.out).write_text(json.dumps(R, indent=1))
    print("SELFTEST_JSON " + json.dumps(R), flush=True)
    print(f"VERDICT: {R['verdict']}", flush=True)
    sys.exit(0 if R["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
