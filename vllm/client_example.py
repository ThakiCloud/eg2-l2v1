#!/usr/bin/env python3
"""Call the server started by serve.sh: one embedding per modality through the OpenAI-compatible /v1/embeddings route.

    python vllm/client_example.py --url http://127.0.0.1:8000 --image cat.png --audio clip.wav --video clip.mp4

Text uses `input`; image, audio and video are sent as chat `messages` with a data URL (or base64 for audio).
Audio should be 16 kHz mono; clips longer than 30 s are truncated to 30 s by the model's processor.
Standard library only.
"""
import argparse, base64, json, math, urllib.request

MODEL = "eg2"


def post(url, body, timeout=300):
    rq = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.load(r)["data"][0]["embedding"]


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _msg(part, prompt, model=MODEL):
    return {"model": model, "messages": [{"role": "user", "content": [part, {"type": "text", "text": prompt}]}]}


def text_body(text, model=MODEL):
    return {"model": model, "input": [text]}


def image_body(path, model=MODEL, mime="image/png"):
    return _msg({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + _b64(path)}}, "Describe this image:", model)


def audio_body(path, model=MODEL):
    return _msg({"type": "input_audio", "input_audio": {"data": _b64(path), "format": "wav"}}, "Describe this audio:", model)


def video_body(path, model=MODEL):
    return _msg({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + _b64(path)}}, "Describe this video:", model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--text", default="task: search result | query: how do heat pumps work in winter?")
    ap.add_argument("--image"); ap.add_argument("--audio"); ap.add_argument("--video")
    a = ap.parse_args()
    bodies = {"text": text_body(a.text)}
    if a.image:
        bodies["image"] = image_body(a.image)
    if a.audio:
        bodies["audio"] = audio_body(a.audio)
    if a.video:
        bodies["video"] = video_body(a.video)
    embs = {m: post(a.url, b) for m, b in bodies.items()}
    for m, e in embs.items():
        print(f"{m:6s} dim={len(e)} norm={math.sqrt(sum(x * x for x in e)):.4f}")
    if len(embs) > 1:
        t = embs["text"]
        for m, e in embs.items():
            if m != "text":
                print(f"cos(text, {m}) = {sum(x * y for x, y in zip(t, e)):.4f}")


if __name__ == "__main__":
    main()
