r"""
app_server.py - Signlytic AI FastAPI Dashboard Server
Runs on port 8000 alongside Gradio (port 7860).

Usage:
    conda activate BSL
    cd D:\Signlytic_AI\code\bsl_translation_project
    python app_server.py

Then open: http://localhost:8000
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "inference"))
sys.path.insert(0, str(project_root / "scripts"))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Signlytic AI", version="2.0")

# ── Lazy-loaded ML singletons ─────────────────────────────────────────────────
_speech_to_bsl = None
_bsl_dict_recognizer = None
_pose_renderer = None
_avatar_3d = None
_tts = None


def get_speech_to_bsl():
    global _speech_to_bsl
    if _speech_to_bsl is None:
        from src.inference.speech_to_bsl import SpeechToBSL
        _speech_to_bsl = SpeechToBSL()
        print("[Server] SpeechToBSL loaded")
    return _speech_to_bsl


def get_bsl_dict_recognizer():
    global _bsl_dict_recognizer
    if _bsl_dict_recognizer is None:
        try:
            import torch
            from src.inference.bsl_dict_recognizer import BSLDictRecognizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _bsl_dict_recognizer = BSLDictRecognizer(device=device)
            print(f"[Server] BSLDictRecognizer loaded on {device}")
        except Exception as e:
            print(f"[Server] BSLDictRecognizer unavailable: {e}")
    return _bsl_dict_recognizer


def get_pose_renderer():
    global _pose_renderer
    if _pose_renderer is None:
        from src.inference.pose_sign_renderer import PoseSignRenderer
        _pose_renderer = PoseSignRenderer(project_root=project_root)
        print("[Server] PoseSignRenderer loaded")
    return _pose_renderer


def get_avatar_3d():
    global _avatar_3d
    if _avatar_3d is None:
        try:
            from src.inference.avatar_3d_renderer import Avatar3DRenderer
            _avatar_3d = Avatar3DRenderer(
                project_root=project_root,
                pose_renderer=get_pose_renderer(),
            )
            print("[Server] Avatar3DRenderer loaded")
        except Exception as e:
            print(f"[Server] Avatar3DRenderer unavailable: {e}")
    return _avatar_3d


def to_wav_16k_mono(input_path: str) -> str:
    """Convert any audio/video to 16kHz mono WAV using ffmpeg."""
    out = tempfile.mktemp(suffix=".wav")
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out,
             "-loglevel", "error"],
            capture_output=True, timeout=60
        )
        if result.returncode == 0 and Path(out).exists() and Path(out).stat().st_size > 100:
            return out
    except Exception as e:
        print(f"[Server] ffmpeg error: {e}")
    return input_path



# ── Startup warmup ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _warmup():
    """Pre-load the BSL recognizer on boot so /api/health reports recognizer_ready
    before the camera starts (the readiness gate would otherwise deadlock)."""
    try:
        get_bsl_dict_recognizer()
    except Exception as e:
        print(f"[Server] warmup failed: {e}")


# ── Favicon ────────────────────────────────────────────────────────────────────
@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(
        b'\x00',
        media_type="image/x-icon",
        headers={"Content-Length": "1"}
    )

# ── Static files ──────────────────────────────────────────────────────────────
avatars_dir = project_root / "data" / "avatars"
avatars_dir.mkdir(parents=True, exist_ok=True)
app.mount("/avatars", StaticFiles(directory=str(avatars_dir)), name="avatars")


# ── HTML dashboard ────────────────────────────────────────────────────────────
DASHBOARD_PATH = project_root / "dashboard.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    if DASHBOARD_PATH.exists():
        return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard.html not found</h1>")


# ── Health ────────────────────────────────────────────────────────────────────

# ── XTTS v2 helper ────────────────────────────────────────────────────────────
_xtts_model = None
_xtts_speaker = None


def _get_xtts(proj_root):
    """Load XTTS v2 once; resample speaker to 22050Hz for best voice quality."""
    global _xtts_model, _xtts_speaker
    if _xtts_model is not None:
        return _xtts_model, _xtts_speaker

    src = Path(proj_root) / "data" / "processed" / "voice_training.wav"
    if not src.exists():
        raise FileNotFoundError(f"Speaker wav not found: {src}")

    # Resample to 22050Hz once — XTTS v2 native sample rate
    dst = Path(proj_root) / "data" / "processed" / "voice_training_22050.wav"
    if not dst.exists():
        import subprocess as _sp
        print("[TTS] Resampling speaker wav to 22050Hz...")
        _sp.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ar", "22050", "-ac", "1", str(dst), "-loglevel", "error"],
            check=True
        )
        print(f"[TTS] Resampled speaker: {dst.stat().st_size} bytes")

    from TTS.api import TTS as _TTS
    import torch as _torch
    _dev = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"[TTS] Loading XTTS v2 on {_dev}...")
    _xtts_model = _TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_dev)
    _xtts_speaker = str(dst)
    print("[TTS] XTTS v2 ready.")
    return _xtts_model, _xtts_speaker


def _natural_text(text: str, is_live: bool = False) -> str:
    """
    Pad text for XTTS voice cloning only when necessary.
    - Full English sentences: return as-is (already natural)
    - Live single gloss words: pad so XTTS clones voice properly
    - Short phrases from translation: return as-is (Groq already made them natural)
    """
    text = text.strip()
    words = text.split()
    # If it ends with punctuation it is already a proper sentence — use as-is
    if text and text[-1] in ".!?":
        return text
    # Long enough — use as-is
    if len(words) >= 6:
        return text
    # Live single-word gloss — needs padding for XTTS voice cloning
    if is_live and len(words) <= 2:
        return f"The sign is {text.lower()}. {text.lower()}."
    # Short non-punctuated phrase — add a period only
    return text + "."


def _synthesize_xtts(text: str, proj_root, is_live: bool = False) -> "str | None":
    """Synthesize with XTTS v2. Returns base64 WAV string or None."""
    if not text or not text.strip():
        return None
    tts_path = tempfile.mktemp(suffix=".wav")
    try:
        model, speaker = _get_xtts(proj_root)
        nat = _natural_text(text, is_live=is_live)
        model.tts_to_file(
            text=nat,
            speaker_wav=speaker,
            language="en",
            file_path=tts_path,
        )
        fp = Path(tts_path)
        if fp.exists() and fp.stat().st_size > 1024:
            result = base64.b64encode(fp.read_bytes()).decode()
            print(f"[Server] TTS OK: {fp.stat().st_size} bytes ({len(nat.split())} words)")
            return result
        print("[Server] TTS: empty output")
        return None
    finally:
        try:
            if Path(tts_path).exists():
                os.unlink(tts_path)
        except Exception:
            pass


@app.get("/api/health")
async def health():
    import torch
    # Check multiple vocab file locations
    vocab_size = 0
    for vp in [
        project_root / "data" / "processed" / "vocabulary_extended.json",
        project_root / "data" / "processed" / "vocabulary.json",
        project_root / "data" / "bsldict" / "bsldict" / "bsldict_v1.pkl",
    ]:
        if vp.exists():
            try:
                if vp.suffix == ".json":
                    data = json.loads(vp.read_text(encoding="utf-8"))
                    vocab_size = len(data) if isinstance(data, (dict, list)) else 0
                elif vp.suffix == ".pkl":
                    import pickle
                    with open(vp, "rb") as _f:
                        data = pickle.load(_f)
                    vocab_size = len(data) if isinstance(data, (dict, list)) else 0
                if vocab_size > 100:
                    break
            except Exception:
                pass
    # Live readiness — recognizer and XTTS are lazy-loaded module-level singletons
    # in app_server.py (set on first /api/live/frame and first /api/live/assemble).
    recognizer_ready = globals().get("_bsl_dict_recognizer") is not None
    tts_ready = globals().get("_xtts_model") is not None
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "vocab_size": vocab_size,
        "bsl_dict": (project_root / "models" / "bsl_dict_recognition" / "retrieval_model.pt").exists(),
        "recognizer_ready": recognizer_ready,
        "tts_ready": tts_ready,
        "avatars": {
            "male": (avatars_dir / "Male.glb").exists(),
            "female": (avatars_dir / "Female.glb").exists(),
        },
    }


# ── Direction 1: BSL Video → English ─────────────────────────────────────────
@app.post("/api/d1/video")
async def d1_video(
    file: UploadFile = File(...),
    mode: str = Form("groq"),
    api_key: str = Form(""),
):
    tmp = tempfile.mktemp(suffix=Path(file.filename).suffix or ".mp4")
    try:
        with open(tmp, "wb") as f:
            f.write(await file.read())
        recognizer = get_bsl_dict_recognizer()
        if recognizer is None:
            raise HTTPException(status_code=503, detail="BSL Dict Recognizer not available")
        import torch
        from src.inference.bsl_dict_recognizer import BSLDictRecognizer
        # recognize() returns List[Tuple[str, float]]
        raw = recognizer.recognize(tmp, top_k=5)
        # Normalise to list of dicts
        results = []
        for item in (raw or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                results.append({"gloss": str(item[0]), "score": float(item[1])})
            elif isinstance(item, dict):
                results.append(item)
        glosses = [r["gloss"] for r in results[:3]]
        english = " ".join(glosses)
        if glosses:
            try:
                from src.inference.gloss_to_text import GlossToText
                english = GlossToText(mode=mode).convert(glosses)
            except Exception as e:
                print(f"[Server] GlossToText error: {e}")
        # TTS — speak the English translation
        audio_b64 = None
        try:
            audio_b64 = _synthesize_xtts(english, project_root)
        except Exception as _e:
            print(f"[Server] TTS error in d1_video: {_e}")
        return {"glosses": glosses, "english": english, "results": results[:5], "audio_b64": audio_b64}
    finally:
        if Path(tmp).exists():
            os.unlink(tmp)


@app.post("/api/d1/glosses")
async def d1_glosses(
    glosses: str = Form(...),
    mode: str = Form("groq"),
    api_key: str = Form(""),
):
    gloss_list = [g.strip().upper() for g in glosses.split() if g.strip()]
    if not gloss_list:
        raise HTTPException(status_code=400, detail="No glosses provided")
    try:
        from src.inference.gloss_to_text import GlossToText
        converter = GlossToText(mode=mode)
        english = converter.convert(gloss_list)
    except Exception as e:
        print(f"[Server] GlossToText error: {e}")
        english = " ".join(g.lower() for g in gloss_list)
    # TTS — Coqui XTTS v2 with 22050Hz resampled speaker reference
    audio_b64 = None
    try:
        audio_b64 = _synthesize_xtts(english, project_root)
    except Exception as _tts_err:
        print(f"[Server] TTS error: {_tts_err}")
    return {"glosses": gloss_list, "english": english, "audio_b64": audio_b64}


# ── Direction 2: English → BSL ────────────────────────────────────────────────
def _collect_pose_frames(glosses: list, speed: float = 1.0) -> list:
    renderer = get_pose_renderer()
    speed = float(np.clip(speed, 0.6, 1.6))
    per_gloss_s = renderer.base_gloss_duration / speed
    frames_per_gloss = max(1, int(round(per_gloss_s * renderer.output_fps)))
    frames = []
    for gloss in glosses:
        sampled, missing = renderer._get_gloss_frames(gloss, frames_per_gloss)
        for frame in sampled:
            frames.append({
                "g": gloss,
                "m": bool(missing),
                "p": frame["pose"].tolist(),
                "l": frame["left_hand"].tolist(),
                "r": frame["right_hand"].tolist(),
            })
    return frames


@app.post("/api/d2/text")
async def d2_text(
    text: str = Form(...),
    speed: float = Form(1.0),
    avatar: str = Form("male"),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    pipeline = get_speech_to_bsl()
    glosses = pipeline.text_to_gloss.convert(text)
    info = pipeline.text_to_gloss.convert_with_info(text)
    coverage = float(info.get("coverage", 0.0)) if info else 0.0
    oov = info.get("out_of_vocab", []) if info else []
    frames = _collect_pose_frames(glosses, speed)
    return {
        "text": text,
        "glosses": glosses,
        "coverage": coverage,
        "out_of_vocab": oov[:10],
        "frames": frames,
        "frame_count": len(frames),
        "fps": get_pose_renderer().output_fps,
    }


@app.post("/api/d2/audio")
async def d2_audio(
    file: UploadFile = File(...),
    speed: float = Form(1.0),
    avatar: str = Form("male"),
):
    tmp = tempfile.mktemp(suffix=Path(file.filename).suffix or ".wav")
    try:
        with open(tmp, "wb") as f:
            f.write(await file.read())
        wav = to_wav_16k_mono(tmp)
        pipeline = get_speech_to_bsl()
        result = pipeline.process(wav, return_intermediate=True)
        text = result.get("text", "")
        glosses = result.get("glosses", [])
        info = pipeline.text_to_gloss.convert_with_info(text)
        coverage = float(info.get("coverage", 0.0)) if info else 0.0
        oov = info.get("out_of_vocab", []) if info else []
        frames = _collect_pose_frames(glosses, speed)
        return {
            "text": text,
            "glosses": glosses,
            "coverage": coverage,
            "out_of_vocab": oov[:10],
            "frames": frames,
            "frame_count": len(frames),
            "fps": get_pose_renderer().output_fps,
        }
    finally:
        for p in [tmp]:
            if Path(p).exists():
                try: os.unlink(p)
                except: pass


# ── Live: single webcam frame → gloss ────────────────────────────────────────
@app.post("/api/live/frame")
async def live_frame(file: UploadFile = File(...)):
    recognizer = get_bsl_dict_recognizer()
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not available")
    tmp = tempfile.mktemp(suffix=".jpg")
    try:
        with open(tmp, "wb") as f:
            f.write(await file.read())
        # recognize() returns List[Tuple[str, float]]
        raw = recognizer.recognize(tmp, top_k=3)
        clean = []
        for item in (raw or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                clean.append({"gloss": str(item[0]), "score": float(item[1])})
            elif isinstance(item, dict) and "gloss" in item:
                clean.append(item)
        # Per-frame TTS disabled — sentence-level TTS only via /api/live/assemble.
        return {"results": clean[:3], "audio_b64": None}
    finally:
        if Path(tmp).exists():
            os.unlink(tmp)


# ── Live: assemble glosses into English sentence + XTTS ──────────────────────
@app.post("/api/live/assemble")
async def live_assemble(payload: dict = Body(...)):
    """
    Take the accumulated live gloss list, run it through GlossToText (Groq llama-3.3-70b),
    then synthesise the English result with the cloned XTTS v2 voice.
    Input:  {"glosses": ["HELLO", "MY", "NAME", "OKE"]}
    Output: {"sentence": "...", "audio_b64": "<b64 wav>" | None, "error": str | None}
    """
    raw = payload.get("glosses") or []
    # Filter empties + dedupe consecutive duplicates (live recognition often double-fires)
    glosses: List[str] = []
    for g in raw:
        if not isinstance(g, str):
            continue
        g = g.strip()
        if not g:
            continue
        if glosses and glosses[-1].lower() == g.lower():
            continue
        glosses.append(g.upper())

    if not glosses:
        return {"sentence": "", "audio_b64": None, "error": "no glosses"}

    # Gloss -> English
    try:
        from src.inference.gloss_to_text import GlossToText
        english = GlossToText(mode="groq").convert(glosses)
    except Exception as e:
        print(f"[Live] GlossToText error: {e}")
        english = " ".join(g.lower() for g in glosses).capitalize() + "."

    print(f"[Live] Assemble: {len(glosses)} glosses -> {english[:60]}...")

    # English -> XTTS
    audio_b64 = None
    err: Optional[str] = None
    try:
        audio_b64 = _synthesize_xtts(english, project_root)
    except Exception as e:
        err = f"tts: {e}"
        print(f"[Live] TTS error in assemble: {e}")

    return {"sentence": english, "audio_b64": audio_b64, "error": err}


# ── Graceful shutdown ──────────────────────────────────────────────────────────
@app.post("/api/shutdown")
async def shutdown():
    """Shut down the FastAPI server."""
    import threading, time
    def _stop():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_stop, daemon=False).start()
    return {"status": "shutting_down", "message": "Server shutting down..."}



# ── Avatar GLB files ───────────────────────────────────────────────────────────
@app.get("/api/avatar/{name}")
async def avatar_file(name: str):
    if not name.endswith(".glb"):
        raise HTTPException(status_code=400, detail="GLB files only")
    path = avatars_dir / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found in data/avatars/")
    return FileResponse(str(path), media_type="model/gltf-binary")


# ── Run ───────────────────────────────────────────────────────────────────────


# Extension sign data endpoint
# Serves pose frame JSON for the Chrome extension's lazy-load fallback
# Only accessible from localhost -- not a public API
EXTENSION_SIGNS_DIR = Path(r"D:\Signlytic_AI\code\bsl_translation_project\extension-data\signs")

@app.get("/api/signs/{gloss}")
async def get_sign_frames(gloss: str, request: Request):
    # Only allow requests from localhost (extension content scripts)
    host = request.headers.get("host", "")
    origin = request.headers.get("origin", "")
    if "localhost" not in host and "127.0.0.1" not in host:
        return JSONResponse({"error": "local only"}, status_code=403)

    sign_file = EXTENSION_SIGNS_DIR / f"{gloss.upper()}.json"
    if not sign_file.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    import json
    frames = json.loads(sign_file.read_text(encoding="utf-8"))
    return JSONResponse(frames)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    print(f"\n{'='*55}")
    print("  Signlytic AI — FastAPI Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*55}\n")
    uvicorn.run(
        "app_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
