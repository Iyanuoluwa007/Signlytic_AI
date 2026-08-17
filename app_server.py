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
from typing import Dict, List, Optional

import hmac
import threading

import numpy as np
import uvicorn
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
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


# ── Access control ────────────────────────────────────────────────────────────
# This server was written to run on localhost, where anything reaching it is
# already you. Exposing it to a network changes that completely, so two things
# are guarded here:
#
#   Destructive endpoints (shutdown) require an admin token, always.
#   Expensive endpoints (video, audio, live frames) are rate limited per client,
#   because they occupy the GPU and spend LLM quota.
#
# Deliberately dependency free: no slowapi, no auth library. A demo box on a
# free tier should not need a package install to be safe.

ADMIN_TOKEN = os.environ.get("SIGNLYTIC_ADMIN_TOKEN", "").strip()

# Requests per window, per client IP, for the endpoints that cost real compute.
RATE_LIMIT_REQUESTS = int(os.environ.get("SIGNLYTIC_RATE_LIMIT", "20"))
RATE_LIMIT_WINDOW_SECONDS = 60.0

_rate_state: Dict[str, list] = {}
_rate_lock = threading.Lock()


def _client_key(request: Request) -> str:
    # Behind a tunnel or reverse proxy every request arrives from the proxy, so
    # the forwarded header is the only thing that distinguishes callers. It is
    # client controlled and therefore spoofable: this is throttling to keep one
    # careless user from monopolising the box, not a security boundary.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Allow RATE_LIMIT_REQUESTS per client per minute, or return 429."""
    if RATE_LIMIT_REQUESTS <= 0:
        return
    key = _client_key(request)
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        hits = [t for t in _rate_state.get(key, []) if t > cutoff]
        if len(hits) >= RATE_LIMIT_REQUESTS:
            retry = int(hits[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Try again in {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        hits.append(now)
        _rate_state[key] = hits
        # Drop idle clients so a long-running server does not accumulate keys.
        if len(_rate_state) > 4096:
            for k in [k for k, v in _rate_state.items() if not [t for t in v if t > cutoff]]:
                _rate_state.pop(k, None)


# On a hosted demo box there is no GPU, so video recognition, speech
# transcription and voice synthesis would take tens of seconds and queue behind
# each other. Rather than let a visitor sit through that and conclude the
# project is broken, demo mode turns those three endpoints into a clear pointer
# at the full download. The text and signing paths, which are what the demo is
# for, stay fully live.
DEMO_MODE = os.environ.get("SIGNLYTIC_DEMO_MODE") == "1"

FULL_APP_URL = os.environ.get(
    "SIGNLYTIC_FULL_APP_URL",
    "https://github.com/Iyanuoluwa007/Signlytic_AI/releases/latest",
)


def require_full_install() -> None:
    """Refuse GPU-bound features on a demo box, and say where to get them."""
    if not DEMO_MODE:
        return
    raise HTTPException(
        status_code=503,
        detail={
            "error": "not_available_in_demo",
            "message": (
                "Video recognition and voice need a local GPU, so they are not "
                "part of this online demo. Text and caption translation to BSL "
                "signing work here in full. Download and run the project "
                "locally for the complete system."
            ),
            "download": FULL_APP_URL,
        },
    )


def require_admin(request: Request) -> None:
    """Gate destructive endpoints behind a shared token."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Disabled. Set SIGNLYTIC_ADMIN_TOKEN to enable admin endpoints.",
        )
    supplied = (
        request.headers.get("x-admin-token")
        or request.query_params.get("token")
        or ""
    )
    # Constant time, so a wrong token cannot be found one character at a time.
    if not hmac.compare_digest(supplied.strip(), ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")

# ── Lazy-loaded ML singletons ─────────────────────────────────────────────────
_speech_to_bsl = None
_text_to_gloss = None
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


def get_text_to_gloss():
    """
    Just the English to gloss converter, without the rest of the pipeline.

    /api/d2/text needs only this, but reaching it through SpeechToBSL builds the
    whole pipeline, and that constructor loads Whisper eagerly. On a demo box
    that meant the busiest endpoint pulled a speech model it never uses into
    memory on the first request. Constructed with the same defaults the pipeline
    would have used.
    """
    global _text_to_gloss
    if _text_to_gloss is None:
        # Imported as a bare module, not as src.inference.speech_to_bsl.
        # The package __init__ imports recognizer.py, which imports torch, so
        # the qualified form drags the whole ML stack in and fails outright on
        # a CPU-only box that deliberately has no torch. src/inference is
        # already on sys.path, and speech_to_bsl has no package-relative
        # imports, so this loads the one class that is actually needed.
        from speech_to_bsl import TextToGloss
        _text_to_gloss = TextToGloss(mode="simple")
        print("[Server] TextToGloss loaded")
    return _text_to_gloss


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
    if DEMO_MODE:
        # The endpoints that need it are turned off, so loading torch and the
        # model here would only cost a slow boot and a few hundred MB on a box
        # that has neither to spare.
        print("[Server] demo mode: skipping recognizer warmup")
        return
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
    # Optional: a demo box has no torch by design, and health failing with a
    # 500 there is worse than useless, because health is the first thing anyone
    # checks when something looks wrong.
    try:
        import torch
    except ImportError:
        torch = None
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
        "gpu": (torch.cuda.get_device_name(0) if torch and torch.cuda.is_available()
                else ("CPU only" if torch else "no torch installed (demo box)")),
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
@app.post("/api/d1/video", dependencies=[Depends(rate_limit), Depends(require_full_install)])
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


@app.post("/api/d1/glosses", dependencies=[Depends(rate_limit)])
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
    # TTS — Coqui XTTS v2 with 22050Hz resampled speaker reference.
    #
    # Skipped in demo mode. The translation above is a hosted model call and
    # costs the box nothing, but XTTS loads a large model and synthesises on
    # the CPU, which would make this endpoint the slowest thing on a demo
    # machine while the useful part, the sentence, is already done.
    audio_b64 = None
    if not DEMO_MODE:
        try:
            audio_b64 = _synthesize_xtts(english, project_root)
        except Exception as _tts_err:
            print(f"[Server] TTS error: {_tts_err}")
    return {
        "glosses": gloss_list,
        "english": english,
        "audio_b64": audio_b64,
        "voice_available": not DEMO_MODE,
        "download": None if not DEMO_MODE else FULL_APP_URL,
    }


# ── Direction 2: English → BSL ────────────────────────────────────────────────
# Where to get sign frames from.
#
# Locally that is data/poses, read through PoseSignRenderer. A hosted box has no
# such directory: the pose data is 2.6 GB and the sign JSON another 1.4 GB, none
# of it in the repository. Setting SIGNLYTIC_SIGNS_API points the lookup at the
# website's sign endpoint instead, which already serves exactly this data and
# already caches it, so the box carries no data at all.
SIGNS_API = os.environ.get("SIGNLYTIC_SIGNS_API", "").strip().rstrip("/")

# Match PoseSignRenderer's defaults, so playback timing is identical whichever
# source is in use.
REMOTE_OUTPUT_FPS = 20
REMOTE_GLOSS_DURATION = 0.9

# Signs are immutable, and a sentence usually repeats common glosses, so one
# fetch each is plenty. Bounded so a long-running box cannot grow without limit.
_sign_frame_cache: Dict[str, Optional[list]] = {}

# Each cached sign is a few hundred kilobytes once parsed into Python lists, so
# 512 of them can reach a few hundred MB. That is nothing on a 24 GB box and
# most of the budget on a 1 GB Always Free shape, hence the knob.
_SIGN_CACHE_MAX = int(os.environ.get("SIGNLYTIC_SIGN_CACHE", "512"))


def _fetch_remote_sign(gloss: str) -> Optional[list]:
    """Fetch one gloss from the sign API. None means no sign for this gloss."""
    if gloss in _sign_frame_cache:
        return _sign_frame_cache[gloss]
    frames = None
    try:
        import requests
        r = requests.get(f"{SIGNS_API}/{gloss}", timeout=8)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                frames = data
    except Exception as e:
        # A sign that cannot be fetched is fingerspelled by the renderer, which
        # is the same outcome as a sign that does not exist. Not fatal.
        print(f"[Server] sign fetch failed for {gloss}: {e}")
    if len(_sign_frame_cache) >= _SIGN_CACHE_MAX:
        _sign_frame_cache.clear()
    _sign_frame_cache[gloss] = frames
    return frames


def _resample(frames: list, target: int) -> list:
    """Pick target frames evenly across the source, matching local behaviour."""
    if not frames:
        return []
    if len(frames) == target:
        return frames
    idx = [min(len(frames) - 1, int(round(i * (len(frames) - 1) / max(1, target - 1))))
           for i in range(target)] if target > 1 else [0]
    return [frames[i] for i in idx]


def _collect_pose_frames_remote(glosses: list, speed: float) -> list:
    per_gloss_s = REMOTE_GLOSS_DURATION / speed
    frames_per_gloss = max(1, int(round(per_gloss_s * REMOTE_OUTPUT_FPS)))
    out = []
    for gloss in glosses:
        src = _fetch_remote_sign(gloss)
        missing = src is None
        if missing:
            # No sign available. Emit an empty-handed frame run so the client
            # keeps its timing and can fingerspell, exactly as the local
            # renderer's neutral frames do.
            for _ in range(frames_per_gloss):
                out.append({"g": gloss, "m": True, "p": None, "l": None, "r": None})
            continue
        for frame in _resample(src, frames_per_gloss):
            out.append({
                "g": gloss,
                "m": False,
                "p": frame.get("body"),
                "l": frame.get("lh"),
                "r": frame.get("rh"),
            })
    return out


def _collect_pose_frames(glosses: list, speed: float = 1.0) -> list:
    speed = float(np.clip(speed, 0.6, 1.6))
    if SIGNS_API:
        return _collect_pose_frames_remote(glosses, speed)

    renderer = get_pose_renderer()
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


@app.post("/api/d2/text", dependencies=[Depends(rate_limit)])
async def d2_text(
    text: str = Form(...),
    speed: float = Form(1.0),
    avatar: str = Form("male"),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    converter = get_text_to_gloss()
    glosses = converter.convert(text)
    info = converter.convert_with_info(text)
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
        "fps": REMOTE_OUTPUT_FPS if SIGNS_API else get_pose_renderer().output_fps,
    }


@app.post("/api/d2/audio", dependencies=[Depends(rate_limit), Depends(require_full_install)])
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
            "fps": REMOTE_OUTPUT_FPS if SIGNS_API else get_pose_renderer().output_fps,
        }
    finally:
        for p in [tmp]:
            if Path(p).exists():
                try: os.unlink(p)
                except: pass


# ── Live: single webcam frame → gloss ────────────────────────────────────────
@app.post("/api/live/frame", dependencies=[Depends(rate_limit), Depends(require_full_install)])
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
@app.post("/api/live/assemble", dependencies=[Depends(rate_limit)])
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

    # English -> XTTS. Skipped on a demo box for the same reason as
    # d1/glosses: XTTS is the slowest thing here and the sentence, which is
    # the useful part, is already done.
    audio_b64 = None
    err: Optional[str] = None
    if not DEMO_MODE:
        try:
            audio_b64 = _synthesize_xtts(english, project_root)
        except Exception as e:
            err = f"tts: {e}"
            print(f"[Live] TTS error in assemble: {e}")

    return {"sentence": english, "audio_b64": audio_b64, "error": err}


# ── Graceful shutdown ──────────────────────────────────────────────────────────
@app.post("/api/shutdown", dependencies=[Depends(require_admin)])
async def shutdown():
    """
    Shut down the FastAPI server.

    Admin token required. Unprotected, this is a single unauthenticated POST
    that kills the process, which is fine on localhost and unacceptable the
    moment the server is reachable from anywhere else.
    """
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
# Serves pose frame JSON for the Chrome extension's lazy-load fallback.
#
# Relative to the project, not an absolute path: this was pinned to a D: drive
# on one Windows machine, so on any other machine, and on any Linux host, every
# lookup here returned 404. Override with SIGNLYTIC_SIGNS_DIR if the data lives
# outside the project, which it will on a server where 1.4 GB of sign JSON is
# mounted separately.
EXTENSION_SIGNS_DIR = Path(
    os.environ.get("SIGNLYTIC_SIGNS_DIR")
    or (project_root / "extension-data" / "signs")
)

# The localhost restriction below is right for a machine serving its own
# extension, and wrong for a hosted demo where every request arrives through a
# proxy. Set this when the server is meant to serve signs publicly.
SIGNS_PUBLIC = os.environ.get("SIGNLYTIC_SIGNS_PUBLIC") == "1"


@app.get("/api/signs/{gloss}")
async def get_sign_frames(gloss: str, request: Request):
    # Only allow requests from localhost (extension content scripts)
    host = request.headers.get("host", "")
    origin = request.headers.get("origin", "")
    if not SIGNS_PUBLIC and "localhost" not in host and "127.0.0.1" not in host:
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

    # Binding to anything other than loopback puts this on a network. Refuse to
    # do that without an admin token: the alternative is a server where anyone
    # who can reach it can stop it, and that mistake is silent until it is
    # exploited. Explicit override for someone who genuinely wants it open.
    loopback = args.host in ("localhost", "127.0.0.1", "::1")
    if not loopback and not ADMIN_TOKEN:
        if os.environ.get("SIGNLYTIC_ALLOW_INSECURE") == "1":
            print("[Server] WARNING: exposed on %s with no admin token "
                  "(SIGNLYTIC_ALLOW_INSECURE=1)" % args.host)
        else:
            print(
                "\nRefusing to start.\n"
                f"  Binding to {args.host} exposes this server beyond this machine,\n"
                "  and admin endpoints such as /api/shutdown would be unprotected.\n\n"
                "  Set an admin token first:\n"
                "    SIGNLYTIC_ADMIN_TOKEN=<a long random string>\n\n"
                "  Or, if you really intend an open server:\n"
                "    SIGNLYTIC_ALLOW_INSECURE=1\n"
            )
            raise SystemExit(2)

    print(f"\n{'='*55}")
    print("  Signlytic AI — FastAPI Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  bind: {args.host}"
          f"{'  (network exposed)' if not loopback else '  (this machine only)'}")
    print(f"  admin token: {'set' if ADMIN_TOKEN else 'not set'}")
    print(f"  rate limit: {RATE_LIMIT_REQUESTS}/min per client"
          f"{'  (disabled)' if RATE_LIMIT_REQUESTS <= 0 else ''}")
    print(f"{'='*55}\n")
    uvicorn.run(
        "app_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
