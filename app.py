"""
Signlytic AI - British Sign Language Translation System
========================================================

Bidirectional BSL translation powered by Video-SWIN-T, Groq LLM, and Coqui TTS.
- Direction 1: BSL Video/Glosses -> English Text -> Speech
- Direction 2: Speech/Text -> BSL Glosses -> Animated Signing

Developed by Oke Iyanuoluwa Enoch
Independent Robotics & AI Systems Engineer

Usage:
    python app.py
    python app.py --share
"""

import argparse
import sys
import os
import tempfile
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(project_root) == "scripts":
    project_root = os.path.dirname(project_root)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "inference"))
sys.path.insert(0, os.path.join(project_root, "scripts"))

try:
    from src.inference.avatar_3d_renderer import Avatar3DRenderer
    AVATAR_3D_AVAILABLE = True
except ImportError as _e3d:
    AVATAR_3D_AVAILABLE = False
    print(f"Avatar3DRenderer unavailable: {_e3d}")

try:
    import gradio as gr
except ImportError:
    print("Gradio required. Install with: pip install gradio")
    sys.exit(1)

# ---------------------------------------------------------------------------
# BSL Dict Recognizer (SWIN-based)
# ---------------------------------------------------------------------------
try:
    from src.inference.bsl_dict_recognizer import BSLDictRecognizer
    BSL_DICT_AVAILABLE = True
except ImportError:
    BSLDictRecognizer = None
    BSL_DICT_AVAILABLE = False
    print("Warning: BSLDictRecognizer not available")

# ---------------------------------------------------------------------------
# Import pipeline components
# ---------------------------------------------------------------------------
try:
    from speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
    from gloss_to_text import GlossToText, BSLToSpeechPipeline
    from avatar_renderer import BSLAvatarRenderer
    from pose_sign_renderer import PoseSignRenderer
except ImportError:
    try:
        from inference.speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
        from inference.gloss_to_text import GlossToText, BSLToSpeechPipeline
        from inference.avatar_renderer import BSLAvatarRenderer
        from inference.pose_sign_renderer import PoseSignRenderer
    except ImportError:
        from src.inference.speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
        from src.inference.gloss_to_text import GlossToText, BSLToSpeechPipeline
        from src.inference.avatar_renderer import BSLAvatarRenderer
        from src.inference.pose_sign_renderer import PoseSignRenderer
        from src.inference.bsl_dict_recognizer import BSLDictRecognizer

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_speech_to_bsl = None
_gloss_to_text_simple = None
_gloss_to_text_groq = None
_groq_key_in_use = None
_tts = None
_tts_disabled_reason = None
_avatar_renderer = None
_pose_sign_renderer = None
_avatar_3d_renderer = None
_sign_recognizer = None
_bsl_dict_recognizer = None
_live_running = False
_live_stop_event = threading.Event()
_live_clear_event = threading.Event()

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_VOCAB = os.path.join(project_root, "data", "processed", "vocabulary_extended.json")
DEFAULT_SPEAKER = os.path.join(project_root, "data", "processed", "voice_training.wav")
DEFAULT_VIDEO_DIR = os.path.join(project_root, "data", "videos", "bsl_signs")
DEFAULT_VIDEO_MAP = os.path.join(project_root, "data", "bsldict", "bsldict", "bsldict_video_map.json")
DEFAULT_SIGN_MODEL = os.path.join(project_root, "models", "sign_recognition", "best_model.pt")
DEFAULT_SIGN_VOCAB = os.path.join(project_root, "models", "sign_recognition", "vocabulary.json")
DEFAULT_SIGN_CLASS_STATS = os.path.join(project_root, "models", "sign_recognition", "class_stats.json")
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()






# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

/* ═══════════════════════════════════════
   GRADIO VARIABLE OVERRIDES
   Light-only. Clean, bright, professional.
   ═══════════════════════════════════════ */
.gradio-container, body, :root {
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8f9fb !important;
    --block-background-fill: #ffffff !important;
    --input-background-fill: #ffffff !important;
    --panel-background-fill: #ffffff !important;
    --body-background-fill: #f4f5f7 !important;
    --block-border-color: #e2e5ea !important;
    --block-label-background-fill: #f5f6f8 !important;
    --block-label-border-color: #e2e5ea !important;
    --block-label-text-color: #475569 !important;
    --block-title-text-color: #0f172a !important;
    --input-border-color: #d1d5db !important;
    --border-color-primary: #e2e5ea !important;
    --color-accent: #0e7c6b !important;
    --color-accent-soft: rgba(14,124,107,0.06) !important;
    --button-primary-background-fill: #1e3a5f !important;
    --button-primary-background-fill-hover: #162d4a !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-background-fill-hover: #f5f6f8 !important;
    --button-secondary-border-color: #d1d5db !important;
    --button-secondary-text-color: #374151 !important;
    --neutral-50: #f9fafb !important;
    --neutral-100: #f5f6f8 !important;
    --neutral-200: #e2e5ea !important;
    --neutral-300: #d1d5db !important;
    --neutral-400: #9ca3af !important;
    --neutral-500: #6b7280 !important;
    --neutral-600: #4b5563 !important;
    --neutral-700: #374151 !important;
    --neutral-800: #1f2937 !important;
    --neutral-900: #0f172a !important;
    --shadow-drop: 0 1px 2px rgba(0,0,0,0.04) !important;
    --shadow-drop-lg: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ─── Global ─── */
.gradio-container {
    max-width: 1260px !important; margin: 0 auto !important;
    font-family: 'Outfit', system-ui, -apple-system, sans-serif !important;
    background: #f4f5f7 !important; color: #0f172a !important;
}
.gradio-container * { font-family: 'Outfit', system-ui, -apple-system, sans-serif !important; }

/* ─── Icons — always visible ─── */
.gradio-container svg { color: #475569 !important; }
.gradio-container .upload-container { color: #6b7280 !important; }
.gradio-container textarea::placeholder,
.gradio-container input::placeholder { color: #9ca3af !important; }

/* ═══════════════
   HERO
   ═══════════════ */
.sig-hero {
    background: linear-gradient(155deg, #0d1a30 0%, #162d4a 35%, #1e3a5f 65%, #254468 100%);
    border-radius: 14px; padding: 2.2rem; margin-bottom: 0.7rem;
    position: relative; overflow: hidden;
}
.sig-hero::before {
    content: ''; position: absolute; top: -50px; right: -30px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(14,124,107,0.14) 0%, transparent 65%);
    pointer-events: none;
}
.sig-hero-top { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.4rem; position: relative; z-index: 1; }
.sig-hero-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; background: rgba(14,124,107,0.25);
    border: 1px solid rgba(14,124,107,0.35); border-radius: 8px;
}
.sig-hero-name { color: #fff; font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }
.sig-hero-desc { color: rgba(255,255,255,0.6); font-size: 0.86rem; line-height: 1.6; max-width: 510px; margin-bottom: 0.9rem; position: relative; z-index: 1; }
.sig-chips { display: flex; gap: 0.25rem; flex-wrap: wrap; position: relative; z-index: 1; }
.sig-chip {
    display: inline-flex; align-items: center; gap: 0.2rem;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px; padding: 0.2rem 0.5rem;
    color: rgba(255,255,255,0.55); font-size: 0.68rem; font-weight: 500;
}
.sig-chip b { color: rgba(255,255,255,0.85); font-weight: 700; }

/* ═══════════════
   TRUST STRIP
   ═══════════════ */
.sig-trust { display: flex; gap: 0.3rem; flex-wrap: wrap; padding: 0.15rem 0 0.45rem 0; }
.sig-badge { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.18rem 0.45rem; border-radius: 100px; font-size: 0.66rem; font-weight: 600; border: 1px solid; }
.sig-badge-ok { background: #ecfdf5; color: #15803d; border-color: #a7f3d0; }
.sig-badge-warn { background: #fefce8; color: #a16207; border-color: #fde68a; }

/* ═══════════════
   TABS — bright white card
   ═══════════════ */
.tabs { background: #ffffff !important; border-radius: 12px !important; overflow: hidden !important; border: 1px solid #e2e5ea !important; box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important; }
.tab-nav { border-bottom: 1px solid #e8eaef !important; padding: 0 0.3rem !important; background: #fafbfc !important; }
.tab-nav button { font-weight: 600 !important; font-size: 0.86rem !important; padding: 0.75rem 1.1rem !important; border: none !important; color: #64748b !important; transition: all 0.15s !important; }
.tab-nav button:hover { color: #0f172a !important; background: rgba(14,124,107,0.04) !important; }
.tab-nav button.selected { color: #0e7c6b !important; font-weight: 700 !important; background: #ffffff !important; border-bottom: 2.5px solid #0e7c6b !important; }

/* ═══════════════
   SECTIONS
   ═══════════════ */
.sig-sh { font-size: 1.18rem; font-weight: 700; color: #0f172a; margin: 0 0 0.1rem 0; letter-spacing: -0.01em; }
.sig-sd { font-size: 0.85rem; color: #475569; line-height: 1.55; margin: 0 0 0.65rem 0; }
.sig-col { font-size: 0.67rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin: 0 0 0.4rem 0; padding-left: 0.4rem; border-left: 3px solid #1e3a5f; }
.sig-method { font-size: 0.87rem; font-weight: 600; color: #0f172a; margin: 0 0 0.28rem 0; display: flex; align-items: center; gap: 0.35rem; }
.sig-rec { display: inline-flex; background: #ecfdf5; border: 1px solid #a7f3d0; color: #15803d; font-size: 0.6rem; font-weight: 700; padding: 0.12rem 0.35rem; border-radius: 100px; text-transform: uppercase; letter-spacing: 0.04em; }
.sig-sep { border: none; border-top: 1px solid #e8eaef; margin: 0.75rem 0; }

/* ═══════════════
   BUTTONS
   ═══════════════ */
.gr-button-primary { background: #1e3a5f !important; border: none !important; font-weight: 600 !important; font-size: 0.84rem !important; border-radius: 8px !important; padding: 0.52rem 1.05rem !important; color: #fff !important; transition: all 0.15s !important; }
.gr-button-primary:hover { background: #162d4a !important; transform: translateY(-1px) !important; box-shadow: 0 2px 8px rgba(30,58,95,0.18) !important; }
.gr-button-secondary { border: 1.5px solid #d1d5db !important; color: #374151 !important; font-weight: 500 !important; border-radius: 8px !important; background: #fff !important; transition: all 0.15s !important; }
.gr-button-secondary:hover { border-color: #9ca3af !important; color: #0f172a !important; background: #f9fafb !important; }

/* ─── Labels ─── */
label, .gr-label { font-weight: 500 !important; font-size: 0.84rem !important; color: #475569 !important; }
details summary { color: #0f172a !important; font-weight: 600 !important; }

/* ─── Skip link ─── */
.sig-skip { position: absolute; left: -9999px; top: 0; z-index: 999; background: #0e7c6b; color: #fff; padding: 0.5rem 1rem; border-radius: 0 0 8px 0; font-weight: 600; font-size: 0.86rem; text-decoration: none; }
.sig-skip:focus { left: 0; }

/* ─── Result cards ─── */
.sig-result { background: #ffffff; border: 1px solid #e2e5ea; border-left: 3px solid #0e7c6b; border-radius: 8px; padding: 0.4rem 0.65rem; margin-bottom: 0.2rem; }
.sig-result-navy { border-left-color: #1e3a5f; }
.sig-result-h { font-size: 0.64rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }

/* ─── BSL note ─── */
.sig-bsl-note { font-size: 0.78rem; color: #64748b; margin-bottom: 0.2rem; font-style: italic; }

/* ─── Step numbers ─── */
.sig-step { display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; background: #1e3a5f; color: #fff; border-radius: 50%; font-size: 0.62rem; font-weight: 700; flex-shrink: 0; margin-right: 0.25rem; }

/* ─── Metrics ─── */
.sig-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.5rem; margin-bottom: 1rem; }
.sig-metric { background: #fff; border: 1px solid #e2e5ea; border-top: 3px solid #0e7c6b; border-radius: 10px; padding: 1rem 0.7rem; text-align: center; }
.sig-metric-val { font-size: 1.55rem; font-weight: 800; color: #1e3a5f; line-height: 1.1; margin-bottom: 0.1rem; }
.sig-metric-label { font-size: 0.64rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

/* ─── Tables ─── */
.sig-table { width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 8px; overflow: hidden; font-size: 0.83rem; border: 1px solid #e2e5ea; margin-bottom: 1rem; }
.sig-table th { background: #1e3a5f; color: #fff; font-weight: 600; padding: 0.52rem 0.75rem; text-align: left; font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.04em; }
.sig-table td { padding: 0.48rem 0.75rem; border-bottom: 1px solid #f0f1f3; color: #374151; background: #fff; }
.sig-table tr:last-child td { border-bottom: none; }
.sig-table tr:nth-child(even) td { background: #f9fafb; }
.sig-table .green { color: #15803d; font-weight: 700; }

/* ─── Help cards ─── */
.sig-help { background: #fff; border: 1px solid #e2e5ea; border-left: 3px solid #0e7c6b; border-radius: 8px; padding: 0.72rem 0.88rem; margin-bottom: 0.48rem; }
.sig-help h4 { font-size: 0.87rem; font-weight: 700; color: #0f172a; margin: 0 0 0.15rem 0; }
.sig-help p { font-size: 0.82rem; color: #475569; line-height: 1.55; margin: 0; }
.sig-help a { color: #0e7c6b; font-weight: 600; text-decoration: underline; text-underline-offset: 2px; }

/* ─── A11y grid ─── */
.sig-a11y { display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 0.48rem; margin-top: 0.48rem; }
.sig-a11y-card { background: #fff; border: 1px solid #e2e5ea; border-left: 3px solid #1e3a5f; border-radius: 8px; padding: 0.72rem 0.88rem; }
.sig-a11y-card h5 { font-size: 0.83rem; font-weight: 600; color: #0f172a; margin: 0 0 0.1rem 0; }
.sig-a11y-card p { font-size: 0.78rem; color: #475569; line-height: 1.5; margin: 0; }

/* ─── Footer ─── */
.sig-footer { background: linear-gradient(160deg, #1e3a5f 0%, #162d4a 100%); border-radius: 0 0 12px 12px; padding: 1.35rem; text-align: center; position: relative; }
.sig-footer::before { content: ''; position: absolute; top: 0; left: 12%; right: 12%; height: 1px; background: linear-gradient(90deg, transparent, rgba(14,124,107,0.3), transparent); }
.sig-footer-brand { font-size: 0.9rem; font-weight: 700; color: rgba(255,255,255,0.9); margin-bottom: 0.1rem; }
.sig-footer-author { font-size: 0.78rem; color: rgba(255,255,255,0.45); margin-bottom: 0.1rem; }
.sig-footer-author a { color: rgba(255,255,255,0.68); text-decoration: none; font-weight: 600; }
.sig-footer-author a:hover { text-decoration: underline; }
.sig-footer-links { font-size: 0.7rem; color: rgba(255,255,255,0.25); }
.sig-footer-links a { color: rgba(255,255,255,0.25); text-decoration: none; margin: 0 0.25rem; transition: color 0.15s; }
.sig-footer-links a:hover { color: rgba(255,255,255,0.6); }
.sig-footer-sub { font-size: 0.66rem; color: rgba(255,255,255,0.15); margin-top: 0.18rem; font-style: italic; }

/* ─── Example buttons ─── */
.gradio-container .gr-sample-textbox,
.gradio-container .gr-samples-table button,
.gradio-container table.gr-samples-table td {
    background: #fff !important; color: #374151 !important;
    border: 1px solid #e2e5ea !important; border-radius: 6px !important;
    font-size: 0.8rem !important;
}

/* ─── Focus ─── */
*:focus-visible { outline: 2px solid #0e7c6b !important; outline-offset: 2px !important; }

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .sig-hero { padding: 1.3rem 1rem; border-radius: 10px; }
    .sig-hero-name { font-size: 1.12rem; }
    .sig-metrics { grid-template-columns: repeat(2, 1fr); }
    .sig-footer { border-radius: 0 0 10px 10px; }
}

/* === Signlytic Dark Theme Override === */
body, .gradio-container, .main, .wrap, .app {
    background: #080a10 !important;
    color: #e2e8f0 !important;
}
.tabitem, .tab-nav, .panel, .form, .block, .box, .gap {
    background: #0c0e16 !important;
    border-color: #1e293b !important;
    color: #e2e8f0 !important;
}
label, .label-wrap span, .block label, .caption-lg, .prose, p, span {
    color: #94a3b8 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0 !important;
}
input, textarea, select, .input-wrap input, .input-wrap textarea {
    background: #111827 !important;
    border-color: #1e3a5f !important;
    color: #f1f5f9 !important;
}
button.primary, .btn-primary, button[variant="primary"] {
    background: #1e3a5f !important;
    border-color: #0e7c6b !important;
    color: #ffffff !important;
}
button.secondary, button[variant="secondary"] {
    background: #0f172a !important;
    border-color: #1e293b !important;
    color: #94a3b8 !important;
}
.tab-nav button { 
    color: #94a3b8 !important; 
    background: #0c0e16 !important; 
    border-color: #1e293b !important;
}
.tab-nav button.selected { 
    color: #5eead4 !important; 
    border-color: #0e7c6b !important;
    background: #0f1f2e !important;
}
.accordion, .accordion > .label-wrap {
    background: #0f172a !important; 
    border-color: #1e293b !important; 
    color: #94a3b8 !important;
}
.gr-radio label, .radio-group label { color: #94a3b8 !important; }
.gr-check-radio:checked { accent-color: #0e7c6b !important; }
.sig-sh { color: #5eead4 !important; }
.sig-col { color: #0e7c6b !important; }
.sig-result { background: #0f172a !important; border-color: #1e3a5f !important; }
.sig-result-h { color: #5eead4 !important; }
.sig-help { background: #0f172a !important; border-color: #1e293b !important; }
.sig-step { background: #0e7c6b !important; color: #fff !important; }
footer, .footer { display: none !important; }

/* -- Gradio 6.x tab panel dark overrides -- */
.tab-nav { background: #0c0e16 !important; border-bottom: 1px solid #1e293b !important; }
.tabitem > div, .tabitem { background: #080a10 !important; }
div.form { background: #111827 !important; border-color: #1e293b !important; }
.contain { background: #111827 !important; }
div[data-testid='block'] { background: #111827 !important; }
.gap-4, .gap-2 { background: #080a10 !important; }
"""



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def to_wav_16k_mono(input_path: str) -> str:
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError("FFmpeg conversion failed")
    return out_path


def synthesize_with_windows_tts(text: str, output_path: str) -> bool:
    if os.name != "nt":
        return False
    if not text or not text.strip():
        return False
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ps_text = text.replace("'", "''")
        ps_out = output_path.replace("'", "''")
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{ps_out}'); "
            f"$s.Speak('{ps_text}'); "
            "$s.Dispose();"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
    except Exception:
        return False


def file_to_path(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return str(uploaded_file)
    if isinstance(uploaded_file, dict):
        if uploaded_file.get("path"):
            return uploaded_file["path"]
        if uploaded_file.get("name"):
            return uploaded_file["name"]
    return uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)


def media_preview_path(media_input):
    media_path = file_to_path(media_input)
    if media_path and os.path.exists(media_path):
        return media_path
    return None


# =============================================================================
# LAZY LOADERS
# =============================================================================

def get_speech_to_bsl():
    global _speech_to_bsl
    if _speech_to_bsl is None:
        vocab_path = DEFAULT_VOCAB if os.path.exists(DEFAULT_VOCAB) else None
        _speech_to_bsl = SpeechToBSL(
            whisper_model="base",
            gloss_mode="simple",
            vocabulary_path=vocab_path,
        )
    return _speech_to_bsl


def get_gloss_converter(mode="simple", api_key=None):
    global _gloss_to_text_simple, _gloss_to_text_groq, _groq_key_in_use
    if mode == "simple":
        if _gloss_to_text_simple is None:
            _gloss_to_text_simple = GlossToText(mode="simple")
        return _gloss_to_text_simple
    if mode == "groq":
        key = (api_key or "").strip() or DEFAULT_GROQ_API_KEY
        if not key:
            raise RuntimeError("Groq API key required")
        if _gloss_to_text_groq is None or _groq_key_in_use != key:
            _gloss_to_text_groq = GlossToText(mode="groq", groq_api_key=key)
            _groq_key_in_use = key
        return _gloss_to_text_groq
    if _gloss_to_text_simple is None:
        _gloss_to_text_simple = GlossToText(mode="simple")
    return _gloss_to_text_simple


def get_tts():
    global _tts, _tts_disabled_reason
    if _tts is False:
        return None
    if _tts is None:
        if not os.path.exists(DEFAULT_SPEAKER):
            _tts_disabled_reason = f"Speaker reference missing: {DEFAULT_SPEAKER}"
            _tts = False
            print(f"TTS disabled: {_tts_disabled_reason}")
            return None
        try:
            _tts = CoquiTTS(speaker_wav=DEFAULT_SPEAKER)
        except Exception as e:
            _tts_disabled_reason = str(e)
            _tts = False
            print(f"TTS disabled: {_tts_disabled_reason}")
            return None
    return _tts


def get_avatar_renderer():
    global _avatar_renderer
    if _avatar_renderer is None:
        video_map = DEFAULT_VIDEO_MAP if os.path.exists(DEFAULT_VIDEO_MAP) else None
        _avatar_renderer = BSLAvatarRenderer(
            video_dir=DEFAULT_VIDEO_DIR,
            video_map_path=video_map,
        )
    return _avatar_renderer


def get_pose_sign_renderer():
    global _pose_sign_renderer
    if _pose_sign_renderer is None:
        _pose_sign_renderer = PoseSignRenderer(project_root=Path(project_root))
    return _pose_sign_renderer


def get_avatar_3d_renderer():
    global _avatar_3d_renderer
    if not AVATAR_3D_AVAILABLE:
        raise RuntimeError("Avatar3DRenderer not available.")
    if _avatar_3d_renderer is None:
        pose_renderer = get_pose_sign_renderer()
        _avatar_3d_renderer = Avatar3DRenderer(
            project_root=Path(project_root),
            pose_renderer=pose_renderer,
        )
    return _avatar_3d_renderer


def get_bsl_dict_recognizer():
    global _bsl_dict_recognizer
    if not BSL_DICT_AVAILABLE:
        print("BSL Dict Recognizer not available")
        return None
    if _bsl_dict_recognizer is None:
        try:
            import torch as th
            device = "cuda" if th.cuda.is_available() else "cpu"
            _bsl_dict_recognizer = BSLDictRecognizer(device=device)
            print(f"BSL Dict Recognizer loaded: {_bsl_dict_recognizer.glosses[:5]}...")
        except Exception as e:
            print(f"Failed to load BSL Dict Recognizer: {e}")
            return None
    return _bsl_dict_recognizer


def get_sign_recognizer():
    global _sign_recognizer
    if _sign_recognizer is None:
        if not os.path.exists(DEFAULT_SIGN_MODEL):
            raise FileNotFoundError(f"Recognition model not found: {DEFAULT_SIGN_MODEL}")
        if not os.path.exists(DEFAULT_SIGN_VOCAB):
            raise FileNotFoundError(f"Recognition vocabulary not found: {DEFAULT_SIGN_VOCAB}")
        import torch
        from realtime_recognition import RealtimeRecognizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_stats_path = DEFAULT_SIGN_CLASS_STATS if os.path.exists(DEFAULT_SIGN_CLASS_STATS) else None
        _sign_recognizer = RealtimeRecognizer(
            model_path=DEFAULT_SIGN_MODEL,
            vocab_path=DEFAULT_SIGN_VOCAB,
            device=device,
            confidence_threshold=0.3,
            ema_alpha=0.65,
            class_stats_path=class_stats_path,
            abstain_threshold=0.12,
            margin_threshold=0.02,
            logit_adjustment_tau=0.7,
            disable_logit_adjustment=False,
        )
    _sign_recognizer.reset()
    return _sign_recognizer


# =============================================================================
# DIRECTION 1: BSL -> English / Speech
# =============================================================================

def direction1_glosses_to_text(glosses_input, mode, api_key):
    if not glosses_input or not glosses_input.strip():
        return "Please enter BSL glosses."
    glosses = glosses_input.upper().split()
    try:
        converter = get_gloss_converter(mode, api_key)
        return converter.convert(glosses)
    except Exception as e:
        return f"Error: {str(e)}"


def direction1_text_to_speech(text):
    if not text or not text.strip():
        return None
    try:
        tts = get_tts()
        output_path = tempfile.mktemp(suffix=".wav")
        if tts is not None:
            tts.synthesize(text, output_path)
            return output_path
        if synthesize_with_windows_tts(text, output_path):
            return output_path
        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        try:
            output_path = tempfile.mktemp(suffix=".wav")
            if synthesize_with_windows_tts(text, output_path):
                return output_path
        except Exception:
            pass
        return None


def direction1_full_pipeline(glosses_input, mode, api_key):
    text = direction1_glosses_to_text(glosses_input, mode, api_key)
    if text.startswith("Error") or text.startswith("Please"):
        return text, None
    audio = direction1_text_to_speech(text)
    return text, audio


def direction1_video_swin(video_input, mode, api_key):
    video_path = file_to_path(video_input)
    if not video_path:
        return "", "Please record or upload a BSL video.", None
    if not os.path.exists(video_path):
        return "", f"Error: video file not found: {video_path}", None
    try:
        recognizer = get_bsl_dict_recognizer()
        if recognizer is None:
            return "", "BSL Dict Recognizer not available.", None
        results = recognizer.recognize(video_path, top_k=5)
        if results:
            top_gloss = results[0][0].upper()
            confidence = results[0][1] * 100
            all_predictions = ", ".join([f"{g.upper()} ({c*100:.0f}%)" for g, c in results[:3]])
            gloss_output = f"Top: {top_gloss} ({confidence:.0f}%)\nAlternatives: {all_predictions}"
            gloss_converter = get_gloss_converter(mode, api_key)
            english_text = gloss_converter.convert(top_gloss)
            tts = get_tts()
            audio_path = None
            if tts and english_text.strip():
                audio_path = tempfile.mktemp(suffix=".wav")
                tts.synthesize(english_text, audio_path)
                if not os.path.exists(audio_path):
                    audio_path = None
            return gloss_output, english_text, audio_path
        else:
            return "No signs detected.", "", None
    except Exception as e:
        return "", f"Error: {str(e)}", None


def direction1_video_to_speech(video_input, mode, api_key):
    video_path = file_to_path(video_input)
    if not video_path:
        return "", "Please record or upload a BSL video.", None
    if not os.path.exists(video_path):
        return "", f"Error: video file not found: {video_path}", None
    try:
        import cv2
        from realtime_recognition import RealtimePoseExtractor, SignBuffer, _hand_activity
        recognizer = get_sign_recognizer()
        pose_extractor = RealtimePoseExtractor(swap_hands=False)
        sign_buffer = SignBuffer(window_size=48, stride=12, min_active_frames=12)
        prediction_history = deque(maxlen=5)
        gloss_history = deque(maxlen=20)
        current_prediction = ""
        prev_pose = None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            pose_extractor.close()
            return "", f"Error: failed to open video: {video_path}", None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pose = pose_extractor.extract(frame)
                left_detected, right_detected, motion_score = _hand_activity(pose, prev_pose)
                has_hands = left_detected or right_detected
                is_active_frame = has_hands and (prev_pose is None or motion_score >= 0.002)
                prev_pose = pose.copy()
                pose_sequence = sign_buffer.add_frame(pose, is_active=is_active_frame)
                if pose_sequence is None:
                    continue
                rec = recognizer.recognize(pose_sequence, top_k=5, return_details=True)
                if rec["abstain"] or not rec["top_k"]:
                    continue
                gloss, confidence = rec["top_k"][0]
                prediction_history.append(gloss)
                if prediction_history.count(gloss) >= 3 and confidence >= 0.3:
                    if gloss != current_prediction:
                        current_prediction = gloss
                        if not gloss_history or gloss_history[-1] != gloss:
                            gloss_history.append(gloss)
        finally:
            cap.release()
            pose_extractor.close()
        if not gloss_history:
            return "", "No stable sign sequence detected. Try a clearer video.", None
        gloss_str = " ".join(gloss_history)
        text = direction1_glosses_to_text(gloss_str, mode, api_key)
        if text.startswith("Error") or text.startswith("Please"):
            return gloss_str, text, None
        audio = direction1_text_to_speech(text)
        return gloss_str, text, audio
    except Exception as e:
        return "", f"Error: {str(e)}", None


def direction1_live_stream(camera_index=0, no_speech=False, mode="simple", api_key=None):
    global _live_running
    if _live_running:
        yield None, "Live stream already running.", "", "", None
        return
    try:
        camera_idx = int(camera_index)
    except Exception:
        yield None, f"Error: invalid camera index: {camera_index}", "", "", None
        return
    try:
        import cv2
        from realtime_recognition import (
            RealtimePoseExtractor, SignBuffer, _hand_activity, draw_landmarks,
        )
    except Exception as e:
        yield None, f"Error: failed to import realtime components: {e}", "", "", None
        return
    try:
        recognizer = get_sign_recognizer()
        recognizer.reset()
    except Exception as e:
        yield None, f"Error: failed to initialize recognizer: {e}", "", "", None
        return
    try:
        pose_extractor = RealtimePoseExtractor(swap_hands=False)
    except Exception as e:
        yield None, f"Error: failed to initialize pose extractor: {e}", "", "", None
        return
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        pose_extractor.close()
        yield None, f"Error: failed to open camera {camera_idx}", "", "", None
        return
    _live_running = True
    _live_stop_event.clear()
    _live_clear_event.clear()
    sign_buffer = SignBuffer(window_size=48, stride=12, min_active_frames=12)
    prediction_history = deque(maxlen=5)
    gloss_history = deque(maxlen=10)
    current_prediction = ""
    current_text = ""
    last_predictions = []
    last_audio_path = None
    prev_pose = None
    left_detected = False
    right_detected = False
    motion_score = 0.0
    target_frame_interval = 1.0 / 15.0
    status = f"Live recognition active on camera {camera_idx}."
    try:
        while not _live_stop_event.is_set():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                status = f"Camera read failed on index {camera_idx}."
                yield None, status, " ".join(gloss_history), current_text, last_audio_path
                break
            if _live_clear_event.is_set():
                _live_clear_event.clear()
                sign_buffer.clear()
                recognizer.reset()
                prediction_history.clear()
                gloss_history.clear()
                current_prediction = ""
                current_text = ""
                last_predictions = []
                last_audio_path = None
                prev_pose = None
                status = "History cleared."
            pose = pose_extractor.extract(frame)
            left_detected, right_detected, motion_score = _hand_activity(pose, prev_pose)
            has_hands = left_detected or right_detected
            is_active_frame = has_hands and (prev_pose is None or motion_score >= 0.002)
            prev_pose = pose.copy()
            pose_sequence = sign_buffer.add_frame(pose, is_active=is_active_frame)
            if pose_sequence is not None:
                rec = recognizer.recognize(pose_sequence, top_k=5, return_details=True)
                last_predictions = rec["top_k"]
                if rec["abstain"]:
                    current_prediction = "NO_SIGN"
                elif rec["top_k"]:
                    gloss, confidence = rec["top_k"][0]
                    prediction_history.append(gloss)
                    if prediction_history.count(gloss) >= 3 and confidence >= 0.3:
                        if gloss != current_prediction:
                            current_prediction = gloss
                            gloss_history.append(gloss)
                            recent_glosses = list(gloss_history)[-5:]
                            try:
                                converter = get_gloss_converter(mode, api_key)
                                current_text = converter.convert(recent_glosses)
                            except Exception:
                                current_text = " ".join(recent_glosses).replace("_", " ")
                            if not no_speech:
                                audio_path = direction1_text_to_speech(current_text)
                                if audio_path:
                                    last_audio_path = audio_path
            draw_frame = frame.copy()
            draw_frame = draw_landmarks(draw_frame, pose_extractor, pose)
            if current_prediction:
                gloss_color = (0, 255, 0) if current_prediction != "NO_SIGN" else (0, 180, 255)
                cv2.putText(draw_frame, f"Gloss: {current_prediction}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, gloss_color, 2)
            if current_text:
                cv2.putText(draw_frame, f"Text: {current_text[:60]}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            history_str = " -> ".join(list(gloss_history)[-5:])
            cv2.putText(draw_frame, f"History: {history_str[:80]}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            hand_status = (
                f"Hands: L={'Y' if left_detected else 'N'} "
                f"R={'Y' if right_detected else 'N'} Motion:{motion_score:.4f}"
            )
            hand_color = (0, 255, 0) if has_hands else (0, 0, 255)
            cv2.putText(draw_frame, hand_status, (10, draw_frame.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
            if last_predictions:
                top_str = " | ".join(
                    [f"[{i+1}]{g}:{c:.2f}" for i, (g, c) in enumerate(last_predictions[:5])]
                )
                cv2.putText(draw_frame, f"Top: {top_str[:90]}", (10, draw_frame.shape[0] - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            stream_status = f"{status} Speech: {'OFF' if no_speech else 'ON'} | Mode: {mode}"
            cv2.putText(draw_frame, stream_status[:95], (10, draw_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            frame_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb, status, " ".join(gloss_history), current_text, last_audio_path
            elapsed = time.time() - loop_start
            if elapsed < target_frame_interval:
                time.sleep(target_frame_interval - elapsed)
    finally:
        _live_running = False
        _live_stop_event.clear()
        cap.release()
        pose_extractor.close()


def direction1_stop_live_realtime():
    _live_stop_event.set()
    return "Stopping live stream..."


def direction1_clear_live_history():
    _live_clear_event.set()
    return "Clearing recognition history..."


# =============================================================================
# DIRECTION 2: Speech / Text -> BSL Signing
# =============================================================================

def _format_vocab_coverage(info: dict) -> str:
    coverage = float(info.get("coverage", 0.0)) if info else 0.0
    text = f"Vocabulary coverage: {coverage:.1f}%"
    out_of_vocab = info.get("out_of_vocab", []) if info else []
    if out_of_vocab:
        text += f"\nOut of vocabulary: {', '.join(out_of_vocab[:20])}"
    return text


def _direction2_render_sequence(
    transcription_text: str,
    glosses: list,
    coverage_text: str,
    render_engine: str,
    render_speed: float,
    avatar_gender: str = "Male",
):
    glosses = [str(g).strip().upper() for g in (glosses or []) if str(g).strip()]
    gloss_str = " ".join(glosses)
    if not glosses:
        yield transcription_text, "", coverage_text, None, None, "No glosses generated.", ""
        return

    engine_raw = (render_engine or "Pose Animator").strip()
    engine_aliases = {
        "3D Avatar": "3D Avatar",
        "2D Pose Animator": "Pose Animator",
    }
    engine = engine_aliases.get(engine_raw, engine_raw)
    try:
        speed = float(render_speed)
    except Exception:
        speed = 1.0
    speed = max(0.6, min(1.6, speed))

    if engine == "3D Avatar":
        if not AVATAR_3D_AVAILABLE:
            yield transcription_text, gloss_str, coverage_text, None, None, "3D Avatar renderer not available.", ""
            return
        avatar_gender = "female" if "female" in (render_engine or "").lower() else "male"
        yield transcription_text, gloss_str, coverage_text, None, None, "Loading 3D renderer...", ""
        try:
            renderer_3d = get_avatar_3d_renderer()
        except Exception as _e3d:
            yield transcription_text, gloss_str, coverage_text, None, None, f"3D renderer failed: {_e3d}", ""
            return
        cov3d = renderer_3d.get_coverage(glosses)
        cov3d_text = (
            f"{coverage_text}\n\n3D coverage: {cov3d['coverage']:.1f}%"
            f" ({cov3d['available_count']}/{len(glosses)})"
        )
        if cov3d["missing"]:
            cov3d_text += f"\nMissing: {', '.join(cov3d['missing'][:10])}"
        yield transcription_text, gloss_str, cov3d_text, None, None, "Generating 3D animation...", ""
        try:
            html_3d = renderer_3d.render_sequence_html(
                glosses=glosses,
                speed=speed,
                avatar=avatar_gender.lower() if avatar_gender else "male",
                gloss_str=gloss_str,
            )
            yield transcription_text, gloss_str, cov3d_text, None, None, "3D avatar ready.", html_3d
        except Exception as _e3d:
            yield transcription_text, gloss_str, cov3d_text, None, None, f"3D render failed: {_e3d}", ""
        return

    if engine == "Legacy Clip Avatar":
        legacy_renderer = get_avatar_renderer()
        legacy_cov = legacy_renderer.get_coverage(glosses)
        legacy_coverage = (
            f"{coverage_text}\n\nLegacy clip coverage: {legacy_cov['coverage']:.1f}%"
        )
        if legacy_cov["missing"]:
            legacy_coverage += f"\nMissing videos: {', '.join(legacy_cov['missing'][:10])}"
        yield transcription_text, gloss_str, legacy_coverage, None, None, "Rendering legacy clip avatar...", ""
        avatar_video = render_avatar_video(glosses)
        status = "Legacy clip avatar rendered." if avatar_video else "Legacy clip avatar unavailable."
        yield transcription_text, gloss_str, legacy_coverage, None, avatar_video, status, ""
        return

    yield transcription_text, gloss_str, coverage_text, None, None, "Initializing pose renderer...", ""
    try:
        pose_renderer = get_pose_sign_renderer()
    except Exception as e:
        yield transcription_text, gloss_str, coverage_text, None, None, f"Pose renderer failed: {e}", ""
        return

    pose_cov = pose_renderer.get_coverage(glosses)
    pose_coverage = (
        f"{coverage_text}\n\n"
        f"Pose coverage: {pose_cov['coverage']:.1f}% "
        f"({pose_cov['available_count']}/{len(glosses)})"
    )
    if pose_cov["missing"]:
        pose_coverage += f"\nMissing: {', '.join(pose_cov['missing'][:10])}"

    yield transcription_text, gloss_str, pose_coverage, None, None, "Animating signs...", ""

    last_frame = None
    last_status = "Animation started."
    frame_count = 0
    timeout_seconds = 90.0
    output_path = tempfile.mktemp(suffix=".mp4")
    max_sequence_seconds = 45.0

    writer = None
    try:
        import cv2
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            pose_renderer.output_fps,
            (pose_renderer.canvas_width, pose_renderer.canvas_height),
        )
        if not writer.isOpened():
            writer.release()
            writer = None
            last_status = "Video writer unavailable; preview only."
        start_time = time.time()
        for frame_rgb, status in pose_renderer.render_sequence_frames(
            glosses=glosses,
            speed=speed,
            max_total_seconds=max_sequence_seconds,
        ):
            if (time.time() - start_time) > timeout_seconds:
                last_status = "Render timeout; returning partial output."
                break
            last_frame = frame_rgb
            last_status = status
            frame_count += 1
            if writer is not None:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            yield transcription_text, gloss_str, pose_coverage, frame_rgb, None, status, ""
    except Exception as e:
        last_status = f"Rendering failed: {e}"
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        yield transcription_text, gloss_str, pose_coverage, None, None, last_status, ""
        return

    final_video = None
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        final_video = output_path
    else:
        try:
            final_video = pose_renderer.render_sequence_video(
                glosses=glosses,
                output_path=tempfile.mktemp(suffix=".mp4"),
                speed=speed,
                max_total_seconds=max_sequence_seconds,
            )
        except Exception as e:
            last_status = f"{last_status} MP4 failed: {e}"

    final_status = f"Complete. {frame_count} frames rendered."
    yield transcription_text, gloss_str, pose_coverage, last_frame, final_video, final_status, ""


def direction2_audio_to_signing(audio_input, render_engine, render_speed, avatar_gender="Male"):
    input_path = file_to_path(audio_input)
    if not input_path:
        yield "", "", "", None, None, "Please record or upload audio.", ""
        return
    wav_path = None
    try:
        wav_path = to_wav_16k_mono(input_path)
        pipeline = get_speech_to_bsl()
        result = pipeline.process(wav_path, return_intermediate=True)
        text = result.get("text", "")
        glosses = result.get("glosses", [])
        info = pipeline.text_to_gloss.convert_with_info(text)
        coverage = _format_vocab_coverage(info)
        yield from _direction2_render_sequence(
            transcription_text=text,
            glosses=glosses,
            coverage_text=coverage,
            render_engine=render_engine,
            render_speed=render_speed,
            avatar_gender=avatar_gender,
        )
    except Exception as e:
        yield f"Error: {e}", "", "", None, None, f"Audio pipeline failed: {e}"
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


def direction2_text_to_signing(text_input, render_engine, render_speed, avatar_gender="Male"):
    text = (text_input or "").strip()
    if not text:
        yield "", "", "", None, None, "Please enter text.", ""
        return
    try:
        pipeline = get_speech_to_bsl()
        glosses = pipeline.text_to_gloss.convert(text)
        info = pipeline.text_to_gloss.convert_with_info(text)
        coverage = _format_vocab_coverage(info)
        yield from _direction2_render_sequence(
            transcription_text=text,
            glosses=glosses,
            coverage_text=coverage,
            render_engine=render_engine,
            render_speed=render_speed,
            avatar_gender=avatar_gender,
        )
    except Exception as e:
        yield text, "", "", None, None, f"Text pipeline failed: {e}", ""


def render_avatar_video(glosses: list) -> str:
    try:
        renderer = get_avatar_renderer()
        coverage = renderer.get_coverage(glosses)
        if not coverage["available"]:
            return None
        output_path = tempfile.mktemp(suffix=".mp4")
        result = renderer.render(glosses, output_path)
        if result and os.path.exists(result):
            return result
        return None
    except Exception as e:
        print(f"Avatar render error: {e}")
        return None



# =============================================================================
# GRADIO UI — POLISHED CREATIVE LAYOUT
# =============================================================================




def create_demo():
    """Create the Signlytic AI Gradio interface."""
    groq_ok = bool(DEFAULT_GROQ_API_KEY)
    groq_label = "Connected" if groq_ok else "Not set"
    ff_ok = ensure_ffmpeg_available()
    swin_ok = BSL_DICT_AVAILABLE
    vid_ok = os.path.exists(DEFAULT_VIDEO_DIR)
    video_count = len(get_avatar_renderer().video_index) if vid_ok else 0

    # User-facing system status (not developer diagnostics)
    all_systems_ok = swin_ok and groq_ok and ff_ok and vid_ok
    if all_systems_ok:
        trust_html = '<div class="sig-trust"><span class="sig-badge sig-badge-ok"><span style="width:6px;height:6px;border-radius:50%;background:#15803d;"></span>All systems ready &mdash; recognition, translation, and speech are available</span></div>'
    else:
        issues = []
        if not swin_ok:
            issues.append("sign recognition")
        if not groq_ok:
            issues.append("advanced translation")
        if not ff_ok:
            issues.append("audio processing")
        if not vid_ok:
            issues.append("signing videos")
        trust_html = f'<div class="sig-trust"><span class="sig-badge sig-badge-warn"><span style="width:6px;height:6px;border-radius:50%;background:#a16207;"></span>Some features limited: {", ".join(issues)}</span></div>'

    with gr.Blocks(
        title="Signlytic AI",
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.gray,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Outfit"),
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # ── Skip link (a11y) ──
        gr.HTML('<a href="#bsl-to-english" class="sig-skip">Skip to main content</a>')

        # ── Hero ──
        gr.HTML(f"""
        <div class="sig-hero" role="banner">
            <div class="sig-hero-top">
                <div class="sig-hero-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
                    </svg>
                </div>
                <span class="sig-hero-name">Signlytic AI</span>
            </div>
            <div class="sig-hero-desc">
                Translate between British Sign Language and English.
                Upload a BSL video, speak into your mic, or type text.
                All outputs are shown as text &mdash; no audio required.
            </div>
            <div class="sig-chips">
                <span class="sig-chip"><b>5,203</b> BSL signs recognised</span>
                <span class="sig-chip"><b>100%</b> accuracy on dictionary signs</span>
                <span class="sig-chip"><b>11,573+</b> words supported</span>
                <span class="sig-chip">AI-powered recognition</span>
            </div>
        </div>
        """)

        # ── Trust strip ──
        gr.HTML(trust_html)

        # ── Main Tabs ──
        with gr.Tabs():

            # ══════════════════════════════════════════
            # TAB 1: BSL to English
            # ══════════════════════════════════════════
            with gr.TabItem("BSL to English", id="bsl-to-english"):
                gr.HTML("""
                <div class="sig-sh">Understand BSL Signs</div>
                <div class="sig-sd">Upload a video of BSL signing, or type BSL signs. You will see the English meaning and can listen to it spoken aloud.</div>
                """)

                with gr.Row(equal_height=False):

                    with gr.Column(scale=5):
                        gr.HTML('<div class="sig-col">Input</div>')

                        with gr.Group():
                            gr.HTML('<div class="sig-method">Upload or record a video <span class="sig-rec">Best accuracy</span></div>')
                            gr.HTML('<div style="font-size:0.82rem;color:#6b7280;margin-bottom:0.35rem;">Record yourself signing, or upload a video file. This takes a few seconds to process.</div>')
                            d1_video_input = gr.Video(
                                label="BSL video",
                                sources=["webcam", "upload"],
                            )
                            with gr.Row():
                                d1_swin_btn = gr.Button("Recognise BSL signs", variant="primary", size="lg")
                                d1_video_btn = gr.Button("Recognise (alternative)", variant="secondary")

                        gr.HTML('<hr class="sig-sep">')

                        with gr.Group():
                            gr.HTML('<div class="sig-method">Or type BSL signs</div>')
                            d1_glosses = gr.Textbox(
                                label="Type BSL signs (one word at a time)",
                                placeholder="Example: HELLO MY NAME SARAH",
                                lines=2,
                            )
                            with gr.Row():
                                d1_convert_btn = gr.Button("Translate to English", variant="primary")
                                d1_speak_btn = gr.Button("Translate & speak aloud", variant="secondary")

                        gr.HTML('<hr class="sig-sep">')

                        with gr.Accordion("Live camera (advanced)", open=False):
                            with gr.Row():
                                d1_live_camera_index = gr.Number(label="Camera number", value=0, precision=0)
                                d1_live_no_speech = gr.Checkbox(label="Mute speech", value=False)
                            with gr.Row():
                                d1_live_start_btn = gr.Button("Start live recognition", variant="primary")
                                d1_live_stop_btn = gr.Button("Stop", variant="secondary")
                                d1_live_clear_btn = gr.Button("Clear history", variant="secondary")

                        with gr.Accordion("Advanced settings", open=False):
                            d1_mode = gr.Radio(
                                choices=["simple", "groq"],
                                value="groq" if DEFAULT_GROQ_API_KEY else "simple",
                                label="Translation quality",
                                info="'groq' gives more natural English sentences. 'simple' is faster but less fluent.",
                            )
                            d1_api_key = gr.Textbox(
                                label="API Key (optional, for advanced translation)",
                                type="password",
                                placeholder="Leave blank to use default",
                            )

                    with gr.Column(scale=5):
                        gr.HTML('<div class="sig-col">Results</div>')

                        d1_recorded_preview = gr.Video(label="Your video", interactive=False)
                        d1_live_preview = gr.Image(label="Live camera", streaming=True, type="numpy", interactive=False)
                        d1_live_status = gr.Textbox(label="Status", lines=1, interactive=False)

                        gr.HTML('<div class="sig-result"><div class="sig-result-h">What was signed (BSL)</div></div>')
                        d1_video_glosses = gr.Textbox(label="BSL signs detected", lines=2, interactive=False, show_label=False)

                        gr.HTML('<div class="sig-result sig-result-navy"><div class="sig-result-h">English meaning</div></div>')
                        d1_text_output = gr.Textbox(label="Translation", lines=3, interactive=False, show_label=False)

                        gr.HTML('<div class="sig-result"><div class="sig-result-h">Listen (optional)</div></div>')
                        d1_audio_output = gr.Audio(label="Speech", type="filepath", show_label=False)

                gr.HTML('<div class="sig-bsl-note">These examples use BSL word order, not English grammar.</div>')
                gr.Examples(
                    examples=[
                        ["TOMORROW MEETING WHAT TIME"],
                        ["MY NAME SARAH"],
                        ["YESTERDAY I GO DOCTOR"],
                        ["THANK YOU MUCH"],
                        ["I NOT UNDERSTAND"],
                    ],
                    inputs=[d1_glosses],
                    label="Try these BSL signs",
                )

                # Wire events
                d1_convert_btn.click(fn=direction1_glosses_to_text, inputs=[d1_glosses, d1_mode, d1_api_key], outputs=[d1_text_output])
                d1_speak_btn.click(fn=direction1_full_pipeline, inputs=[d1_glosses, d1_mode, d1_api_key], outputs=[d1_text_output, d1_audio_output])
                d1_video_input.change(fn=media_preview_path, inputs=[d1_video_input], outputs=[d1_recorded_preview])
                d1_video_btn.click(fn=direction1_video_to_speech, inputs=[d1_video_input, d1_mode, d1_api_key], outputs=[d1_video_glosses, d1_text_output, d1_audio_output])
                d1_swin_btn.click(fn=direction1_video_swin, inputs=[d1_video_input, d1_mode, d1_api_key], outputs=[d1_video_glosses, d1_text_output, d1_audio_output])
                d1_live_event = d1_live_start_btn.click(fn=direction1_live_stream, inputs=[d1_live_camera_index, d1_live_no_speech, d1_mode, d1_api_key], outputs=[d1_live_preview, d1_live_status, d1_video_glosses, d1_text_output, d1_audio_output], show_progress="hidden")
                d1_live_stop_btn.click(fn=direction1_stop_live_realtime, outputs=[d1_live_status], cancels=[d1_live_event])
                d1_live_clear_btn.click(fn=direction1_clear_live_history, outputs=[d1_live_status])

            # ══════════════════════════════════════════
            # TAB 2: English to BSL
            # ══════════════════════════════════════════
            with gr.TabItem("English to BSL", id="english-to-bsl"):
                gr.HTML("""
                <div class="sig-sh">Show Me in BSL</div>
                <div class="sig-sd">Speak or type in English. You will see BSL signs and a signing animation.</div>
                """)

                with gr.Row(equal_height=False):

                    with gr.Column(scale=5):
                        gr.HTML('<div class="sig-col">Input</div>')

                        with gr.Group():
                            gr.HTML('<div class="sig-method">Record or upload audio</div>')
                            gr.HTML('<div style="font-size:0.82rem;color:#6b7280;margin-bottom:0.35rem;">Speak clearly in a quiet place. This takes a few seconds to process.</div>')
                            d2_audio_input = gr.Audio(label="Your audio", type="filepath", sources=["microphone", "upload"])
                            d2_audio_btn = gr.Button("Convert to BSL", variant="primary", size="lg")

                        gr.HTML('<hr class="sig-sep">')

                        with gr.Group():
                            gr.HTML('<div class="sig-method">Or type in English</div>')
                            d2_text = gr.Textbox(label="Type what you want to say", placeholder="Example: What time is the meeting tomorrow?", lines=2)
                            d2_text_btn = gr.Button("Convert to BSL", variant="primary")

                        with gr.Accordion("Animation settings", open=False):
                            d2_render_engine = gr.Radio(choices=["Pose Animator", "3D Avatar", "Legacy Clip Avatar"], value="Pose Animator", label="Animation style", info="Pose Animator: skeleton. 3D Avatar: rigged character. Legacy: video clips.")
                            d2_render_speed = gr.Slider(minimum=0.6, maximum=1.6, value=1.0, step=0.1, label="Signing speed")

                    with gr.Column(scale=5):
                        gr.HTML('<div class="sig-col">Results</div>')

                        d2_audio_preview = gr.Audio(label="Your audio", type="filepath", interactive=False)

                        gr.HTML('<div class="sig-result"><div class="sig-result-h">What you said</div></div>')
                        d2_transcription = gr.Textbox(label="Transcription", lines=2, interactive=False, show_label=False)

                        gr.HTML('<div class="sig-result sig-result-navy"><div class="sig-result-h">BSL signs</div></div>')
                        d2_glosses_output = gr.Textbox(label="BSL signs", lines=2, interactive=False, show_label=False)

                        d2_coverage = gr.Textbox(label="Coverage", lines=2, interactive=False)
                        d2_live_preview = gr.Image(label="Signing preview", streaming=True, type="numpy", interactive=False)

                        gr.HTML('<div class="sig-result"><div class="sig-result-h">Signing video</div></div>')
                        d2_avatar_video = gr.Video(label="BSL animation", show_label=False)
                        d2_render_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        d2_avatar_gender = gr.Radio(
                            choices=["Male", "Female"],
                            value="Male",
                            label="Avatar gender",
                            visible=False,
                        )
                        d2_avatar_3d = gr.HTML(value="")

                gr.Examples(
                    examples=[["Hello, my name is John."], ["What time is the meeting?"], ["Thank you very much."], ["I need help please."]],
                    inputs=[d2_text],
                    label="Try these phrases",
                )

                d2_audio_btn.click(fn=direction2_audio_to_signing, inputs=[d2_audio_input, d2_render_engine, d2_render_speed, d2_avatar_gender], outputs=[d2_transcription, d2_glosses_output, d2_coverage, d2_live_preview, d2_avatar_video, d2_render_status, d2_avatar_3d], show_progress="hidden")
                d2_audio_input.change(fn=media_preview_path, inputs=[d2_audio_input], outputs=[d2_audio_preview])
                d2_text_btn.click(fn=direction2_text_to_signing, inputs=[d2_text, d2_render_engine, d2_render_speed, d2_avatar_gender], outputs=[d2_transcription, d2_glosses_output, d2_coverage, d2_live_preview, d2_avatar_video, d2_render_status, d2_avatar_3d], show_progress="hidden")
                d2_render_engine.change(
                    fn=lambda e: gr.update(visible=(e == "3D Avatar")),
                    inputs=[d2_render_engine],
                    outputs=[d2_avatar_gender],
                )

            # ══════════════════════════════════════════
            # TAB 3: Help & Accessibility
            # ══════════════════════════════════════════
            with gr.TabItem("Help & Accessibility", id="help"):
                gr.HTML('<div class="sig-sh">How to Use This App</div><div class="sig-sd">Simple guides for BSL users and hearing users. All outputs are shown as text.</div>')

                gr.HTML("""
                <div class="sig-help">
                    <h4>Understand BSL signs (BSL to English)</h4>
                    <p>
                        <span class="sig-step">1</span> Go to the <strong>BSL to English</strong> tab.<br>
                        <span class="sig-step">2</span> Record yourself signing, or upload a video file.<br>
                        <span class="sig-step">3</span> Click <strong>Recognise BSL signs</strong>.<br>
                        <span class="sig-step">4</span> You will see: the BSL sign name, the English meaning, and a speech player.
                    </p>
                </div>

                <div class="sig-help">
                    <h4>Show me in BSL (English to BSL)</h4>
                    <p>
                        <span class="sig-step">1</span> Go to the <strong>English to BSL</strong> tab.<br>
                        <span class="sig-step">2</span> Record your voice, or type in English.<br>
                        <span class="sig-step">3</span> Click <strong>Convert to BSL</strong>.<br>
                        <span class="sig-step">4</span> You will see: BSL signs and a signing animation video.
                    </p>
                </div>

                <div class="sig-help">
                    <h4>What to expect</h4>
                    <p>
                        Processing takes a few seconds. You will always see text results &mdash; you never need to rely on audio.
                        The system works best with clear, short phrases. Results may not be perfect for complex sentences.
                    </p>
                </div>

                <div class="sig-help">
                    <h4>What are BSL signs / glosses?</h4>
                    <p>
                        A "gloss" is the written name of a BSL sign. For example, the glosses
                        <strong>TOMORROW MEETING WHAT TIME</strong> mean "What time is the meeting tomorrow?"
                        BSL uses a different word order from English &mdash; this is normal.
                    </p>
                </div>

                <div class="sig-help">
                    <h4>Tips for best results</h4>
                    <p>
                        &bull; Sign clearly against a plain, well-lit background.<br>
                        &bull; Keep both hands visible in the camera frame.<br>
                        &bull; For speech input, speak clearly in a quiet place.<br>
                        &bull; Short phrases give better results than long sentences.
                    </p>
                </div>

                <div class="sig-help">
                    <h4>BSL video guides</h4>
                    <p>
                        BSL video instructions for this app are planned for a future update.
                        We are working with BSL users to create clear video guides.
                    </p>
                </div>

                <div class="sig-help">
                    <h4>Contact</h4>
                    <p>
                        Questions or feedback? Email is the easiest way to reach us.<br>
                        You can also contact the developer on
                        <a href="https://www.linkedin.com/in/iyanuoluwa-enoch-oke/" target="_blank">LinkedIn</a>
                        or open an issue on
                        <a href="https://github.com/Iyanuoluwa007/Signlytic_AI/issues" target="_blank">GitHub</a>.
                    </p>
                </div>
                """)

                gr.HTML('<div style="margin-top:1rem;"><div class="sig-sh" style="font-size:1.1rem;">Accessibility Statement</div></div>')
                gr.HTML("""
                <div class="sig-a11y">
                    <div class="sig-a11y-card"><h5>Text for Everything</h5><p>All outputs are shown as text. Speech output is optional &mdash; you never need to hear audio to use this app.</p></div>
                    <div class="sig-a11y-card"><h5>High Contrast</h5><p>Dark text on light backgrounds. Clear visual hierarchy throughout the interface.</p></div>
                    <div class="sig-a11y-card"><h5>Keyboard Navigation</h5><p>All controls are reachable using Tab and Enter. Visible focus rings on every element.</p></div>
                    <div class="sig-a11y-card"><h5>Plain Language</h5><p>Short, clear labels. Technical details are kept in the System tab, not in the main interface.</p></div>
                    <div class="sig-a11y-card"><h5>Clear Output Separation</h5><p>BSL signs, English translation, speech, and video are each shown in their own labelled section.</p></div>
                    <div class="sig-a11y-card"><h5>BSL-Friendly Layout</h5><p>Minimal text density. Large buttons. Clear actions. Designed for BSL-first users.</p></div>
                    <div class="sig-a11y-card"><h5>Responsive Design</h5><p>Works on desktop and tablet screens. Layout adapts without losing readability.</p></div>
                    <div class="sig-a11y-card"><h5>Consistent Structure</h5><p>Every tab follows the same Input / Results pattern. Predictable layout throughout.</p></div>
                </div>
                """)

            # ══════════════════════════════════════════
            # TAB 4: About & System
            # ══════════════════════════════════════════
            with gr.TabItem("About & System", id="about"):
                gr.HTML('<div class="sig-sh">System Overview</div><div class="sig-sd">Architecture, models, and performance benchmarks.</div>')

                gr.HTML("""
                <div class="sig-metrics">
                    <div class="sig-metric"><div class="sig-metric-val">100%</div><div class="sig-metric-label">Top-1 Accuracy</div></div>
                    <div class="sig-metric"><div class="sig-metric-val">5,203</div><div class="sig-metric-label">BSL Signs</div></div>
                    <div class="sig-metric"><div class="sig-metric-val">11,573+</div><div class="sig-metric-label">Glosses</div></div>
                    <div class="sig-metric"><div class="sig-metric-val">GPU</div><div class="sig-metric-label">Accelerated</div></div>
                </div>
                """)

                gr.HTML("""
                <div style="font-size:0.95rem;font-weight:700;color:#111827;margin-bottom:0.4rem;">Technical Architecture</div>
                <table class="sig-table"><thead><tr><th>Component</th><th>Technology</th><th>Details</th></tr></thead><tbody>
                <tr><td><strong>Sign Recognition</strong></td><td>Video-SWIN-T</td><td>Retrieval on 5,203 pre-extracted 768-dim features</td></tr>
                <tr><td><strong>Speech Recognition</strong></td><td>OpenAI Whisper</td><td>Base model, 16 kHz mono</td></tr>
                <tr><td><strong>Text-to-Speech</strong></td><td>Coqui XTTS v2</td><td>Voice cloning with speaker reference</td></tr>
                <tr><td><strong>Language Model</strong></td><td>Groq gpt-oss-120b</td><td>Gloss to natural English</td></tr>
                <tr><td><strong>Avatar</strong></td><td>2D Pose Animator</td><td>Skeleton signing + video export</td></tr>
                <tr><td><strong>Vocabulary</strong></td><td>11,573+ glosses</td><td>BSL-1K + BSLDict datasets</td></tr>
                </tbody></table>
                """)

                gr.HTML("""
                <div style="font-size:0.95rem;font-weight:700;color:#111827;margin-bottom:0.4rem;">Trained Models</div>
                <table class="sig-table"><thead><tr><th>Model</th><th>Language</th><th>Top-1</th><th>Top-5</th></tr></thead><tbody>
                <tr><td><strong>BSL Dict Retrieval</strong></td><td>British</td><td class="green">100%</td><td class="green">100%</td></tr>
                <tr><td>BSL-100</td><td>British</td><td>72.34%</td><td>95.03%</td></tr>
                <tr><td>BSL-500</td><td>British</td><td>59.26%</td><td>89.04%</td></tr>
                <tr><td>Pose Recognition</td><td>ASL</td><td>44.44%</td><td>81.62%</td></tr>
                <tr><td>Multi-Lingual</td><td>ASL+LSF</td><td>20.95%</td><td>49.17%</td></tr>
                </tbody></table>
                """)

                gr.HTML("""
                <div style="font-size:0.95rem;font-weight:700;color:#111827;margin-bottom:0.4rem;">Key Insights</div>
                <div style="font-size:0.86rem;color:#4b5563;line-height:1.6;max-width:760px;">
                <p style="margin-bottom:0.4rem;"><strong>Retrieval over classification.</strong> With one sample per class, cosine similarity on 768-dim SWIN features achieves perfect accuracy.</p>
                <p style="margin-bottom:0.4rem;"><strong>Feature extraction</strong> takes about one hour for 5,203 videos on RTX 4060 (8 GB).</p>
                <p><strong>End-to-end pipeline</strong> unifies vision, language, speech, and animation in one bidirectional system.</p>
                </div>
                """)

        # ── Footer ──
        gr.HTML("""
        <div class="sig-footer" role="contentinfo">
            <div class="sig-footer-brand">Signlytic AI</div>
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.55);margin-bottom:0.35rem;">Bridging communication between BSL users and hearing communities</div>
            <div class="sig-footer-author">Developed by <a href="https://www.linkedin.com/in/iyanuoluwa-enoch-oke/" target="_blank" rel="noopener">Oke Iyanuoluwa Enoch</a></div>
            <div class="sig-footer-links">
                <a href="https://github.com/Iyanuoluwa007/Signlytic_AI" target="_blank">GitHub</a>
                <a href="https://signlytic-ai-website.vercel.app" target="_blank">Website</a>
                <a href="https://huggingface.co/spaces/Iyanuoluwa007/signlytic-ai" target="_blank">HuggingFace</a>
            </div>
            <div class="sig-footer-sub">Independent Robotics & AI Systems Engineer &middot; v2.0 &middot; March 2026</div>
        </div>
        """)

    return demo


# =============================================================================
# MAIN
# =============================================================================



def main():
    parser = argparse.ArgumentParser(description="Signlytic AI - BSL Translation System")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()

    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("=" * 60)
    print("  Signlytic AI - BSL Translation System")
    print("  Developed by Oke Iyanuoluwa Enoch")
    print("=" * 60)
    print(f"  Vocabulary:    {DEFAULT_VOCAB}")
    print(f"  Video dir:     {DEFAULT_VIDEO_DIR}")
    print(f"  FFmpeg:        {'OK' if ensure_ffmpeg_available() else 'NOT FOUND'}")
    print(f"  GROQ_API_KEY:  {'FOUND' if DEFAULT_GROQ_API_KEY else 'NOT FOUND'}")
    print(f"  BSL Dict:      {'AVAILABLE' if BSL_DICT_AVAILABLE else 'NOT AVAILABLE'}")
    print("=" * 60)

    demo = create_demo()
    demo.launch(allowed_paths=[str(Path(project_root) / 'data' / 'avatars')], 
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        pwa=True,
    )


if __name__ == "__main__":
    main()
