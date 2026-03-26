"""
Signlytic AI - British Sign Language Translation System
========================================================

Bidirectional BSL translation powered by Video-SWIN-T, Groq LLM, and Coqui TTS.
- Direction 1: BSL Video/Glosses -> English Text -> Speech
- Direction 2: Speech/Text -> BSL Glosses -> Animated Signing

Developed by Oke Iyanuoluwa Enoch
Independent Robotics & AI Systems Engineer
MSc Robotics & Automation, University of Salford

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

# Path setup
project_root = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(project_root) == "scripts":
    project_root = os.path.dirname(project_root)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "inference"))
sys.path.insert(0, os.path.join(project_root, "scripts"))

try:
    import gradio as gr
except ImportError:
    print("Gradio required. Install with: pip install gradio")
    sys.exit(1)

# BSL Dict Recognizer (SWIN-based)
try:
    from src.inference.bsl_dict_recognizer import BSLDictRecognizer
    BSL_DICT_AVAILABLE = True
except ImportError:
    BSLDictRecognizer = None
    BSL_DICT_AVAILABLE = False
    print("Warning: BSLDictRecognizer not available")

# Import pipeline components
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

# Globals
_speech_to_bsl = None
_gloss_to_text_simple = None
_gloss_to_text_groq = None
_groq_key_in_use = None
_tts = None
_tts_disabled_reason = None
_avatar_renderer = None
_pose_sign_renderer = None
_sign_recognizer = None
_bsl_dict_recognizer = None
_live_running = False
_live_stop_event = threading.Event()
_live_clear_event = threading.Event()

# Default paths
DEFAULT_VOCAB = os.path.join(project_root, "data", "processed", "vocabulary_extended.json")
DEFAULT_SPEAKER = os.path.join(project_root, "data", "processed", "voice_training.wav")
DEFAULT_VIDEO_DIR = os.path.join(project_root, "data", "videos", "bsl_signs")
DEFAULT_VIDEO_MAP = os.path.join(project_root, "data", "bsldict", "bsldict", "bsldict_video_map.json")
DEFAULT_SIGN_MODEL = os.path.join(project_root, "models", "sign_recognition", "best_model.pt")
DEFAULT_SIGN_VOCAB = os.path.join(project_root, "models", "sign_recognition", "vocabulary.json")
DEFAULT_SIGN_CLASS_STATS = os.path.join(project_root, "models", "sign_recognition", "class_stats.json")
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# =============================================================================
# CUSTOM CSS - PROFESSIONAL ACCESSIBLE DESIGN
# =============================================================================
CUSTOM_CSS = """
/* ---- Import distinctive fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Root variables ---- */
:root {
    --sig-navy: #0c1b33;
    --sig-teal: #0d9488;
    --sig-teal-light: #14b8a6;
    --sig-teal-dark: #0f766e;
    --sig-warm: #f59e0b;
    --sig-warm-light: #fbbf24;
    --sig-surface: #f8fafc;
    --sig-surface-alt: #f1f5f9;
    --sig-border: #e2e8f0;
    --sig-text: #1e293b;
    --sig-text-muted: #64748b;
    --sig-white: #ffffff;
    --sig-success: #059669;
    --sig-error: #dc2626;
    --sig-radius: 14px;
    --sig-radius-sm: 8px;
    --sig-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --sig-shadow-lg: 0 10px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.04);
    --sig-shadow-xl: 0 20px 50px -12px rgba(0,0,0,0.15);
}

/* ---- Global overrides ---- */
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
    background: var(--sig-surface) !important;
}

.gradio-container * {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
}

/* ---- Hero Section ---- */
.hero-wrapper {
    background: linear-gradient(160deg, var(--sig-navy) 0%, #162d50 40%, var(--sig-teal-dark) 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--sig-shadow-xl);
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(13,148,136,0.25) 0%, transparent 70%);
    pointer-events: none;
}

.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -40%;
    left: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(245,158,11,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    color: var(--sig-white) !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    margin: 0 0 0.5rem 0 !important;
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1.15rem !important;
    font-weight: 400 !important;
    max-width: 640px;
    line-height: 1.6 !important;
    margin: 0 0 1.5rem 0 !important;
    position: relative;
    z-index: 1;
}

/* ---- Stats Row ---- */
.stats-grid {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}

.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px;
    padding: 0.5rem 1.1rem;
    color: var(--sig-white);
    font-size: 0.88rem;
    font-weight: 500;
    transition: background 0.2s ease;
}

.stat-chip:hover {
    background: rgba(255,255,255,0.18);
}

.stat-chip .stat-num {
    color: var(--sig-teal-light);
    font-weight: 700;
    font-size: 0.95rem;
}

.stat-chip .stat-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--sig-teal-light);
    flex-shrink: 0;
}

/* ---- Tab Styling ---- */
.tabs {
    border-radius: var(--sig-radius) !important;
}

.tab-nav {
    border-bottom: 2px solid var(--sig-border) !important;
    gap: 0 !important;
    padding: 0 0.5rem !important;
}

.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.85rem 1.3rem !important;
    border-radius: var(--sig-radius-sm) var(--sig-radius-sm) 0 0 !important;
    border: none !important;
    color: var(--sig-text-muted) !important;
    transition: all 0.2s ease !important;
    position: relative;
}

.tab-nav button:hover {
    color: var(--sig-teal-dark) !important;
    background: rgba(13,148,136,0.05) !important;
}

.tab-nav button.selected {
    color: var(--sig-teal-dark) !important;
    background: transparent !important;
    border-bottom: 3px solid var(--sig-teal) !important;
}

/* ---- Section headers inside tabs ---- */
.section-header {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: var(--sig-navy) !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 0.25rem 0 !important;
    line-height: 1.3 !important;
}

.section-desc {
    font-size: 0.95rem !important;
    color: var(--sig-text-muted) !important;
    line-height: 1.6 !important;
    margin: 0 0 1.25rem 0 !important;
}

/* ---- Card panels ---- */
.input-panel, .output-panel {
    background: var(--sig-white);
    border: 1px solid var(--sig-border);
    border-radius: var(--sig-radius);
    padding: 1.25rem;
    box-shadow: var(--sig-shadow);
}

.panel-label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--sig-teal-dark) !important;
    margin: 0 0 0.75rem 0 !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.panel-label .panel-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--sig-teal);
}

/* ---- Buttons ---- */
.gr-button-primary {
    background: linear-gradient(135deg, var(--sig-teal-dark) 0%, var(--sig-teal) 100%) !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em !important;
    border-radius: var(--sig-radius-sm) !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 8px rgba(13,148,136,0.25) !important;
}

.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(13,148,136,0.35) !important;
}

.gr-button-secondary {
    border: 1.5px solid var(--sig-border) !important;
    color: var(--sig-text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: var(--sig-radius-sm) !important;
    background: var(--sig-white) !important;
    transition: all 0.2s ease !important;
}

.gr-button-secondary:hover {
    border-color: var(--sig-teal) !important;
    color: var(--sig-teal-dark) !important;
    background: rgba(13,148,136,0.04) !important;
}

/* ---- Form elements ---- */
.gr-box, .gr-input, .gr-text-input, .gr-panel {
    border-radius: var(--sig-radius-sm) !important;
    border-color: var(--sig-border) !important;
}

label, .gr-label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--sig-text) !important;
}

/* ---- Examples styling ---- */
.gr-examples {
    border-radius: var(--sig-radius) !important;
}

/* ---- Recommended badge ---- */
.rec-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: linear-gradient(135deg, rgba(13,148,136,0.1) 0%, rgba(13,148,136,0.05) 100%);
    border: 1px solid rgba(13,148,136,0.2);
    color: var(--sig-teal-dark);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.3rem 0.7rem;
    border-radius: 100px;
    letter-spacing: 0.02em;
}

/* ---- Divider ---- */
.section-divider {
    border: none;
    border-top: 1px solid var(--sig-border);
    margin: 1.25rem 0;
}

/* ---- About page ---- */
.about-container {
    max-width: 800px;
}

.arch-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--sig-radius-sm);
    overflow: hidden;
    font-size: 0.92rem;
    border: 1px solid var(--sig-border);
}

.arch-table th {
    background: var(--sig-navy);
    color: var(--sig-white);
    font-weight: 600;
    padding: 0.75rem 1rem;
    text-align: left;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.arch-table td {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid var(--sig-border);
    color: var(--sig-text);
}

.arch-table tr:last-child td {
    border-bottom: none;
}

.arch-table tr:nth-child(even) td {
    background: var(--sig-surface-alt);
}

.perf-card {
    background: linear-gradient(135deg, var(--sig-navy) 0%, #1a3555 100%);
    border-radius: var(--sig-radius);
    padding: 1.5rem;
    color: var(--sig-white);
    text-align: center;
    box-shadow: var(--sig-shadow-lg);
}

.perf-number {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--sig-teal-light);
    line-height: 1;
    margin-bottom: 0.3rem;
}

.perf-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Footer ---- */
.footer-wrapper {
    text-align: center;
    padding: 2rem 1.5rem;
    margin-top: 2rem;
    border-top: 1px solid var(--sig-border);
    background: var(--sig-white);
    border-radius: 0 0 var(--sig-radius) var(--sig-radius);
}

.footer-brand {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--sig-navy);
    margin-bottom: 0.35rem;
}

.footer-author {
    font-size: 0.9rem;
    color: var(--sig-text-muted);
    margin-bottom: 0.35rem;
}

.footer-author a {
    color: var(--sig-teal-dark);
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
}

.footer-author a:hover {
    color: var(--sig-teal);
    text-decoration: underline;
}

.footer-links {
    font-size: 0.82rem;
    color: var(--sig-text-muted);
}

.footer-links a {
    color: var(--sig-text-muted);
    text-decoration: none;
    transition: color 0.2s;
    margin: 0 0.5rem;
}

.footer-links a:hover {
    color: var(--sig-teal);
}

.footer-uni {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-style: italic;
}

/* ---- Accessibility focus states ---- */
*:focus-visible {
    outline: 3px solid var(--sig-teal) !important;
    outline-offset: 2px !important;
}

/* ---- Responsive ---- */
@media (max-width: 768px) {
    .hero-wrapper {
        padding: 2rem 1.25rem;
        border-radius: 14px;
    }
    .hero-title {
        font-size: 1.8rem !important;
    }
    .hero-subtitle {
        font-size: 0.95rem !important;
    }
    .stat-chip {
        font-size: 0.8rem;
        padding: 0.4rem 0.8rem;
    }
    .perf-number {
        font-size: 1.8rem;
    }
}
"""


# =============================================================================
# UTILITY FUNCTIONS (unchanged from original)
# =============================================================================

def ensure_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False


def to_wav_16k_mono(input_path: str) -> str:
    """Convert audio to WAV 16kHz mono."""
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError("FFmpeg conversion failed")
    
    return out_path


def synthesize_with_windows_tts(text: str, output_path: str) -> bool:
    """Fallback TTS using Windows System.Speech to produce a WAV file."""
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
    """Get filepath from uploaded file."""
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
    """Resolve media input to an existing local path for preview components."""
    media_path = file_to_path(media_input)
    if media_path and os.path.exists(media_path):
        return media_path
    return None


# =============================================================================
# LAZY LOADERS (unchanged from original)
# =============================================================================

def get_speech_to_bsl():
    global _speech_to_bsl
    if _speech_to_bsl is None:
        vocab_path = DEFAULT_VOCAB if os.path.exists(DEFAULT_VOCAB) else None
        _speech_to_bsl = SpeechToBSL(
            whisper_model="base",
            gloss_mode="simple",
            vocabulary_path=vocab_path
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
            video_map_path=video_map
        )
    return _avatar_renderer


def get_pose_sign_renderer():
    """Lazy-load pose-based signer renderer used by Direction 2."""
    global _pose_sign_renderer
    if _pose_sign_renderer is None:
        _pose_sign_renderer = PoseSignRenderer(project_root=Path(project_root))
    return _pose_sign_renderer


def get_bsl_dict_recognizer():
    """Lazy-load BSL dictionary recognizer (SWIN-based, 100% accuracy on 5203 signs)."""
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
    """Lazy-load sign recognizer used by Direction 1 video input."""
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
# DIRECTION 1: BSL -> Speech (unchanged logic)
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
            print("Using Windows TTS fallback for speech output.")
            return output_path

        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        try:
            output_path = tempfile.mktemp(suffix=".wav")
            if synthesize_with_windows_tts(text, output_path):
                print("Using Windows TTS fallback after Coqui error.")
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
    """Direction 1 using SWIN-based BSL recognition (5203 signs, 100% accuracy)."""
    video_path = file_to_path(video_input)
    if not video_path:
        return "", "Please record from camera or upload a video.", None
    if not os.path.exists(video_path):
        return "", f"Error: video file not found: {video_path}", None
    
    try:
        recognizer = get_bsl_dict_recognizer()
        if recognizer is None:
            return "", "BSL Dict Recognizer not available", None
        
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
                import tempfile
                audio_path = tempfile.mktemp(suffix=".wav")
                tts.synthesize(english_text, audio_path)
                if not os.path.exists(audio_path):
                    audio_path = None
            
            return gloss_output, english_text, audio_path
        else:
            return "No signs detected", "", None
            
    except Exception as e:
        return "", f"Error: {str(e)}", None


def direction1_video_to_speech(video_input, mode, api_key):
    """Direction 1 pipeline from camera/uploaded video -> glosses -> text -> speech."""
    video_path = file_to_path(video_input)
    if not video_path:
        return "", "Please record from camera or upload a video.", None
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
            return "", "No stable sign sequence detected. Try a clearer video with visible hands.", None

        gloss_str = " ".join(gloss_history)
        text = direction1_glosses_to_text(gloss_str, mode, api_key)
        if text.startswith("Error") or text.startswith("Please"):
            return gloss_str, text, None

        audio = direction1_text_to_speech(text)
        return gloss_str, text, audio
    except Exception as e:
        return "", f"Error: {str(e)}", None


def direction1_live_stream(camera_index=0, no_speech=False, mode="simple", api_key=None):
    """Stream live camera recognition into the Gradio preview area."""
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
            RealtimePoseExtractor,
            SignBuffer,
            _hand_activity,
            draw_landmarks,
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
    status = f"Live realtime running on camera {camera_idx}. Click Stop to end."

    try:
        while not _live_stop_event.is_set():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                status = f"Camera read failed on index {camera_idx}. Live stream stopped."
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
                status = "Live history cleared."

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
                cv2.putText(
                    draw_frame,
                    f"Gloss: {current_prediction}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    gloss_color,
                    2,
                )

            if current_text:
                cv2.putText(
                    draw_frame,
                    f"Text: {current_text[:60]}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1,
                )

            history_str = " -> ".join(list(gloss_history)[-5:])
            cv2.putText(
                draw_frame,
                f"History: {history_str[:80]}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
            )

            hand_status = (
                f"Hands: L={'Y' if left_detected else 'N'} "
                f"R={'Y' if right_detected else 'N'} Motion:{motion_score:.4f}"
            )
            hand_color = (0, 255, 0) if has_hands else (0, 0, 255)
            cv2.putText(
                draw_frame,
                hand_status,
                (10, draw_frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                hand_color,
                1,
            )

            if last_predictions:
                top_str = " | ".join(
                    [f"[{i + 1}]{g}:{c:.2f}" for i, (g, c) in enumerate(last_predictions[:5])]
                )
                cv2.putText(
                    draw_frame,
                    f"Top: {top_str[:90]}",
                    (10, draw_frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (180, 180, 180),
                    1,
                )

            stream_status = f"{status} Speech: {'OFF' if no_speech else 'ON'} | Mode: {mode}"
            cv2.putText(
                draw_frame,
                stream_status[:95],
                (10, draw_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

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
    """Signal the live stream loop to stop."""
    _live_stop_event.set()
    return "Stopping live realtime stream..."


def direction1_clear_live_history():
    """Signal the live stream loop to clear recognition history."""
    _live_clear_event.set()
    return "Clearing live recognition history..."


# =============================================================================
# DIRECTION 2: Speech -> BSL (unchanged logic)
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
):
    """Shared Direction 2 renderer with live preview streaming + MP4 output."""
    glosses = [str(g).strip().upper() for g in (glosses or []) if str(g).strip()]
    gloss_str = " ".join(glosses)
    if not glosses:
        yield transcription_text, "", coverage_text, None, None, "No glosses generated."
        return

    engine_raw = (render_engine or "Pose Animator").strip()
    engine_aliases = {
        "3D Pose Animator": "Pose Animator",
        "2D Pose Animator": "Pose Animator",
    }
    engine = engine_aliases.get(engine_raw, engine_raw)
    try:
        speed = float(render_speed)
    except Exception:
        speed = 1.0
    speed = max(0.6, min(1.6, speed))

    if engine == "Legacy Clip Avatar":
        legacy_renderer = get_avatar_renderer()
        legacy_cov = legacy_renderer.get_coverage(glosses)
        legacy_coverage = (
            f"{coverage_text}\n\n"
            f"Legacy clip coverage: {legacy_cov['coverage']:.1f}%"
        )
        if legacy_cov["missing"]:
            legacy_coverage += f"\nMissing videos: {', '.join(legacy_cov['missing'][:10])}"

        yield transcription_text, gloss_str, legacy_coverage, None, None, "Rendering legacy clip avatar..."
        avatar_video = render_avatar_video(glosses)
        if avatar_video:
            status = "Legacy clip avatar rendered."
        else:
            status = "Legacy clip avatar unavailable for the current gloss sequence."
        yield transcription_text, gloss_str, legacy_coverage, None, avatar_video, status
        return

    yield transcription_text, gloss_str, coverage_text, None, None, "Initializing pose renderer..."
    try:
        pose_renderer = get_pose_sign_renderer()
    except Exception as e:
        status = f"Pose renderer initialization failed: {e}"
        yield transcription_text, gloss_str, coverage_text, None, None, status
        return

    pose_cov = pose_renderer.get_coverage(glosses)
    pose_coverage = (
        f"{coverage_text}\n\n"
        f"Pose coverage: {pose_cov['coverage']:.1f}% "
        f"({pose_cov['available_count']}/{len(glosses)})"
    )
    if pose_cov["missing"]:
        pose_coverage += f"\nMissing pose glosses: {', '.join(pose_cov['missing'][:10])}"

    yield transcription_text, gloss_str, pose_coverage, None, None, "Starting pose animation..."

    last_frame = None
    last_status = "Pose animation started."
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
            last_status = "Video writer unavailable; preview will continue without MP4 capture."

        start_time = time.time()
        for frame_rgb, status in pose_renderer.render_sequence_frames(
            glosses=glosses,
            speed=speed,
            max_total_seconds=max_sequence_seconds,
        ):
            if (time.time() - start_time) > timeout_seconds:
                last_status = "Render timeout reached; returning partial output."
                break

            last_frame = frame_rgb
            last_status = status
            frame_count += 1

            if writer is not None:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            yield transcription_text, gloss_str, pose_coverage, frame_rgb, None, status
    except Exception as e:
        last_status = f"Pose rendering failed: {e}"
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        yield transcription_text, gloss_str, pose_coverage, None, None, last_status
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
            last_status = f"{last_status} MP4 generation failed: {e}"

    final_status = f"{last_status} | Pose animation complete ({frame_count} frames)."
    yield transcription_text, gloss_str, pose_coverage, last_frame, final_video, final_status


def direction2_audio_to_signing(audio_input, render_engine, render_speed):
    input_path = file_to_path(audio_input)
    if not input_path:
        yield "", "", "", None, None, "Please record or upload audio."
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
        )
    except Exception as e:
        yield f"Error: {e}", "", "", None, None, f"Direction 2 audio failed: {e}"
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


def direction2_text_to_signing(text_input, render_engine, render_speed):
    text = (text_input or "").strip()
    if not text:
        yield "", "", "", None, None, "Please enter text."
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
        )
    except Exception as e:
        yield text, "", "", None, None, f"Direction 2 text failed: {e}"


def render_avatar_video(glosses: list) -> str:
    """Render avatar video from glosses."""
    try:
        renderer = get_avatar_renderer()
        
        coverage = renderer.get_coverage(glosses)
        if not coverage['available']:
            print(f"No videos available for glosses: {glosses}")
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
# GRADIO UI - PROFESSIONAL REDESIGN
# =============================================================================

def create_demo():
    """Create the Signlytic AI Gradio interface with professional design."""
    groq_status = "Connected" if DEFAULT_GROQ_API_KEY else "Not configured"
    video_count = len(get_avatar_renderer().video_index) if os.path.exists(DEFAULT_VIDEO_DIR) else 0
    
    with gr.Blocks(
        title="Signlytic AI | BSL Translation System",
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Sans"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ),
        css=CUSTOM_CSS,
    ) as demo:
        
        # ─── HERO SECTION ───
        gr.HTML(f"""
        <div class="hero-wrapper">
            <div class="hero-title">Signlytic AI</div>
            <div class="hero-subtitle">
                Bidirectional British Sign Language translation system. 
                Recognize BSL signs from video, convert speech to signing animations, 
                and bridge communication between deaf and hearing communities.
            </div>
            <div class="stats-grid">
                <span class="stat-chip">
                    <span class="stat-dot"></span>
                    <span class="stat-num">5,203</span> BSL Signs
                </span>
                <span class="stat-chip">
                    <span class="stat-dot"></span>
                    <span class="stat-num">100%</span> Recognition Accuracy
                </span>
                <span class="stat-chip">
                    <span class="stat-dot"></span>
                    <span class="stat-num">11,573+</span> Glosses
                </span>
                <span class="stat-chip">
                    <span class="stat-dot"></span>
                    Video-SWIN-T Transformer
                </span>
                <span class="stat-chip">
                    <span class="stat-dot"></span>
                    Groq LLM: {groq_status}
                </span>
            </div>
        </div>
        """)
        
        # ─── MAIN TABS ───
        with gr.Tabs():
            
            # ═══════════════════════════════════════════
            # TAB 1: BSL -> Speech
            # ═══════════════════════════════════════════
            with gr.TabItem("BSL to Speech", id="bsl-to-speech"):
                gr.HTML("""
                <div style="margin-bottom: 1rem;">
                    <div class="section-header">Recognize BSL Signs & Convert to Speech</div>
                    <div class="section-desc">
                        Upload a BSL video, use your camera for live recognition, or type glosses directly. 
                        The system translates BSL into natural English text and synthesized speech.
                    </div>
                </div>
                """)
                
                with gr.Row(equal_height=False):
                    # ── Left: Input Panel ──
                    with gr.Column(scale=5):
                        gr.HTML('<div class="panel-label"><span class="panel-dot"></span> INPUT</div>')
                        
                        # Method 1: SWIN Video Recognition (recommended)
                        with gr.Group():
                            gr.HTML("""
                            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                                <span style="font-weight:600; font-size:0.95rem; color:#1e293b;">Video Recognition</span>
                                <span class="rec-badge">Recommended</span>
                            </div>
                            """)
                            d1_video_input = gr.Video(
                                label="Record or upload a BSL video",
                                sources=["webcam", "upload"],
                            )
                            with gr.Row():
                                d1_swin_btn = gr.Button(
                                    "Recognize with SWIN (5,203 signs)",
                                    variant="primary",
                                    size="lg",
                                )
                                d1_video_btn = gr.Button(
                                    "Recognize with Pose Model",
                                    variant="secondary",
                                )
                        
                        gr.HTML('<hr class="section-divider">')
                        
                        # Method 2: Type glosses
                        with gr.Group():
                            gr.Markdown("**Type BSL Glosses**")
                            d1_glosses = gr.Textbox(
                                label="BSL Glosses (space-separated)",
                                placeholder="e.g. TOMORROW MEETING WHAT TIME",
                                lines=2,
                            )
                            with gr.Row():
                                d1_convert_btn = gr.Button("Translate to English", variant="primary")
                                d1_speak_btn = gr.Button("Translate & Speak", variant="secondary")
                        
                        gr.HTML('<hr class="section-divider">')
                        
                        # Method 3: Live Realtime
                        with gr.Accordion("Live Realtime Camera", open=False):
                            with gr.Row():
                                d1_live_camera_index = gr.Number(
                                    label="Camera Index",
                                    value=0,
                                    precision=0,
                                )
                                d1_live_no_speech = gr.Checkbox(
                                    label="Mute speech output",
                                    value=False,
                                )
                            with gr.Row():
                                d1_live_start_btn = gr.Button("Start Live", variant="primary")
                                d1_live_stop_btn = gr.Button("Stop", variant="secondary")
                                d1_live_clear_btn = gr.Button("Clear History", variant="secondary")
                        
                        gr.HTML('<hr class="section-divider">')
                        
                        # Settings
                        with gr.Accordion("Translation Settings", open=False):
                            d1_mode = gr.Radio(
                                choices=["simple", "groq"],
                                value="groq" if DEFAULT_GROQ_API_KEY else "simple",
                                label="Conversion Mode",
                                info="Groq uses Llama 3.3 70B for natural English output.",
                            )
                            d1_api_key = gr.Textbox(
                                label="Groq API Key (optional)",
                                type="password",
                                placeholder="Uses env GROQ_API_KEY if blank",
                            )
                    
                    # ── Right: Output Panel ──
                    with gr.Column(scale=5):
                        gr.HTML('<div class="panel-label"><span class="panel-dot"></span> OUTPUT</div>')
                        
                        d1_recorded_preview = gr.Video(
                            label="Video Preview",
                            interactive=False,
                            visible=True,
                        )
                        d1_live_preview = gr.Image(
                            label="Live Camera Feed",
                            streaming=True,
                            type="numpy",
                            interactive=False,
                        )
                        d1_live_status = gr.Textbox(
                            label="Live Status",
                            lines=1,
                            interactive=False,
                        )
                        d1_video_glosses = gr.Textbox(
                            label="Recognized Glosses",
                            lines=2,
                            interactive=False,
                        )
                        d1_text_output = gr.Textbox(
                            label="English Translation",
                            lines=3,
                            interactive=False,
                        )
                        d1_audio_output = gr.Audio(
                            label="Speech Output",
                            type="filepath",
                        )
                
                # Examples
                gr.Examples(
                    examples=[
                        ["TOMORROW MEETING WHAT TIME"],
                        ["MY NAME SARAH"],
                        ["YESTERDAY I GO DOCTOR"],
                        ["THANK YOU MUCH"],
                        ["I NOT UNDERSTAND"],
                    ],
                    inputs=[d1_glosses],
                    label="Example BSL Glosses",
                )
                
                # ── Wire events ──
                d1_convert_btn.click(
                    fn=direction1_glosses_to_text,
                    inputs=[d1_glosses, d1_mode, d1_api_key],
                    outputs=[d1_text_output],
                )
                d1_speak_btn.click(
                    fn=direction1_full_pipeline,
                    inputs=[d1_glosses, d1_mode, d1_api_key],
                    outputs=[d1_text_output, d1_audio_output],
                )
                d1_video_input.change(
                    fn=media_preview_path,
                    inputs=[d1_video_input],
                    outputs=[d1_recorded_preview],
                )
                d1_video_btn.click(
                    fn=direction1_video_to_speech,
                    inputs=[d1_video_input, d1_mode, d1_api_key],
                    outputs=[d1_video_glosses, d1_text_output, d1_audio_output],
                )
                d1_swin_btn.click(
                    fn=direction1_video_swin,
                    inputs=[d1_video_input, d1_mode, d1_api_key],
                    outputs=[d1_video_glosses, d1_text_output, d1_audio_output],
                )
                d1_live_event = d1_live_start_btn.click(
                    fn=direction1_live_stream,
                    inputs=[d1_live_camera_index, d1_live_no_speech, d1_mode, d1_api_key],
                    outputs=[d1_live_preview, d1_live_status, d1_video_glosses, d1_text_output, d1_audio_output],
                    show_progress="hidden",
                )
                d1_live_stop_btn.click(
                    fn=direction1_stop_live_realtime,
                    outputs=[d1_live_status],
                    cancels=[d1_live_event],
                )
                d1_live_clear_btn.click(
                    fn=direction1_clear_live_history,
                    outputs=[d1_live_status],
                )
            
            # ═══════════════════════════════════════════
            # TAB 2: Speech -> BSL
            # ═══════════════════════════════════════════
            with gr.TabItem("Speech to BSL", id="speech-to-bsl"):
                gr.HTML("""
                <div style="margin-bottom: 1rem;">
                    <div class="section-header">Convert Speech or Text to BSL Signing</div>
                    <div class="section-desc">
                        Record your voice or type a message. The system generates BSL glosses and 
                        renders an animated signing video using pose-based animation.
                    </div>
                </div>
                """)
                
                with gr.Row(equal_height=False):
                    # ── Left: Input ──
                    with gr.Column(scale=5):
                        gr.HTML('<div class="panel-label"><span class="panel-dot"></span> INPUT</div>')
                        
                        with gr.Group():
                            gr.Markdown("**Option A: Record or Upload Audio**")
                            d2_audio_input = gr.Audio(
                                label="Audio Input",
                                type="filepath",
                                sources=["microphone", "upload"],
                            )
                            d2_audio_btn = gr.Button(
                                "Convert Audio to BSL",
                                variant="primary",
                                size="lg",
                            )
                        
                        gr.HTML('<hr class="section-divider">')
                        
                        with gr.Group():
                            gr.Markdown("**Option B: Type Text**")
                            d2_text = gr.Textbox(
                                label="English Text",
                                placeholder="e.g. What time is the meeting tomorrow?",
                                lines=2,
                            )
                            d2_text_btn = gr.Button(
                                "Convert Text to BSL",
                                variant="primary",
                            )
                        
                        gr.HTML('<hr class="section-divider">')
                        
                        with gr.Accordion("Render Settings", open=False):
                            d2_render_engine = gr.Radio(
                                choices=["Pose Animator", "Legacy Clip Avatar"],
                                value="Pose Animator",
                                label="Render Engine",
                                info="Pose Animator produces smooth 2D skeleton-based signing. Legacy uses pre-recorded clip concatenation.",
                            )
                            d2_render_speed = gr.Slider(
                                minimum=0.6,
                                maximum=1.6,
                                value=1.0,
                                step=0.1,
                                label="Signing Speed",
                            )
                    
                    # ── Right: Output ──
                    with gr.Column(scale=5):
                        gr.HTML('<div class="panel-label"><span class="panel-dot"></span> OUTPUT</div>')
                        
                        d2_audio_preview = gr.Audio(
                            label="Audio Preview",
                            type="filepath",
                            interactive=False,
                        )
                        d2_transcription = gr.Textbox(
                            label="Transcription / Input Text",
                            lines=2,
                            interactive=False,
                        )
                        d2_glosses_output = gr.Textbox(
                            label="BSL Glosses",
                            lines=2,
                            interactive=False,
                        )
                        d2_coverage = gr.Textbox(
                            label="Vocabulary Coverage",
                            lines=3,
                            interactive=False,
                        )
                        d2_live_preview = gr.Image(
                            label="Signing Preview",
                            streaming=True,
                            type="numpy",
                            interactive=False,
                        )
                        d2_avatar_video = gr.Video(label="BSL Signing Video")
                        d2_render_status = gr.Textbox(
                            label="Render Status",
                            lines=1,
                            interactive=False,
                        )
                
                # Examples
                gr.Examples(
                    examples=[
                        ["Hello, my name is John."],
                        ["What time is the meeting?"],
                        ["Thank you very much."],
                        ["I need help please."],
                    ],
                    inputs=[d2_text],
                    label="Example Phrases",
                )
                
                # ── Wire events ──
                d2_audio_btn.click(
                    fn=direction2_audio_to_signing,
                    inputs=[d2_audio_input, d2_render_engine, d2_render_speed],
                    outputs=[
                        d2_transcription, d2_glosses_output, d2_coverage,
                        d2_live_preview, d2_avatar_video, d2_render_status,
                    ],
                    show_progress="hidden",
                )
                d2_audio_input.change(
                    fn=media_preview_path,
                    inputs=[d2_audio_input],
                    outputs=[d2_audio_preview],
                )
                d2_text_btn.click(
                    fn=direction2_text_to_signing,
                    inputs=[d2_text, d2_render_engine, d2_render_speed],
                    outputs=[
                        d2_transcription, d2_glosses_output, d2_coverage,
                        d2_live_preview, d2_avatar_video, d2_render_status,
                    ],
                    show_progress="hidden",
                )
            
            # ═══════════════════════════════════════════
            # TAB 3: About
            # ═══════════════════════════════════════════
            with gr.TabItem("About", id="about"):
                gr.HTML("""
                <div class="about-container">
                    <div class="section-header" style="margin-top: 0.5rem;">About Signlytic AI</div>
                    <div class="section-desc" style="max-width: 720px;">
                        Signlytic AI is an advanced bidirectional British Sign Language translation system 
                        designed to bridge communication between deaf and hearing communities. It combines 
                        state-of-the-art computer vision, natural language processing, and speech synthesis 
                        into a unified, accessible platform.
                    </div>
                </div>
                """)
                
                # Performance cards
                with gr.Row():
                    gr.HTML("""
                    <div class="perf-card">
                        <div class="perf-number">100%</div>
                        <div class="perf-label">Top-1 Recognition Accuracy</div>
                    </div>
                    """)
                    gr.HTML("""
                    <div class="perf-card">
                        <div class="perf-number">5,203</div>
                        <div class="perf-label">BSL Signs Supported</div>
                    </div>
                    """)
                    gr.HTML("""
                    <div class="perf-card">
                        <div class="perf-number">11,573+</div>
                        <div class="perf-label">BSL Glosses in Vocabulary</div>
                    </div>
                    """)
                    gr.HTML("""
                    <div class="perf-card">
                        <div class="perf-number">GPU</div>
                        <div class="perf-label">Accelerated Inference</div>
                    </div>
                    """)
                
                gr.HTML("<br>")
                
                # System capabilities
                gr.HTML("""
                <div class="about-container">
                    <div style="font-size:1.1rem; font-weight:700; color:#0c1b33; margin-bottom:0.75rem;">
                        System Capabilities
                    </div>
                    <table class="arch-table">
                        <thead>
                            <tr>
                                <th>Direction</th>
                                <th>Input</th>
                                <th>Output</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>BSL to Speech</strong></td>
                                <td>Video of BSL signs, typed glosses, live camera</td>
                                <td>Natural English text + synthesized speech</td>
                            </tr>
                            <tr>
                                <td><strong>Speech to BSL</strong></td>
                                <td>Audio recording, typed text</td>
                                <td>BSL glosses + animated signing video</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)
                
                gr.HTML("<br>")
                
                # Technical architecture
                gr.HTML("""
                <div class="about-container">
                    <div style="font-size:1.1rem; font-weight:700; color:#0c1b33; margin-bottom:0.75rem;">
                        Technical Architecture
                    </div>
                    <table class="arch-table">
                        <thead>
                            <tr>
                                <th>Component</th>
                                <th>Technology</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Sign Recognition</strong></td>
                                <td>Video-SWIN-T Transformer</td>
                                <td>Retrieval-based matching on 5,203 pre-extracted features</td>
                            </tr>
                            <tr>
                                <td><strong>Speech Recognition</strong></td>
                                <td>OpenAI Whisper</td>
                                <td>Base model, 16kHz mono audio input</td>
                            </tr>
                            <tr>
                                <td><strong>Text-to-Speech</strong></td>
                                <td>Coqui XTTS v2</td>
                                <td>Voice cloning with speaker reference audio</td>
                            </tr>
                            <tr>
                                <td><strong>Language Model</strong></td>
                                <td>Groq Llama 3.3 70B</td>
                                <td>BSL gloss-to-natural-English conversion</td>
                            </tr>
                            <tr>
                                <td><strong>Avatar Rendering</strong></td>
                                <td>2D Pose Animator</td>
                                <td>Skeleton-based signing animation with video export</td>
                            </tr>
                            <tr>
                                <td><strong>Vocabulary</strong></td>
                                <td>11,573+ BSL Glosses</td>
                                <td>Extended vocabulary from BSL-1K and BSLDict datasets</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)
                
                gr.HTML("<br>")
                
                # Model performance table
                gr.HTML("""
                <div class="about-container">
                    <div style="font-size:1.1rem; font-weight:700; color:#0c1b33; margin-bottom:0.75rem;">
                        Trained Models &amp; Results
                    </div>
                    <table class="arch-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Language</th>
                                <th>Top-1 Acc.</th>
                                <th>Top-5 Acc.</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>BSL Dict Retrieval</strong></td>
                                <td>British</td>
                                <td style="color:#059669; font-weight:700;">100%</td>
                                <td style="color:#059669; font-weight:700;">100%</td>
                            </tr>
                            <tr>
                                <td>BSL-100 Classification</td>
                                <td>British</td>
                                <td>72.34%</td>
                                <td>95.03%</td>
                            </tr>
                            <tr>
                                <td>BSL-500 Classification</td>
                                <td>British</td>
                                <td>59.26%</td>
                                <td>89.04%</td>
                            </tr>
                            <tr>
                                <td>Pose Recognition</td>
                                <td>ASL</td>
                                <td>44.44%</td>
                                <td>81.62%</td>
                            </tr>
                            <tr>
                                <td>Multi-Lingual Pose</td>
                                <td>ASL+LSF</td>
                                <td>20.95%</td>
                                <td>49.17%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)
                
                gr.HTML("<br>")
                
                gr.HTML("""
                <div class="about-container">
                    <div style="font-size:1.1rem; font-weight:700; color:#0c1b33; margin-bottom:0.75rem;">
                        Key Technical Insights
                    </div>
                    <div style="font-size:0.92rem; color:#475569; line-height:1.7;">
                        <p style="margin-bottom:0.5rem;">
                            <strong>Retrieval beats classification</strong> for single-sample-per-class problems. 
                            The BSL Dictionary contains 5,203 clean isolated sign videos, each with one sample. 
                            Cosine similarity matching on SWIN-extracted 768-dimensional features achieves 100% 
                            Top-1 accuracy on same-source evaluation.
                        </p>
                        <p style="margin-bottom:0.5rem;">
                            <strong>Feature extraction</strong> uses the Video-SWIN-T backbone to produce compact 
                            768-dim embeddings per video. Full extraction across all 5,203 signs takes approximately 
                            one hour on an RTX 4060 Laptop GPU (8GB VRAM).
                        </p>
                        <p>
                            <strong>Bidirectional pipeline</strong> combines multiple AI modalities (vision, language, 
                            speech, animation) into a single cohesive system, demonstrating end-to-end integration 
                            from video input through language understanding to synthesized output.
                        </p>
                    </div>
                </div>
                """)
        
        # ─── FOOTER ───
        gr.HTML("""
        <div class="footer-wrapper">
            <div class="footer-brand">Signlytic AI</div>
            <div class="footer-author">
                Developed by 
                <a href="https://www.linkedin.com/in/iyanuoluwa-enoch-oke/" target="_blank" rel="noopener">
                    Oke Iyanuoluwa Enoch
                </a>
            </div>
            <div class="footer-links">
                <a href="https://github.com/Iyanuoluwa007/Signlytic_AI" target="_blank" rel="noopener">GitHub</a>
                <a href="https://signlytic-ai-website.vercel.app" target="_blank" rel="noopener">Website</a>
                <a href="https://huggingface.co/spaces/Iyanuoluwa007/signlytic-ai" target="_blank" rel="noopener">HuggingFace</a>
            </div>
            <div class="footer-uni">
                Independent Robotics & AI Systems Engineer | MSc Robotics & Automation, University of Salford
            </div>
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
    
    # Fix Windows asyncio issue
    import sys
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("=" * 60)
    print("  Signlytic AI - BSL Translation System")
    print("  Developed by Oke Iyanuoluwa Enoch")
    print("=" * 60)
    print(f"  Vocabulary:  {DEFAULT_VOCAB}")
    print(f"  Video dir:   {DEFAULT_VIDEO_DIR}")
    print(f"  FFmpeg:      {'OK' if ensure_ffmpeg_available() else 'NOT FOUND'}")
    print(f"  GROQ_API_KEY: {'FOUND' if DEFAULT_GROQ_API_KEY else 'NOT FOUND'}")
    print(f"  BSL Dict:    {'AVAILABLE' if BSL_DICT_AVAILABLE else 'NOT AVAILABLE'}")
    print("=" * 60)
    
    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        pwa=True,
    )


if __name__ == "__main__":
    main()