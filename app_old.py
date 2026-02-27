"""
BSL Translation Demo - Gradio Interface

Bidirectional British Sign Language translation system:
- Direction 1: BSL Glosses -> Text -> Speech
- Direction 2: Speech/Text -> Glosses -> (Avatar placeholder)

Usage:
    python app.py
    python app.py --share    # Create public link
    python app.py --port 7860
"""

import argparse
import sys
import os
import tempfile
import subprocess
from pathlib import Path

# -----------------------------
# Path setup
# -----------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(project_root) == "scripts":
    project_root = os.path.dirname(project_root)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "inference"))

try:
    import gradio as gr
except ImportError:
    print("Gradio required. Install with: pip install gradio")
    sys.exit(1)

# Import pipeline components
try:
    from speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
    from gloss_to_text import GlossToText, BSLToSpeechPipeline
except ImportError:
    try:
        from inference.speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
        from inference.gloss_to_text import GlossToText, BSLToSpeechPipeline
    except ImportError:
        from src.inference.speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
        from src.inference.gloss_to_text import GlossToText, BSLToSpeechPipeline

# -----------------------------
# Globals / Defaults
# -----------------------------
_speech_to_bsl = None
_gloss_to_text_simple = None
_gloss_to_text_groq = None
_groq_key_in_use = None
_tts = None

DEFAULT_VOCAB = os.path.join(project_root, "data", "processed", "vocabulary_extended.json")
DEFAULT_SPEAKER = os.path.join(project_root, "data", "processed", "voice_training.wav")
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


# -----------------------------
# Lazy loaders
# -----------------------------
def get_speech_to_bsl():
    """Lazy load Speech-to-BSL pipeline."""
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
    """
    Get gloss-to-text converter.

    Behavior:
    - simple: always works (rule-based)
    - groq: uses env GROQ_API_KEY if present; otherwise requires the UI textbox
    """
    global _gloss_to_text_simple, _gloss_to_text_groq, _groq_key_in_use

    if mode == "simple":
        if _gloss_to_text_simple is None:
            _gloss_to_text_simple = GlossToText(mode="simple")
        return _gloss_to_text_simple

    if mode == "groq":
        key = (api_key or "").strip() or DEFAULT_GROQ_API_KEY
        if not key:
            raise RuntimeError(
                "Groq mode selected but no GROQ_API_KEY found. "
                "Set GROQ_API_KEY in your environment or enter a key in the box."
            )

        # If key changes, rebuild the Groq converter instance
        if _gloss_to_text_groq is None or _groq_key_in_use != key:
            _gloss_to_text_groq = GlossToText(mode="groq", groq_api_key=key)
            _groq_key_in_use = key

        return _gloss_to_text_groq

    # fallback
    if _gloss_to_text_simple is None:
        _gloss_to_text_simple = GlossToText(mode="simple")
    return _gloss_to_text_simple


def get_tts():
    """Lazy load TTS."""
    global _tts
    if _tts is None and os.path.exists(DEFAULT_SPEAKER):
        _tts = CoquiTTS(speaker_wav=DEFAULT_SPEAKER)
    return _tts


# -----------------------------
# Audio conversion helpers
# -----------------------------
def ensure_ffmpeg_available() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception:
        return False


def to_wav_16k_mono(input_path: str) -> str:
    """
    Convert any audio container (webm/mp3/m4a/ogg/wav...) to WAV 16kHz mono PCM.
    Returns a temp wav file path.
    """
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if not ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure your terminal can run `ffmpeg -version`."
        )

    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-err_detect",
        "ignore_err",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        out_path,
    ]

    # Some webm/opus inputs emit warnings; conversion can still be OK.
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    if (not os.path.exists(out_path)) or (os.path.getsize(out_path) < 1024):
        raise RuntimeError("FFmpeg conversion failed or produced an empty WAV.")

    return out_path


def file_to_path(uploaded_file):
    """Helper: turn gr.File object into a filepath string for preview."""
    if uploaded_file is None:
        return None
    return uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)


# -----------------------------
# Direction 1: BSL Glosses -> Text -> Speech
# -----------------------------
def direction1_glosses_to_text(glosses_input, mode, api_key):
    """Convert BSL glosses to natural English text."""
    if not glosses_input or not glosses_input.strip():
        return "Please enter BSL glosses."

    glosses = glosses_input.upper().split()

    try:
        converter = get_gloss_converter(mode, api_key)
        text = converter.convert(glosses)
        return text
    except Exception as e:
        return f"Error: {str(e)}"


def direction1_text_to_speech(text):
    """Convert text to speech audio."""
    if not text or not text.strip():
        return None

    try:
        tts = get_tts()
        if tts is None:
            return None

        output_path = tempfile.mktemp(suffix=".wav")
        tts.synthesize(text, output_path)
        return output_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def direction1_full_pipeline(glosses_input, mode, api_key):
    """Full Direction 1 pipeline: Glosses -> Text -> Speech."""
    text = direction1_glosses_to_text(glosses_input, mode, api_key)

    if text.startswith("Error") or text.startswith("Please"):
        return text, None

    audio = direction1_text_to_speech(text)
    return text, audio


# -----------------------------
# Direction 2: Speech/Text -> Glosses
# -----------------------------
def direction2_audio_to_glosses(mic_audio_path, uploaded_file):
    """
    Convert speech audio to BSL glosses + vocabulary coverage.

    - mic_audio_path: filepath from gr.Audio (microphone)
    - uploaded_file: file object from gr.File (upload) to avoid MIME rejections for .webm
    """
    if (not mic_audio_path) and (uploaded_file is None):
        return "", "", ""

    try:
        # Prefer microphone if provided, else use uploaded file path
        if mic_audio_path:
            input_path = mic_audio_path
        else:
            input_path = file_to_path(uploaded_file)

        wav_path = to_wav_16k_mono(input_path)

        pipeline = get_speech_to_bsl()
        result = pipeline.process(wav_path, return_intermediate=True)

        glosses = " ".join(result["glosses"])
        text = result["text"]

        # Coverage info computed from the transcription text
        info = pipeline.text_to_gloss.convert_with_info(text)
        coverage = f"Vocabulary coverage: {info['coverage']:.1f}%"
        if info["out_of_vocab"]:
            coverage += f"\nOut of vocabulary: {', '.join(info['out_of_vocab'])}"

        return text, glosses, coverage

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def direction2_text_to_glosses(text_input):
    """Convert text to BSL glosses + vocabulary coverage."""
    if not text_input or not text_input.strip():
        return "Please enter text.", ""

    try:
        pipeline = get_speech_to_bsl()
        glosses = pipeline.text_to_gloss.convert(text_input)

        info = pipeline.text_to_gloss.convert_with_info(text_input)

        gloss_str = " ".join(glosses)
        coverage = f"Vocabulary coverage: {info['coverage']:.1f}%"

        if info["out_of_vocab"]:
            coverage += f"\nOut of vocabulary: {', '.join(info['out_of_vocab'])}"

        return gloss_str, coverage

    except Exception as e:
        return f"Error: {str(e)}", ""


# -----------------------------
# UI
# -----------------------------
def create_demo():
    """Create Gradio interface."""
    groq_env_status = "FOUND" if DEFAULT_GROQ_API_KEY else "NOT FOUND"

    with gr.Blocks(title="BSL Translation System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
        # BSL Translation System
        
        Bidirectional British Sign Language translation powered by AI.

        **Groq API Key (GROQ_API_KEY) status:** `{groq_env_status}`
        """
        )

        with gr.Tabs():
            # -------------------------
            # Direction 1
            # -------------------------
            with gr.TabItem("Direction 1: BSL to Speech"):
                gr.Markdown(
                    """
                ### BSL Glosses → Natural English → Speech
                
                Enter BSL glosses (space-separated, e.g., `TOMORROW MEETING WHAT TIME`) 
                to convert to natural English and synthesize speech.
                """
                )

                with gr.Row():
                    with gr.Column():
                        d1_glosses = gr.Textbox(
                            label="BSL Glosses",
                            placeholder="TOMORROW MEETING WHAT TIME",
                            lines=2,
                        )

                        with gr.Row():
                            d1_mode = gr.Radio(
                                choices=["simple", "groq"],
                                value="simple",
                                label="Conversion Mode",
                            )
                            d1_api_key = gr.Textbox(
                                label="Groq API Key (optional: used only if GROQ_API_KEY is not set)",
                                type="password",
                                placeholder="Leave blank to use GROQ_API_KEY from environment",
                            )

                        d1_convert_btn = gr.Button("Convert to English", variant="primary")
                        d1_speak_btn = gr.Button("Convert & Speak", variant="secondary")

                    with gr.Column():
                        d1_text_output = gr.Textbox(label="English Text", lines=3)
                        d1_audio_output = gr.Audio(label="Speech Output", type="filepath")

                gr.Examples(
                    examples=[
                        ["TOMORROW MEETING WHAT TIME"],
                        ["MY NAME SARAH"],
                        ["YESTERDAY I GO DOCTOR"],
                        ["HELP LONDON LIVE FIND"],
                        ["THANK YOU MUCH"],
                        ["WEATHER TODAY BEAUTIFUL"],
                        ["I NOT UNDERSTAND"],
                        ["YOU HELP ME PLEASE"],
                    ],
                    inputs=[d1_glosses],
                    label="Example BSL Glosses",
                )

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

            # -------------------------
            # Direction 2
            # -------------------------
            with gr.TabItem("Direction 2: Speech to BSL"):
                gr.Markdown(
                    """
                ### Speech/Text → BSL Glosses
                
                - Microphone recording works as before (and is playable).
                - File upload accepts `.webm` and shows a playable preview so you can confirm the right file.
                - Backend auto-converts everything to WAV (16kHz mono) before ASR.
                - Vocabulary Info now updates for BOTH audio and text.
                """
                )

                with gr.Row():
                    with gr.Column():
                        # Mic input (playable)
                        d2_mic = gr.Audio(
                            label="Record from Microphone",
                            type="filepath",
                            sources=["microphone"],
                        )

                        # Upload input (accept webm etc.)
                        d2_upload = gr.File(
                            label="Or Upload Audio File (.webm/.wav/.mp3/.m4a/.ogg)",
                            file_types=[".webm", ".wav", ".mp3", ".m4a", ".ogg", ".aac", ".flac"],
                        )

                        # Playable preview of uploaded file
                        d2_upload_preview = gr.Audio(
                            label="Uploaded Audio Preview (play to confirm)",
                            type="filepath",
                            interactive=False,
                        )

                        d2_audio_btn = gr.Button("Convert Audio to Glosses", variant="primary")

                        gr.Markdown("---")

                        d2_text = gr.Textbox(
                            label="Or Enter Text",
                            placeholder="What time is the meeting tomorrow?",
                            lines=2,
                        )
                        d2_text_btn = gr.Button("Convert Text to Glosses", variant="secondary")

                    with gr.Column():
                        d2_transcription = gr.Textbox(label="Transcription", lines=2)
                        d2_glosses_output = gr.Textbox(label="BSL Glosses", lines=2)
                        d2_coverage = gr.Textbox(label="Vocabulary Info", lines=2)

                # When a file is selected, show it in the preview player
                d2_upload.change(
                    fn=file_to_path,
                    inputs=[d2_upload],
                    outputs=[d2_upload_preview],
                )

                gr.Examples(
                    examples=[
                        ["What time is the meeting tomorrow?"],
                        ["My name is Sarah."],
                        ["I need help finding somewhere to live in London."],
                        ["Can you please repeat that?"],
                        ["Thank you very much."],
                    ],
                    inputs=[d2_text],
                    label="Example Text",
                )

                d2_audio_btn.click(
                    fn=direction2_audio_to_glosses,
                    inputs=[d2_mic, d2_upload],
                    outputs=[d2_transcription, d2_glosses_output, d2_coverage],
                )

                d2_text_btn.click(
                    fn=direction2_text_to_glosses,
                    inputs=[d2_text],
                    outputs=[d2_glosses_output, d2_coverage],
                )

            # -------------------------
            # About
            # -------------------------
            with gr.TabItem("About"):
                gr.Markdown(
                    """
                ## BSL Translation System
                
                This system provides bidirectional translation between British Sign Language (BSL) 
                and spoken English.
                
                ### Direction 1: BSL → Speech
                - Input: BSL glosses (individual sign labels)
                - Processing: Gloss-to-text conversion using rule-based or LLM methods
                - Output: Natural English text and synthesized speech
                
                ### Direction 2: Speech → BSL
                - Input: Speech audio (mic or upload) or text
                - Processing: Whisper ASR + Text-to-gloss conversion with lemmatization
                - Output: BSL glosses for avatar rendering
                
                ### Technical Details
                - Speech Recognition: OpenAI Whisper
                - Text-to-Speech: Coqui XTTS v2 with voice cloning
                - Vocabulary: 11,573 BSL glosses (BOBSL + BslDict)
                - Gloss-to-Text: Rule-based or Groq Llama 3.1
                
                ### Notes on BSL Grammar
                BSL has different grammar from English:
                - Topic-comment structure (topic first)
                - Time markers at the beginning
                - No articles (a, an, the)
                - Different word order
                
                Example:
                - BSL: `TOMORROW MEETING WHAT TIME`
                - English: "What time is the meeting tomorrow?"
                """
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="BSL Translation Demo")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")

    args = parser.parse_args()

    print("Starting BSL Translation Demo...")
    print(f"Vocabulary: {DEFAULT_VOCAB}")
    print(f"Speaker reference: {DEFAULT_SPEAKER}")
    print("FFmpeg available:", "YES" if ensure_ffmpeg_available() else "NO")
    print("GROQ_API_KEY:", "FOUND" if DEFAULT_GROQ_API_KEY else "NOT FOUND")

    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
