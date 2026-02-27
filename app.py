"""
BSL Translation Demo - Gradio Interface with Avatar Rendering

Bidirectional British Sign Language translation system:
- Direction 1: BSL Glosses/Video -> Text -> Speech
- Direction 2: Speech/Text -> Glosses -> Animated Signing

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


# Lazy loaders
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


# Direction 1: BSL -> Speech
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

        # Coqui unavailable: fallback to Windows built-in TTS.
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


# Direction 2: Speech -> BSL (Pose Animator + Legacy Clip Avatar)
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
        "3D Pose Animator": "Pose Animator",   # Backward compatibility for cached UI state
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
        
        # Check coverage first
        coverage = renderer.get_coverage(glosses)
        if not coverage['available']:
            print(f"No videos available for glosses: {glosses}")
            return None
        
        # Render video
        output_path = tempfile.mktemp(suffix=".mp4")
        result = renderer.render(glosses, output_path)
        
        if result and os.path.exists(result):
            return result
        return None
        
    except Exception as e:
        print(f"Avatar render error: {e}")
        return None


def create_demo():
    """Create Gradio interface."""
    groq_status = "FOUND" if DEFAULT_GROQ_API_KEY else "NOT FOUND"
    video_count = len(get_avatar_renderer().video_index) if os.path.exists(DEFAULT_VIDEO_DIR) else 0
    
    with gr.Blocks(title="BSL Translation System") as demo:
        gr.Markdown(f"""
        # BSL Translation System
        
        Bidirectional British Sign Language translation with pose animation and avatar fallback.
        
        **Status:** GROQ_API_KEY: `{groq_status}` | Avatar Videos: `{video_count}` available
        """)
        
        with gr.Tabs():
            # Direction 1: BSL -> Speech
            with gr.TabItem("Direction 1: BSL to Speech"):
                gr.Markdown("""
                ### BSL Glosses → Natural English → Speech
                
                Choose one input style below for a cleaner workflow.
                - Option A: Type glosses
                - Option B: Camera/upload video (record then recognize)
                - Option C: Live realtime camera in-app preview (no popup window)
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Option A: Type BSL Glosses")
                        d1_glosses = gr.Textbox(
                            label="BSL Glosses",
                            placeholder="TOMORROW MEETING WHAT TIME",
                            lines=2
                        )
                        
                        with gr.Row():
                            d1_mode = gr.Radio(
                                choices=["simple", "groq"],
                                value="groq" if DEFAULT_GROQ_API_KEY else "simple",
                                label="Conversion Mode"
                            )
                            d1_api_key = gr.Textbox(
                                label="Groq API Key (optional)",
                                type="password",
                                placeholder="Uses env GROQ_API_KEY if blank"
                            )

                        gr.Markdown("#### Option B: Camera or Upload Video")
                        d1_video_input = gr.Video(
                            label="Camera / Upload BSL Video",
                            sources=["webcam", "upload"]
                        )

                        gr.Markdown("#### Option C: Live Realtime Camera")
                        with gr.Row():
                            d1_live_camera_index = gr.Number(
                                label="Live Camera Index",
                                value=0,
                                precision=0
                            )
                            d1_live_no_speech = gr.Checkbox(
                                label="Disable Speech in Live Mode",
                                value=False
                            )
                        with gr.Row():
                            d1_live_start_btn = gr.Button("Start Live Realtime", variant="secondary")
                            d1_live_stop_btn = gr.Button("Stop Live Realtime", variant="secondary")
                            d1_live_clear_btn = gr.Button("Clear Live History", variant="secondary")
                        
                        d1_convert_btn = gr.Button("Convert to English", variant="primary")
                        d1_speak_btn = gr.Button("Convert & Speak", variant="secondary")
                        d1_video_btn = gr.Button("Recognize from Camera/Video", variant="secondary")
                    
                    with gr.Column():
                        d1_recorded_preview = gr.Video(label="Recorded Preview", interactive=False)
                        d1_live_preview = gr.Image(
                            label="Live Preview",
                            streaming=True,
                            type="numpy",
                            interactive=False,
                        )
                        d1_live_status = gr.Textbox(label="Live Realtime Status", lines=2, interactive=False)
                        d1_video_glosses = gr.Textbox(label="Recognized Glosses (from Video)", lines=2)
                        d1_text_output = gr.Textbox(label="English Text", lines=3)
                        d1_audio_output = gr.Audio(label="Speech Output", type="filepath")
                
                gr.Examples(
                    examples=[
                        ["TOMORROW MEETING WHAT TIME"],
                        ["MY NAME SARAH"],
                        ["YESTERDAY I GO DOCTOR"],
                        ["THANK YOU MUCH"],
                        ["I NOT UNDERSTAND"],
                    ],
                    inputs=[d1_glosses],
                    label="Example BSL Glosses"
                )
                
                d1_convert_btn.click(
                    fn=direction1_glosses_to_text,
                    inputs=[d1_glosses, d1_mode, d1_api_key],
                    outputs=[d1_text_output]
                )
                
                d1_speak_btn.click(
                    fn=direction1_full_pipeline,
                    inputs=[d1_glosses, d1_mode, d1_api_key],
                    outputs=[d1_text_output, d1_audio_output]
                )

                d1_video_input.change(
                    fn=media_preview_path,
                    inputs=[d1_video_input],
                    outputs=[d1_recorded_preview]
                )

                d1_video_btn.click(
                    fn=direction1_video_to_speech,
                    inputs=[d1_video_input, d1_mode, d1_api_key],
                    outputs=[d1_video_glosses, d1_text_output, d1_audio_output]
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
            
            # Direction 2: Speech -> BSL with Pose Animator / Legacy Avatar
            with gr.TabItem("Direction 2: Speech to BSL"):
                gr.Markdown("""
                ### Speech/Text -> BSL Glosses -> Animated Signing

                Choose one input style below, then pick a render engine.
                - Pose Animator (default): in-app 2D hand-sign animation + MP4 output
                - Legacy Clip Avatar: existing sign video concatenation flow
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Option A: Record or Upload Speech")
                        d2_audio_input = gr.Audio(
                            label="Record / Upload Audio",
                            type="filepath",
                            sources=["microphone", "upload"]
                        )
                        
                        d2_audio_btn = gr.Button("Convert Audio to BSL", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("#### Option B: Enter Text")
                        
                        d2_text = gr.Textbox(
                            label="Or Enter Text",
                            placeholder="What time is the meeting tomorrow?",
                            lines=2
                        )
                        d2_text_btn = gr.Button("Convert Text to BSL", variant="secondary")

                        gr.Markdown("---")
                        d2_render_engine = gr.Radio(
                            choices=["Pose Animator", "Legacy Clip Avatar", "3D Pose Animator"],
                            value="Pose Animator",
                            label="Render Engine"
                        )
                        gr.Markdown("Note: `3D Pose Animator` currently maps to the 2D pose renderer for compatibility.")
                        d2_render_speed = gr.Slider(
                            minimum=0.6,
                            maximum=1.6,
                            value=1.0,
                            step=0.1,
                            label="Render Speed"
                        )
                    
                    with gr.Column():
                        d2_audio_preview = gr.Audio(label="Audio Preview", type="filepath", interactive=False)
                        d2_transcription = gr.Textbox(label="Transcription / Input Text", lines=2)
                        d2_glosses_output = gr.Textbox(label="BSL Glosses", lines=2)
                        d2_coverage = gr.Textbox(label="Coverage Info", lines=5)
                        d2_live_preview = gr.Image(
                            label="Live Signing Preview",
                            streaming=True,
                            type="numpy",
                            interactive=False,
                        )
                        d2_avatar_video = gr.Video(label="BSL Avatar Video")
                        d2_render_status = gr.Textbox(label="Render Status", lines=2, interactive=False)
                
                gr.Examples(
                    examples=[
                        ["Hello, my name is John."],
                        ["What time is the meeting?"],
                        ["Thank you very much."],
                        ["I need help please."],
                    ],
                    inputs=[d2_text],
                    label="Example Text"
                )
                
                d2_audio_btn.click(
                    fn=direction2_audio_to_signing,
                    inputs=[d2_audio_input, d2_render_engine, d2_render_speed],
                    outputs=[
                        d2_transcription,
                        d2_glosses_output,
                        d2_coverage,
                        d2_live_preview,
                        d2_avatar_video,
                        d2_render_status,
                    ],
                    show_progress="hidden",
                )

                d2_audio_input.change(
                    fn=media_preview_path,
                    inputs=[d2_audio_input],
                    outputs=[d2_audio_preview]
                )
                
                d2_text_btn.click(
                    fn=direction2_text_to_signing,
                    inputs=[d2_text, d2_render_engine, d2_render_speed],
                    outputs=[
                        d2_transcription,
                        d2_glosses_output,
                        d2_coverage,
                        d2_live_preview,
                        d2_avatar_video,
                        d2_render_status,
                    ],
                    show_progress="hidden",
                )
            
            # About Tab
            with gr.TabItem("About"):
                gr.Markdown("""
                ## BSL Translation System
                
                ### Direction 1: BSL → Speech
                - Input: BSL glosses, camera recording/upload, or live realtime camera
                - Output: Natural English text + synthesized speech (Coqui TTS)
                
                ### Direction 2: Speech → BSL
                - Input: Speech via record/upload selector, or text
                - Output: BSL glosses + Pose animation preview + MP4 signing video
                
                ### Technical Stack
                - **ASR:** OpenAI Whisper
                - **TTS:** Coqui XTTS v2 with voice cloning
                - **Gloss-to-Text:** Groq Llama 3.3 70B
                - **Signing Renderer:** 2D pose animator (default) + legacy clip avatar fallback
                - **Vocabulary:** 11,573 BSL glosses
                
                ### Download Videos
                To enable avatar rendering, download BSL sign videos:
                ```
                python scripts/download_bsl_videos.py --limit 500
                ```
                """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="BSL Translation Demo")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    
    args = parser.parse_args()
    
    # Fix Windows asyncio issue
    import sys
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("Starting BSL Translation Demo...")
    print(f"Vocabulary: {DEFAULT_VOCAB}")
    print(f"Video directory: {DEFAULT_VIDEO_DIR}")
    print(f"FFmpeg: {'OK' if ensure_ffmpeg_available() else 'NOT FOUND'}")
    print(f"GROQ_API_KEY: {'FOUND' if DEFAULT_GROQ_API_KEY else 'NOT FOUND'}")
    
    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()



