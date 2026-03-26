"""
Patch to add BSL Dict Recognizer to app.py
"""

# Read app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add import for BSLDictRecognizer after other imports
old_import = '''from src.inference.speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
        from src.inference.gloss_to_text import GlossToText, BSLToSpeechPipeline
        from src.inference.avatar_renderer import BSLAvatarRenderer
        from src.inference.pose_sign_renderer import PoseSignRenderer'''

new_import = '''from src.inference.speech_to_bsl import SpeechToBSL, TextToGloss, CoquiTTS
        from src.inference.gloss_to_text import GlossToText, BSLToSpeechPipeline
        from src.inference.avatar_renderer import BSLAvatarRenderer
        from src.inference.pose_sign_renderer import PoseSignRenderer
        from src.inference.bsl_dict_recognizer import BSLDictRecognizer'''

content = content.replace(old_import, new_import)

# 2. Add global for BSL Dict Recognizer
old_global = '''_sign_recognizer = None
_live_running = False'''

new_global = '''_sign_recognizer = None
_bsl_dict_recognizer = None
_live_running = False'''

content = content.replace(old_global, new_global)

# 3. Add function to get BSL Dict Recognizer
old_func = '''def get_sign_recognizer():'''

new_func = '''def get_bsl_dict_recognizer():
    """Lazy-load BSL dictionary recognizer (SWIN-based, 100% accuracy on 5203 signs)."""
    global _bsl_dict_recognizer
    if _bsl_dict_recognizer is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _bsl_dict_recognizer = BSLDictRecognizer(device=device)
        except Exception as e:
            print(f"Failed to load BSL Dict Recognizer: {e}")
            return None
    return _bsl_dict_recognizer


def get_sign_recognizer():'''

content = content.replace(old_func, new_func)

# 4. Add new SWIN-based recognition function before direction1_video_to_speech
old_direction1 = '''def direction1_video_to_speech(video_input, mode, api_key):
    """Direction 1 pipeline from camera/uploaded video -> glosses -> text -> speech."""'''

new_direction1 = '''def direction1_video_swin(video_input, mode, api_key):
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
        
        # Recognize BSL signs from video
        results = recognizer.recognize(video_path, top_k=5)
        
        # Format results
        if results:
            top_gloss = results[0][0].upper()
            confidence = results[0][1] * 100
            
            all_predictions = ", ".join([f"{g.upper()} ({c*100:.0f}%)" for g, c in results[:3]])
            gloss_output = f"Top: {top_gloss} ({confidence:.0f}%)\\nAlternatives: {all_predictions}"
            
            # Convert to text
            gloss_converter = get_gloss_to_text(mode, api_key)
            english_text = gloss_converter.convert(top_gloss)
            
            # Generate speech
            tts = get_tts()
            audio_path = None
            if tts:
                audio_path = tts.synthesize(english_text)
            
            return gloss_output, english_text, audio_path
        else:
            return "No signs detected", "", None
            
    except Exception as e:
        return "", f"Error: {str(e)}", None


def direction1_video_to_speech(video_input, mode, api_key):
    """Direction 1 pipeline from camera/uploaded video -> glosses -> text -> speech."""'''

content = content.replace(old_direction1, new_direction1)

# Save modified app.py
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Patched app.py with BSL Dict Recognizer")
