"""Fix TTS call in direction1_video_swin"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the TTS call to include output_path
old_tts = '''# Generate speech
            tts = get_tts()
            audio_path = None
            if tts:
                audio_path = tts.synthesize(english_text)'''

new_tts = '''# Generate speech
            tts = get_tts()
            audio_path = None
            if tts and english_text.strip():
                import tempfile
                audio_path = tempfile.mktemp(suffix=".wav")
                tts.synthesize(english_text, audio_path)
                if not os.path.exists(audio_path):
                    audio_path = None'''

content = content.replace(old_tts, new_tts)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixed TTS synthesize call")
