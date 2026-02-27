from TTS.api import TTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(
    text="Hello there, I hope you’re having a pleasant day. This message demonstrates realistic British English speech synthesis.",
    speaker_wav=r"E:\Signlytic_AI\code\bsl_translation_project\data\processed\voice_training.wav",
    language="en",
    file_path=r"E:\Signlytic_AI\code\bsl_translation_project\outputs\voice_output.wav"
)

print("Done! Audio saved.")
