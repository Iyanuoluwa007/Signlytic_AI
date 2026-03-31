"""
gen_test_audio.py - Generate test audio using XTTS v2
Run from project root: python scripts/gen_test_audio.py
"""
import sys, os
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "src/inference")

text = (
    "One sunny morning, Barnaby decided to have a picnic. "
    "He packed his favorite jar of honey and a crusty loaf of bread. "
    "Just as he sat down, he heard a rustling in the bushes. "
    "Out popped Bella Badger, looking sad."
)

speaker = Path("data/processed/voice_training_22050.wav")
if not speaker.exists():
    speaker = Path("data/processed/voice_training.wav")
if not speaker.exists():
    print("[ERR] No speaker wav found at data/processed/")
    sys.exit(1)

out = Path("outputs/barnaby_test_audio.wav")
out.parent.mkdir(exist_ok=True)

print(f"Speaker: {speaker}")
print(f"Output:  {out}")
print(f"Text:    {text[:60]}...")

from TTS.api import TTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:  {device}")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts.tts_to_file(
    text=text,
    speaker_wav=str(speaker),
    language="en",
    file_path=str(out),
)
print(f"[OK] Generated: {out.stat().st_size:,} bytes")
print(f"[OK] Saved to:  {out.resolve()}")
