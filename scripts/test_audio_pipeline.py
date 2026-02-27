"""
Test Speech-to-BSL pipeline with audio input.

Usage:
    python scripts/test_audio_pipeline.py                           
    python scripts/test_audio_pipeline.py --audio path/to/file.wav  
    python scripts/test_audio_pipeline.py --record                  
"""

import argparse
import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'inference'))

try:
    from speech_to_bsl import SpeechToBSL, TextToGloss, WhisperASR, GlossRenderer, CoquiTTS
except ImportError:
    try:
        from inference.speech_to_bsl import SpeechToBSL, TextToGloss, WhisperASR, GlossRenderer, CoquiTTS
    except ImportError:
        from src.inference.speech_to_bsl import SpeechToBSL, TextToGloss, WhisperASR, GlossRenderer, CoquiTTS


def record_audio(output_path: str, duration: int = 5):
    """Record audio from microphone."""
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
        import numpy as np
    except ImportError:
        print("Recording requires: pip install sounddevice scipy")
        return False
    
    sample_rate = 16000
    print(f"Recording for {duration} seconds...")
    
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    
    wav.write(output_path, sample_rate, audio)
    print(f"Saved recording to: {output_path}")
    return True


def test_full_pipeline(
    audio_path: str,
    vocab_path: str = None,
    whisper_model: str = "base",
    gloss_mode: str = "simple"
):
    """Test the complete Speech-to-BSL pipeline."""
    
    print("\n" + "=" * 60)
    print("Speech-to-BSL Pipeline Test")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        return None
    
    print(f"\nAudio file: {audio_path}")
    print(f"Whisper model: {whisper_model}")
    print(f"Gloss mode: {gloss_mode}")
    if vocab_path:
        print(f"Vocabulary: {vocab_path}")
    
    # Initialize pipeline
    print("\n--- Initializing Pipeline ---")
    start_time = time.time()
    
    pipeline = SpeechToBSL(
        whisper_model=whisper_model,
        gloss_mode=gloss_mode,
        vocabulary_path=vocab_path
    )
    
    init_time = time.time() - start_time
    print(f"Pipeline initialized in {init_time:.1f}s")
    
    # Process audio
    print("\n--- Processing Audio ---")
    start_time = time.time()
    
    result = pipeline.process(audio_path, return_intermediate=True)
    
    process_time = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print(f"\nTranscribed Text:")
    print(f"  \"{result['text']}\"")
    
    print(f"\nBSL Glosses:")
    print(f"  {' '.join(result['glosses'])}")
    
    print(f"\nTimeline ({result['render']['total_duration']:.1f}s total):")
    for item in result['render']['timeline']:
        render_type = "video" if item['render_type'] == 'video' else "text"
        print(f"  [{item['start_time']:.1f}s - {item['end_time']:.1f}s] {item['gloss']} ({render_type})")
    
    print(f"\nProcessing time: {process_time:.2f}s")
    
    # Vocabulary coverage
    if vocab_path:
        converter = pipeline.text_to_gloss
        info = converter.convert_with_info(result['text'])
        print(f"\nVocabulary Coverage: {info['coverage']:.1f}%")
        if info['out_of_vocab']:
            print(f"Out of vocabulary: {', '.join(info['out_of_vocab'])}")
    
    return result


def test_text_only(text: str, vocab_path: str = None):
    """Test text-to-gloss conversion without audio."""
    print("\n" + "=" * 60)
    print("Text-to-Gloss Test")
    print("=" * 60)
    
    converter = TextToGloss(
        mode="simple",
        vocabulary_path=vocab_path
    )
    
    result = converter.convert_with_info(text)
    
    print(f"\nInput: \"{text}\"")
    print(f"Glosses: {' '.join(result['glosses'])}")
    print(f"Coverage: {result['coverage']:.1f}%")
    
    if result['out_of_vocab']:
        print(f"Out of vocabulary: {', '.join(result['out_of_vocab'])}")
    
    return result


def test_tts(text: str, speaker_wav: str, output_path: str):
    """Test text-to-speech synthesis."""
    print("\n" + "=" * 60)
    print("Text-to-Speech Test (Coqui XTTS v2)")
    print("=" * 60)
    
    if not os.path.exists(speaker_wav):
        print(f"ERROR: Speaker reference not found: {speaker_wav}")
        return None
    
    print(f"\nText: \"{text}\"")
    print(f"Speaker reference: {speaker_wav}")
    print(f"Output: {output_path}")
    
    print("\nInitializing TTS...")
    tts = CoquiTTS(speaker_wav=speaker_wav)
    
    print("Synthesizing speech...")
    start_time = time.time()
    result_path = tts.synthesize(text, output_path)
    duration = time.time() - start_time
    
    print(f"\nSpeech generated in {duration:.2f}s")
    print(f"Saved to: {result_path}")
    
    return result_path


def test_round_trip(
    audio_path: str,
    speaker_wav: str,
    output_audio: str,
    vocab_path: str = None,
    whisper_model: str = "base"
):
    """
    Test complete round-trip: Audio -> Glosses -> TTS Audio.
    Useful for verifying the full pipeline.
    """
    print("\n" + "=" * 60)
    print("Round-Trip Test: Audio -> BSL Glosses -> TTS Audio")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Input audio not found: {audio_path}")
        return None
    
    if not os.path.exists(speaker_wav):
        print(f"ERROR: Speaker reference not found: {speaker_wav}")
        return None
    
    # Step 1: Speech to Glosses
    print("\n--- Step 1: Speech to BSL Glosses ---")
    pipeline = SpeechToBSL(
        whisper_model=whisper_model,
        gloss_mode="simple",
        vocabulary_path=vocab_path
    )
    
    result = pipeline.process(audio_path, return_intermediate=True)
    print(f"Transcribed: \"{result['text']}\"")
    print(f"Glosses: {' '.join(result['glosses'])}")
    
    # Step 2: Glosses to TTS
    print("\n--- Step 2: Glosses to Speech ---")
    tts = CoquiTTS(speaker_wav=speaker_wav)
    
    # Convert glosses back to text for TTS
    gloss_text = " ".join(g.lower() for g in result['glosses'] 
                         if g not in ['<unk>', '<pad>', '<sos>', '<eos>'])
    
    print(f"TTS input: \"{gloss_text}\"")
    tts.synthesize(gloss_text, output_audio)
    print(f"Output saved: {output_audio}")
    
    return {
        "original_text": result['text'],
        "glosses": result['glosses'],
        "gloss_text": gloss_text,
        "output_audio": output_audio
    }


def main():
    parser = argparse.ArgumentParser(description="Test Speech-to-BSL Pipeline")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--text", type=str, help="Test with text input directly")
    parser.add_argument("--record", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration (seconds)")
    parser.add_argument("--vocab", type=str, default=None, help="Path to vocabulary.json")
    parser.add_argument("--whisper", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--mode", type=str, default="simple",
                       choices=["simple", "llm", "groq"],
                       help="Text-to-gloss conversion mode")
    
    # TTS arguments
    parser.add_argument("--tts", action="store_true", help="Generate TTS output from text")
    parser.add_argument("--tts-text", type=str, help="Text to synthesize with TTS")
    parser.add_argument("--speaker-wav", type=str, default=None, 
                       help="Reference voice for TTS (default: voice_training.wav)")
    parser.add_argument("--tts-output", type=str, default=None,
                       help="Output path for TTS audio")
    parser.add_argument("--round-trip", action="store_true",
                       help="Test full round-trip: Audio -> Glosses -> TTS")
    
    args = parser.parse_args()
    
    # Auto-detect paths
    if args.vocab is None:
        vocab_candidates = [
            os.path.join(project_root, "data", "processed", "vocabulary_extended.json"),
            os.path.join(project_root, "data", "processed", "vocabulary.json"),
            "data/processed/vocabulary_extended.json",
            "data/processed/vocabulary.json",
        ]
        for v in vocab_candidates:
            if os.path.exists(v):
                args.vocab = v
                print(f"Using vocabulary: {v}")
                break
    
    if args.speaker_wav is None:
        speaker_candidates = [
            os.path.join(project_root, "data", "processed", "voice_training.wav"),
            "data/processed/voice_training.wav",
        ]
        for s in speaker_candidates:
            if os.path.exists(s):
                args.speaker_wav = s
                break
    
    if args.tts_output is None:
        args.tts_output = os.path.join(project_root, "outputs", "tts_output.wav")
    
    # Run appropriate test
    if args.tts_text:
        # TTS synthesis test
        if not args.speaker_wav:
            print("ERROR: --speaker-wav required for TTS")
            return
        test_tts(args.tts_text, args.speaker_wav, args.tts_output)
    
    elif args.round_trip:
        # Full round-trip test
        audio_path = args.audio
        if not audio_path:
            audio_path = os.path.join(project_root, "data", "processed", "voice_training.wav")
        
        if not args.speaker_wav:
            print("ERROR: --speaker-wav required for round-trip test")
            return
        
        test_round_trip(
            audio_path=audio_path,
            speaker_wav=args.speaker_wav,
            output_audio=args.tts_output,
            vocab_path=args.vocab,
            whisper_model=args.whisper
        )
    
    elif args.text:
        test_text_only(args.text, args.vocab)
    
    elif args.record:
        temp_audio = os.path.join(project_root, "outputs", "recorded_audio.wav")
        os.makedirs(os.path.dirname(temp_audio), exist_ok=True)
        
        if record_audio(temp_audio, args.duration):
            test_full_pipeline(temp_audio, args.vocab, args.whisper, args.mode)
    
    elif args.audio:
        test_full_pipeline(args.audio, args.vocab, args.whisper, args.mode)
    
    else:
        default_audio = os.path.join(project_root, "data", "processed", "voice_training.wav")
        if os.path.exists(default_audio):
            print(f"Using default audio: {default_audio}")
            test_full_pipeline(default_audio, args.vocab, args.whisper, args.mode)
        else:
            print("Usage:")
            print("  python scripts/test_audio_pipeline.py --audio path/to/file.wav")
            print("  python scripts/test_audio_pipeline.py --text 'Hello, how are you?'")
            print("  python scripts/test_audio_pipeline.py --record --duration 5")
            print("  python scripts/test_audio_pipeline.py --tts-text 'Hello world'")
            print("  python scripts/test_audio_pipeline.py --round-trip --audio input.wav")


if __name__ == "__main__":
    main()
