#!/usr/bin/env python3
"""
Setup and test the BSL to Speech inference pipeline.

Steps:
1. Save vocabulary to JSON (for inference)
2. Test recognition on sample data
3. Test gloss-to-text conversion
4. Test text-to-speech (if available)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np


def setup_vocabulary(data_dir: str, output_path: str):
    """Save vocabulary for inference use."""
    from data.annotation_parser import BOBSLAnnotationParser
    from data.datasets import Vocabulary
    
    print("Setting up vocabulary...")
    
    # Check if already exists
    if Path(output_path).exists():
        print(f"  Vocabulary already exists: {output_path}")
        vocab = Vocabulary.load(output_path)
        print(f"  Size: {len(vocab)} classes")
        return vocab
    
    # Create from parser
    parser = BOBSLAnnotationParser(data_dir)
    parser.parse_isolated_signs()
    
    vocab = Vocabulary.from_parser(parser)
    vocab.save(output_path)
    
    print(f"  Saved to: {output_path}")
    print(f"  Size: {len(vocab)} classes")
    
    return vocab


def test_recognizer(model_path: str, vocab_path: str, data_dir: str, model_type: str):
    """Test the BSL recognizer."""
    from inference.recognizer import BSLRecognizer
    
    print("\n" + "=" * 60)
    print("TESTING BSL RECOGNIZER")
    print("=" * 60)
    
    # Initialize
    recognizer = BSLRecognizer(
        model_path=model_path,
        vocab_path=vocab_path,
        model_type=model_type,
    )
    
    # Find sample feature file
    feature_base = Path(data_dir) / 'features' / 'bobsl' / 'v1.4' / 'video_features' / 'swin_v1'
    feature_dir = None
    
    for item in feature_base.iterdir():
        if item.is_dir():
            feature_dir = item
            break
    
    if feature_dir is None:
        print("No feature directory found!")
        return None
    
    feature_files = list(feature_dir.glob('*.npy'))
    if not feature_files:
        print("No feature files found!")
        return None
    
    sample_file = feature_files[0]
    print(f"\nSample: {sample_file.name}")
    
    # Load features info
    features = np.load(sample_file, mmap_mode='r')
    print(f"Shape: {features.shape}")
    print(f"Duration: {features.shape[0] / 25.0:.1f}s")
    
    # Test single prediction
    print("\n--- Single Prediction (middle of video) ---")
    timestamp = features.shape[0] / 25.0 / 2
    window = recognizer.extract_window(features, timestamp)
    predictions = recognizer.predict(window, top_k=5)
    
    print(f"Timestamp: {timestamp:.1f}s")
    for gloss, prob in predictions:
        print(f"  {gloss}: {prob:.2%}")
    
    # Test continuous prediction (limit to first 30 seconds for speed)
    print("\n--- Continuous Prediction (first 30s) ---")
    
    # Load features and limit duration
    features_full = np.load(sample_file, mmap_mode='r')
    max_test_frames = int(30 * 25)  # 30 seconds at 25fps
    
    # Save a temporary short version
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        temp_path = f.name
        features_short = np.array(features_full[:max_test_frames])
        if features_short.dtype == np.float16:
            features_short = features_short.astype(np.float32)
        np.save(f.name, features_short)
    
    glosses = recognizer.recognize_sequence(
        temp_path,
        window_stride=1.0,
        confidence_threshold=0.3,
    )
    
    # Clean up
    import os
    os.unlink(temp_path)
    
    print(f"Detected {len(glosses)} signs")
    if glosses:
        print(f"Glosses: {' '.join(glosses[:15])}")
    else:
        print("No glosses detected above threshold")
    
    return glosses


def test_gloss_to_text(glosses: list = None, gloss_mode: str = 'simple'):
    """Test gloss to text conversion."""
    from inference.bsl_to_speech import GlossToText, GlossToTextWithGroq
    
    print("\n" + "=" * 60)
    print(f"TESTING GLOSS TO TEXT (mode: {gloss_mode})")
    print("=" * 60)
    
    # Initialize converter based on mode
    if gloss_mode == 'groq':
        converter = GlossToTextWithGroq()
    else:
        converter = GlossToText(mode=gloss_mode)
    
    # Test cases
    test_cases = [
        ['hello', 'how', 'you'],
        ['i', 'need', 'help'],
        ['thank', 'you'],
        ['good', 'morning'],
        ['please', 'wait'],
    ]
    
    if glosses and len(glosses) > 0:
        test_cases.append(glosses[:10])
    
    print("\nConverting glosses to English:")
    for gloss_list in test_cases:
        text = converter.process(gloss_list)
        print(f"  {gloss_list}")
        print(f"  → {text}\n")


def test_tts(speaker_wav: str):
    """Test text to speech with Coqui TTS."""
    print("\n" + "=" * 60)
    print("TESTING TEXT TO SPEECH (Coqui XTTS v2)")
    print("=" * 60)
    
    try:
        from inference.bsl_to_speech import TextToSpeech
        
        print(f"Reference voice: {speaker_wav}")
        
        tts = TextToSpeech(
            engine='coqui',
            speaker_wav=speaker_wav,
        )
        
        test_text = "Hello. This is a test of the BSL to speech system using voice cloning."
        print(f"Text: {test_text}")
        
        # Save to file
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'test_speech_coqui.wav'
        
        print("Generating speech (this may take a moment)...")
        tts.save(test_text, str(output_path))
        print(f"Saved: {output_path}")
        
        print("TTS test passed!")
        return True
        
    except ImportError as e:
        print(f"Coqui TTS not available: {e}")
        print("Install with: pip install TTS")
        return False
    except Exception as e:
        print(f"TTS error: {e}")
        return False


def test_full_pipeline(model_path: str, vocab_path: str, data_dir: str, model_type: str, speaker_wav: str, gloss_mode: str = 'simple'):
    """Test the complete pipeline."""
    from inference.bsl_to_speech import BSLToSpeechPipeline
    from inference.recognizer import BSLRecognizer
    
    print("\n" + "=" * 60)
    print(f"TESTING FULL PIPELINE (gloss_mode: {gloss_mode})")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = BSLRecognizer(
        model_path=model_path,
        vocab_path=vocab_path,
        model_type=model_type,
    )
    
    # Initialize pipeline with Coqui TTS
    print(f"Using voice: {speaker_wav}")
    print(f"Gloss-to-text mode: {gloss_mode}")
    try:
        pipeline = BSLToSpeechPipeline(
            recognizer=recognizer,
            tts_engine='coqui',
            speaker_wav=speaker_wav,
            gloss_mode=gloss_mode,
        )
    except Exception as e:
        print(f"Coqui TTS initialization failed: {e}")
        print("Falling back to pyttsx3...")
        pipeline = BSLToSpeechPipeline(
            recognizer=recognizer,
            tts_engine='pyttsx3',
            gloss_mode=gloss_mode,
        )
    
    # Find sample file
    feature_base = Path(data_dir) / 'features' / 'bobsl' / 'v1.4' / 'video_features' / 'swin_v1'
    feature_dir = None
    for item in feature_base.iterdir():
        if item.is_dir():
            feature_dir = item
            break
    
    sample_file = list(feature_dir.glob('*.npy'))[0]
    
    # Process
    print(f"\nProcessing: {sample_file.name}")
    result = pipeline.process_features(
        feature_path=str(sample_file),
        window_stride=1.0,
        confidence_threshold=0.3,
        speak=False,
        save_audio=None,
    )
    
    print(f"\n--- Result ---")
    print(f"Glosses: {' '.join(result['glosses'][:15])}...")
    print(f"Text: {result['text']}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup and test BSL inference')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--model', type=str, default='outputs/best_model.pt')
    parser.add_argument('--vocab', type=str, default='data/processed/vocabulary.json')
    parser.add_argument('--model_type', type=str, default='temporal_mlp',
                       choices=['transformer', 'transformer_cls', 'temporal_mlp'])
    parser.add_argument('--speaker_wav', type=str, default='data/processed/voice_training.wav',
                       help='Reference voice file for Coqui TTS')
    parser.add_argument('--gloss_mode', type=str, default='simple',
                       choices=['simple', 'llm', 'groq'],
                       help='Gloss-to-text mode: simple (join), llm (local FLAN-T5), groq (Groq API)')
    parser.add_argument('--test', type=str, default='all',
                       choices=['vocab', 'recognizer', 'gloss', 'tts', 'pipeline', 'all'])
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BSL TO SPEECH PIPELINE SETUP")
    print("=" * 60)
    
    # Always setup vocabulary first
    vocab = setup_vocabulary(args.data_dir, args.vocab)
    
    glosses = None
    
    if args.test in ['recognizer', 'all']:
        glosses = test_recognizer(args.model, args.vocab, args.data_dir, args.model_type)
    
    if args.test in ['gloss', 'all']:
        test_gloss_to_text(glosses, args.gloss_mode)
    
    if args.test in ['tts', 'all']:
        test_tts(args.speaker_wav)
    
    if args.test in ['pipeline', 'all']:
        test_full_pipeline(args.model, args.vocab, args.data_dir, args.model_type, args.speaker_wav, args.gloss_mode)
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()