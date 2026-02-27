"""
Test Gloss-to-Text Converter (Direction 1)

Usage:
    python scripts/test_gloss_to_text.py                      # Simple mode demo
    python scripts/test_gloss_to_text.py --mode groq          # Groq API (best quality)
    python scripts/test_gloss_to_text.py --mode llm           # Local FLAN-T5
    python scripts/test_gloss_to_text.py --glosses "HELLO MY NAME JOHN"
    python scripts/test_gloss_to_text.py --tts --speaker-wav path/to/voice.wav
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
    from gloss_to_text import GlossToText, BSLToSpeechPipeline
except ImportError:
    try:
        from inference.gloss_to_text import GlossToText, BSLToSpeechPipeline
    except ImportError:
        from src.inference.gloss_to_text import GlossToText, BSLToSpeechPipeline


# Test sequences representing BSL gloss order
TEST_SEQUENCES = [
    ["TOMORROW", "MEETING", "WHAT", "TIME"],
    ["YESTERDAY", "I", "GO", "DOCTOR"],
    ["HELP", "LONDON", "LIVE", "FIND"],
    ["THANK", "YOU", "MUCH"],
    ["MY", "NAME", "SARAH"],
    ["WEATHER", "TODAY", "BEAUTIFUL"],
    ["TRAIN", "MANCHESTER", "LEAVE", "THREE", "AFTERNOON"],
    ["I", "NOT", "UNDERSTAND"],
    ["YOU", "HELP", "ME", "PLEASE"],
    ["WHAT", "YOUR", "NAME"],
]


def test_converter(mode: str, api_key: str = None):
    """Test gloss-to-text conversion."""
    print("\n" + "=" * 60)
    print(f"Gloss-to-Text Test (mode: {mode})")
    print("=" * 60)
    
    try:
        converter = GlossToText(mode=mode, groq_api_key=api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize converter: {e}")
        return
    
    print("\n{:<45} | {}".format("BSL Glosses", "English"))
    print("-" * 90)
    
    total_time = 0
    for glosses in TEST_SEQUENCES:
        gloss_str = " ".join(glosses)
        
        start = time.time()
        text = converter.convert(glosses)
        elapsed = time.time() - start
        total_time += elapsed
        
        # Truncate for display if needed
        if len(gloss_str) > 44:
            display_gloss = gloss_str[:41] + "..."
        else:
            display_gloss = gloss_str
        
        print(f"{display_gloss:<45} | {text}")
    
    avg_time = total_time / len(TEST_SEQUENCES)
    print("-" * 90)
    print(f"Average conversion time: {avg_time*1000:.1f}ms")


def test_single(glosses_str: str, mode: str, api_key: str = None):
    """Test single gloss sequence."""
    print("\n" + "=" * 60)
    print(f"Single Conversion Test (mode: {mode})")
    print("=" * 60)
    
    glosses = glosses_str.upper().split()
    
    try:
        converter = GlossToText(mode=mode, groq_api_key=api_key)
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    print(f"\nInput glosses: {' '.join(glosses)}")
    
    start = time.time()
    text = converter.convert(glosses)
    elapsed = time.time() - start
    
    print(f"Output text: {text}")
    print(f"Time: {elapsed*1000:.1f}ms")
    
    return text


def test_with_tts(glosses_str: str, speaker_wav: str, output_path: str, mode: str, api_key: str = None):
    """Test full pipeline with TTS output."""
    print("\n" + "=" * 60)
    print("BSL-to-Speech Pipeline Test")
    print("=" * 60)
    
    if not os.path.exists(speaker_wav):
        print(f"ERROR: Speaker reference not found: {speaker_wav}")
        return
    
    glosses = glosses_str.upper().split()
    
    print(f"\nInput glosses: {' '.join(glosses)}")
    print(f"Speaker reference: {speaker_wav}")
    print(f"Output: {output_path}")
    
    try:
        pipeline = BSLToSpeechPipeline(
            gloss_mode=mode,
            groq_api_key=api_key,
            speaker_wav=speaker_wav
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    print("\nProcessing...")
    start = time.time()
    result = pipeline.process(glosses, output_audio=output_path)
    elapsed = time.time() - start
    
    print(f"\nGenerated text: {result['text']}")
    if result.get('audio_path'):
        print(f"Audio saved: {result['audio_path']}")
    print(f"Total time: {elapsed:.2f}s")
    
    return result


def compare_modes(api_key: str = None):
    """Compare different conversion modes."""
    print("\n" + "=" * 60)
    print("Mode Comparison")
    print("=" * 60)
    
    modes = ['simple']
    
    # Check if Groq API key available
    if api_key or os.environ.get("GROQ_API_KEY"):
        modes.append('groq')
    
    # Check if transformers available
    try:
        import transformers
        modes.append('llm')
    except ImportError:
        pass
    
    test_glosses = [
        ["TOMORROW", "MEETING", "WHAT", "TIME"],
        ["HELP", "LONDON", "LIVE", "FIND"],
        ["I", "NOT", "UNDERSTAND"],
    ]
    
    converters = {}
    for mode in modes:
        try:
            converters[mode] = GlossToText(mode=mode, groq_api_key=api_key)
        except Exception as e:
            print(f"Could not initialize {mode}: {e}")
    
    for glosses in test_glosses:
        gloss_str = " ".join(glosses)
        print(f"\n{'='*60}")
        print(f"Glosses: {gloss_str}")
        print("-" * 60)
        
        for mode, conv in converters.items():
            text = conv.convert(glosses)
            print(f"  {mode:8s}: {text}")


def main():
    parser = argparse.ArgumentParser(description="Test Gloss-to-Text Converter")
    parser.add_argument("--mode", type=str, default="simple",
                       choices=["simple", "llm", "groq"],
                       help="Conversion mode")
    parser.add_argument("--glosses", type=str, 
                       help="Gloss sequence to convert (space-separated)")
    parser.add_argument("--api-key", type=str, 
                       help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all available modes")
    parser.add_argument("--tts", action="store_true",
                       help="Generate TTS audio output")
    parser.add_argument("--speaker-wav", type=str,
                       help="Speaker reference audio for TTS")
    parser.add_argument("--output", type=str,
                       help="Output audio path for TTS")
    
    args = parser.parse_args()
    
    # Auto-detect speaker wav
    if args.tts and not args.speaker_wav:
        candidates = [
            os.path.join(project_root, "data", "processed", "voice_training.wav"),
            "data/processed/voice_training.wav",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.speaker_wav = c
                break
    
    # Default output path
    if args.tts and not args.output:
        args.output = os.path.join(project_root, "outputs", "gloss_to_speech.wav")
    
    # Run tests
    if args.compare:
        compare_modes(args.api_key)
    elif args.glosses and args.tts:
        test_with_tts(args.glosses, args.speaker_wav, args.output, args.mode, args.api_key)
    elif args.glosses:
        test_single(args.glosses, args.mode, args.api_key)
    else:
        test_converter(args.mode, args.api_key)


if __name__ == "__main__":
    main()
