"""
Test Script for Speech-to-BSL Pipeline (Direction 2)

Usage:
    python scripts/test_speech_to_bsl.py                    # Run demo with sample sentences
    python scripts/test_speech_to_bsl.py --audio path.wav   # Process audio file
    python scripts/test_speech_to_bsl.py --text "sentence"  # Process text directly
    python scripts/test_speech_to_bsl.py --mode groq --api_key YOUR_KEY  # Use Groq API
"""

import argparse
import sys
import os

# Add project root and src to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'inference'))

# Try different import paths
try:
    from speech_to_bsl import SpeechToBSL, TextToGloss, WhisperASR, GlossRenderer, CoquiTTS
except ImportError:
    try:
        from inference.speech_to_bsl import SpeechToBSL, TextToGloss, WhisperASR, GlossRenderer, CoquiTTS
    except ImportError:
        from src.inference.speech_to_bsl import SpeechToBSL, TextToGloss, WhisperASR, GlossRenderer, CoquiTTS


def test_text_to_gloss(mode: str = "simple", api_key: str = None, vocab_path: str = None, strict: bool = False):
    """Test the text-to-gloss component."""
    print("\n" + "=" * 60)
    print(f"Testing Text-to-Gloss (mode: {mode}, strict_vocab: {strict})")
    print("=" * 60)
    
    converter = TextToGloss(
        mode=mode,
        vocabulary_path=vocab_path,
        groq_api_key=api_key,
        strict_vocab=strict
    )
    
    test_sentences = [
        "Hello, how are you?",
        "My name is Sarah.",
        "What time is the meeting tomorrow morning?",
        "I need help finding somewhere to live in London.",
        "The train to Manchester leaves at 3pm.",
        "Can you please repeat that?",
        "I don't understand.",
        "Thank you very much.",
    ]
    
    print("\n{:<50} | {}".format("English", "BSL Glosses"))
    print("-" * 80)
    
    total_coverage = []
    for sentence in test_sentences:
        if vocab_path:
            result = converter.convert_with_info(sentence)
            glosses = result['glosses']
            coverage = result['coverage']
            total_coverage.append(coverage)
            
            # Mark out-of-vocab glosses with asterisk
            marked_glosses = []
            for g in glosses:
                if g in result['out_of_vocab']:
                    marked_glosses.append(f"{g}*")
                else:
                    marked_glosses.append(g)
            gloss_str = " ".join(marked_glosses)
        else:
            glosses = converter.convert(sentence)
            gloss_str = " ".join(glosses)
        
        print(f"{sentence:<50} | {gloss_str}")
    
    if vocab_path and total_coverage:
        avg_coverage = sum(total_coverage) / len(total_coverage)
        print("-" * 80)
        print(f"Average vocabulary coverage: {avg_coverage:.1f}%")
        print("(* = gloss not in BOBSL vocabulary)")
    
    return converter


def test_full_pipeline(audio_path: str, mode: str = "simple", api_key: str = None, vocab_path: str = None):
    """Test the full speech-to-BSL pipeline with an audio file."""
    print("\n" + "=" * 60)
    print("Testing Full Speech-to-BSL Pipeline")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        return
    
    print(f"\nInput audio: {audio_path}")
    print(f"Gloss mode: {mode}")
    
    # Initialize pipeline
    pipeline = SpeechToBSL(
        whisper_model="base",  # Use 'small' or 'medium' for better accuracy
        gloss_mode=mode,
        vocabulary_path=vocab_path,
        groq_api_key=api_key
    )
    
    # Process
    result = pipeline.process(audio_path, return_intermediate=True)
    
    print("\n--- Results ---")
    print(f"Transcribed Text: {result['text']}")
    print(f"BSL Glosses: {' '.join(result['glosses'])}")
    print(f"\nRendering Info:")
    print(f"  Total Duration: {result['render']['total_duration']:.1f}s")
    print(f"  Gloss Count: {len(result['glosses'])}")
    
    print("\nTimeline:")
    for item in result['render']['timeline']:
        print(f"  [{item['start_time']:.1f}s - {item['end_time']:.1f}s] {item['gloss']}")
    
    return result


def test_text_input(text: str, mode: str = "simple", api_key: str = None, vocab_path: str = None):
    """Test pipeline with direct text input (no audio)."""
    print("\n" + "=" * 60)
    print("Testing Text-to-BSL (Direct Input)")
    print("=" * 60)
    
    # Just use the text-to-gloss + renderer
    converter = TextToGloss(
        mode=mode,
        vocabulary_path=vocab_path,
        groq_api_key=api_key
    )
    renderer = GlossRenderer()
    
    glosses = converter.convert(text)
    render_data = renderer.render(glosses)
    
    print(f"\nInput: {text}")
    print(f"BSL Glosses: {' '.join(glosses)}")
    print(f"Duration: {render_data['total_duration']:.1f}s")
    
    return glosses, render_data


def compare_modes(vocab_path: str = None, api_key: str = None):
    """Compare different text-to-gloss modes."""
    print("\n" + "=" * 60)
    print("Comparing Text-to-Gloss Modes")
    print("=" * 60)
    
    test_sentences = [
        "I need help finding somewhere to live in London.",
        "What time is the meeting tomorrow?",
        "Thank you very much for your help.",
    ]
    
    modes = ["simple"]
    if api_key:
        modes.append("groq")
    
    # Check if transformers is available for llm mode
    try:
        import transformers
        modes.append("llm")
    except ImportError:
        print("Note: 'llm' mode unavailable (transformers not installed)")
    
    converters = {}
    for mode in modes:
        try:
            converters[mode] = TextToGloss(
                mode=mode,
                vocabulary_path=vocab_path,
                groq_api_key=api_key
            )
        except Exception as e:
            print(f"Could not initialize {mode} mode: {e}")
    
    for sentence in test_sentences:
        print(f"\n{'='*60}")
        print(f"English: {sentence}")
        print("-" * 60)
        
        for mode, converter in converters.items():
            glosses = converter.convert(sentence)
            print(f"  {mode:8s}: {' '.join(glosses)}")


def main():
    parser = argparse.ArgumentParser(description="Test Speech-to-BSL Pipeline")
    parser.add_argument("--audio", type=str, help="Path to audio file to process")
    parser.add_argument("--text", type=str, help="Text to convert directly")
    parser.add_argument("--mode", type=str, default="simple", 
                       choices=["simple", "llm", "groq"],
                       help="Text-to-gloss conversion mode")
    parser.add_argument("--api_key", type=str, help="Groq API key (for mode=groq)")
    parser.add_argument("--vocab", type=str, help="Path to vocabulary.json")
    parser.add_argument("--strict", action="store_true", 
                       help="Only output glosses that exist in vocabulary")
    parser.add_argument("--compare", action="store_true", help="Compare all available modes")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample sentences")
    
    args = parser.parse_args()
    
    # Run appropriate test
    if args.compare:
        compare_modes(vocab_path=args.vocab, api_key=args.api_key)
    elif args.audio:
        test_full_pipeline(
            args.audio, 
            mode=args.mode, 
            api_key=args.api_key,
            vocab_path=args.vocab
        )
    elif args.text:
        test_text_input(
            args.text,
            mode=args.mode,
            api_key=args.api_key,
            vocab_path=args.vocab
        )
    else:
        # Default: run text-to-gloss demo
        test_text_to_gloss(
            mode=args.mode,
            api_key=args.api_key,
            vocab_path=args.vocab,
            strict=args.strict
        )


if __name__ == "__main__":
    main()