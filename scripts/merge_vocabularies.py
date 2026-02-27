"""
Merge BslDict (9,283 words) with BOBSL (1,843 glosses) vocabulary.

Usage:
    python scripts/merge_vocabularies.py

Output:
    data/processed/vocabulary_extended.json
"""

import os
import json
import pickle
from pathlib import Path


def load_bobsl_vocabulary(vocab_path: str) -> set:
    """Load BOBSL vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    # Handle different formats
    if isinstance(vocab_data, dict):
        if 'gloss_to_idx' in vocab_data:
            glosses = set(vocab_data['gloss_to_idx'].keys())
        else:
            glosses = set(vocab_data.keys())
    elif isinstance(vocab_data, list):
        glosses = set(vocab_data)
    else:
        glosses = set()
    
    # Normalize to lowercase
    return {g.lower() for g in glosses if isinstance(g, str)}


def load_bsldict_vocabulary(pkl_path: str) -> tuple:
    """Load BslDict vocabulary from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract words list
    words = data.get('words', [])
    words_normalised = data.get('words_normalised', [])
    
    # Also get the word descriptions for context
    words_to_id = data.get('words_to_id', {})
    
    # Clean up words (remove special characters, normalize)
    cleaned_words = set()
    for word in words:
        # Remove hyphens used for multi-word phrases, convert to space
        clean = word.replace('-', ' ').replace('?', '').strip().lower()
        if clean:
            cleaned_words.add(clean)
            # Also add individual words from phrases
            for w in clean.split():
                if len(w) > 1:  # Skip single letters
                    cleaned_words.add(w)
    
    # Also add normalised versions
    for word in words_normalised:
        if word:
            clean = str(word).lower().strip()
            if clean:
                cleaned_words.add(clean)
    
    return cleaned_words, data


def create_extended_vocabulary(
    bobsl_path: str,
    bsldict_path: str,
    output_path: str
) -> dict:
    """
    Merge BOBSL and BslDict vocabularies.
    
    Returns statistics about the merge.
    """
    print("=" * 60)
    print("Vocabulary Merger: BOBSL + BslDict")
    print("=" * 60)
    
    # Load BOBSL
    print(f"\nLoading BOBSL vocabulary from: {bobsl_path}")
    bobsl_glosses = load_bobsl_vocabulary(bobsl_path)
    print(f"  BOBSL glosses: {len(bobsl_glosses)}")
    
    # Load BslDict
    print(f"\nLoading BslDict vocabulary from: {bsldict_path}")
    bsldict_words, bsldict_data = load_bsldict_vocabulary(bsldict_path)
    print(f"  BslDict words: {len(bsldict_words)}")
    
    # Find overlap and unique
    overlap = bobsl_glosses & bsldict_words
    only_bobsl = bobsl_glosses - bsldict_words
    only_bsldict = bsldict_words - bobsl_glosses
    
    print(f"\n--- Vocabulary Analysis ---")
    print(f"  Overlap (in both):     {len(overlap)}")
    print(f"  Only in BOBSL:         {len(only_bobsl)}")
    print(f"  Only in BslDict:       {len(only_bsldict)}")
    
    # Merge
    combined = bobsl_glosses | bsldict_words
    print(f"\n  Combined vocabulary:   {len(combined)}")
    
    # Create output structure
    # Sort alphabetically
    sorted_vocab = sorted(combined)
    
    # Create gloss_to_idx mapping (compatible with existing format)
    # Reserve special tokens
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    
    gloss_to_idx = {}
    for i, token in enumerate(special_tokens):
        gloss_to_idx[token] = i
    
    for i, gloss in enumerate(sorted_vocab):
        gloss_to_idx[gloss] = i + len(special_tokens)
    
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    
    output_data = {
        'gloss_to_idx': gloss_to_idx,
        'idx_to_gloss': idx_to_gloss,
        'num_classes': len(gloss_to_idx),
        'special_tokens': special_tokens,
        'metadata': {
            'bobsl_count': len(bobsl_glosses),
            'bsldict_count': len(bsldict_words),
            'overlap_count': len(overlap),
            'combined_count': len(combined),
            'sources': ['BOBSL v1.4', 'BslDict v1']
        }
    }
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved extended vocabulary to: {output_path}")
    print(f"  Total entries: {len(gloss_to_idx)} (including {len(special_tokens)} special tokens)")
    
    # Also save a simple list version
    list_output_path = output_path.replace('.json', '_list.json')
    with open(list_output_path, 'w') as f:
        json.dump(sorted_vocab, f, indent=2)
    print(f"Saved word list to: {list_output_path}")
    
    # Print some example words that were added
    print("\n--- Sample new words from BslDict ---")
    common_words = ['hello', 'you', 'i', 'me', 'please', 'thank', 'sorry', 
                    'yes', 'no', 'what', 'where', 'when', 'how', 'why',
                    'name', 'my', 'your', 'good', 'bad', 'help']
    
    added_common = [w for w in common_words if w in only_bsldict]
    print(f"  Common words now available: {', '.join(added_common[:15])}")
    
    return {
        'bobsl_count': len(bobsl_glosses),
        'bsldict_count': len(bsldict_words),
        'overlap': len(overlap),
        'combined': len(combined),
        'output_path': output_path
    }


def main():
    # Default paths (relative to project root)
    project_root = Path(__file__).parent.parent
    
    bobsl_vocab = project_root / "data" / "processed" / "vocabulary.json"
    bsldict_pkl = project_root / "data" / "bsldict" / "bsldict" / "bsldict_v1.pkl"
    output_path = project_root / "data" / "processed" / "vocabulary_extended.json"
    
    # Check if running from different location
    if not bobsl_vocab.exists():
        # Try current directory structure
        bobsl_vocab = Path("data/processed/vocabulary.json")
        bsldict_pkl = Path("data/bsldict/bsldict/bsldict_v1.pkl")
        output_path = Path("data/processed/vocabulary_extended.json")
    
    # Validate inputs exist
    if not bobsl_vocab.exists():
        print(f"ERROR: BOBSL vocabulary not found: {bobsl_vocab}")
        print("Please provide the correct path.")
        return
    
    if not bsldict_pkl.exists():
        print(f"ERROR: BslDict pickle not found: {bsldict_pkl}")
        print("Please download it first:")
        print("  curl.exe -L -o bsldict_v1.pkl https://www.robots.ox.ac.uk/~vgg/research/bsldict/data/bsldict_v1.pkl")
        return
    
    # Run merge
    stats = create_extended_vocabulary(
        str(bobsl_vocab),
        str(bsldict_pkl),
        str(output_path)
    )
    
    print("\n" + "=" * 60)
    print("Done! Update your pipeline to use the extended vocabulary:")
    print("=" * 60)
    print(f"""
# In your code, change:
vocabulary_path = "data/processed/vocabulary.json"
# To:
vocabulary_path = "data/processed/vocabulary_extended.json"

# Or test with:
python scripts/test_speech_to_bsl.py --vocab data/processed/vocabulary_extended.json
""")


if __name__ == "__main__":
    main()