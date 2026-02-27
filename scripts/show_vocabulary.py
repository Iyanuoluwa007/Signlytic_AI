"""
Show the glosses that the recognition model was trained on.
"""

import json
from pathlib import Path

vocab_path = Path("D:/Signlytic_AI/code/bsl_translation_project/models/sign_recognition/vocabulary.json")

if vocab_path.exists():
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"\nModel trained on {len(vocab)} glosses:\n")
    print("-" * 50)
    
    for i, gloss in enumerate(sorted(vocab.keys()), 1):
        print(f"{i:3}. {gloss}")
    
    print("-" * 50)
    print(f"\nTotal: {len(vocab)} signs")
    print("\nNOTE: Only these signs can be recognized!")
    print("Signs like 'HELLO' are NOT in this vocabulary.")
else:
    print(f"Vocabulary not found: {vocab_path}")
