import urllib.request
import json
from pathlib import Path
import re

print("="*70)
print("SEARCHING FOR PHOENIX ANNOTATIONS")
print("="*70)

output_dir = Path("data/signavatars/phoenix/annotations")
output_dir.mkdir(parents=True, exist_ok=True)

# Try GitHub sources that host PHOENIX annotations
urls_to_try = [
    # Common GitHub repos that host PHOENIX data
    ("https://raw.githubusercontent.com/neccam/slt/master/data/phoenix2014T.train", "train.txt"),
    ("https://raw.githubusercontent.com/neccam/slt/master/data/phoenix2014T.dev", "dev.txt"),
    ("https://raw.githubusercontent.com/neccam/slt/master/data/phoenix2014T.test", "test.txt"),
]

for url, filename in urls_to_try:
    try:
        print(f"Trying: {filename}...")
        output_path = output_dir / filename
        urllib.request.urlretrieve(url, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"  Downloaded: {len(lines)} lines")
        
        # Show sample
        if lines:
            print(f"  Sample: {lines[0][:100]}...")
    except Exception as e:
        print(f"  Failed: {e}")

# Try alternative source
print("\nTrying alternative sources...")
alt_urls = [
    ("https://raw.githubusercontent.com/AlanJiang98/OneChart/main/Data/phoenix-2014t/train.corpus.csv", "train.corpus.csv"),
    ("https://raw.githubusercontent.com/Spoken-language-understanding/phoenix14t/main/annotations/manual/train.corpus.csv", "train2.corpus.csv"),
]

for url, filename in alt_urls:
    try:
        print(f"Trying: {url[:60]}...")
        output_path = output_dir / filename
        urllib.request.urlretrieve(url, output_path)
        
        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f"  Downloaded: {len(content)} bytes")
        print(f"  First 200 chars: {content[:200]}")
    except Exception as e:
        print(f"  Failed: {e}")

# Check what we have
print("\n" + "="*70)
print("Downloaded files:")
for f in output_dir.iterdir():
    print(f"  {f.name}: {f.stat().st_size} bytes")
