import urllib.request
import json
from pathlib import Path

print("="*70)
print("TRYING MORE SOURCES FOR PHOENIX")
print("="*70)

output_dir = Path("data/signavatars/phoenix/annotations")

# Try HuggingFace datasets
urls = [
    # HuggingFace datasets often host this
    "https://huggingface.co/datasets/aoxo/phoenix-2014t/raw/main/train.csv",
    "https://huggingface.co/datasets/aoxo/phoenix-2014t/resolve/main/train.csv",
    # SignLLM project
    "https://raw.githubusercontent.com/SignLLM/SignLLM/main/data/phoenix/train.json",
    # SLT benchmarks
    "https://raw.githubusercontent.com/OpenNMT/OpenNMT-py/master/data/phoenix/train.de-en.de",
]

for url in urls:
    try:
        print(f"Trying: {url[:70]}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=10)
        content = response.read().decode('utf-8', errors='ignore')
        
        filename = url.split('/')[-1]
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  SUCCESS: {len(content)} bytes")
        print(f"  Preview: {content[:150]}...")
        break
    except Exception as e:
        print(f"  Failed: {type(e).__name__}")

print("\n" + "="*70)
print("SUMMARY: PHOENIX requires manual download")
print("="*70)
print("""
To get PHOENIX-2014T annotations:

1. Visit: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
2. Register and download the corpus files
3. Place them in: data/signavatars/phoenix/annotations/

The corpus files contain:
  - Video ID (matches our SignAvatars files)
  - Gloss sequence (German Sign Language glosses)
  - German text translation

For now, we can proceed with:
  - WLASL (1000 samples) - DONE
  - French LSF (710 samples) - DONE
  - Total: 1710 samples, 478 classes
""")
