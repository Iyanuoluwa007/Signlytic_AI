from pathlib import Path
import os

print("="*70)
print("CHECKING FOR ACTUAL BSL-1K LABELS")
print("="*70)

# Check raw data folder
raw_path = Path("data/raw")
if raw_path.exists():
    print("\n1. RAW DATA FILES:")
    for f in raw_path.iterdir():
        size_mb = f.stat().st_size / (1024*1024)
        print(f"   {f.name}: {size_mb:.1f} MB")

# Check for any tar/zip files that might have labels
print("\n2. LOOKING FOR LABEL ARCHIVES:")
for f in Path("data").rglob("*.tar*"):
    size_mb = f.stat().st_size / (1024*1024)
    print(f"   {f}: {size_mb:.1f} MB")

for f in Path("data").rglob("*.zip"):
    size_mb = f.stat().st_size / (1024*1024)
    print(f"   {f}: {size_mb:.1f} MB")

# Check BSL-1K folder structure
print("\n3. BSL-1K FULL STRUCTURE:")
bsl1k = Path("data/BSL-1K")
for root, dirs, files in os.walk(bsl1k):
    level = root.replace(str(bsl1k), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        fpath = Path(root) / file
        size = fpath.stat().st_size
        print(f"{subindent}{file} ({size} bytes)")

print("\n4. SOLUTION OPTIONS:")
print("""
Since BSL videos have 1 sample per gloss (dictionary format), we have 3 options:

Option A: POSE EXTRACTION FROM BSL VIDEOS
   - Extract MediaPipe pose from 5,203 BSL videos
   - Use for similarity/retrieval-based recognition
   - Match input sign to nearest BSL dictionary entry

Option B: BOBSL PSEUDO-LABEL DOWNLOAD
   - Download BSL-1K pseudo-labels from VGG website
   - These provide weak labels for BOBSL continuous videos
   - URL: https://www.robots.ox.ac.uk/~vgg/research/bsl1k/

Option C: TRANSFER LEARNING
   - Use our multi-lingual model (ASL+LSF) as base
   - Fine-tune on BSL with 1-shot learning techniques
   - Use the 5203 BSL videos as support set
""")
