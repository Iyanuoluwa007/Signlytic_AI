from pathlib import Path
import os

print("="*70)
print("BSL (BRITISH SIGN LANGUAGE) DATA CHECK")
print("="*70)

# Check what BSL data exists
data_root = Path("data")

print("\n1. Checking for BSL-specific folders...")
bsl_keywords = ['bsl', 'bobsl', 'british', 'bsldict', 'bsl1k', 'bsl-1k']

for folder in data_root.rglob("*"):
    if folder.is_dir():
        name_lower = folder.name.lower()
        for kw in bsl_keywords:
            if kw in name_lower:
                file_count = len(list(folder.rglob("*")))
                print(f"  Found: {folder} ({file_count} files)")

print("\n2. Checking for BSL in file names...")
bsl_files = []
for f in data_root.rglob("*"):
    if f.is_file():
        name_lower = f.name.lower()
        for kw in bsl_keywords:
            if kw in name_lower:
                bsl_files.append(f)
                break

print(f"  BSL-related files: {len(bsl_files)}")
for f in bsl_files[:10]:
    print(f"    {f}")

print("\n3. SignAvatars data (current):")
signavatars = Path("data/signavatars")
if signavatars.exists():
    for d in signavatars.iterdir():
        if d.is_dir():
            count = len(list(d.rglob("*.pkl")))
            lang = {
                'wlasl': 'ASL (American)',
                'how2sign': 'ASL (American)',
                'phoenix': 'DGS (German)',
                'hamnosys': 'Mixed (French/Polish/Greek)',
            }.get(d.name, 'Unknown')
            print(f"  {d.name}: {count} files - {lang}")

print("\n4. BSL Datasets Available Online:")
print("""
  BOBSL (BBC-Oxford BSL):
    - 1.2M signs, 2,281 hours of BBC content
    - URL: https://www.robots.ox.ac.uk/~vgg/data/bobsl/
    - Requires: License agreement
    
  BSL-1K:
    - 1,000 sign vocabulary
    - URL: https://www.robots.ox.ac.uk/~vgg/research/bsl1k/
    - Requires: Request access
    
  BSL Signbank:
    - Dictionary with video examples
    - URL: https://bslsignbank.ucl.ac.uk/
    - Public access
""")
