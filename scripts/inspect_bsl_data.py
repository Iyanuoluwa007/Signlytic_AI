from pathlib import Path
import json
import pickle

print("="*70)
print("BSL DATA INSPECTION")
print("="*70)

# 1. BOBSL Annotations
print("\n1. BOBSL ANNOTATIONS:")
bobsl_ann = Path("data/processed/annotations/bobsl")
if bobsl_ann.exists():
    files = list(bobsl_ann.iterdir())
    print(f"   Files: {len(files)}")
    for f in files[:10]:
        print(f"   - {f.name} ({f.stat().st_size} bytes)")
    
    # Try to read one
    for f in files[:3]:
        if f.suffix == '.json':
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                print(f"\n   Sample from {f.name}:")
                if isinstance(data, dict):
                    print(f"   Keys: {list(data.keys())[:10]}")
                elif isinstance(data, list):
                    print(f"   Items: {len(data)}, First: {data[0] if data else 'empty'}")
            except Exception as e:
                print(f"   Error reading {f.name}: {e}")

# 2. BSL-1K Labels
print("\n" + "="*70)
print("2. BSL-1K DATA:")
bsl1k = Path("data/BSL-1K")
if bsl1k.exists():
    for subdir in bsl1k.iterdir():
        if subdir.is_dir():
            print(f"\n   {subdir.name}/")
            for f in list(subdir.rglob("*"))[:5]:
                if f.is_file():
                    print(f"     - {f.relative_to(subdir)} ({f.stat().st_size} bytes)")

# 3. BSL Dictionary
print("\n" + "="*70)
print("3. BSLDICT DATA:")
bsldict = Path("data/bsldict/bsldict")
if bsldict.exists():
    for f in bsldict.iterdir():
        print(f"   - {f.name} ({f.stat().st_size} bytes)")
        
        if f.suffix == '.json':
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                if isinstance(data, dict):
                    print(f"     Keys: {list(data.keys())[:5]}")
                    first_key = list(data.keys())[0]
                    print(f"     Sample [{first_key}]: {data[first_key]}")
            except:
                pass
        elif f.suffix == '.pkl':
            try:
                with open(f, 'rb') as fp:
                    data = pickle.load(fp)
                if isinstance(data, dict):
                    print(f"     Keys: {list(data.keys())[:5]}")
            except:
                pass

# 4. BOBSL Features (SWIN)
print("\n" + "="*70)
print("4. BOBSL FEATURES:")
bobsl_features = Path("data/processed/features/bobsl")
if bobsl_features.exists():
    files = list(bobsl_features.rglob("*.npy"))
    print(f"   NPY files: {len(files)}")
    if files:
        print(f"   Sample: {files[0].name}")

# 5. BOBSL Subtitles
print("\n" + "="*70)
print("5. BOBSL SUBTITLES:")
subtitles = Path("data/processed/subtitles/bobsl")
if subtitles.exists():
    files = list(subtitles.iterdir())
    print(f"   Files: {len(files)}")
    for f in files[:3]:
        print(f"   - {f.name}")
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                content = fp.read()[:300]
            print(f"     Preview: {content[:150]}...")
        except:
            pass

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
