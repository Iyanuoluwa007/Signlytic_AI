import json
import urllib.request
from pathlib import Path

print("Downloading WLASL annotations...")

# Official WLASL annotation URL
url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
output_path = Path("data/signavatars/wlasl/WLASL_v0.3.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")
    
    # Parse and show stats
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total glosses: {len(data)}")
    
    # Build video_id to gloss mapping
    video_to_gloss = {}
    gloss_counts = {}
    
    for item in data:
        gloss = item['gloss']
        gloss_counts[gloss] = 0
        for instance in item.get('instances', []):
            video_id = instance.get('video_id', '')
            if video_id:
                video_to_gloss[video_id] = gloss
                gloss_counts[gloss] += 1
    
    print(f"Total video mappings: {len(video_to_gloss)}")
    
    # Save mapping
    mapping_path = Path("data/signavatars/wlasl/video_to_gloss.json")
    with open(mapping_path, 'w') as f:
        json.dump(video_to_gloss, f, indent=2)
    print(f"Saved mapping to {mapping_path}")
    
    # Show sample
    print("\nSample mappings:")
    for vid, gloss in list(video_to_gloss.items())[:10]:
        print(f"  {vid} -> {gloss}")
    
    # Check which of our SignAvatars files have mappings
    signavatars_files = list(Path("data/signavatars/wlasl").rglob("*.pkl"))
    matched = 0
    unmatched = []
    
    for f in signavatars_files:
        vid = f.stem
        if vid in video_to_gloss:
            matched += 1
        else:
            unmatched.append(vid)
    
    print(f"\nSignAvatars files matched: {matched}/{len(signavatars_files)}")
    if unmatched[:5]:
        print(f"Unmatched samples: {unmatched[:5]}")

except Exception as e:
    print(f"Error: {e}")
