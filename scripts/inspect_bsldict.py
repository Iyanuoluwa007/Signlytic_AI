"""
Deep inspect BslDict structure - focus on 'videos' key.
"""

import pickle
import os
from pathlib import Path

def deep_inspect(pkl_path: str):
    """Deep inspection of BslDict."""
    
    print(f"Loading: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nTop-level keys: {list(data.keys())}")
    print("=" * 60)
    
    # Inspect each top-level key
    for key in data.keys():
        value = data[key]
        print(f"\n[{key}]")
        print(f"  Type: {type(value)}")
        
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First item type: {type(value[0])}")
                print(f"  First 5 items: {value[:5]}")
                
        elif isinstance(value, dict):
            print(f"  Number of keys: {len(value)}")
            sample_keys = list(value.keys())[:5]
            print(f"  Sample keys: {sample_keys}")
            
            # Check first item's structure
            if sample_keys:
                first_key = sample_keys[0]
                first_val = value[first_key]
                print(f"  First item [{first_key}]: {type(first_val)}")
                
                if isinstance(first_val, dict):
                    print(f"    Keys: {list(first_val.keys())}")
                    for k, v in list(first_val.items())[:5]:
                        v_str = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
                        print(f"      {k}: {v_str}")
                elif isinstance(first_val, (list, tuple)):
                    print(f"    Length: {len(first_val)}")
                    if len(first_val) > 0:
                        print(f"    First element: {first_val[0]}")
                else:
                    print(f"    Value: {first_val}")
        else:
            print(f"  Value: {str(value)[:200]}")
    
    # Deep dive into 'videos' specifically
    print("\n" + "=" * 60)
    print("DEEP DIVE: 'videos' key")
    print("=" * 60)
    
    if 'videos' in data:
        videos = data['videos']
        print(f"Type: {type(videos)}")
        
        if isinstance(videos, dict):
            print(f"Number of video entries: {len(videos)}")
            
            # Get a sample entry
            sample_key = list(videos.keys())[0]
            sample_val = videos[sample_key]
            
            print(f"\nSample entry key: {sample_key}")
            print(f"Sample entry type: {type(sample_val)}")
            
            if isinstance(sample_val, dict):
                print(f"Sample entry fields:")
                for k, v in sample_val.items():
                    v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                    print(f"  {k} ({type(v).__name__}): {v_str}")
                
                # Check for URL patterns
                print(f"\nLooking for video URLs in sample...")
                for k, v in sample_val.items():
                    if isinstance(v, str) and ('http' in v or 'mp4' in v or 'video' in v.lower()):
                        print(f"  FOUND in '{k}': {v}")
                    if isinstance(v, list) and len(v) > 0:
                        if isinstance(v[0], str) and ('http' in v[0] or 'mp4' in v[0]):
                            print(f"  FOUND in '{k}' (list): {v[0]}")
            
            # Check multiple entries for URLs
            print(f"\nScanning first 100 entries for video URLs...")
            url_count = 0
            url_samples = []
            
            for key, val in list(videos.items())[:100]:
                if isinstance(val, dict):
                    for field, content in val.items():
                        if isinstance(content, str) and 'http' in content:
                            url_count += 1
                            if len(url_samples) < 5:
                                url_samples.append((key, field, content))
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, str) and 'http' in item:
                                    url_count += 1
                                    if len(url_samples) < 5:
                                        url_samples.append((key, field, item))
            
            print(f"URLs found in first 100 entries: {url_count}")
            if url_samples:
                print("Sample URLs:")
                for word, field, url in url_samples:
                    print(f"  [{word}] {field}: {url}")
        
        elif isinstance(videos, list):
            print(f"Number of videos: {len(videos)}")
            if len(videos) > 0:
                print(f"First video entry: {videos[0]}")
    
    # Check words_to_id mapping
    print("\n" + "=" * 60)
    print("WORDS TO ID MAPPING")
    print("=" * 60)
    
    if 'words_to_id' in data:
        w2id = data['words_to_id']
        print(f"Type: {type(w2id)}")
        if isinstance(w2id, dict):
            print(f"Number of words: {len(w2id)}")
            sample_items = list(w2id.items())[:10]
            print(f"Sample mappings: {sample_items}")
    
    return data


def main():
    pkl_path = Path("E:/Signlytic_AI/code/bsl_translation_project/data/bsldict/bsldict/bsldict_v1.pkl")
    
    if not pkl_path.exists():
        pkl_path = Path("data/bsldict/bsldict/bsldict_v1.pkl")
    
    if not pkl_path.exists():
        print("ERROR: bsldict_v1.pkl not found")
        return
    
    deep_inspect(str(pkl_path))


if __name__ == "__main__":
    main()