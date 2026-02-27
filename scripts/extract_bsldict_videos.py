"""
Extract video data from BslDict - handle nested dict structure.
"""

import pickle
import json
from pathlib import Path

def extract_videos(pkl_path: str):
    """Extract video info from BslDict."""
    
    print(f"Loading: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    videos = data['videos']
    
    print(f"\nColumns in 'videos': {list(videos.keys())}")
    
    # Check structure of each column
    print("\n" + "=" * 60)
    print("Column structures:")
    print("=" * 60)
    
    for col_name in videos.keys():
        col_data = videos[col_name]
        print(f"\n{col_name}:")
        print(f"  Type: {type(col_data)}")
        
        if isinstance(col_data, list):
            print(f"  Length: {len(col_data)}")
            if len(col_data) > 0:
                print(f"  First item type: {type(col_data[0])}")
                sample = str(col_data[0])[:80]
                print(f"  First item: {sample}")
        elif isinstance(col_data, dict):
            print(f"  Keys: {list(col_data.keys())[:10]}")
            first_key = list(col_data.keys())[0]
            print(f"  First key '{first_key}': {type(col_data[first_key])}")
            sample = str(col_data[first_key])[:100]
            print(f"  First value: {sample}")
    
    # Focus on video-related columns
    print("\n" + "=" * 60)
    print("VIDEO URL COLUMNS")
    print("=" * 60)
    
    for col_name in ['videos_original', 'videos_360h_25fps', 'video_link_db', 'youtube_identifier_db']:
        if col_name in videos:
            col_data = videos[col_name]
            print(f"\n{col_name}:")
            print(f"  Type: {type(col_data)}")
            
            if isinstance(col_data, dict):
                print(f"  Keys: {list(col_data.keys())}")
                for k, v in col_data.items():
                    print(f"    {k}: {type(v)}")
                    if isinstance(v, list) and len(v) > 0:
                        print(f"      Length: {len(v)}")
                        print(f"      Sample[0]: {str(v[0])[:150]}")
                        print(f"      Sample[1]: {str(v[1])[:150] if len(v) > 1 else 'N/A'}")
            elif isinstance(col_data, list):
                print(f"  Length: {len(col_data)}")
                for i in range(min(3, len(col_data))):
                    print(f"  [{i}]: {str(col_data[i])[:150]}")
    
    # Build word -> video mapping using video_link_db
    print("\n" + "=" * 60)
    print("Building word -> video URL mapping")
    print("=" * 60)
    
    words = videos['word']
    video_links = videos.get('video_link_db', [])
    youtube_ids = videos.get('youtube_identifier_db', [])
    
    print(f"Words: {len(words)}")
    print(f"Video links: {len(video_links) if isinstance(video_links, list) else type(video_links)}")
    print(f"YouTube IDs: {len(youtube_ids) if isinstance(youtube_ids, list) else type(youtube_ids)}")
    
    # Create mapping
    video_map = {}
    has_video = 0
    has_youtube = 0
    
    for i in range(len(words)):
        word = words[i]
        entry = {'word': word, 'index': i}
        
        if isinstance(video_links, list) and i < len(video_links) and video_links[i]:
            entry['video_link'] = video_links[i]
            has_video += 1
        
        if isinstance(youtube_ids, list) and i < len(youtube_ids) and youtube_ids[i]:
            entry['youtube_id'] = youtube_ids[i]
            has_youtube += 1
        
        video_map[word] = entry
    
    print(f"\nWords with video_link: {has_video}")
    print(f"Words with youtube_id: {has_youtube}")
    
    # Show samples with URLs
    print("\nSample entries with video URLs:")
    count = 0
    for word, entry in video_map.items():
        if 'video_link' in entry or 'youtube_id' in entry:
            print(f"  {word}: {entry}")
            count += 1
            if count >= 10:
                break
    
    # Save full mapping
    output_path = Path(pkl_path).parent / "bsldict_video_map.json"
    
    # Filter to only entries with video data
    video_map_filtered = {w: e for w, e in video_map.items() 
                         if 'video_link' in e or 'youtube_id' in e}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(video_map_filtered, f, indent=2)
    
    print(f"\nSaved {len(video_map_filtered)} entries with video data to: {output_path}")
    
    return videos, video_map


def main():
    pkl_path = Path("E:/Signlytic_AI/code/bsl_translation_project/data/bsldict/bsldict/bsldict_v1.pkl")
    
    if not pkl_path.exists():
        pkl_path = Path("data/bsldict/bsldict/bsldict_v1.pkl")
    
    if not pkl_path.exists():
        print("ERROR: bsldict_v1.pkl not found")
        return
    
    extract_videos(str(pkl_path))


if __name__ == "__main__":
    main()