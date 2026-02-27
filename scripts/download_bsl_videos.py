"""
BSL Video Downloader

Downloads BSL sign videos from signbsl.com based on BslDict mapping.
Selective download - only fetches videos for glosses in vocabulary.

Usage:
    python scripts/download_bsl_videos.py                    # Download all vocabulary videos
    python scripts/download_bsl_videos.py --limit 100        # Download first 100
    python scripts/download_bsl_videos.py --word hello       # Download specific word
    python scripts/download_bsl_videos.py --resume           # Resume interrupted download
"""

import os
import json
import time
import pickle
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class BSLVideoDownloader:
    """Download BSL sign videos from signbsl.com."""
    
    def __init__(
        self,
        video_map_path: str,
        output_dir: str,
        vocabulary_path: Optional[str] = None
    ):
        """
        Initialize downloader.
        
        Args:
            video_map_path: Path to bsldict_video_map.json
            output_dir: Directory to save videos
            vocabulary_path: Optional - only download videos for these glosses
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load video mapping
        print(f"Loading video map: {video_map_path}")
        with open(video_map_path, 'r') as f:
            self.video_map = json.load(f)
        print(f"Loaded {len(self.video_map)} video entries")
        
        # Load vocabulary if provided
        self.vocabulary = None
        if vocabulary_path and os.path.exists(vocabulary_path):
            print(f"Loading vocabulary: {vocabulary_path}")
            with open(vocabulary_path, 'r') as f:
                vocab_data = json.load(f)
            
            if isinstance(vocab_data, dict):
                if 'gloss_to_idx' in vocab_data:
                    self.vocabulary = set(k.lower() for k in vocab_data['gloss_to_idx'].keys())
                else:
                    self.vocabulary = set(k.lower() for k in vocab_data.keys())
            elif isinstance(vocab_data, list):
                self.vocabulary = set(v.lower() for v in vocab_data)
            
            print(f"Vocabulary size: {len(self.vocabulary)}")
        
        # Track download progress
        self.progress_file = self.output_dir / "download_progress.json"
        self.downloaded = self._load_progress()
        
    def _load_progress(self) -> set:
        """Load previously downloaded files."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_progress(self):
        """Save download progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.downloaded), f)
    
    def _get_download_list(self, limit: Optional[int] = None) -> List[Dict]:
        """Get list of videos to download."""
        to_download = []
        
        for word, info in self.video_map.items():
            # Skip if already downloaded
            if word in self.downloaded:
                continue
            
            # Check vocabulary filter
            if self.vocabulary is not None:
                if word.lower() not in self.vocabulary:
                    continue
            
            # Get video URL
            video_url = info.get('video_link', '')
            
            # Skip YouTube embeds (need different handling)
            if 'youtube' in video_url.lower():
                continue
            
            # Skip invalid URLs
            if not video_url.startswith('http'):
                continue
            
            to_download.append({
                'word': word,
                'url': video_url,
                'filename': f"{word.replace('/', '_').replace(' ', '_')}.mp4"
            })
        
        if limit:
            to_download = to_download[:limit]
        
        return to_download
    
    def download_video(self, word: str, url: str, filename: str) -> bool:
        """Download a single video."""
        output_path = self.output_dir / filename
        
        if output_path.exists():
            return True
        
        try:
            # Add headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            request = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(request, timeout=30) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            
            return True
            
        except Exception as e:
            print(f"  Error downloading {word}: {e}")
            return False
    
    def download_all(
        self,
        limit: Optional[int] = None,
        workers: int = 4,
        delay: float = 0.5
    ) -> Dict:
        """
        Download all videos.
        
        Args:
            limit: Maximum number of videos to download
            workers: Number of parallel downloads
            delay: Delay between downloads (be nice to server)
        """
        to_download = self._get_download_list(limit)
        
        print(f"\nVideos to download: {len(to_download)}")
        print(f"Already downloaded: {len(self.downloaded)}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        if not to_download:
            print("Nothing to download!")
            return {'downloaded': 0, 'failed': 0, 'skipped': len(self.downloaded)}
        
        success = 0
        failed = 0
        failed_words = []
        
        for i, item in enumerate(to_download):
            word = item['word']
            url = item['url']
            filename = item['filename']
            
            print(f"[{i+1}/{len(to_download)}] Downloading: {word}")
            
            if self.download_video(word, url, filename):
                success += 1
                self.downloaded.add(word)
                
                # Save progress periodically
                if success % 10 == 0:
                    self._save_progress()
            else:
                failed += 1
                failed_words.append(word)
            
            # Be nice to the server
            time.sleep(delay)
        
        # Final save
        self._save_progress()
        
        print("-" * 60)
        print(f"Downloaded: {success}")
        print(f"Failed: {failed}")
        print(f"Total available: {len(self.downloaded)}")
        
        if failed_words:
            print(f"\nFailed words: {failed_words[:20]}...")
        
        return {
            'downloaded': success,
            'failed': failed,
            'total': len(self.downloaded),
            'failed_words': failed_words
        }
    
    def download_word(self, word: str) -> bool:
        """Download video for a specific word."""
        word_lower = word.lower()
        
        # Find in video map
        info = self.video_map.get(word_lower) or self.video_map.get(word)
        
        if not info:
            print(f"Word not found in video map: {word}")
            return False
        
        url = info.get('video_link', '')
        if not url.startswith('http'):
            print(f"Invalid URL for {word}: {url}")
            return False
        
        filename = f"{word.replace('/', '_').replace(' ', '_')}.mp4"
        print(f"Downloading: {word} -> {filename}")
        
        return self.download_video(word, url, filename)
    
    def get_stats(self) -> Dict:
        """Get download statistics."""
        total_in_map = len(self.video_map)
        in_vocab = 0
        downloadable = 0
        
        for word, info in self.video_map.items():
            url = info.get('video_link', '')
            
            if self.vocabulary and word.lower() in self.vocabulary:
                in_vocab += 1
            
            if url.startswith('http') and 'youtube' not in url.lower():
                downloadable += 1
        
        return {
            'total_in_map': total_in_map,
            'in_vocabulary': in_vocab if self.vocabulary else 'N/A',
            'downloadable': downloadable,
            'downloaded': len(self.downloaded),
            'output_dir': str(self.output_dir)
        }


def main():
    parser = argparse.ArgumentParser(description="Download BSL sign videos")
    parser.add_argument("--video-map", type=str, default=None,
                       help="Path to bsldict_video_map.json")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for videos")
    parser.add_argument("--vocab", type=str, default=None,
                       help="Vocabulary file (only download these words)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Maximum videos to download")
    parser.add_argument("--word", type=str, default=None,
                       help="Download specific word only")
    parser.add_argument("--workers", type=int, default=1,
                       help="Parallel downloads (be careful)")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between downloads")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics only")
    parser.add_argument("--resume", action="store_true",
                       help="Resume interrupted download")
    
    args = parser.parse_args()
    
    # Auto-detect paths
    project_root = Path(__file__).parent.parent
    
    if args.video_map is None:
        args.video_map = project_root / "data" / "bsldict" / "bsldict" / "bsldict_video_map.json"
    
    if args.output is None:
        args.output = project_root / "data" / "videos" / "bsl_signs"
    
    if args.vocab is None:
        vocab_path = project_root / "data" / "processed" / "vocabulary_extended.json"
        if vocab_path.exists():
            args.vocab = vocab_path
    
    # Initialize downloader
    downloader = BSLVideoDownloader(
        video_map_path=str(args.video_map),
        output_dir=str(args.output),
        vocabulary_path=str(args.vocab) if args.vocab else None
    )
    
    if args.stats:
        stats = downloader.get_stats()
        print("\nDownload Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return
    
    if args.word:
        downloader.download_word(args.word)
        return
    
    # Download all
    downloader.download_all(
        limit=args.limit,
        workers=args.workers,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
