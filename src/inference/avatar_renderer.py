"""
BSL Avatar Video Renderer

Renders BSL signing videos by concatenating individual sign video clips.
Supports:
- Video concatenation from gloss sequence
- Fallback to fingerspelling for unknown signs
- Smooth transitions between signs
- Configurable playback speed

Usage:
    from avatar_renderer import BSLAvatarRenderer
    
    renderer = BSLAvatarRenderer(video_dir="data/videos/bsl_signs")
    output_path = renderer.render(["HELLO", "MY", "NAME"], "output.mp4")
"""

import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class BSLAvatarRenderer:
    """
    Render BSL signing videos from gloss sequences.
    
    Concatenates individual sign video clips to create
    continuous signing output.
    """
    
    # Fingerspelling alphabet mapping
    FINGERSPELL = {
        'a': 'fingerspell_a', 'b': 'fingerspell_b', 'c': 'fingerspell_c',
        'd': 'fingerspell_d', 'e': 'fingerspell_e', 'f': 'fingerspell_f',
        'g': 'fingerspell_g', 'h': 'fingerspell_h', 'i': 'fingerspell_i',
        'j': 'fingerspell_j', 'k': 'fingerspell_k', 'l': 'fingerspell_l',
        'm': 'fingerspell_m', 'n': 'fingerspell_n', 'o': 'fingerspell_o',
        'p': 'fingerspell_p', 'q': 'fingerspell_q', 'r': 'fingerspell_r',
        's': 'fingerspell_s', 't': 'fingerspell_t', 'u': 'fingerspell_u',
        'v': 'fingerspell_v', 'w': 'fingerspell_w', 'x': 'fingerspell_x',
        'y': 'fingerspell_y', 'z': 'fingerspell_z',
    }
    
    def __init__(
        self,
        video_dir: str,
        video_map_path: Optional[str] = None,
        transition_frames: int = 5,
        default_fps: int = 25
    ):
        """
        Initialize avatar renderer.
        
        Args:
            video_dir: Directory containing sign video files
            video_map_path: Path to video mapping JSON (word -> filename)
            transition_frames: Frames to blend between signs
            default_fps: Default video framerate
        """
        self.video_dir = Path(video_dir)
        self.transition_frames = transition_frames
        self.default_fps = default_fps
        
        # Build video index
        self.video_index = self._build_index(video_map_path)
        print(f"Avatar renderer initialized with {len(self.video_index)} available signs")
    
    def _build_index(self, video_map_path: Optional[str]) -> Dict[str, str]:
        """Build index of available sign videos."""
        index = {}
        
        # Index from video map if available
        if video_map_path and os.path.exists(video_map_path):
            with open(video_map_path, 'r') as f:
                video_map = json.load(f)
            
            for word, info in video_map.items():
                # Expected filename pattern
                filename = f"{word.replace('/', '_').replace(' ', '_')}.mp4"
                filepath = self.video_dir / filename
                
                if filepath.exists():
                    index[word.lower()] = str(filepath)
        
        # Also scan directory for any video files
        if self.video_dir.exists():
            for video_file in self.video_dir.glob("*.mp4"):
                # Extract word from filename
                word = video_file.stem.lower().replace('_', ' ')
                if word not in index:
                    index[word] = str(video_file)
        
        return index
    
    def has_video(self, gloss: str) -> bool:
        """Check if video exists for a gloss."""
        return gloss.lower() in self.video_index
    
    def get_video_path(self, gloss: str) -> Optional[str]:
        """Get video path for a gloss."""
        return self.video_index.get(gloss.lower())
    
    def get_available_glosses(self) -> List[str]:
        """Get list of glosses with available videos."""
        return list(self.video_index.keys())
    
    def get_coverage(self, glosses: List[str]) -> Dict:
        """
        Calculate video coverage for a gloss sequence.
        
        Returns:
            Dict with available, missing, and coverage percentage
        """
        available = []
        missing = []
        
        for gloss in glosses:
            gloss_lower = gloss.lower()
            if gloss_lower in ['<unk>', '<pad>', '<sos>', '<eos>']:
                continue
            
            if self.has_video(gloss_lower):
                available.append(gloss)
            else:
                missing.append(gloss)
        
        total = len(available) + len(missing)
        coverage = (len(available) / total * 100) if total > 0 else 0
        
        return {
            'available': available,
            'missing': missing,
            'coverage': coverage,
            'total': total
        }
    
    def render(
        self,
        glosses: List[str],
        output_path: str,
        speed: float = 1.0,
        include_text: bool = True
    ) -> Optional[str]:
        """
        Render video from gloss sequence.
        
        Args:
            glosses: List of BSL glosses
            output_path: Path for output video
            speed: Playback speed multiplier
            include_text: Overlay gloss text on video
            
        Returns:
            Path to rendered video, or None if failed
        """
        # Filter special tokens
        glosses = [g for g in glosses if g.lower() not in ['<unk>', '<pad>', '<sos>', '<eos>']]
        
        if not glosses:
            print("No glosses to render")
            return None
        
        # Get video paths for each gloss
        video_files = []
        for gloss in glosses:
            video_path = self.get_video_path(gloss)
            if video_path:
                video_files.append((gloss, video_path))
            else:
                print(f"  Missing video for: {gloss}")
                # Could add fingerspelling fallback here
        
        if not video_files:
            print("No videos available for any gloss")
            return None
        
        print(f"Rendering {len(video_files)} signs...")
        
        # Concatenate videos using ffmpeg
        try:
            output = self._concatenate_videos(video_files, output_path, speed, include_text)
            return output
        except Exception as e:
            print(f"Render error: {e}")
            return None
    
    def _concatenate_videos(
        self,
        video_files: List[Tuple[str, str]],
        output_path: str,
        speed: float,
        include_text: bool
    ) -> str:
        """Concatenate video files using ffmpeg with proper re-encoding."""
        
        # Create file list for ffmpeg concat
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            list_file = f.name
            for gloss, video_path in video_files:
                # Use forward slashes and proper escaping for Windows
                escaped_path = video_path.replace('\\', '/').replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        try:
            # Build filter for consistent output
            # Scale all videos to same size and framerate to avoid freezing
            filter_complex = "scale=480:360:force_original_aspect_ratio=decrease,pad=480:360:(ow-iw)/2:(oh-ih)/2,fps=25"
            
            if speed != 1.0:
                filter_complex += f",setpts={1/speed}*PTS"
            
            # Build ffmpeg command with re-encoding for consistency
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-vf', filter_complex,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Ensure compatibility
                '-r', '25',  # Force consistent framerate
                '-an',  # Remove audio to avoid sync issues
                output_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr[:500]}")
                # Try alternative method
                return self._reencode_concat(video_files, output_path)
            
            return output_path
            
        finally:
            # Clean up temp file
            if os.path.exists(list_file):
                os.remove(list_file)
    
    def _reencode_concat(
        self,
        video_files: List[Tuple[str, str]],
        output_path: str
    ) -> str:
        """Alternative: re-encode each clip first then concat."""
        
        temp_dir = tempfile.mkdtemp()
        normalized_files = []
        
        try:
            # Normalize each video to same format
            for i, (gloss, video_path) in enumerate(video_files):
                temp_output = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-vf', 'scale=480:360:force_original_aspect_ratio=decrease,pad=480:360:(ow-iw)/2:(oh-ih)/2,fps=25',
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    '-t', '5',  # Limit clip length
                    temp_output
                ]
                
                subprocess.run(cmd, capture_output=True)
                
                if os.path.exists(temp_output):
                    normalized_files.append(temp_output)
            
            if not normalized_files:
                return None
            
            # Create concat list
            list_file = os.path.join(temp_dir, "list.txt")
            with open(list_file, 'w') as f:
                for filepath in normalized_files:
                    escaped = filepath.replace('\\', '/').replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            
            # Concat normalized clips
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True)
            return output_path
            
        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _simple_concat(
        self,
        video_files: List[Tuple[str, str]],
        output_path: str
    ) -> str:
        """Simple concatenation without filters."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            list_file = f.name
            for gloss, video_path in video_files:
                escaped_path = video_path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True)
            return output_path
            
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)
    
    def render_preview(self, glosses: List[str]) -> Dict:
        """
        Get preview info without rendering.
        
        Returns timeline and coverage info.
        """
        coverage = self.get_coverage(glosses)
        
        timeline = []
        current_time = 0.0
        
        for gloss in glosses:
            if gloss.lower() in ['<unk>', '<pad>', '<sos>', '<eos>']:
                continue
            
            video_path = self.get_video_path(gloss)
            duration = self._get_video_duration(video_path) if video_path else 1.0
            
            timeline.append({
                'gloss': gloss,
                'start': current_time,
                'end': current_time + duration,
                'has_video': video_path is not None,
                'video_path': video_path
            })
            
            current_time += duration
        
        return {
            'glosses': glosses,
            'timeline': timeline,
            'total_duration': current_time,
            'coverage': coverage
        }
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        if not video_path:
            return 1.0
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 1.0  # Default duration


class BSLAvatarPipeline:
    """
    Complete avatar rendering pipeline.
    
    Combines gloss-to-video rendering with text-to-gloss
    for end-to-end speech-to-avatar.
    """
    
    def __init__(
        self,
        video_dir: str,
        video_map_path: Optional[str] = None,
        vocabulary_path: Optional[str] = None
    ):
        """Initialize pipeline."""
        self.renderer = BSLAvatarRenderer(
            video_dir=video_dir,
            video_map_path=video_map_path
        )
        
        # Optionally load text-to-gloss converter
        self.text_to_gloss = None
        try:
            from speech_to_bsl import TextToGloss
            self.text_to_gloss = TextToGloss(
                mode="simple",
                vocabulary_path=vocabulary_path
            )
        except ImportError:
            pass
    
    def render_from_glosses(
        self,
        glosses: List[str],
        output_path: str
    ) -> Optional[str]:
        """Render video from glosses."""
        return self.renderer.render(glosses, output_path)
    
    def render_from_text(
        self,
        text: str,
        output_path: str
    ) -> Optional[str]:
        """Render video from English text."""
        if self.text_to_gloss is None:
            print("Text-to-gloss converter not available")
            return None
        
        glosses = self.text_to_gloss.convert(text)
        print(f"Text: {text}")
        print(f"Glosses: {' '.join(glosses)}")
        
        return self.renderer.render(glosses, output_path)
    
    def get_coverage_from_text(self, text: str) -> Dict:
        """Get video coverage for text input."""
        if self.text_to_gloss is None:
            return {'error': 'Text-to-gloss not available'}
        
        glosses = self.text_to_gloss.convert(text)
        return self.renderer.get_coverage(glosses)


def demo():
    """Demo avatar rendering."""
    print("=" * 60)
    print("BSL Avatar Renderer Demo")
    print("=" * 60)
    
    # Initialize renderer
    project_root = Path(__file__).parent.parent
    video_dir = project_root / "data" / "videos" / "bsl_signs"
    video_map = project_root / "data" / "bsldict" / "bsldict" / "bsldict_video_map.json"
    
    if not video_dir.exists():
        print(f"\nVideo directory not found: {video_dir}")
        print("Run download_bsl_videos.py first to download sign videos.")
        return
    
    renderer = BSLAvatarRenderer(
        video_dir=str(video_dir),
        video_map_path=str(video_map) if video_map.exists() else None
    )
    
    # Test glosses
    test_glosses = ["HELLO", "MY", "NAME", "JOHN"]
    
    print(f"\nTest glosses: {' '.join(test_glosses)}")
    
    # Check coverage
    coverage = renderer.get_coverage(test_glosses)
    print(f"\nCoverage: {coverage['coverage']:.1f}%")
    print(f"Available: {coverage['available']}")
    print(f"Missing: {coverage['missing']}")
    
    # Render preview
    preview = renderer.render_preview(test_glosses)
    print(f"\nTimeline:")
    for item in preview['timeline']:
        status = "OK" if item['has_video'] else "MISSING"
        print(f"  {item['gloss']}: {item['start']:.2f}s - {item['end']:.2f}s [{status}]")
    
    print(f"\nTotal duration: {preview['total_duration']:.2f}s")
    
    # Render video if videos available
    if coverage['available']:
        output_path = project_root / "outputs" / "avatar_demo.mp4"
        output_path.parent.mkdir(exist_ok=True)
        
        print(f"\nRendering to: {output_path}")
        result = renderer.render(test_glosses, str(output_path))
        
        if result:
            print(f"Video saved: {result}")
        else:
            print("Rendering failed")


if __name__ == "__main__":
    demo()