"""
Integrated Speech → BSL 3D Avatar Pipeline

Complete Direction 2 implementation:
    Speech → Text (Whisper) → Gloss (TextToGloss) → Pose → 3D Avatar

Usage:
    python scripts/speech_to_avatar.py --audio input.wav
    python scripts/speech_to_avatar.py --text "Hello, how are you?"
    python scripts/speech_to_avatar.py --interactive
"""

import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Optional

# Add project paths
project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "inference"))

import numpy as np


class SpeechToBSLAvatar:
    """
    Complete Speech → 3D Avatar Pipeline
    
    Pipeline:
        1. Speech → Text (Whisper)
        2. Text → Gloss (TextToGloss with Groq LLM)
        3. Gloss → Pose (Lookup from BOBSL extracted poses)
        4. Pose → 3D Avatar (Three.js or Pygame)
    """
    
    def __init__(
        self,
        poses_dir: Optional[str] = None,
        use_whisper: bool = True,
        whisper_model: str = "base"
    ):
        self.project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
        self.poses_dir = Path(poses_dir) if poses_dir else self.project_root / "data" / "poses"
        
        # Initialize components
        self.whisper_model = None
        self.text_to_gloss = None
        self.pose_lookup = None
        
        print("Initializing Speech → BSL Avatar Pipeline...")
        
        # 1. Speech-to-Text (Whisper)
        if use_whisper:
            self._init_whisper(whisper_model)
        
        # 2. Text-to-Gloss
        self._init_text_to_gloss()
        
        # 3. Pose Lookup
        self._init_pose_lookup()
        
        print("Pipeline ready!")
    
    def _init_whisper(self, model_size: str):
        """Initialize Whisper for speech recognition."""
        try:
            import whisper
            print(f"  Loading Whisper ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            print("  ✓ Whisper ready")
        except Exception as e:
            print(f"  ✗ Whisper failed: {e}")
            self.whisper_model = None
    
    def _init_text_to_gloss(self):
        """Initialize Text-to-Gloss converter."""
        try:
            from text_to_gloss import TextToGloss
            print("  Loading TextToGloss...")
            self.text_to_gloss = TextToGloss(
                vocab_path=str(self.project_root / "data" / "processed" / "vocabulary.json"),
                strict_vocab=False
            )
            print("  ✓ TextToGloss ready")
        except Exception as e:
            print(f"  ✗ TextToGloss failed: {e}")
            # Create simple fallback
            self.text_to_gloss = None
    
    def _init_pose_lookup(self):
        """Initialize pose sequence lookup."""
        print("  Building pose index...")
        
        self.gloss_to_files = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.poses_dir / split
            if not split_dir.exists():
                continue
            
            for json_file in split_dir.glob("*.json"):
                try:
                    gloss = json_file.stem.split('_')[0].upper()
                    if gloss not in self.gloss_to_files:
                        self.gloss_to_files[gloss] = []
                    self.gloss_to_files[gloss].append(json_file)
                except:
                    continue
        
        print(f"  ✓ Indexed {len(self.gloss_to_files)} glosses")
        self.pose_lookup = True
    
    def speech_to_text(self, audio_path: str) -> str:
        """Convert speech to text using Whisper."""
        if self.whisper_model is None:
            raise RuntimeError("Whisper not available")
        
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()
    
    def text_to_glosses(self, text: str) -> List[str]:
        """Convert text to BSL glosses."""
        if self.text_to_gloss:
            glosses = self.text_to_gloss.convert(text)
            return glosses if isinstance(glosses, list) else glosses.split()
        else:
            # Simple fallback: uppercase words
            import re
            words = re.findall(r'\b\w+\b', text.upper())
            return words
    
    def get_pose_sequence(self, gloss: str) -> Optional[List]:
        """Get pose sequence for a single gloss."""
        gloss = gloss.upper().strip()
        
        if gloss not in self.gloss_to_files:
            return None
        
        import random
        pose_file = random.choice(self.gloss_to_files[gloss])
        
        try:
            with open(pose_file, 'r') as f:
                data = json.load(f)
            return data.get('poses', [])
        except:
            return None
    
    def glosses_to_poses(self, glosses: List[str]) -> List:
        """Convert glosses to concatenated pose sequence."""
        all_poses = []
        found_glosses = []
        missing_glosses = []
        
        for gloss in glosses:
            poses = self.get_pose_sequence(gloss)
            if poses:
                all_poses.extend(poses)
                found_glosses.append(gloss)
            else:
                missing_glosses.append(gloss)
        
        return {
            'poses': all_poses,
            'found': found_glosses,
            'missing': missing_glosses
        }
    
    def process_speech(self, audio_path: str) -> dict:
        """Full pipeline: Speech → Text → Gloss → Poses."""
        print(f"\n{'='*50}")
        print("SPEECH → BSL AVATAR PIPELINE")
        print(f"{'='*50}")
        
        # Step 1: Speech → Text
        print("\n[1] Speech → Text (Whisper)")
        text = self.speech_to_text(audio_path)
        print(f"    Text: \"{text}\"")
        
        # Step 2: Text → Gloss
        print("\n[2] Text → Gloss")
        glosses = self.text_to_glosses(text)
        print(f"    Glosses: {' → '.join(glosses)}")
        
        # Step 3: Gloss → Poses
        print("\n[3] Gloss → Poses")
        result = self.glosses_to_poses(glosses)
        print(f"    Found: {result['found']}")
        print(f"    Missing: {result['missing']}")
        print(f"    Total frames: {len(result['poses'])}")
        
        return {
            'text': text,
            'glosses': glosses,
            'poses': result['poses'],
            'found_glosses': result['found'],
            'missing_glosses': result['missing']
        }
    
    def process_text(self, text: str) -> dict:
        """Pipeline from text: Text → Gloss → Poses."""
        print(f"\n{'='*50}")
        print("TEXT → BSL AVATAR PIPELINE")
        print(f"{'='*50}")
        
        print(f"\n[1] Input Text: \"{text}\"")
        
        # Step 2: Text → Gloss
        print("\n[2] Text → Gloss")
        glosses = self.text_to_glosses(text)
        print(f"    Glosses: {' → '.join(glosses)}")
        
        # Step 3: Gloss → Poses
        print("\n[3] Gloss → Poses")
        result = self.glosses_to_poses(glosses)
        print(f"    Found: {result['found']}")
        print(f"    Missing: {result['missing']}")
        print(f"    Total frames: {len(result['poses'])}")
        
        return {
            'text': text,
            'glosses': glosses,
            'poses': result['poses'],
            'found_glosses': result['found'],
            'missing_glosses': result['missing']
        }
    
    def generate_avatar_html(self, poses: List, title: str = "BSL Avatar") -> str:
        """Generate Three.js HTML for the poses."""
        from avatar_threejs import generate_avatar_html
        return generate_avatar_html(poses, title)
    
    def render_2d(self, poses: List, title: str = ""):
        """Render poses with 2D Pygame avatar."""
        try:
            from pose_avatar_3d import Avatar2D
            
            # Helper to extract x,y,z from keypoint
            def get_xyz(kp):
                if isinstance(kp, dict):
                    return [kp.get('x', 0), kp.get('y', 0), kp.get('z', 0)]
                elif isinstance(kp, (list, tuple)):
                    return [kp[0] if len(kp) > 0 else 0,
                            kp[1] if len(kp) > 1 else 0,
                            kp[2] if len(kp) > 2 else 0]
                return [0, 0, 0]
            
            # Convert poses to numpy array
            frames = []
            for pose in poses:
                frame = []
                
                # Body stored as 'pose' key, not 'body_pose'
                for kp in pose.get('pose', []):
                    frame.extend(get_xyz(kp))
                while len(frame) < 33 * 3:
                    frame.extend([0, 0, 0])
                
                for kp in pose.get('left_hand', []):
                    frame.extend(get_xyz(kp))
                while len(frame) < (33 + 21) * 3:
                    frame.extend([0, 0, 0])
                
                for kp in pose.get('right_hand', []):
                    frame.extend(get_xyz(kp))
                while len(frame) < (33 + 21 + 21) * 3:
                    frame.extend([0, 0, 0])
                
                frames.append(frame)
            
            pose_array = np.array(frames, dtype=np.float32)
            
            avatar = Avatar2D()
            avatar.run(pose_array, title)
            
        except Exception as e:
            print(f"2D rendering failed: {e}")
            import traceback
            traceback.print_exc()
            print("Install pygame: pip install pygame")


def main():
    parser = argparse.ArgumentParser(description="Speech → BSL 3D Avatar")
    parser.add_argument("--audio", type=str, help="Audio file path")
    parser.add_argument("--text", type=str, help="Text input (skip speech recognition)")
    parser.add_argument("--glosses", type=str, help="Glosses directly (skip text conversion)")
    parser.add_argument("--output", type=str, default="avatar_output.html", help="Output HTML file")
    parser.add_argument("--render", choices=["html", "2d", "both"], default="html", help="Rendering mode")
    parser.add_argument("--serve", action="store_true", help="Open in browser")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpeechToBSLAvatar(whisper_model=args.whisper_model)
    
    if args.interactive:
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("="*50)
        print("Enter text to convert to BSL avatar, or 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("\nText: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                result = pipeline.process_text(user_input)
                
                if result['poses']:
                    if args.render in ['2d', 'both']:
                        print("\nRendering 2D avatar...")
                        pipeline.render_2d(result['poses'], " → ".join(result['found_glosses']))
                    
                    if args.render in ['html', 'both']:
                        html = pipeline.generate_avatar_html(result['poses'], user_input)
                        with open(args.output, 'w') as f:
                            f.write(html)
                        print(f"\nSaved: {args.output}")
                        
                        if args.serve:
                            import webbrowser
                            webbrowser.open(args.output)
                else:
                    print("No poses generated!")
            
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
        return
    
    # Process input
    result = None
    
    if args.glosses:
        glosses = args.glosses.upper().split()
        result = {
            'glosses': glosses,
            **pipeline.glosses_to_poses(glosses)
        }
        result['text'] = args.glosses
        result['found_glosses'] = result['found']
        result['missing_glosses'] = result['missing']
        
    elif args.text:
        result = pipeline.process_text(args.text)
        
    elif args.audio:
        result = pipeline.process_speech(args.audio)
        
    else:
        # Demo
        result = pipeline.process_text("Hello, how are you today?")
    
    if not result or not result.get('poses'):
        print("No poses generated!")
        return
    
    # Render
    if args.render in ['2d', 'both']:
        print("\nRendering 2D avatar...")
        pipeline.render_2d(result['poses'], " → ".join(result.get('found_glosses', [])))
    
    if args.render in ['html', 'both']:
        print(f"\nGenerating HTML avatar...")
        html = pipeline.generate_avatar_html(result['poses'], result.get('text', 'BSL Avatar'))
        
        with open(args.output, 'w') as f:
            f.write(html)
        print(f"Saved: {args.output}")
        
        if args.serve:
            import webbrowser
            webbrowser.open(args.output)


if __name__ == "__main__":
    main()