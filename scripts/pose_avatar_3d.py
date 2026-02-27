"""
3D BSL Signing Avatar - Pose-based Animation

Renders a 3D avatar from pose sequences extracted from BOBSL data.
Uses PyGame + OpenGL for real-time 3D rendering.

Pipeline: Gloss → Pose Lookup → 3D Animation → Display

Usage:
    python scripts/pose_avatar_3d.py --gloss HELLO
    python scripts/pose_avatar_3d.py --glosses "HELLO GOOD YOU"
    python scripts/pose_avatar_3d.py --interactive
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Check for pygame and OpenGL
try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Install pygame and PyOpenGL: pip install pygame PyOpenGL PyOpenGL_accelerate")


class PoseLookup:
    """Look up pose sequences for glosses from extracted data."""
    
    def __init__(self, poses_dir: str):
        self.poses_dir = Path(poses_dir)
        self.gloss_to_files: Dict[str, List[Path]] = {}
        self._build_index()
    
    def _build_index(self):
        """Build index of gloss -> pose files."""
        print("Building pose index...")
        
        for split in ['train', 'val', 'test']:
            split_dir = self.poses_dir / split
            if not split_dir.exists():
                continue
            
            for json_file in split_dir.glob("*.json"):
                try:
                    # Extract gloss from filename (format: GLOSS_video_time.json)
                    gloss = json_file.stem.split('_')[0].upper()
                    
                    if gloss not in self.gloss_to_files:
                        self.gloss_to_files[gloss] = []
                    self.gloss_to_files[gloss].append(json_file)
                except:
                    continue
        
        print(f"Indexed {len(self.gloss_to_files)} glosses")
    
    def get_pose_sequence(self, gloss: str) -> Optional[np.ndarray]:
        """Get pose sequence for a gloss."""
        gloss = gloss.upper().strip()
        
        if gloss not in self.gloss_to_files:
            print(f"Gloss not found in index: {gloss}")
            return None
        
        # Pick a random example for variety
        pose_file = random.choice(self.gloss_to_files[gloss])
        
        try:
            with open(pose_file, 'r') as f:
                data = json.load(f)
            
            poses = data.get('poses', [])
            if not poses:
                print(f"No poses in file: {pose_file}")
                return None
            
            # Convert to numpy array
            # Each pose has pose (body), left_hand, right_hand
            frames = []
            for pose in poses:
                frame = []
                
                # Body pose (33 keypoints x 3) - stored as 'pose' not 'body_pose'
                body = pose.get('pose', [])
                for kp in body:
                    # Handle both dict and list formats
                    if isinstance(kp, dict):
                        frame.extend([kp.get('x', 0), kp.get('y', 0), kp.get('z', 0)])
                    elif isinstance(kp, (list, tuple)):
                        frame.extend([kp[0] if len(kp) > 0 else 0, 
                                     kp[1] if len(kp) > 1 else 0, 
                                     kp[2] if len(kp) > 2 else 0])
                    else:
                        frame.extend([0, 0, 0])
                
                # Pad if needed
                while len(frame) < 33 * 3:
                    frame.extend([0, 0, 0])
                
                # Left hand (21 keypoints x 3)
                left = pose.get('left_hand', [])
                for kp in left:
                    if isinstance(kp, dict):
                        frame.extend([kp.get('x', 0), kp.get('y', 0), kp.get('z', 0)])
                    elif isinstance(kp, (list, tuple)):
                        frame.extend([kp[0] if len(kp) > 0 else 0, 
                                     kp[1] if len(kp) > 1 else 0, 
                                     kp[2] if len(kp) > 2 else 0])
                    else:
                        frame.extend([0, 0, 0])
                while len(frame) < (33 + 21) * 3:
                    frame.extend([0, 0, 0])
                
                # Right hand (21 keypoints x 3)
                right = pose.get('right_hand', [])
                for kp in right:
                    if isinstance(kp, dict):
                        frame.extend([kp.get('x', 0), kp.get('y', 0), kp.get('z', 0)])
                    elif isinstance(kp, (list, tuple)):
                        frame.extend([kp[0] if len(kp) > 0 else 0, 
                                     kp[1] if len(kp) > 1 else 0, 
                                     kp[2] if len(kp) > 2 else 0])
                    else:
                        frame.extend([0, 0, 0])
                while len(frame) < (33 + 21 + 21) * 3:
                    frame.extend([0, 0, 0])
                
                frames.append(frame)
            
            return np.array(frames, dtype=np.float32)
        
        except Exception as e:
            print(f"Error loading pose from {pose_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_available_glosses(self) -> List[str]:
        """Get list of available glosses."""
        return sorted(self.gloss_to_files.keys())


class Avatar3D:
    """3D avatar renderer using OpenGL."""
    
    # MediaPipe pose connections (body)
    BODY_CONNECTIONS = [
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face left
        (0, 4), (4, 5), (5, 6), (6, 8),  # Face right
    ]
    
    # Hand connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17),           # Palm
    ]
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.current_frame = 0
        self.pose_sequence = None
        self.playing = False
        self.fps = 25
        
        if not OPENGL_AVAILABLE:
            raise RuntimeError("OpenGL not available")
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("BSL 3D Avatar")
        
        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, width/height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 5, 10, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
    
    def set_pose_sequence(self, poses: np.ndarray):
        """Set the pose sequence to animate."""
        self.pose_sequence = poses
        self.current_frame = 0
        self.playing = True
    
    def _draw_sphere(self, x: float, y: float, z: float, radius: float, color: Tuple[float, float, float]):
        """Draw a sphere at position."""
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        quad = gluNewQuadric()
        gluSphere(quad, radius, 16, 16)
        gluDeleteQuadric(quad)
        glPopMatrix()
    
    def _draw_cylinder(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], 
                       radius: float, color: Tuple[float, float, float]):
        """Draw a cylinder between two points."""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        # Direction vector
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if length < 0.001:
            return
        
        glPushMatrix()
        glTranslatef(x1, y1, z1)
        
        # Calculate rotation to align cylinder with direction
        if abs(dz) < 0.99 * length:
            ax = -dy
            ay = dx
            az = 0
            angle = np.degrees(np.arccos(dz / length))
            glRotatef(angle, ax, ay, az)
        elif dz < 0:
            glRotatef(180, 1, 0, 0)
        
        glColor3f(*color)
        quad = gluNewQuadric()
        gluCylinder(quad, radius, radius, length, 16, 1)
        gluDeleteQuadric(quad)
        glPopMatrix()
    
    def _extract_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract body, left hand, right hand keypoints from frame."""
        # Reshape: (225,) -> body(33,3), left(21,3), right(21,3)
        frame = frame.reshape(-1)
        
        body = frame[:33*3].reshape(33, 3)
        left_hand = frame[33*3:(33+21)*3].reshape(21, 3)
        right_hand = frame[(33+21)*3:(33+21+21)*3].reshape(21, 3)
        
        # Transform coordinates for OpenGL (flip Y, scale)
        scale = 5.0
        
        body_gl = body.copy()
        body_gl[:, 0] = (body[:, 0] - 0.5) * scale  # X centered
        body_gl[:, 1] = (0.5 - body[:, 1]) * scale  # Y flipped
        body_gl[:, 2] = body[:, 2] * scale - 5      # Z depth
        
        left_gl = left_hand.copy()
        left_gl[:, 0] = (left_hand[:, 0] - 0.5) * scale
        left_gl[:, 1] = (0.5 - left_hand[:, 1]) * scale
        left_gl[:, 2] = left_hand[:, 2] * scale - 5
        
        right_gl = right_hand.copy()
        right_gl[:, 0] = (right_hand[:, 0] - 0.5) * scale
        right_gl[:, 1] = (0.5 - right_hand[:, 1]) * scale
        right_gl[:, 2] = right_hand[:, 2] * scale - 5
        
        return body_gl, left_gl, right_gl
    
    def _draw_body(self, body: np.ndarray):
        """Draw body skeleton."""
        # Draw joints
        for i, (x, y, z) in enumerate(body):
            if abs(x) < 10 and abs(y) < 10:  # Valid keypoint
                self._draw_sphere(x, y, z, 0.08, (0.3, 0.6, 1.0))  # Blue
        
        # Draw bones
        for i, j in self.BODY_CONNECTIONS:
            if i < len(body) and j < len(body):
                p1, p2 = body[i], body[j]
                if abs(p1[0]) < 10 and abs(p2[0]) < 10:
                    self._draw_cylinder(tuple(p1), tuple(p2), 0.04, (0.2, 0.5, 0.9))
    
    def _draw_hand(self, hand: np.ndarray, color: Tuple[float, float, float]):
        """Draw hand skeleton."""
        # Check if hand detected
        if np.sum(np.abs(hand)) < 0.1:
            return
        
        # Draw joints
        for x, y, z in hand:
            if abs(x) < 10 and abs(y) < 10:
                self._draw_sphere(x, y, z, 0.03, color)
        
        # Draw connections
        darker = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        for i, j in self.HAND_CONNECTIONS:
            if i < len(hand) and j < len(hand):
                p1, p2 = hand[i], hand[j]
                if abs(p1[0]) < 10 and abs(p2[0]) < 10:
                    self._draw_cylinder(tuple(p1), tuple(p2), 0.015, darker)
    
    def render_frame(self):
        """Render current frame."""
        if self.pose_sequence is None:
            return
        
        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera position
        gluLookAt(0, 0, 5,   # Eye
                  0, 0, 0,   # Center
                  0, 1, 0)   # Up
        
        # Get current pose
        frame = self.pose_sequence[self.current_frame]
        body, left_hand, right_hand = self._extract_keypoints(frame)
        
        # Draw avatar
        self._draw_body(body)
        self._draw_hand(left_hand, (0.2, 0.8, 0.2))   # Green for left
        self._draw_hand(right_hand, (0.8, 0.2, 0.2))  # Red for right
        
        # Draw ground plane
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(-3, -3, -6)
        glVertex3f(3, -3, -6)
        glVertex3f(3, -3, -2)
        glVertex3f(-3, -3, -2)
        glEnd()
        
        pygame.display.flip()
        
        # Advance frame
        if self.playing:
            self.current_frame = (self.current_frame + 1) % len(self.pose_sequence)
    
    def run(self, pose_sequence: np.ndarray, loop: bool = True):
        """Run animation loop."""
        self.set_pose_sequence(pose_sequence)
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_r:
                        self.current_frame = 0
                    elif event.key == pygame.K_LEFT:
                        self.current_frame = max(0, self.current_frame - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.current_frame = min(len(self.pose_sequence) - 1, self.current_frame + 1)
            
            self.render_frame()
            clock.tick(self.fps)
            
            # Stop if not looping and reached end
            if not loop and self.current_frame == 0 and not self.playing:
                break
        
        pygame.quit()


class Avatar2D:
    """Simpler 2D avatar using pygame (no OpenGL required)."""
    
    BODY_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28),
    ]
    
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    
    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("BSL 2D Avatar")
        self.font = pygame.font.Font(None, 36)
        self.current_frame = 0
        self.pose_sequence = None
        self.playing = False
        self.fps = 25
        self.current_gloss = ""
    
    def set_pose_sequence(self, poses: np.ndarray, gloss: str = ""):
        """Set the pose sequence to animate."""
        self.pose_sequence = poses
        self.current_frame = 0
        self.playing = True
        self.current_gloss = gloss
    
    def _to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized coords to screen coords."""
        sx = int(x * self.width)
        sy = int(y * self.height)
        return sx, sy
    
    def _draw_skeleton(self, keypoints: np.ndarray, connections: List[Tuple[int, int]], 
                       color: Tuple[int, int, int], radius: int = 5, thickness: int = 2):
        """Draw skeleton on screen."""
        # Draw connections
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints):
                x1, y1 = keypoints[i, 0], keypoints[i, 1]
                x2, y2 = keypoints[j, 0], keypoints[j, 1]
                
                if 0 < x1 < 1 and 0 < y1 < 1 and 0 < x2 < 1 and 0 < y2 < 1:
                    p1 = self._to_screen(x1, y1)
                    p2 = self._to_screen(x2, y2)
                    pygame.draw.line(self.screen, color, p1, p2, thickness)
        
        # Draw joints
        for x, y, z in keypoints:
            if 0 < x < 1 and 0 < y < 1:
                pos = self._to_screen(x, y)
                pygame.draw.circle(self.screen, color, pos, radius)
    
    def render_frame(self):
        """Render current frame."""
        if self.pose_sequence is None:
            return
        
        # Clear screen
        self.screen.fill((30, 30, 40))
        
        # Get current pose
        frame = self.pose_sequence[self.current_frame]
        frame = frame.reshape(-1)
        
        body = frame[:33*3].reshape(33, 3)
        left_hand = frame[33*3:(33+21)*3].reshape(21, 3)
        right_hand = frame[(33+21)*3:(33+21+21)*3].reshape(21, 3)
        
        # Draw body (blue)
        self._draw_skeleton(body, self.BODY_CONNECTIONS, (100, 150, 255), radius=8, thickness=4)
        
        # Draw hands
        if np.sum(np.abs(left_hand)) > 0.1:
            self._draw_skeleton(left_hand, self.HAND_CONNECTIONS, (100, 255, 100), radius=4, thickness=2)
        if np.sum(np.abs(right_hand)) > 0.1:
            self._draw_skeleton(right_hand, self.HAND_CONNECTIONS, (255, 100, 100), radius=4, thickness=2)
        
        # Draw UI
        text = self.font.render(f"Gloss: {self.current_gloss}", True, (255, 255, 255))
        self.screen.blit(text, (20, 20))
        
        frame_text = self.font.render(f"Frame: {self.current_frame + 1}/{len(self.pose_sequence)}", True, (200, 200, 200))
        self.screen.blit(frame_text, (20, 60))
        
        status = "Playing" if self.playing else "Paused"
        status_text = self.font.render(f"[SPACE] {status} | [Q] Quit", True, (150, 150, 150))
        self.screen.blit(status_text, (20, self.height - 40))
        
        pygame.display.flip()
        
        # Advance frame
        if self.playing:
            self.current_frame = (self.current_frame + 1) % len(self.pose_sequence)
    
    def run(self, pose_sequence: np.ndarray, gloss: str = "", loop: bool = True):
        """Run animation loop."""
        self.set_pose_sequence(pose_sequence, gloss)
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_r:
                        self.current_frame = 0
            
            self.render_frame()
            clock.tick(self.fps)
            
            if not loop and self.current_frame == 0 and not self.playing:
                break
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="BSL 3D Avatar Animation")
    parser.add_argument("--poses-dir", type=str, default=None, help="Pose data directory")
    parser.add_argument("--gloss", type=str, default=None, help="Single gloss to animate")
    parser.add_argument("--glosses", type=str, default=None, help="Space-separated glosses")
    parser.add_argument("--mode", type=str, choices=["2d", "3d"], default="2d", help="Rendering mode")
    parser.add_argument("--list-glosses", action="store_true", help="List available glosses")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    poses_dir = Path(args.poses_dir) if args.poses_dir else project_root / "data" / "poses"
    
    # Initialize pose lookup
    pose_lookup = PoseLookup(str(poses_dir))
    
    if args.list_glosses:
        glosses = pose_lookup.get_available_glosses()
        print(f"\nAvailable glosses ({len(glosses)}):")
        for i, g in enumerate(glosses[:50]):
            print(f"  {g}", end="\t")
            if (i + 1) % 5 == 0:
                print()
        print(f"\n  ... and {len(glosses) - 50} more")
        return
    
    if args.interactive:
        print("\n" + "="*50)
        print("BSL 3D Avatar - Interactive Mode")
        print("="*50)
        print("Enter gloss(es) to animate, or 'quit' to exit")
        print("Example: HELLO GOOD YOU")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("Gloss(es): ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                glosses = user_input.upper().split()
                if not glosses:
                    continue
                
                # Concatenate pose sequences
                all_poses = []
                for gloss in glosses:
                    poses = pose_lookup.get_pose_sequence(gloss)
                    if poses is not None:
                        all_poses.append(poses)
                        print(f"  {gloss}: {len(poses)} frames")
                
                if all_poses:
                    combined = np.concatenate(all_poses, axis=0)
                    print(f"Total: {len(combined)} frames")
                    
                    if args.mode == "3d" and OPENGL_AVAILABLE:
                        avatar = Avatar3D()
                    else:
                        avatar = Avatar2D()
                    
                    avatar.run(combined, gloss=" ".join(glosses))
            
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
        return
    
    # Single/multiple gloss animation
    glosses = []
    if args.gloss:
        glosses = [args.gloss.upper()]
    elif args.glosses:
        glosses = args.glosses.upper().split()
    else:
        # Demo with common glosses
        glosses = ["HELLO", "GOOD", "YOU"]
    
    print(f"Animating: {' → '.join(glosses)}")
    
    # Get pose sequences
    all_poses = []
    for gloss in glosses:
        poses = pose_lookup.get_pose_sequence(gloss)
        if poses is not None:
            all_poses.append(poses)
            print(f"  {gloss}: {len(poses)} frames")
        else:
            print(f"  {gloss}: NOT FOUND")
    
    if not all_poses:
        print("No poses found!")
        return
    
    # Concatenate
    combined = np.concatenate(all_poses, axis=0)
    print(f"Total: {len(combined)} frames")
    
    # Run avatar
    if args.mode == "3d" and OPENGL_AVAILABLE:
        avatar = Avatar3D()
    else:
        avatar = Avatar2D()
    
    avatar.run(combined, gloss=" → ".join(glosses))


if __name__ == "__main__":
    main()