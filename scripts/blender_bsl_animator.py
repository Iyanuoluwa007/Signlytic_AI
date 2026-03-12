"""
BSL Pose to Blender Animation

Converts extracted BSL pose sequences to Blender armature animations.
Works with Mixamo characters or custom rigs with proper bone naming.

SETUP:
1. Import a rigged character (Mixamo FBX or custom)
2. Ensure armature has these bones (or remap in BONE_MAPPING)
3. Run this script in Blender's Scripting workspace

USAGE IN BLENDER:
    1. Open Blender
    2. Import your rigged character (File → Import → FBX)
    3. Select the armature
    4. Go to Scripting workspace
    5. Open this script
    6. Edit POSE_FILE path
    7. Run script (Alt+P)

REQUIREMENTS:
    - Blender 3.0+
    - Rigged character with hand bones
    - Extracted pose JSON from BSL project
"""

import bpy
import json
import math
from mathutils import Vector, Euler, Quaternion, Matrix
from pathlib import Path

# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================

# Path to your pose JSON file(s)
POSE_DIR = r"D:\Signlytic_AI\code\bsl_translation_project\data\poses"

# Glosses to animate (will be concatenated)
GLOSSES = ["HELLO", "GOOD", "YOU"]

# Or single pose file
POSE_FILE = None  # Set to specific file path to override glosses

# Frame rate
FPS = 25

# Scale factor (MediaPipe coords are 0-1, adjust for your character)
SCALE = 1.0

# ============================================================
# BONE MAPPING - MediaPipe to Blender/Mixamo bone names
# ============================================================

# Mixamo bone naming convention
BONE_MAPPING_MIXAMO = {
    # Body
    'hips': 'mixamorig:Hips',
    'spine': 'mixamorig:Spine',
    'spine1': 'mixamorig:Spine1',
    'spine2': 'mixamorig:Spine2',
    'neck': 'mixamorig:Neck',
    'head': 'mixamorig:Head',
    
    # Left Arm
    'left_shoulder': 'mixamorig:LeftShoulder',
    'left_arm': 'mixamorig:LeftArm',
    'left_forearm': 'mixamorig:LeftForeArm',
    'left_hand': 'mixamorig:LeftHand',
    
    # Right Arm
    'right_shoulder': 'mixamorig:RightShoulder',
    'right_arm': 'mixamorig:RightArm',
    'right_forearm': 'mixamorig:RightForeArm',
    'right_hand': 'mixamorig:RightHand',
    
    # Left Hand Fingers
    'left_thumb_1': 'mixamorig:LeftHandThumb1',
    'left_thumb_2': 'mixamorig:LeftHandThumb2',
    'left_thumb_3': 'mixamorig:LeftHandThumb3',
    'left_index_1': 'mixamorig:LeftHandIndex1',
    'left_index_2': 'mixamorig:LeftHandIndex2',
    'left_index_3': 'mixamorig:LeftHandIndex3',
    'left_middle_1': 'mixamorig:LeftHandMiddle1',
    'left_middle_2': 'mixamorig:LeftHandMiddle2',
    'left_middle_3': 'mixamorig:LeftHandMiddle3',
    'left_ring_1': 'mixamorig:LeftHandRing1',
    'left_ring_2': 'mixamorig:LeftHandRing2',
    'left_ring_3': 'mixamorig:LeftHandRing3',
    'left_pinky_1': 'mixamorig:LeftHandPinky1',
    'left_pinky_2': 'mixamorig:LeftHandPinky2',
    'left_pinky_3': 'mixamorig:LeftHandPinky3',
    
    # Right Hand Fingers
    'right_thumb_1': 'mixamorig:RightHandThumb1',
    'right_thumb_2': 'mixamorig:RightHandThumb2',
    'right_thumb_3': 'mixamorig:RightHandThumb3',
    'right_index_1': 'mixamorig:RightHandIndex1',
    'right_index_2': 'mixamorig:RightHandIndex2',
    'right_index_3': 'mixamorig:RightHandIndex3',
    'right_middle_1': 'mixamorig:RightHandMiddle1',
    'right_middle_2': 'mixamorig:RightHandMiddle2',
    'right_middle_3': 'mixamorig:RightHandMiddle3',
    'right_ring_1': 'mixamorig:RightHandRing1',
    'right_ring_2': 'mixamorig:RightHandRing2',
    'right_ring_3': 'mixamorig:RightHandRing3',
    'right_pinky_1': 'mixamorig:RightHandPinky1',
    'right_pinky_2': 'mixamorig:RightHandPinky2',
    'right_pinky_3': 'mixamorig:RightHandPinky3',
}

# MediaPipe landmark indices
MEDIAPIPE_BODY = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}

# MediaPipe hand landmark indices (0-20 for each hand)
MEDIAPIPE_HAND = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_mcp': 5,
    'index_pip': 6,
    'index_dip': 7,
    'index_tip': 8,
    'middle_mcp': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    'middle_tip': 12,
    'ring_mcp': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    'ring_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20,
}


# ============================================================
# POSE LOADING
# ============================================================

def find_pose_files(pose_dir: str, gloss: str):
    """Find all pose files for a gloss."""
    pose_dir = Path(pose_dir)
    files = []
    
    for split in ['train', 'val', 'test']:
        split_dir = pose_dir / split
        if split_dir.exists():
            for f in split_dir.glob(f"{gloss}_*.json"):
                files.append(f)
            # Also try lowercase
            for f in split_dir.glob(f"{gloss.lower()}_*.json"):
                files.append(f)
    
    return files


def load_pose_sequence(pose_file: str):
    """Load pose sequence from JSON file."""
    with open(pose_file, 'r') as f:
        data = json.load(f)
    return data.get('poses', [])


def load_glosses_poses(pose_dir: str, glosses: list):
    """Load and concatenate poses for multiple glosses."""
    all_poses = []
    
    for gloss in glosses:
        files = find_pose_files(pose_dir, gloss)
        if files:
            # Use first file found
            poses = load_pose_sequence(str(files[0]))
            all_poses.extend(poses)
            print(f"Loaded {gloss}: {len(poses)} frames from {files[0].name}")
        else:
            print(f"Warning: No poses found for {gloss}")
    
    return all_poses


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def mediapipe_to_blender(x, y, z, scale=1.0):
    """
    Convert MediaPipe coordinates to Blender coordinates.
    MediaPipe: X right, Y down, Z towards camera (0-1 normalized)
    Blender: X right, Y forward, Z up
    """
    bx = (x - 0.5) * scale
    by = -z * scale  # Depth becomes Y
    bz = (0.5 - y) * scale  # Flip Y to Z
    return Vector((bx, by, bz))


def calculate_bone_rotation(start_pos, end_pos, rest_direction=Vector((0, 1, 0))):
    """
    Calculate rotation to align bone from start to end position.
    """
    direction = (end_pos - start_pos).normalized()
    
    # Calculate rotation from rest pose to target
    rotation = rest_direction.rotation_difference(direction)
    
    return rotation


# ============================================================
# ARMATURE ANIMATION
# ============================================================

class BSLAnimator:
    """Animates a Blender armature from BSL pose data."""
    
    def __init__(self, armature_name=None, bone_mapping=None):
        """
        Initialize animator.
        
        Args:
            armature_name: Name of armature object (auto-detect if None)
            bone_mapping: Dict mapping generic names to actual bone names
        """
        self.armature = None
        self.bone_mapping = bone_mapping or BONE_MAPPING_MIXAMO
        
        # Find armature
        if armature_name:
            self.armature = bpy.data.objects.get(armature_name)
        else:
            # Auto-detect first armature
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE':
                    self.armature = obj
                    break
        
        if not self.armature:
            raise ValueError("No armature found! Import a rigged character first.")
        
        print(f"Using armature: {self.armature.name}")
        
        # List available bones
        print(f"Available bones ({len(self.armature.data.bones)}):")
        for bone in self.armature.data.bones:
            print(f"  - {bone.name}")
    
    def get_bone(self, generic_name):
        """Get pose bone by generic name."""
        bone_name = self.bone_mapping.get(generic_name)
        if bone_name and bone_name in self.armature.pose.bones:
            return self.armature.pose.bones[bone_name]
        return None
    
    def set_bone_rotation(self, bone, rotation, frame):
        """Set bone rotation and insert keyframe."""
        if bone is None:
            return
        
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = rotation
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    
    def animate_frame(self, pose_data, frame_num):
        """Animate a single frame from pose data."""
        body = pose_data.get('pose', [])
        left_hand = pose_data.get('left_hand', [])
        right_hand = pose_data.get('right_hand', [])
        
        # Get body landmarks
        def get_landmark(landmarks, idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                if isinstance(lm, list):
                    return mediapipe_to_blender(lm[0], lm[1], lm[2] if len(lm) > 2 else 0, SCALE)
            return None
        
        # ============ ARM ANIMATION ============
        
        # Left arm
        left_shoulder_pos = get_landmark(body, MEDIAPIPE_BODY['left_shoulder'])
        left_elbow_pos = get_landmark(body, MEDIAPIPE_BODY['left_elbow'])
        left_wrist_pos = get_landmark(body, MEDIAPIPE_BODY['left_wrist'])
        
        if left_shoulder_pos and left_elbow_pos:
            rot = calculate_bone_rotation(left_shoulder_pos, left_elbow_pos, Vector((0, 0, -1)))
            self.set_bone_rotation(self.get_bone('left_arm'), rot, frame_num)
        
        if left_elbow_pos and left_wrist_pos:
            rot = calculate_bone_rotation(left_elbow_pos, left_wrist_pos, Vector((0, 0, -1)))
            self.set_bone_rotation(self.get_bone('left_forearm'), rot, frame_num)
        
        # Right arm
        right_shoulder_pos = get_landmark(body, MEDIAPIPE_BODY['right_shoulder'])
        right_elbow_pos = get_landmark(body, MEDIAPIPE_BODY['right_elbow'])
        right_wrist_pos = get_landmark(body, MEDIAPIPE_BODY['right_wrist'])
        
        if right_shoulder_pos and right_elbow_pos:
            rot = calculate_bone_rotation(right_shoulder_pos, right_elbow_pos, Vector((0, 0, -1)))
            self.set_bone_rotation(self.get_bone('right_arm'), rot, frame_num)
        
        if right_elbow_pos and right_wrist_pos:
            rot = calculate_bone_rotation(right_elbow_pos, right_wrist_pos, Vector((0, 0, -1)))
            self.set_bone_rotation(self.get_bone('right_forearm'), rot, frame_num)
        
        # ============ HAND ANIMATION ============
        
        self.animate_hand(left_hand, 'left', frame_num)
        self.animate_hand(right_hand, 'right', frame_num)
        
        # ============ HEAD ANIMATION ============
        
        nose_pos = get_landmark(body, MEDIAPIPE_BODY['nose'])
        if nose_pos and left_shoulder_pos and right_shoulder_pos:
            mid_shoulder = (left_shoulder_pos + right_shoulder_pos) / 2
            head_dir = (nose_pos - mid_shoulder).normalized()
            
            # Simple head rotation based on nose position
            head_bone = self.get_bone('head')
            if head_bone:
                yaw = math.atan2(head_dir.x, head_dir.y) * 0.5
                pitch = math.asin(max(-1, min(1, head_dir.z))) * 0.3
                
                euler = Euler((pitch, 0, yaw), 'XYZ')
                self.set_bone_rotation(head_bone, euler.to_quaternion(), frame_num)
    
    def animate_hand(self, hand_landmarks, side, frame_num):
        """Animate hand fingers."""
        if len(hand_landmarks) < 21:
            return
        
        def get_hand_landmark(idx):
            if idx < len(hand_landmarks):
                lm = hand_landmarks[idx]
                if isinstance(lm, list):
                    return mediapipe_to_blender(lm[0], lm[1], lm[2] if len(lm) > 2 else 0, SCALE * 0.5)
            return None
        
        # Finger mapping: (mcp, pip, dip, tip) indices
        fingers = {
            'thumb': (1, 2, 3, 4),
            'index': (5, 6, 7, 8),
            'middle': (9, 10, 11, 12),
            'ring': (13, 14, 15, 16),
            'pinky': (17, 18, 19, 20),
        }
        
        wrist_pos = get_hand_landmark(0)
        
        for finger_name, indices in fingers.items():
            positions = [get_hand_landmark(i) for i in indices]
            
            # Skip if missing landmarks
            if None in positions or wrist_pos is None:
                continue
            
            # Animate each joint
            for joint_idx in range(3):
                bone_name = f'{side}_{finger_name}_{joint_idx + 1}'
                bone = self.get_bone(bone_name)
                
                if bone:
                    start = positions[joint_idx]
                    end = positions[joint_idx + 1]
                    rot = calculate_bone_rotation(start, end)
                    self.set_bone_rotation(bone, rot, frame_num)
    
    def animate_sequence(self, poses, start_frame=1):
        """Animate full pose sequence."""
        print(f"Animating {len(poses)} frames...")
        
        # Set scene frame range
        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = start_frame + len(poses) - 1
        bpy.context.scene.render.fps = FPS
        
        for i, pose in enumerate(poses):
            frame = start_frame + i
            self.animate_frame(pose, frame)
            
            if (i + 1) % 10 == 0:
                print(f"  Frame {i + 1}/{len(poses)}")
        
        print(f"Animation complete! Frames {start_frame} to {start_frame + len(poses) - 1}")
        
        # Set current frame to start
        bpy.context.scene.frame_set(start_frame)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main entry point - run this in Blender."""
    
    print("\n" + "="*60)
    print("BSL POSE TO BLENDER ANIMATION")
    print("="*60)
    
    # Load poses
    if POSE_FILE:
        poses = load_pose_sequence(POSE_FILE)
        print(f"Loaded {len(poses)} frames from {POSE_FILE}")
    else:
        poses = load_glosses_poses(POSE_DIR, GLOSSES)
    
    if not poses:
        print("ERROR: No poses loaded!")
        return
    
    print(f"Total frames: {len(poses)}")
    
    # Create animator
    try:
        animator = BSLAnimator()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nPlease import a rigged character first:")
        print("  1. File → Import → FBX")
        print("  2. Select a Mixamo character")
        print("  3. Run this script again")
        return
    
    # Animate
    animator.animate_sequence(poses)
    
    print("\n" + "="*60)
    print("DONE! Press Space to play animation")
    print("="*60)


# Run when script is executed
if __name__ == "__main__":
    main()
