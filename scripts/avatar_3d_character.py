"""
3D Humanoid BSL Avatar - Full Character Animation

Uses Three.js with a rigged humanoid model (Ready Player Me style)
to create a realistic 3D signing avatar.

Pipeline: Gloss → Pose → 3D Character Animation

Usage:
    python scripts/avatar_3d_character.py --glosses "HELLO GOOD YOU" --serve
    python scripts/avatar_3d_character.py --text "Hello, how are you?" --serve
"""

import json
import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path
from typing import List, Optional
import numpy as np


class PoseLookup:
    """Look up pose sequences for glosses."""
    
    def __init__(self, poses_dir: str):
        self.poses_dir = Path(poses_dir)
        self.gloss_to_files = {}
        self._build_index()
    
    def _build_index(self):
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
        
        print(f"Indexed {len(self.gloss_to_files)} glosses")
    
    def get_pose_sequence(self, gloss: str) -> Optional[List]:
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
    
    def get_available_glosses(self) -> List[str]:
        return sorted(self.gloss_to_files.keys())


def generate_3d_avatar_html(poses_data: List, gloss_text: str = "") -> str:
    """Generate HTML with Three.js 3D humanoid avatar."""
    
    poses_json = json.dumps(poses_data)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BSL 3D Avatar - {gloss_text}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
        }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            z-index: 100;
            background: rgba(0,0,0,0.6);
            padding: 20px 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        #info h1 {{ 
            font-size: 28px;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        #info .gloss {{
            font-size: 20px;
            color: #4ecdc4;
            font-weight: 600;
            letter-spacing: 2px;
        }}
        #controls {{
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 15px;
            z-index: 100;
        }}
        .btn {{
            background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
            border: none;
            color: white;
            padding: 15px 30px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(78, 205, 196, 0.6);
        }}
        #progress-container {{
            position: absolute;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            width: 300px;
            z-index: 100;
        }}
        #progress-bar {{
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.3);
            border-radius: 3px;
            overflow: hidden;
        }}
        #progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4ecdc4, #44a08d);
            width: 0%;
            transition: width 0.1s;
        }}
        #frame-info {{
            color: white;
            text-align: center;
            margin-top: 10px;
            font-size: 14px;
            opacity: 0.8;
        }}
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 24px;
            z-index: 200;
        }}
        .loading-spinner {{
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top-color: #4ecdc4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="loading" class="loading">
        <div class="loading-spinner"></div>
        <div>Loading 3D Avatar...</div>
    </div>
    
    <div id="info" style="display:none;">
        <h1>BSL 3D Avatar</h1>
        <div class="gloss">{gloss_text}</div>
    </div>
    
    <div id="progress-container" style="display:none;">
        <div id="progress-bar">
            <div id="progress-fill"></div>
        </div>
        <div id="frame-info">Frame: <span id="frame-num">1</span> / <span id="total-frames">0</span></div>
    </div>
    
    <div id="controls" style="display:none;">
        <button class="btn" onclick="togglePlay()">Play / Pause</button>
        <button class="btn" onclick="restart()">Restart</button>
        <button class="btn" onclick="toggleSpeed()">Speed: <span id="speed">1x</span></button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Pose data
        const posesData = {poses_json};
        
        // Animation state
        let currentFrame = 0;
        let playing = true;
        let speed = 1;
        const fps = 25;
        
        // Three.js objects
        let scene, camera, renderer;
        let avatar, skeleton;
        let mixer, clock;
        
        // Body part meshes
        const bodyParts = {{}};
        
        // MediaPipe landmark indices
        const LANDMARKS = {{
            NOSE: 0,
            LEFT_SHOULDER: 11,
            RIGHT_SHOULDER: 12,
            LEFT_ELBOW: 13,
            RIGHT_ELBOW: 14,
            LEFT_WRIST: 15,
            RIGHT_WRIST: 16,
            LEFT_HIP: 23,
            RIGHT_HIP: 24,
            LEFT_KNEE: 25,
            RIGHT_KNEE: 26,
            LEFT_ANKLE: 27,
            RIGHT_ANKLE: 28
        }};
        
        init();
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 1.2, 3);
            camera.lookAt(0, 1, 0);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
            mainLight.position.set(5, 10, 7);
            mainLight.castShadow = true;
            mainLight.shadow.mapSize.width = 2048;
            mainLight.shadow.mapSize.height = 2048;
            scene.add(mainLight);
            
            const fillLight = new THREE.DirectionalLight(0x4ecdc4, 0.3);
            fillLight.position.set(-5, 5, -5);
            scene.add(fillLight);
            
            // Ground
            const groundGeometry = new THREE.CircleGeometry(3, 64);
            const groundMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x2a2a4a,
                roughness: 0.8
            }});
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.receiveShadow = true;
            scene.add(ground);
            
            // Create humanoid avatar
            createAvatar();
            
            // Hide loading, show UI
            document.getElementById('loading').style.display = 'none';
            document.getElementById('info').style.display = 'block';
            document.getElementById('controls').style.display = 'flex';
            document.getElementById('progress-container').style.display = 'block';
            document.getElementById('total-frames').textContent = posesData.length;
            
            // Clock
            clock = new THREE.Clock();
            
            // Start animation
            animate();
        }}
        
        function createAvatar() {{
            avatar = new THREE.Group();
            
            // Skin material
            const skinMaterial = new THREE.MeshStandardMaterial({{
                color: 0xffdbac,
                roughness: 0.5,
                metalness: 0.1
            }});
            
            // Clothing material
            const clothMaterial = new THREE.MeshStandardMaterial({{
                color: 0x4a90a4,
                roughness: 0.7,
                metalness: 0
            }});
            
            // Hair material
            const hairMaterial = new THREE.MeshStandardMaterial({{
                color: 0x2c1810,
                roughness: 0.8
            }});
            
            // Head
            const headGroup = new THREE.Group();
            
            // Main head
            const headGeom = new THREE.SphereGeometry(0.12, 32, 32);
            const head = new THREE.Mesh(headGeom, skinMaterial);
            head.scale.set(1, 1.1, 1);
            headGroup.add(head);
            
            // Hair
            const hairGeom = new THREE.SphereGeometry(0.13, 32, 32, 0, Math.PI * 2, 0, Math.PI / 2);
            const hair = new THREE.Mesh(hairGeom, hairMaterial);
            hair.position.y = 0.02;
            headGroup.add(hair);
            
            // Eyes
            const eyeGeom = new THREE.SphereGeometry(0.02, 16, 16);
            const eyeMaterial = new THREE.MeshStandardMaterial({{ color: 0x3d2314 }});
            const leftEye = new THREE.Mesh(eyeGeom, eyeMaterial);
            leftEye.position.set(-0.04, 0.02, 0.1);
            headGroup.add(leftEye);
            
            const rightEye = new THREE.Mesh(eyeGeom, eyeMaterial);
            rightEye.position.set(0.04, 0.02, 0.1);
            headGroup.add(rightEye);
            
            // Nose
            const noseGeom = new THREE.ConeGeometry(0.015, 0.03, 8);
            const nose = new THREE.Mesh(noseGeom, skinMaterial);
            nose.position.set(0, -0.01, 0.11);
            nose.rotation.x = Math.PI / 2;
            headGroup.add(nose);
            
            // Neck
            const neckGeom = new THREE.CylinderGeometry(0.04, 0.05, 0.1, 16);
            const neck = new THREE.Mesh(neckGeom, skinMaterial);
            neck.position.y = -0.17;
            headGroup.add(neck);
            
            headGroup.position.y = 1.55;
            avatar.add(headGroup);
            bodyParts.head = headGroup;
            
            // Torso
            const torsoGroup = new THREE.Group();
            
            // Upper body
            const upperTorsoGeom = new THREE.CylinderGeometry(0.15, 0.12, 0.3, 16);
            const upperTorso = new THREE.Mesh(upperTorsoGeom, clothMaterial);
            upperTorso.position.y = 1.25;
            torsoGroup.add(upperTorso);
            
            // Lower body
            const lowerTorsoGeom = new THREE.CylinderGeometry(0.12, 0.1, 0.25, 16);
            const lowerTorso = new THREE.Mesh(lowerTorsoGeom, clothMaterial);
            lowerTorso.position.y = 0.975;
            torsoGroup.add(lowerTorso);
            
            avatar.add(torsoGroup);
            bodyParts.torso = torsoGroup;
            
            // Arms
            createArm('left', -1);
            createArm('right', 1);
            
            // Legs
            createLeg('left', -1);
            createLeg('right', 1);
            
            // Shadow
            avatar.traverse((obj) => {{
                if (obj.isMesh) {{
                    obj.castShadow = true;
                    obj.receiveShadow = true;
                }}
            }});
            
            scene.add(avatar);
        }}
        
        function createArm(side, direction) {{
            const skinMaterial = new THREE.MeshStandardMaterial({{
                color: 0xffdbac,
                roughness: 0.5
            }});
            
            const clothMaterial = new THREE.MeshStandardMaterial({{
                color: 0x4a90a4,
                roughness: 0.7
            }});
            
            const armGroup = new THREE.Group();
            
            // Shoulder joint
            const shoulderGeom = new THREE.SphereGeometry(0.045, 16, 16);
            const shoulder = new THREE.Mesh(shoulderGeom, clothMaterial);
            armGroup.add(shoulder);
            
            // Upper arm
            const upperArmGroup = new THREE.Group();
            const upperArmGeom = new THREE.CylinderGeometry(0.035, 0.03, 0.25, 12);
            const upperArm = new THREE.Mesh(upperArmGeom, clothMaterial);
            upperArm.position.y = -0.125;
            upperArmGroup.add(upperArm);
            
            // Elbow
            const elbowGeom = new THREE.SphereGeometry(0.035, 12, 12);
            const elbow = new THREE.Mesh(elbowGeom, skinMaterial);
            elbow.position.y = -0.25;
            upperArmGroup.add(elbow);
            
            armGroup.add(upperArmGroup);
            
            // Forearm group (will be rotated)
            const forearmGroup = new THREE.Group();
            forearmGroup.position.y = -0.25;
            
            const forearmGeom = new THREE.CylinderGeometry(0.03, 0.025, 0.22, 12);
            const forearm = new THREE.Mesh(forearmGeom, skinMaterial);
            forearm.position.y = -0.11;
            forearmGroup.add(forearm);
            
            // Wrist
            const wristGeom = new THREE.SphereGeometry(0.025, 12, 12);
            const wrist = new THREE.Mesh(wristGeom, skinMaterial);
            wrist.position.y = -0.22;
            forearmGroup.add(wrist);
            
            upperArmGroup.add(forearmGroup);
            
            // Hand group
            const handGroup = new THREE.Group();
            handGroup.position.y = -0.22;
            
            // Palm
            const palmGeom = new THREE.BoxGeometry(0.06, 0.08, 0.02);
            const palm = new THREE.Mesh(palmGeom, skinMaterial);
            palm.position.y = -0.04;
            handGroup.add(palm);
            
            // Fingers (simplified)
            for (let i = 0; i < 5; i++) {{
                const fingerGroup = new THREE.Group();
                const x = (i - 2) * 0.012;
                const fingerLen = i === 0 ? 0.03 : 0.04;
                
                const fingerGeom = new THREE.CylinderGeometry(0.005, 0.004, fingerLen, 8);
                const finger = new THREE.Mesh(fingerGeom, skinMaterial);
                finger.position.y = -fingerLen / 2;
                fingerGroup.add(finger);
                
                fingerGroup.position.set(x, -0.08, 0);
                if (i === 0) {{ // Thumb
                    fingerGroup.position.set(direction * 0.025, -0.05, 0.01);
                    fingerGroup.rotation.z = direction * 0.5;
                }}
                
                handGroup.add(fingerGroup);
            }}
            
            forearmGroup.add(handGroup);
            
            // Position arm
            armGroup.position.set(direction * 0.18, 1.35, 0);
            
            avatar.add(armGroup);
            
            if (side === 'left') {{
                bodyParts.leftArm = armGroup;
                bodyParts.leftUpperArm = upperArmGroup;
                bodyParts.leftForearm = forearmGroup;
                bodyParts.leftHand = handGroup;
            }} else {{
                bodyParts.rightArm = armGroup;
                bodyParts.rightUpperArm = upperArmGroup;
                bodyParts.rightForearm = forearmGroup;
                bodyParts.rightHand = handGroup;
            }}
        }}
        
        function createLeg(side, direction) {{
            const clothMaterial = new THREE.MeshStandardMaterial({{
                color: 0x2c3e50,
                roughness: 0.8
            }});
            
            const shoeMaterial = new THREE.MeshStandardMaterial({{
                color: 0x1a1a1a,
                roughness: 0.6
            }});
            
            const legGroup = new THREE.Group();
            
            // Upper leg
            const upperLegGeom = new THREE.CylinderGeometry(0.06, 0.05, 0.4, 12);
            const upperLeg = new THREE.Mesh(upperLegGeom, clothMaterial);
            upperLeg.position.y = -0.2;
            legGroup.add(upperLeg);
            
            // Knee
            const kneeGeom = new THREE.SphereGeometry(0.05, 12, 12);
            const knee = new THREE.Mesh(kneeGeom, clothMaterial);
            knee.position.y = -0.4;
            legGroup.add(knee);
            
            // Lower leg
            const lowerLegGeom = new THREE.CylinderGeometry(0.05, 0.04, 0.4, 12);
            const lowerLeg = new THREE.Mesh(lowerLegGeom, clothMaterial);
            lowerLeg.position.y = -0.6;
            legGroup.add(lowerLeg);
            
            // Foot
            const footGeom = new THREE.BoxGeometry(0.08, 0.05, 0.15);
            const foot = new THREE.Mesh(footGeom, shoeMaterial);
            foot.position.set(0, -0.825, 0.03);
            legGroup.add(foot);
            
            legGroup.position.set(direction * 0.08, 0.85, 0);
            
            avatar.add(legGroup);
            
            if (side === 'left') {{
                bodyParts.leftLeg = legGroup;
            }} else {{
                bodyParts.rightLeg = legGroup;
            }}
        }}
        
        function updateAvatarPose() {{
            if (posesData.length === 0) return;
            
            const pose = posesData[currentFrame];
            const bodyLandmarks = pose.pose || [];
            const leftHand = pose.left_hand || [];
            const rightHand = pose.right_hand || [];
            
            if (bodyLandmarks.length < 17) return;
            
            // Get key landmarks
            function getLandmark(idx) {{
                if (idx < bodyLandmarks.length) {{
                    const lm = bodyLandmarks[idx];
                    return {{
                        x: (lm[0] - 0.5) * 2,
                        y: -(lm[1] - 0.5) * 2 + 1,
                        z: (lm[2] || 0) * 0.5
                    }};
                }}
                return {{ x: 0, y: 1, z: 0 }};
            }}
            
            // Calculate arm angles from landmarks
            const leftShoulder = getLandmark(11);
            const rightShoulder = getLandmark(12);
            const leftElbow = getLandmark(13);
            const rightElbow = getLandmark(14);
            const leftWrist = getLandmark(15);
            const rightWrist = getLandmark(16);
            
            // Left arm
            if (bodyParts.leftUpperArm) {{
                const shoulderToElbow = {{
                    x: leftElbow.x - leftShoulder.x,
                    y: leftElbow.y - leftShoulder.y,
                    z: leftElbow.z - leftShoulder.z
                }};
                
                const armAngleZ = Math.atan2(shoulderToElbow.x, -shoulderToElbow.y);
                const armAngleX = Math.atan2(shoulderToElbow.z, Math.sqrt(shoulderToElbow.x * shoulderToElbow.x + shoulderToElbow.y * shoulderToElbow.y));
                
                bodyParts.leftUpperArm.rotation.z = armAngleZ + 0.2;
                bodyParts.leftUpperArm.rotation.x = armAngleX;
            }}
            
            if (bodyParts.leftForearm) {{
                const elbowToWrist = {{
                    x: leftWrist.x - leftElbow.x,
                    y: leftWrist.y - leftElbow.y,
                    z: leftWrist.z - leftElbow.z
                }};
                
                const forearmAngleZ = Math.atan2(elbowToWrist.x, -elbowToWrist.y);
                const forearmAngleX = Math.atan2(elbowToWrist.z, Math.sqrt(elbowToWrist.x * elbowToWrist.x + elbowToWrist.y * elbowToWrist.y));
                
                bodyParts.leftForearm.rotation.z = forearmAngleZ * 0.5;
                bodyParts.leftForearm.rotation.x = forearmAngleX;
            }}
            
            // Right arm
            if (bodyParts.rightUpperArm) {{
                const shoulderToElbow = {{
                    x: rightElbow.x - rightShoulder.x,
                    y: rightElbow.y - rightShoulder.y,
                    z: rightElbow.z - rightShoulder.z
                }};
                
                const armAngleZ = Math.atan2(shoulderToElbow.x, -shoulderToElbow.y);
                const armAngleX = Math.atan2(shoulderToElbow.z, Math.sqrt(shoulderToElbow.x * shoulderToElbow.x + shoulderToElbow.y * shoulderToElbow.y));
                
                bodyParts.rightUpperArm.rotation.z = armAngleZ - 0.2;
                bodyParts.rightUpperArm.rotation.x = armAngleX;
            }}
            
            if (bodyParts.rightForearm) {{
                const elbowToWrist = {{
                    x: rightWrist.x - rightElbow.x,
                    y: rightWrist.y - rightElbow.y,
                    z: rightWrist.z - rightElbow.z
                }};
                
                const forearmAngleZ = Math.atan2(elbowToWrist.x, -elbowToWrist.y);
                const forearmAngleX = Math.atan2(elbowToWrist.z, Math.sqrt(elbowToWrist.x * elbowToWrist.x + elbowToWrist.y * elbowToWrist.y));
                
                bodyParts.rightForearm.rotation.z = forearmAngleZ * 0.5;
                bodyParts.rightForearm.rotation.x = forearmAngleX;
            }}
            
            // Head follows nose/face direction
            const nose = getLandmark(0);
            if (bodyParts.head) {{
                const midShoulder = {{
                    x: (leftShoulder.x + rightShoulder.x) / 2,
                    y: (leftShoulder.y + rightShoulder.y) / 2
                }};
                bodyParts.head.rotation.y = (nose.x - midShoulder.x) * 0.5;
                bodyParts.head.rotation.x = (nose.y - midShoulder.y - 0.3) * 0.3;
            }}
            
            // Update UI
            document.getElementById('frame-num').textContent = currentFrame + 1;
            document.getElementById('progress-fill').style.width = ((currentFrame + 1) / posesData.length * 100) + '%';
        }}
        
        let lastTime = 0;
        const frameInterval = 1000 / fps;
        
        function animate(time) {{
            requestAnimationFrame(animate);
            
            if (playing && time - lastTime > frameInterval / speed) {{
                lastTime = time;
                currentFrame = (currentFrame + 1) % posesData.length;
                updateAvatarPose();
            }}
            
            renderer.render(scene, camera);
        }}
        
        function togglePlay() {{
            playing = !playing;
        }}
        
        function restart() {{
            currentFrame = 0;
            updateAvatarPose();
        }}
        
        function toggleSpeed() {{
            const speeds = [0.5, 1, 1.5, 2];
            const idx = speeds.indexOf(speed);
            speed = speeds[(idx + 1) % speeds.length];
            document.getElementById('speed').textContent = speed + 'x';
        }}
        
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
            if (e.code === 'KeyR') restart();
        }});
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>'''
    
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate 3D Humanoid BSL Avatar")
    parser.add_argument("--poses-dir", type=str, default=None)
    parser.add_argument("--gloss", type=str, default=None)
    parser.add_argument("--glosses", type=str, default=None)
    parser.add_argument("--text", type=str, default=None, help="Convert text to glosses first")
    parser.add_argument("--output", type=str, default="avatar_3d.html")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    poses_dir = Path(args.poses_dir) if args.poses_dir else project_root / "data" / "poses"
    
    pose_lookup = PoseLookup(str(poses_dir))
    
    # Get glosses
    glosses = []
    if args.gloss:
        glosses = [args.gloss.upper()]
    elif args.glosses:
        glosses = args.glosses.upper().split()
    elif args.text:
        # Simple text to gloss (uppercase words)
        import re
        glosses = re.findall(r'\\b\\w+\\b', args.text.upper())
        print(f"Text: {args.text}")
        print(f"Glosses: {' -> '.join(glosses)}")
    else:
        glosses = ["HELLO", "GOOD", "YOU"]
    
    print(f"Generating 3D avatar for: {' -> '.join(glosses)}")
    
    # Collect poses
    all_poses = []
    found = []
    missing = []
    
    for gloss in glosses:
        poses = pose_lookup.get_pose_sequence(gloss)
        if poses:
            all_poses.extend(poses)
            found.append(gloss)
            print(f"  {gloss}: {len(poses)} frames")
        else:
            missing.append(gloss)
            print(f"  {gloss}: NOT FOUND")
    
    if not all_poses:
        print("No poses found!")
        print(f"\\nAvailable glosses sample: {pose_lookup.get_available_glosses()[:20]}")
        return
    
    print(f"Total: {len(all_poses)} frames")
    if missing:
        print(f"Missing: {missing}")
    
    # Generate HTML
    gloss_text = " -> ".join(found)
    html = generate_3d_avatar_html(all_poses, gloss_text)
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Saved: {output_path}")
    
    if args.serve:
        import os
        os.chdir(output_path.parent)
        
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            url = f"http://localhost:{args.port}/{output_path.name}"
            print(f"\\nOpening: {url}")
            webbrowser.open(url)
            print("Press Ctrl+C to stop server")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\\nServer stopped")


if __name__ == "__main__":
    main()
