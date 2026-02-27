"""
Web-based 3D BSL Avatar using Three.js

Generates an HTML page with a 3D animated signer.
Can be embedded in Gradio or served standalone.

Pipeline: Gloss → Pose Lookup → Three.js Animation

Usage:
    python scripts/avatar_threejs.py --gloss HELLO --output avatar.html
    python scripts/avatar_threejs.py --glosses "HELLO GOOD YOU" --serve
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
        """Build index of gloss -> pose files."""
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
    
    def get_pose_sequence(self, gloss: str) -> Optional[List]:
        """Get pose sequence for a gloss."""
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


def generate_avatar_html(poses_data: List, gloss_text: str = "") -> str:
    """Generate HTML with Three.js 3D avatar animation."""
    
    # Convert poses to JSON for JavaScript
    poses_json = json.dumps(poses_data)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BSL 3D Avatar - {gloss_text}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            z-index: 100;
            background: rgba(0,0,0,0.5);
            padding: 15px 25px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        #info h2 {{ 
            margin-bottom: 10px;
            font-weight: 300;
        }}
        #info .gloss {{
            font-size: 24px;
            color: #4ecdc4;
            font-weight: 600;
        }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            z-index: 100;
        }}
        .btn {{
            background: rgba(78, 205, 196, 0.8);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
        }}
        .btn:hover {{
            background: rgba(78, 205, 196, 1);
            transform: scale(1.05);
        }}
        #frame-info {{
            position: absolute;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-size: 14px;
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h2>BSL 3D Avatar</h2>
        <div class="gloss">{gloss_text}</div>
    </div>
    <div id="frame-info">Frame: <span id="frame-num">1</span> / <span id="total-frames">0</span></div>
    <div id="controls">
        <button class="btn" onclick="togglePlay()">⏯ Play/Pause</button>
        <button class="btn" onclick="restart()">🔄 Restart</button>
        <button class="btn" onclick="toggleSpeed()">⚡ Speed: <span id="speed">1x</span></button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Pose data from Python
        const posesData = {poses_json};
        
        // Animation state
        let currentFrame = 0;
        let playing = true;
        let speed = 1;
        const fps = 25;
        
        // Three.js setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x000000, 0);
        container.appendChild(renderer.domElement);
        
        // Camera position
        camera.position.set(0, 0, 3);
        camera.lookAt(0, 0, 0);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        scene.add(directionalLight);
        
        // Materials
        const bodyMaterial = new THREE.MeshPhongMaterial({{ color: 0x4ecdc4, shininess: 100 }});
        const leftHandMaterial = new THREE.MeshPhongMaterial({{ color: 0x45b7aa, shininess: 100 }});
        const rightHandMaterial = new THREE.MeshPhongMaterial({{ color: 0xff6b6b, shininess: 100 }});
        const boneMaterial = new THREE.MeshPhongMaterial({{ color: 0x3d9d94, shininess: 50 }});
        
        // Body connections
        const bodyConnections = [
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
            [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]
        ];
        
        // Hand connections
        const handConnections = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17]
        ];
        
        // Create spheres and cylinders
        const bodyJoints = [];
        const bodyBones = [];
        const leftHandJoints = [];
        const leftHandBones = [];
        const rightHandJoints = [];
        const rightHandBones = [];
        
        // Create body joints
        for (let i = 0; i < 33; i++) {{
            const sphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.02, 16, 16),
                bodyMaterial
            );
            scene.add(sphere);
            bodyJoints.push(sphere);
        }}
        
        // Create body bones
        for (let conn of bodyConnections) {{
            const cylinder = new THREE.Mesh(
                new THREE.CylinderGeometry(0.01, 0.01, 1, 8),
                boneMaterial
            );
            scene.add(cylinder);
            bodyBones.push({{ mesh: cylinder, start: conn[0], end: conn[1] }});
        }}
        
        // Create hand joints and bones
        function createHand(joints, bones, material) {{
            for (let i = 0; i < 21; i++) {{
                const sphere = new THREE.Mesh(
                    new THREE.SphereGeometry(0.008, 12, 12),
                    material
                );
                scene.add(sphere);
                joints.push(sphere);
            }}
            
            for (let conn of handConnections) {{
                const cylinder = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.004, 0.004, 1, 6),
                    material
                );
                scene.add(cylinder);
                bones.push({{ mesh: cylinder, start: conn[0], end: conn[1] }});
            }}
        }}
        
        createHand(leftHandJoints, leftHandBones, leftHandMaterial);
        createHand(rightHandJoints, rightHandBones, rightHandMaterial);
        
        // Update UI
        document.getElementById('total-frames').textContent = posesData.length;
        
        function extractKeypoints(pose) {{
            const body = [];
            const leftHand = [];
            const rightHand = [];
            
            // Helper to extract x,y,z from keypoint (handles both dict and list)
            function getXYZ(kp) {{
                if (Array.isArray(kp)) {{
                    return [kp[0] || 0, kp[1] || 0, kp[2] || 0];
                }} else if (kp && typeof kp === 'object') {{
                    return [kp.x || 0, kp.y || 0, kp.z || 0];
                }}
                return [0, 0, 0];
            }}
            
            // Body (33 keypoints) - stored as 'pose' not 'body_pose'
            const bodyPose = pose.pose || [];
            for (let kp of bodyPose) {{
                const [x, y, z] = getXYZ(kp);
                body.push([
                    (x - 0.5) * 2,
                    -(y - 0.5) * 2,
                    z * 2
                ]);
            }}
            
            // Left hand (21 keypoints)
            const left = pose.left_hand || [];
            for (let kp of left) {{
                const [x, y, z] = getXYZ(kp);
                leftHand.push([
                    (x - 0.5) * 2,
                    -(y - 0.5) * 2,
                    z * 2
                ]);
            }}
            
            // Right hand (21 keypoints)
            const right = pose.right_hand || [];
            for (let kp of right) {{
                const [x, y, z] = getXYZ(kp);
                rightHand.push([
                    (x - 0.5) * 2,
                    -(y - 0.5) * 2,
                    z * 2
                ]);
            }}
            
            return {{ body, leftHand, rightHand }};
        }}
        
        function updateBone(bone, joints) {{
            const start = joints[bone.start];
            const end = joints[bone.end];
            
            if (!start || !end) return;
            
            const startPos = start.position;
            const endPos = end.position;
            
            // Calculate midpoint
            const midpoint = new THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5);
            bone.mesh.position.copy(midpoint);
            
            // Calculate length
            const length = startPos.distanceTo(endPos);
            bone.mesh.scale.y = length;
            
            // Calculate rotation
            const direction = new THREE.Vector3().subVectors(endPos, startPos).normalize();
            const quaternion = new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0),
                direction
            );
            bone.mesh.quaternion.copy(quaternion);
            
            bone.mesh.visible = length > 0.01 && length < 1;
        }}
        
        function updateFrame() {{
            if (posesData.length === 0) return;
            
            const pose = posesData[currentFrame];
            const {{ body, leftHand, rightHand }} = extractKeypoints(pose);
            
            // Update body joints
            for (let i = 0; i < body.length && i < bodyJoints.length; i++) {{
                bodyJoints[i].position.set(body[i][0], body[i][1], body[i][2]);
                bodyJoints[i].visible = true;
            }}
            
            // Update body bones
            for (let bone of bodyBones) {{
                updateBone(bone, bodyJoints);
            }}
            
            // Check if hands are detected
            const leftDetected = leftHand.some(kp => Math.abs(kp[0]) > 0.01);
            const rightDetected = rightHand.some(kp => Math.abs(kp[0]) > 0.01);
            
            // Update left hand
            for (let i = 0; i < leftHand.length && i < leftHandJoints.length; i++) {{
                leftHandJoints[i].position.set(leftHand[i][0], leftHand[i][1], leftHand[i][2]);
                leftHandJoints[i].visible = leftDetected;
            }}
            for (let bone of leftHandBones) {{
                updateBone(bone, leftHandJoints);
                bone.mesh.visible = bone.mesh.visible && leftDetected;
            }}
            
            // Update right hand
            for (let i = 0; i < rightHand.length && i < rightHandJoints.length; i++) {{
                rightHandJoints[i].position.set(rightHand[i][0], rightHand[i][1], rightHand[i][2]);
                rightHandJoints[i].visible = rightDetected;
            }}
            for (let bone of rightHandBones) {{
                updateBone(bone, rightHandJoints);
                bone.mesh.visible = bone.mesh.visible && rightDetected;
            }}
            
            // Update frame counter
            document.getElementById('frame-num').textContent = currentFrame + 1;
        }}
        
        // Animation loop
        let lastTime = 0;
        const frameInterval = 1000 / fps;
        
        function animate(time) {{
            requestAnimationFrame(animate);
            
            if (playing && time - lastTime > frameInterval / speed) {{
                lastTime = time;
                currentFrame = (currentFrame + 1) % posesData.length;
                updateFrame();
            }}
            
            renderer.render(scene, camera);
        }}
        
        // Controls
        function togglePlay() {{
            playing = !playing;
        }}
        
        function restart() {{
            currentFrame = 0;
            updateFrame();
        }}
        
        function toggleSpeed() {{
            const speeds = [0.5, 1, 1.5, 2];
            const idx = speeds.indexOf(speed);
            speed = speeds[(idx + 1) % speeds.length];
            document.getElementById('speed').textContent = speed + 'x';
        }}
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') togglePlay();
            if (e.code === 'KeyR') restart();
        }});
        
        // Resize handler
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Start
        updateFrame();
        animate(0);
    </script>
</body>
</html>'''
    
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate 3D BSL Avatar HTML")
    parser.add_argument("--poses-dir", type=str, default=None)
    parser.add_argument("--gloss", type=str, default=None)
    parser.add_argument("--glosses", type=str, default=None)
    parser.add_argument("--output", type=str, default="avatar_3d.html")
    parser.add_argument("--serve", action="store_true", help="Start local server and open browser")
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
    else:
        glosses = ["HELLO", "GOOD", "YOU"]
    
    print(f"Generating avatar for: {' → '.join(glosses)}")
    
    # Collect poses
    all_poses = []
    for gloss in glosses:
        poses = pose_lookup.get_pose_sequence(gloss)
        if poses:
            all_poses.extend(poses)
            print(f"  {gloss}: {len(poses)} frames")
        else:
            print(f"  {gloss}: NOT FOUND")
    
    if not all_poses:
        print("No poses found!")
        return
    
    print(f"Total: {len(all_poses)} frames")
    
    # Generate HTML
    html = generate_avatar_html(all_poses, " → ".join(glosses))
    
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
            print(f"\nServing at: {url}")
            webbrowser.open(url)
            print("Press Ctrl+C to stop")
            httpd.serve_forever()


if __name__ == "__main__":
    main()