"""
3D GLB avatar renderer for Signlytic AI Direction 2 (English -> BSL).

Serves files via a local HTTP server on a random free port — bypasses
Gradio's /file= path normalisation issues on Windows entirely.

Avatars (Mixamo, full finger bones):
  Male   - data/avatars/Male.glb    bone prefix: mixamorig9
  Female - data/avatars/Female.glb  bone prefix: mixamorig8
"""

from __future__ import annotations

import functools
import http.server
import json
import random
import shutil
import socket
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_DOWNLOADS = Path.home() / "Downloads"
MALE_GLB_SRC = _DOWNLOADS / "Male.glb"
FEMALE_GLB_SRC = _DOWNLOADS / "Female.glb"
MALE_BONE_PREFIX = "mixamorig9"
FEMALE_BONE_PREFIX = "mixamorig8"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Avatar3DRenderer:
    """Render BSL gloss sequences as a 3D signed avatar (HTML iframe)."""

    _server_thread: Optional[threading.Thread] = None
    _server_port: int = 0
    _server_started: bool = False

    def __init__(
        self,
        project_root: Optional[Path] = None,
        pose_renderer=None,
    ) -> None:
        self.project_root = (
            Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        )
        self.pose_renderer = pose_renderer
        self.avatars_dir = self.project_root / "data" / "avatars"
        self._ensure_avatars()
        self._start_server()

    def _ensure_avatars(self) -> None:
        self.avatars_dir.mkdir(parents=True, exist_ok=True)
        for src, name in [(MALE_GLB_SRC, "Male.glb"), (FEMALE_GLB_SRC, "Female.glb")]:
            dst = self.avatars_dir / name
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)
                print(f"[Avatar3D] Copied {name} -> {dst}")
            elif not dst.exists():
                print(f"[Avatar3D] WARNING: {src} not found. Place GLB in data/avatars/")

    def _start_server(self) -> None:
        if Avatar3DRenderer._server_started:
            return
        port = _free_port()
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(self.avatars_dir),
        )
        srv = http.server.HTTPServer(("127.0.0.1", port), handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        Avatar3DRenderer._server_thread = t
        Avatar3DRenderer._server_port = port
        Avatar3DRenderer._server_started = True
        print(f"[Avatar3D] File server on http://127.0.0.1:{port}/")

    @property
    def _base_url(self) -> str:
        return f"http://127.0.0.1:{Avatar3DRenderer._server_port}"

    # ── Public API ────────────────────────────────────────────────────────────

    def get_coverage(self, glosses: Sequence[str]) -> Dict:
        if self.pose_renderer:
            return self.pose_renderer.get_coverage(glosses)
        normalized = [str(g).strip().upper() for g in glosses if str(g).strip()]
        return {
            "coverage": 0.0, "available": [], "missing": normalized,
            "available_count": 0, "missing_count": len(normalized),
        }

    def render_sequence_html(
        self,
        glosses: Sequence[str],
        speed: float = 1.0,
        avatar: str = "male",
        gloss_str: str = "",
    ) -> str:
        frames = self._collect_frames(glosses, speed)
        if not frames:
            return _error_html("No pose data available for the requested glosses.")

        is_female = avatar == "female"
        bone_prefix = FEMALE_BONE_PREFIX if is_female else MALE_BONE_PREFIX
        glb_name = "Female.glb" if is_female else "Male.glb"
        glb_path = self.avatars_dir / glb_name

        if not glb_path.exists():
            return _error_html(f"Avatar file not found: {glb_path}. Place {glb_name} in data/avatars/.")

        label = gloss_str or " > ".join(
            dict.fromkeys(f.get("g", "") for f in frames[::max(1, len(frames) // 10)])
        )

        page_html = _build_threejs_page(
            frames=frames,
            glb_url=f"{self._base_url}/{glb_name}",
            bone_prefix=bone_prefix,
            gloss_label=label[:80],
        )

        html_file = self.avatars_dir / "signing_preview.html"
        html_file.write_text(page_html, encoding="utf-8")

        iframe_src = f"{self._base_url}/signing_preview.html"
        return (
            f'''<div style="width:100%;border-radius:8px;overflow:hidden;'''
            f'''border:1px solid #1e293b;background:#0c0e14">'''
            f'''<iframe src="{iframe_src}" '''
            f'''style="width:100%;height:420px;border:none;display:block" '''
            f'''allow="autoplay" title="3D BSL Signing Avatar"></iframe>'''
            f'''</div>'''
            f'''<p style="margin:4px 0 0;font-size:0.70rem;color:#475569;font-family:system-ui">'''
            f'''<a href="{iframe_src}" target="_blank" '''
            f'''style="color:#0e7c6b;text-decoration:none">Open fullscreen</a>'''
            f''' &nbsp;&bull;&nbsp; Drag to orbit &nbsp;&bull;&nbsp; '''
            f'''Scroll to zoom &nbsp;&bull;&nbsp; Space to pause</p>'''
        )

    # ── Frame collection ──────────────────────────────────────────────────────

    def _collect_frames(self, glosses: Sequence[str], speed: float) -> List[Dict]:
        if not self.pose_renderer:
            return []
        normalized = [str(g).strip().upper() for g in glosses if str(g).strip()]
        if not normalized:
            return []
        speed = float(np.clip(speed, 0.6, 1.6))
        per_gloss_s = self.pose_renderer.base_gloss_duration / speed
        frames_per_gloss = max(1, int(round(per_gloss_s * self.pose_renderer.output_fps)))
        result: List[Dict] = []
        for gloss in normalized:
            sampled, missing = self.pose_renderer._get_gloss_frames(gloss, frames_per_gloss)
            for frame in sampled:
                result.append({
                    "g": gloss, "m": bool(missing),
                    "p": frame["pose"].tolist(),
                    "l": frame["left_hand"].tolist(),
                    "r": frame["right_hand"].tolist(),
                })
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_html(msg: str) -> str:
    return (
        f'''<div style="padding:1rem;color:#f87171;font-size:0.82rem;'''
        f'''font-family:system-ui;background:#0f172a;border-radius:8px;'''
        f'''border:1px solid #1e293b">{msg}</div>'''
    )


def _build_threejs_page(
    frames: List[Dict],
    glb_url: str,
    bone_prefix: str,
    gloss_label: str,
) -> str:
    frames_json = json.dumps(frames, separators=(",", ":"))
    return (
        '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#0b0d14;
  font-family:system-ui,sans-serif;color:#e2e8f0}
#c{display:block;width:100%;height:100%}
#gloss{position:fixed;top:10px;left:50%;transform:translateX(-50%);
  background:rgba(14,124,107,0.90);color:#fff;padding:3px 14px;
  border-radius:20px;font-size:11px;font-weight:600;letter-spacing:.05em;
  pointer-events:none;max-width:76%;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;backdrop-filter:blur(4px)}
#hud{position:fixed;bottom:44px;left:0;right:0;text-align:center;
  font-size:10px;color:#475569;pointer-events:none}
#bar{position:fixed;bottom:8px;left:50%;transform:translateX(-50%);
  display:flex;gap:4px;align-items:center;
  background:rgba(12,14,20,0.85);padding:4px 8px;border-radius:8px;
  border:1px solid #1e293b;backdrop-filter:blur(6px)}
.btn{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);
  color:#94a3b8;padding:3px 10px;border-radius:5px;cursor:pointer;
  font-size:11px;transition:background .15s}
.btn:hover{background:rgba(255,255,255,0.12);color:#e2e8f0}
#spd{font-size:11px;color:#64748b;min-width:32px;text-align:center}
#stat{position:fixed;top:10px;right:10px;font-size:10px;color:#334155;
  background:rgba(12,14,20,0.7);padding:2px 7px;border-radius:4px}
</style>
</head>
<body>
<canvas id="c"></canvas>
''' + f'''<div id="gloss">{gloss_label}</div>''' + '''
<div id="stat">Loading...</div>
<div id="hud"></div>
<div id="bar">
  <button class="btn" onclick="step(-1)">&#9664;</button>
  <button class="btn" id="pb" onclick="togglePlay()">&#9646;&#9646;</button>
  <button class="btn" onclick="step(1)">&#9654;</button>
  <span style="width:1px;height:14px;background:#1e293b;margin:0 2px"></span>
  <button class="btn" onclick="adj(-0.25)">&#8722;</button>
  <span id="spd">1.0x</span>
  <button class="btn" onclick="adj(0.25)">+</button>
  <span style="width:1px;height:14px;background:#1e293b;margin:0 2px"></span>
  <button class="btn" onclick="resetCam()">&#8635;</button>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
"use strict";
''' + f'''const FRAMES={frames_json};
const GLB_URL="{glb_url}";
const BP="{bone_prefix}";
''' + '''
const FPS=20,IVL=1000/FPS;
const canvas=document.getElementById("c");
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputEncoding=THREE.sRGBEncoding;
renderer.shadowMap.enabled=true;
doResize();

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0b0d14);
scene.fog=new THREE.FogExp2(0x0b0d14,0.035);

const cam=new THREE.PerspectiveCamera(40,canvas.width/canvas.height,0.01,60);
cam.position.set(0,1.4,2.8);

const ctrl=new THREE.OrbitControls(cam,renderer.domElement);
ctrl.target.set(0,1.1,0);
ctrl.enablePan=false;
ctrl.minDistance=0.8;
ctrl.maxDistance=7;
ctrl.update();

// Lighting
const amb=new THREE.AmbientLight(0xffffff,0.5);scene.add(amb);
const key=new THREE.DirectionalLight(0xffffff,1.0);
key.position.set(2,5,3);key.castShadow=true;scene.add(key);
const fill=new THREE.DirectionalLight(0x8ecffc,0.4);
fill.position.set(-3,2,-2);scene.add(fill);
const rim=new THREE.DirectionalLight(0x0e7c6b,0.3);
rim.position.set(0,3,-4);scene.add(rim);

// Floor
const floor=new THREE.Mesh(
  new THREE.CircleGeometry(3,64),
  new THREE.MeshStandardMaterial({color:0x0f172a,roughness:0.9,metalness:0.1})
);
floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;scene.add(floor);

// Subtle grid
const grid=new THREE.GridHelper(6,24,0x1e293b,0x1e293b);
grid.material.opacity=0.5;grid.material.transparent=true;scene.add(grid);

let skel=null,bmap={},bq={},playing=true,fidx=0,lt=0,smult=1.0;

const bn=n=>BP+":"+n;
const gb=n=>bmap[bn(n)]||null;

function mp3(kp){
  return new THREE.Vector3(-(kp[0]-.5),-(kp[1]-.5),-(kp[2]||0));
}

// GLB Load
const loader=new THREE.GLTFLoader();
loader.load(GLB_URL,
  gltf=>{
    const model=gltf.scene;
    // Auto-detect scale: Mixamo uses cm
    const box=new THREE.Box3().setFromObject(model);
    const size=box.getSize(new THREE.Vector3());
    const sc=size.y>5?0.01:1.0;
    model.scale.setScalar(sc);
    scene.add(model);
    model.traverse(o=>{
      if(o.isSkinnedMesh){
        skel=o.skeleton;
        skel.bones.forEach(b=>{bmap[b.name]=b;bq[b.name]=b.quaternion.clone();});
        o.castShadow=true;
        const m=o.material;
        if(Array.isArray(m))m.forEach(x=>{x.side=THREE.DoubleSide;});
        else if(m)m.side=THREE.DoubleSide;
      }
    });
    // Reposition model above floor
    const box2=new THREE.Box3().setFromObject(model);
    model.position.y=-box2.min.y;
    // Aim camera at mid-torso
    const mid=new THREE.Vector3();
    box2.getCenter(mid);
    ctrl.target.set(0,mid.y*sc+model.position.y,0);
    ctrl.update();
    document.getElementById("stat").textContent=
      skel?`Ready | ${FRAMES.length} frames | ${skel.bones.length} bones`:"Loaded";
    applyFrame(0);
    requestAnimationFrame(loop);
  },
  xhr=>{
    const pct=xhr.total?Math.round(xhr.loaded/xhr.total*100):"...";
    document.getElementById("stat").textContent=`Loading ${pct}%`;
  },
  err=>{
    document.getElementById("stat").textContent="Load failed - check console";
    console.error("GLB:",err);
  }
);

function resetBind(){
  for(const[n,b]of Object.entries(bmap))if(bq[n])b.quaternion.copy(bq[n]);
}

function driveBone(name,from,to){
  const bone=gb(name);
  if(!bone||!from||!to)return;
  const dir=new THREE.Vector3().subVectors(mp3(to),mp3(from));
  if(dir.lengthSq()<1e-9)return;
  dir.normalize();
  const pq=new THREE.Quaternion();
  if(bone.parent)bone.parent.getWorldQuaternion(pq);
  const ld=dir.clone().applyQuaternion(pq.invert()).normalize();
  bone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),ld);
}

function df(name,hand,a,b){
  if(hand&&hand[a]&&hand[b])driveBone(name,hand[a],hand[b]);
}

function applyFrame(idx){
  if(!skel||!FRAMES.length)return;
  const f=FRAMES[idx%FRAMES.length];
  const p=f.p,lh=f.l,rh=f.r;
  resetBind();
  driveBone("LeftArm",     p[11],p[13]);
  driveBone("RightArm",    p[12],p[14]);
  driveBone("LeftForeArm", p[13],p[15]);
  driveBone("RightForeArm",p[14],p[16]);
  if(lh&&lh[0]&&lh[9])driveBone("LeftHand", lh[0],lh[9]);
  if(rh&&rh[0]&&rh[9])driveBone("RightHand",rh[0],rh[9]);
  df("LeftHandThumb1",lh,1,2);df("LeftHandThumb2",lh,2,3);df("LeftHandThumb3",lh,3,4);
  df("LeftHandIndex1",lh,5,6);df("LeftHandIndex2",lh,6,7);df("LeftHandIndex3",lh,7,8);
  df("LeftHandMiddle1",lh,9,10);df("LeftHandMiddle2",lh,10,11);df("LeftHandMiddle3",lh,11,12);
  df("LeftHandRing1",lh,13,14);df("LeftHandRing2",lh,14,15);df("LeftHandRing3",lh,15,16);
  df("LeftHandPinky1",lh,17,18);df("LeftHandPinky2",lh,18,19);df("LeftHandPinky3",lh,19,20);
  df("RightHandThumb1",rh,1,2);df("RightHandThumb2",rh,2,3);df("RightHandThumb3",rh,3,4);
  df("RightHandIndex1",rh,5,6);df("RightHandIndex2",rh,6,7);df("RightHandIndex3",rh,7,8);
  df("RightHandMiddle1",rh,9,10);df("RightHandMiddle2",rh,10,11);df("RightHandMiddle3",rh,11,12);
  df("RightHandRing1",rh,13,14);df("RightHandRing2",rh,14,15);df("RightHandRing3",rh,15,16);
  df("RightHandPinky1",rh,17,18);df("RightHandPinky2",rh,18,19);df("RightHandPinky3",rh,19,20);
  const cur=idx%FRAMES.length;
  document.getElementById("hud").textContent=
    `${cur+1} / ${FRAMES.length}  —  ${f.g}${f.m?" (MISSING)":""}`;
}

function loop(t){
  requestAnimationFrame(loop);
  ctrl.update();
  if(playing&&skel&&FRAMES.length){
    if(t-lt>=IVL/smult){lt=t;fidx=(fidx+1)%FRAMES.length;applyFrame(fidx);}
  }
  renderer.render(scene,cam);
}

function togglePlay(){
  playing=!playing;
  document.getElementById("pb").innerHTML=playing?"&#9646;&#9646;":"&#9654;";
}
function step(d){fidx=(fidx+d+FRAMES.length)%FRAMES.length;applyFrame(fidx);}
function adj(d){
  smult=Math.max(.25,Math.min(3,+(smult+d).toFixed(2)));
  document.getElementById("spd").textContent=smult.toFixed(2)+"x";
}
function resetCam(){
  cam.position.set(0,1.4,2.8);
  ctrl.target.set(0,1.1,0);ctrl.update();
}
function doResize(){
  const w=innerWidth,h=innerHeight;
  renderer.setSize(w,h);
  if(cam){cam.aspect=w/h;cam.updateProjectionMatrix();}
}
window.addEventListener("resize",doResize);
document.addEventListener("keydown",e=>{
  if(e.code==="Space"){e.preventDefault();togglePlay();}
  if(e.code==="ArrowLeft")step(-1);
  if(e.code==="ArrowRight")step(1);
});
</script>
</body>
</html>'''
    )
