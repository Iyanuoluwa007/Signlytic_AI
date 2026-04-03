// overlay/avatar3d.js — Signlytic AI Extension — 3D Avatar Renderer
// Depends on THREE (r128) and THREE.GLTFLoader loaded before this script.
//
// Bone prefix:
//   Male avatar   →  mixamorig9:
//   Female avatar →  mixamorig8:
//
// Pipeline per frame:
//   MediaPipe Holistic landmarks → compute limb vectors →
//   convert to bone-local quaternions → apply to skeleton bones

// ─── CDN / GitHub paths ───────────────────────────────────────────────────────
const AVATAR_CDN = {
  male:   'https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases/download/v0.3.5/Male.glb',
  female: 'https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases/download/v0.3.5/Female.glb',
};

const BONE_PREFIX = { male: 'mixamorig9', female: 'mixamorig8' };

// MediaPipe Holistic body landmark indices (upper body only — BSL relevant)
const MP = {
  NOSE:          0,
  L_SHOULDER:   11,  R_SHOULDER:  12,
  L_ELBOW:      13,  R_ELBOW:     14,
  L_WRIST:      15,  R_WRIST:     16,
  L_HIP:        23,  R_HIP:       24,
};

// MediaPipe Hand landmark indices
const MH = {
  WRIST:    0,
  THUMB:    [1,2,3,4],
  INDEX:    [5,6,7,8],
  MIDDLE:   [9,10,11,12],
  RING:     [13,14,15,16],
  PINKY:    [17,18,19,20],
};

// Mixamo upper-body bone names (appended to prefix)
const BONE_NAMES = {
  hips:        'Hips',
  spine:       'Spine',
  spine1:      'Spine1',
  spine2:      'Spine2',
  neck:        'Neck',
  head:        'Head',
  lShoulder:   'LeftShoulder',
  lArm:        'LeftArm',
  lForeArm:    'LeftForeArm',
  lHand:       'LeftHand',
  rShoulder:   'RightShoulder',
  rArm:        'RightArm',
  rForeArm:    'RightForeArm',
  rHand:       'RightHand',
  // Left hand fingers (Index, Middle, Ring, Pinky, Thumb — 3 bones each)
  lIndex1: 'LeftHandIndex1', lIndex2: 'LeftHandIndex2', lIndex3: 'LeftHandIndex3',
  lMiddle1:'LeftHandMiddle1',lMiddle2:'LeftHandMiddle2',lMiddle3:'LeftHandMiddle3',
  lRing1:  'LeftHandRing1',  lRing2:  'LeftHandRing2',  lRing3:  'LeftHandRing3',
  lPinky1: 'LeftHandPinky1', lPinky2: 'LeftHandPinky2', lPinky3: 'LeftHandPinky3',
  lThumb1: 'LeftHandThumb1', lThumb2: 'LeftHandThumb2', lThumb3: 'LeftHandThumb3',
  // Right hand fingers
  rIndex1: 'RightHandIndex1', rIndex2: 'RightHandIndex2', rIndex3: 'RightHandIndex3',
  rMiddle1:'RightHandMiddle1',rMiddle2:'RightHandMiddle2',rMiddle3:'RightHandMiddle3',
  rRing1:  'RightHandRing1',  rRing2:  'RightHandRing2',  rRing3:  'RightHandRing3',
  rPinky1: 'RightHandPinky1', rPinky2: 'RightHandPinky2', rPinky3: 'RightHandPinky3',
  rThumb1: 'RightHandThumb1', rThumb2: 'RightHandThumb2', rThumb3: 'RightHandThumb3',
};

// ─── IDB helpers for GLB blob caching ────────────────────────────────────────
let _glbDb = null;

async function openGlbDB() {
  if (_glbDb) return _glbDb;
  return new Promise((res, rej) => {
    const req = indexedDB.open('signlytic-glb-cache', 1);
    req.onupgradeneeded = e => e.target.result.createObjectStore('glb', { keyPath: 'key' });
    req.onsuccess = e => { _glbDb = e.target.result; res(_glbDb); };
    req.onerror   = e => rej(e);
  });
}

async function glbCacheGet(key) {
  try {
    const db = await openGlbDB();
    return new Promise(res => {
      const req = db.transaction('glb','readonly').objectStore('glb').get(key);
      req.onsuccess = () => res(req.result?.data || null);
      req.onerror   = () => res(null);
    });
  } catch { return null; }
}

async function glbCacheSet(key, arrayBuffer) {
  try {
    const db = await openGlbDB();
    const tx  = db.transaction('glb','readwrite');
    tx.objectStore('glb').put({ key, data: arrayBuffer });
  } catch {}
}

// ─── ThreeAvatarRenderer ─────────────────────────────────────────────────────
class ThreeAvatarRenderer {
  constructor(canvas, options = {}) {
    this.canvas  = canvas;
    this.gender  = options.gender || 'male';
    this.speed   = options.speed  || 1.0;

    // Three.js objects
    this.renderer = null;
    this.scene    = null;
    this.camera   = null;
    this.model    = null;
    this.mixer    = null;
    this.bones    = {};       // key → THREE.Bone
    this.restQ    = {};       // key → THREE.Quaternion (rest pose)
    this.clock    = new THREE.Clock();

    // Sign queue playback
    this._queue   = [];
    this._signIdx = 0;
    this._frameIdx= 0;
    this._rafId   = null;
    this._onGlossChange = null;  // callback(idx)
    this._onDone        = null;  // callback()

    // State
    this.ready    = false;
    this.loading  = false;
  }

  // ── Init Three.js scene ────────────────────────────────────────────────────
  initScene() {
    const W = this.canvas.parentElement?.clientWidth  || 320;
    const H = this.canvas.parentElement?.clientHeight || 180;

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(W, H);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = false;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.renderer.setClearColor(0x06080f, 1);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x06080f);
    // Subtle fog for depth
    this.scene.fog = new THREE.Fog(0x06080f, 8, 20);

    // Camera — orthographic-ish perspective, framed on upper body
    this.camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 100);
    this.camera.position.set(0, 1.35, 3.2);
    this.camera.lookAt(0, 1.1, 0);

    // Lighting
    const ambient = new THREE.AmbientLight(0x5eead4, 0.4);
    this.scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(1.5, 3, 2);
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x0e7c6b, 0.35);
    fill.position.set(-2, 1, 1);
    this.scene.add(fill);

    const rim = new THREE.DirectionalLight(0x1e3a5f, 0.5);
    rim.position.set(0, -1, -3);
    this.scene.add(rim);

    // Render loop (runs even without animation — keeps scene live)
    const loop = () => {
      this._rafId = requestAnimationFrame(loop);
      const delta = this.clock.getDelta();
      if (this.mixer) this.mixer.update(delta);
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  // ── Load GLB ──────────────────────────────────────────────────────────────
  async load(onProgress) {
    if (this.loading || this.ready) return;
    this.loading = true;

    const url   = AVATAR_CDN[this.gender];
    const cacheKey = `glb-${this.gender}`;

    // 1. Try IDB cache
    let arrayBuffer = await glbCacheGet(cacheKey);

    // 2. Fetch from GitHub CDN
    if (!arrayBuffer) {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const total = parseInt(res.headers.get('content-length') || '0', 10);
        const reader = res.body.getReader();
        const chunks = [];
        let received = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.length;
          if (onProgress && total) onProgress(received / total);
        }

        const blob = new Blob(chunks);
        arrayBuffer = await blob.arrayBuffer();
        glbCacheSet(cacheKey, arrayBuffer);
      } catch (err) {
        console.error('[Signlytic 3D] GLB fetch failed:', err);
        this.loading = false;
        return false;
      }
    } else {
      onProgress && onProgress(1);
    }

    // 3. Parse with GLTFLoader
    return new Promise((resolve) => {
      const loader = new THREE.GLTFLoader();
      loader.parse(arrayBuffer, '', (gltf) => {
        this._onGLTFLoaded(gltf);
        resolve(true);
      }, (err) => {
        console.error('[Signlytic 3D] GLTFLoader parse error:', err);
        resolve(false);
      });
    });
  }

  // ── GLTF loaded handler ────────────────────────────────────────────────────
  _onGLTFLoaded(gltf) {
    this.model = gltf.scene;

    // Scale and position — Mixamo avatars are typically very tall
    // Auto-scale: measure model height and fit to ~1.8 units
    const autoBox = new THREE.Box3().setFromObject(this.model);
    const autoH = autoBox.max.y - autoBox.min.y;
    const targetH = 1.8;
    const sc = autoH > 0.01 ? targetH / autoH : 1.0;
    this.model.scale.setScalar(sc);
    // Position after scaling: feet on y=0
    const posBox = new THREE.Box3().setFromObject(this.model);
    this.model.position.y = -posBox.min.y;
    this.model.rotation.y = 0; // face forward

    this.scene.add(this.model);

    // Build bone map — match by name prefix (isBone unreliable in r128 GLTFLoader)
    const prefix = BONE_PREFIX[this.gender];
    this.model.traverse(obj => {
      if (!obj.name || !obj.name.startsWith(prefix)) return;
      const shortName = obj.name.replace(prefix, '');
      for (const [key, bname] of Object.entries(BONE_NAMES)) {
        if (bname === shortName) {
          this.bones[key] = obj;
          this.restQ[key] = obj.quaternion.clone();
          break;
        }
      }
    });

    const boneCount = Object.keys(this.bones).length;
    console.log(`[Signlytic 3D] Loaded ${this.gender} avatar. Bones mapped: ${boneCount}/${Object.keys(BONE_NAMES).length}`);

    this.loading = false;
    this.ready   = true;
  }

  // ── Apply one pose frame to skeleton ──────────────────────────────────────
  // frame: { body: [[x,y,z]×33], lh: [[x,y,z]×21], rh: [[x,y,z]×21] }
  applyFrame(frame) {
    if (!this.ready || !frame) return;
    if (frame.body) this._drivePose(frame.body);
    if (frame.lh)   this._driveHand(frame.lh,  'r');  // mirror: left data -> right bone
    if (frame.rh)   this._driveHand(frame.rh,  'l');  // mirror: right data -> left bone
  }

  // ── Reset to T-pose ────────────────────────────────────────────────────────
  resetPose() {
    if (!this.ready) return;
    for (const [key, bone] of Object.entries(this.bones)) {
      if (this.restQ[key]) bone.quaternion.copy(this.restQ[key]);
    }
  }

  // ── Upper body pose driving ────────────────────────────────────────────────
  _drivePose(body) {
    // Helper: convert normalised MediaPipe [x,y,z] to THREE.Vector3
    // MediaPipe: x=right, y=down, z=depth. Three.js: x=right, y=up, z=toward camera
    const lm = (i) => {
      if (!body[i]) return null;
      return new THREE.Vector3(
        -(body[i][0] * 2 - 1),  // negate X to mirror (image-space to avatar-space)
        -(body[i][1] * 2 - 1),  // flip Y
        -(body[i][2] || 0)      // negate Z (depth toward camera)
      );
    };

    const lShoulder = lm(MP.L_SHOULDER);
    const rShoulder = lm(MP.R_SHOULDER);
    const lElbow    = lm(MP.L_ELBOW);
    const rElbow    = lm(MP.R_ELBOW);
    const lWrist    = lm(MP.L_WRIST);
    const rWrist    = lm(MP.R_WRIST);

    // ── Left arm (driven by RIGHT landmarks - mirror for face-to-face) ──
    if (rShoulder && rElbow && this.bones.lArm) {
      this._driveSegment('lArm', rShoulder, rElbow, new THREE.Vector3(-1, 0, 0));
    }
    if (rElbow && rWrist && this.bones.lForeArm) {
      this._driveSegment('lForeArm', rElbow, rWrist, new THREE.Vector3(-1, 0, 0));
    }

    // ── Right arm (driven by LEFT landmarks - mirror for face-to-face) ──
    if (lShoulder && lElbow && this.bones.rArm) {
      this._driveSegment('rArm', lShoulder, lElbow, new THREE.Vector3(1, 0, 0));
    }
    if (lElbow && lWrist && this.bones.rForeArm) {
      this._driveSegment('rForeArm', lElbow, lWrist, new THREE.Vector3(1, 0, 0));
    }

    // ── Spine lean (from shoulder midpoint vs hip midpoint) ──
    const lHip = lm(MP.L_HIP);
    const rHip = lm(MP.R_HIP);
    if (lShoulder && rShoulder && lHip && rHip && this.bones.spine1) {
      const shoulderMid = lShoulder.clone().add(rShoulder).multiplyScalar(0.5);
      const hipMid      = lHip.clone().add(rHip).multiplyScalar(0.5);
      this._driveSegment('spine1', hipMid, shoulderMid, new THREE.Vector3(0, 1, 0));
    }
  }

  // ── Drive a single bone segment toward a target direction ──────────────────
  // boneName:  key in this.bones
  // from, to:  THREE.Vector3 world positions of parent/child joints
  // restDir:   the bone's rest direction in local parent space (e.g. arm points left)
  _driveSegment(boneName, from, to, restDir) {
    const bone = this.bones[boneName];
    if (!bone) return;

    // Target direction (world space)
    const targetDir = to.clone().sub(from).normalize();
    if (targetDir.length() < 0.001) return;

    // World-space rotation from rest direction to target direction
    const worldQ = new THREE.Quaternion().setFromUnitVectors(restDir.normalize(), targetDir);

    // Convert to local space: localQ = parentWorldQ⁻¹ × worldQ × restQ
    const parentWorldQ = new THREE.Quaternion();
    if (bone.parent) {
      bone.parent.getWorldQuaternion(parentWorldQ);
    }

    const localQ = parentWorldQ.clone().invert()
      .multiply(worldQ)
      .multiply(this.restQ[boneName] || new THREE.Quaternion());

    // Smooth interpolation (slerp 0.6 for responsiveness without jitter)
    bone.quaternion.slerp(localQ, 0.6);
  }

  // ── Hand landmark → finger bone driving ───────────────────────────────────
  // side: 'l' | 'r'
  // hand: [[x,y,z] × 21] MediaPipe hand landmarks
  _driveHand(hand, side) {
    if (!hand || hand.length < 21) return;

    const FINGERS = {
      Index:  { key: `${side}Index`,  lms: MH.INDEX  },
      Middle: { key: `${side}Middle`, lms: MH.MIDDLE },
      Ring:   { key: `${side}Ring`,   lms: MH.RING   },
      Pinky:  { key: `${side}Pinky`,  lms: MH.PINKY  },
      Thumb:  { key: `${side}Thumb`,  lms: MH.THUMB  },
    };

    // Rest direction for each finger phalanx — points along finger axis
    // For left hand: +x is toward fingertip; right hand: -x
    const fingerAxis = side === 'l'
      ? new THREE.Vector3(1, 0, 0)
      : new THREE.Vector3(-1, 0, 0);

    for (const [, finger] of Object.entries(FINGERS)) {
      const lms = finger.lms;
      for (let phalanx = 0; phalanx < 3; phalanx++) {
        const boneKey  = `${finger.key}${phalanx + 1}`; // e.g. lIndex1
        const fromIdx  = lms[phalanx];
        const toIdx    = lms[phalanx + 1];

        if (!hand[fromIdx] || !hand[toIdx]) continue;

        const from = new THREE.Vector3(
          -(hand[fromIdx][0] * 2 - 1),
          -(hand[fromIdx][1] * 2 - 1),
          -(hand[fromIdx][2] || 0)
        );
        const to = new THREE.Vector3(
          -(hand[toIdx][0] * 2 - 1),
          -(hand[toIdx][1] * 2 - 1),
          -(hand[toIdx][2] || 0)
        );

        this._driveSegment(boneKey, from, to, fingerAxis.clone());
      }
    }
  }

  // ── Sign queue playback ────────────────────────────────────────────────────
  // queue: [{ gloss, frames: [{ body, lh, rh }, ...] }, ...]
  playQueue(queue, onGlossChange, onDone) {
    this.stopQueue();
    this._queue         = queue;
    this._signIdx       = 0;
    this._frameIdx      = 0;
    this._onGlossChange = onGlossChange;
    this._onDone        = onDone;
    this._scheduleNextFrame();
  }

  stopQueue() {
    clearTimeout(this._playTimer);
    this._playTimer = null;
    this._queue = [];
    this.resetPose();
  }

  _scheduleNextFrame() {
    const FPS = 25 * (this.speed || 1.0);
    this._playTimer = setTimeout(() => this._tick(), 1000 / FPS);
  }

  _tick() {
    if (!this._queue.length) { this._onDone && this._onDone(); return; }

    const sign = this._queue[this._signIdx];
    if (!sign) { this._onDone && this._onDone(); return; }

    // Notify gloss change
    if (this._frameIdx === 0 && this._onGlossChange) {
      this._onGlossChange(this._signIdx);
    }

    // Apply frame
    const frame = sign.frames[this._frameIdx];
    this.applyFrame(frame);

    this._frameIdx++;
    if (this._frameIdx >= sign.frames.length) {
      this._frameIdx = 0;
      this._signIdx++;
      if (this._signIdx >= this._queue.length) {
        this._onDone && this._onDone();
        this.resetPose();
        return;
      }
    }

    this._scheduleNextFrame();
  }

  // ── Change gender (reloads GLB) ────────────────────────────────────────────
  async changeGender(gender, onProgress) {
    if (gender === this.gender && this.ready) return;
    this.gender = gender;
    this.ready  = false;
    this.bones  = {};
    this.restQ  = {};

    if (this.model) {
      this.scene.remove(this.model);
      this.model = null;
    }

    return this.load(onProgress);
  }

  // ── Resize handler ────────────────────────────────────────────────────────
  resize(w, h) {
    if (!this.renderer || !this.camera) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ── Destroy ───────────────────────────────────────────────────────────────
  destroy() {
    this.stopQueue();
    if (this._rafId) cancelAnimationFrame(this._rafId);
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer = null;
    }
    this.scene  = null;
    this.camera = null;
    this.model  = null;
  }
}

// Export to window for overlay.js
window.ThreeAvatarRenderer = ThreeAvatarRenderer;
