// overlay/avatar3d.js ÔÇö Signlytic AI Extension ÔÇö 3D Avatar Renderer
// Depends on THREE (r128) and THREE.GLTFLoader loaded before this script.
//
// Bone prefix:
//   Male avatar   ÔåÆ  mixamorig9:
//   Female avatar ÔåÆ  mixamorig8:
//
// Pipeline per frame:
//   MediaPipe Holistic landmarks ÔåÆ compute limb vectors ÔåÆ
//   convert to bone-local quaternions ÔåÆ apply to skeleton bones

// ÔöÇÔöÇÔöÇ CDN / GitHub paths ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
// Served through the Signlytic API rather than as public release assets. The
// avatar models are Mixamo characters and are not licensed for
// redistribution, so they are not offered as standalone downloads; the API
// keeps them in a private store and streams them to the app.
// Bump whenever the model files themselves change. Two caches would otherwise
// pin the old model indefinitely: the GLB is stored in IndexedDB with no expiry
// and checked before the network, and the endpoint serves it with
// "immutable, max-age=31536000". Without a new key and a new URL, anyone who
// already loaded a model would keep it for a year.
const AVATAR_VERSION = '2';

const AVATAR_CDN = {
  male:   `https://signlytic-ai-website.vercel.app/api/avatar/male?v=${AVATAR_VERSION}`,
  female: `https://signlytic-ai-website.vercel.app/api/avatar/female?v=${AVATAR_VERSION}`,
};

// Mixamo exports do not agree on a prefix. The male rig here uses
// "mixamorigHips" and the female "mixamorig8Hips". A hardcoded "mixamorig"
// still matched the female names, but stripping it left "8LeftArm", which
// equals no entry in BONE_NAMES, so not one bone was mapped. The avatar
// loaded, reported itself ready, and then stood still through every sign.
// Read the prefix off the rig rather than assuming it.
function detectBonePrefix(root) {
  let prefix = null;
  root.traverse(o => {
    if (prefix !== null || !o.name) return;
    const m = /^(.*?)Hips$/.exec(o.name);
    if (m) prefix = m[1];
  });
  return prefix;
}

// Depth scaling for body landmarks.
// BSL is signed in the space in front of the chest, so this is what lets the
// hands come forward instead of collapsing onto the torso plane. 0 reproduces
// the old flat behaviour, where hands could only reach the chest by
// intersecting it.
// Scale 2.0 is the physically consistent value (MediaPipe z is roughly the
// scale of its raw x, and raw x is doubled by the x * 2 - 1 mapping below),
// but it drives the forearm to ~0.97 along the view axis, which foreshortens
// the arm into a stub for a head-on camera. 0.75 keeps the hands clearly in
// front of the chest while the forearm stays readable.
const BODY_Z_SCALE = 0.75;

// Solve the wrist to a position rather than pointing the bones along a
// direction. Direction-only retargeting takes limb LENGTH from the rig, so the
// wrist lands at shoulder + rigArmLength * direction and can only reach where
// the avatar's proportions allow. In BSL the location of the hand carries
// meaning, a sign at the chin is not the same sign at the chest, so the wrist
// has to arrive where the capture puts it.
const ARM_IK = true;

// The capture is not in metric units. Its torso measures 1.79 shoulder-widths
// where an adult, and this rig, measure about 1.34, so vertical offsets are
// stretched by roughly half as much again. Mapping through TORSO fractions
// rather than raw shoulder-widths removes that, and needs no magic number:
// the ratio is measured from the rig and from each frame.
//
// Depth gets an explicit gain because MediaPipe's z from a single camera is
// only loosely scaled. 0.38 puts the wrist a median 0.7 shoulder-widths in
// front of the chest, which is where the previous direction-driven code landed
// and is anatomically sensible for signing space.
const IK_Z_GAIN = 0.22;

// ─── Pose normalisation ──────────────────────────────────────────────────────
// Raw sign capture data is unreliable: across a 250-sign sample, 62% of signs
// never lift the wrist above the shoulder and 27% of frames place it below the
// bottom of the source video (y > 1). Consumed raw, the arms hang at hip height.
// This repairs each frame before it drives anything: re-anchor on the shoulder
// midpoint, rescale by shoulder width, clamp the extremities back into signing
// space, fill dropouts from the previous frame, and smooth temporally.
// Stateful across frames, so each consumer owns its own instance.
const POSE_NORM = {
  HAND_SMOOTH:     0.4,   // 0 = no smoothing, 1 = frozen
  BODY_EXT_SMOOTH: 0.6,   // smoothing for body landmarks 15-22 (wrists/fingers)
  BODY_EXT_START:  15,    // first extremity landmark index
  BODY_EXT_END:    22,    // last extremity landmark index
  TARGET_W:        0.18,  // shoulder width is normalised to this
  TARGET_X:        0.5,
  TARGET_Y:        0.35,
  MAX_EXT_DIST:    0.35,  // max extremity distance from the shoulder midpoint
};

class PoseNormaliser {
  constructor() {
    this.prevBody = null;
    this.prevLH   = null;
    this.prevRH   = null;
  }

  reset() {
    this.prevBody = null;
    this.prevLH   = null;
    this.prevRH   = null;
  }

  normalise(frame) {
    if (!frame || !frame.body) return frame;
    const body = frame.body;
    const lShoulder = body[11];
    const rShoulder = body[12];
    if (!lShoulder || !rShoulder) return frame;

    const anchorX = (lShoulder[0] + rShoulder[0]) / 2;
    const anchorY = (lShoulder[1] + rShoulder[1]) / 2;

    const shoulderW = Math.abs(lShoulder[0] - rShoulder[0]);
    if (shoulderW < 0.01) return frame;
    const scale = POSE_NORM.TARGET_W / shoulderW;

    // Z is scaled by the same factor as X/Y. The 2D renderer ignores Z, but the
    // avatar drives depth from it, and leaving it unscaled would make the depth
    // of a sign depend on how far the signer stood from the camera.
    const normLms = (lms) => {
      if (!lms) return null;
      return lms.map(lm => lm ? [
        POSE_NORM.TARGET_X + (lm[0] - anchorX) * scale,
        POSE_NORM.TARGET_Y + (lm[1] - anchorY) * scale,
        (lm[2] || 0) * scale,
      ] : lm);
    };

    // Pull outlier finger landmarks back toward the hand centroid
    const clampHand = (lms) => {
      if (!lms || lms.length < 5) return lms;
      const anchors = [0, 5, 9, 13, 17].map(i => lms[i]).filter(Boolean);
      if (anchors.length < 3) return lms;
      const cx = anchors.reduce((s, p) => s + p[0], 0) / anchors.length;
      const cy = anchors.reduce((s, p) => s + p[1], 0) / anchors.length;
      const avgR = anchors.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / anchors.length;
      const maxR = Math.max(avgR * 2.5, 0.02);
      return lms.map(lm => {
        if (!lm) return lm;
        const d = Math.hypot(lm[0] - cx, lm[1] - cy);
        if (d > maxR) {
          const ratio = maxR / d;
          return [cx + (lm[0] - cx) * ratio, cy + (lm[1] - cy) * ratio, lm[2] || 0];
        }
        return lm;
      });
    };

    const smoothHand = (cur, prev) => {
      if (!cur || !prev || prev.length !== cur.length) return cur;
      const a = POSE_NORM.HAND_SMOOTH;
      return cur.map((lm, i) => (lm && prev[i]) ? [
        lm[0] * (1 - a) + prev[i][0] * a,
        lm[1] * (1 - a) + prev[i][1] * a,
        (lm[2] || 0) * (1 - a) + (prev[i][2] || 0) * a,
      ] : lm);
    };

    let normBody = normLms(frame.body);
    let normLH   = smoothHand(clampHand(normLms(frame.lh)), this.prevLH);
    let normRH   = smoothHand(clampHand(normLms(frame.rh)), this.prevRH);
    this.prevLH = normLH;
    this.prevRH = normRH;

    const S = POSE_NORM.BODY_EXT_START;
    const E = POSE_NORM.BODY_EXT_END;

    // Fill dropped extremity points from the previous frame
    if (normBody && this.prevBody) {
      for (let i = S; i <= E && i < normBody.length; i++) {
        if (!normBody[i] && this.prevBody[i]) normBody[i] = this.prevBody[i].slice();
      }
    }
    if (normLH && this.prevLH) {
      normLH = normLH.map((lm, i) => (!lm && this.prevLH[i]) ? this.prevLH[i].slice() : lm);
    }
    if (normRH && this.prevRH) {
      normRH = normRH.map((lm, i) => (!lm && this.prevRH[i]) ? this.prevRH[i].slice() : lm);
    }

    // Clamp extremities into signing space around the torso. This is what pulls
    // off-frame wrists back up in front of the chest.
    if (normBody) {
      const midX = (normBody[11] && normBody[12]) ? (normBody[11][0] + normBody[12][0]) / 2 : POSE_NORM.TARGET_X;
      const midY = (normBody[11] && normBody[12]) ? (normBody[11][1] + normBody[12][1]) / 2 : POSE_NORM.TARGET_Y;
      for (let i = S; i <= E && i < normBody.length; i++) {
        if (!normBody[i]) continue;
        const dx = normBody[i][0] - midX;
        const dy = normBody[i][1] - midY;
        const d = Math.hypot(dx, dy);
        if (d > POSE_NORM.MAX_EXT_DIST) {
          const ratio = POSE_NORM.MAX_EXT_DIST / d;
          normBody[i] = [midX + dx * ratio, midY + dy * ratio, normBody[i][2] || 0];
        }
      }
    }

    // Temporal smoothing on the extremities (wrists/fingers), which is what
    // makes the noisy depth channel usable instead of having to discard it.
    if (normBody && this.prevBody && this.prevBody.length === normBody.length) {
      const a = POSE_NORM.BODY_EXT_SMOOTH;
      for (let i = S; i <= E && i < normBody.length; i++) {
        if (normBody[i] && this.prevBody[i]) {
          normBody[i] = [
            normBody[i][0] * (1 - a) + this.prevBody[i][0] * a,
            normBody[i][1] * (1 - a) + this.prevBody[i][1] * a,
            (normBody[i][2] || 0) * (1 - a) + (this.prevBody[i][2] || 0) * a,
          ];
        }
      }
    }
    this.prevBody = normBody;

    return { body: normBody, lh: normLH, rh: normRH };
  }
}

// MediaPipe Holistic body landmark indices (upper body only ÔÇö BSL relevant)
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
  lUpLeg:      'LeftUpLeg',
  rUpLeg:      'RightUpLeg',
  lShoulder:   'LeftShoulder',
  lArm:        'LeftArm',
  lForeArm:    'LeftForeArm',
  lHand:       'LeftHand',
  rShoulder:   'RightShoulder',
  rArm:        'RightArm',
  rForeArm:    'RightForeArm',
  rHand:       'RightHand',
  // Left hand fingers (Index, Middle, Ring, Pinky, Thumb ÔÇö 3 bones each)
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

// ÔöÇÔöÇÔöÇ IDB helpers for GLB blob caching ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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

// Removes models cached under an older scheme or version, so bumping the
// version does not simply accumulate copies in the user's browser.
async function glbCacheDeleteLegacy(gender) {
  try {
    const db = await openGlbDB();
    const tx = db.transaction('glb', 'readwrite');
    const store = tx.objectStore('glb');
    store.delete(`glb-${gender}`);
    for (let v = 1; v < Number(AVATAR_VERSION); v++) {
      store.delete(`glb-${gender}-v${v}`);
    }
  } catch {}
}

async function glbCacheSet(key, arrayBuffer) {
  try {
    const db = await openGlbDB();
    const tx  = db.transaction('glb','readwrite');
    tx.objectStore('glb').put({ key, data: arrayBuffer });
  } catch {}
}

// ÔöÇÔöÇÔöÇ ThreeAvatarRenderer ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
class ThreeAvatarRenderer {
  constructor(canvas, options = {}) {
    this.canvas  = canvas;
    this.gender  = options.gender || 'male';
    this.speed   = options.speed  || 1.0;
    // Optional override for hosts that must serve the GLB same-origin
    // (browser pages cannot fetch GitHub release assets cross-origin)
    this.modelUrl = options.modelUrl || null;
    // Draw on a clear background so the host can sit over other windows.
    // The desktop overlay needs this; the in-page panel does not.
    this.transparent = options.transparent === true;

    // Repairs raw capture data before it drives the skeleton. Per-instance
    // because it carries frame-to-frame state. Pass normalise:false only if
    // the caller has already normalised the frames it supplies.
    this._normaliser = options.normalise === false ? null : new PoseNormaliser();

    // Three.js objects
    this.renderer = null;
    this.scene    = null;
    this.camera   = null;
    this.model    = null;
    this.mixer    = null;
    this.bones    = {};       // key ÔåÆ THREE.Bone
    this.restQ    = {};       // key ÔåÆ THREE.Quaternion (rest pose)
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

  // ÔöÇÔöÇ Init Three.js scene ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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
    this.renderer.setClearColor(0x06080f, this.transparent ? 0 : 1);

    this.scene = new THREE.Scene();
    if (!this.transparent) {
      this.scene.background = new THREE.Color(0x06080f);
      // Subtle fog for depth. Skipped when transparent: fog blends toward a
      // background colour that is not being drawn, which greys out the avatar.
      this.scene.fog = new THREE.Fog(0x06080f, 8, 20);
    }

    // Camera ÔÇö orthographic-ish perspective, framed on upper body
    this.camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 100);
    this.camera.position.set(0, 1.55, 2.2);
    this.camera.lookAt(0, 1.3, 0);

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

    // Render loop (runs even without animation ÔÇö keeps scene live)
    const loop = () => {
      this._rafId = requestAnimationFrame(loop);
      const delta = this.clock.getDelta();
      if (this.mixer) this.mixer.update(delta);
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  // ÔöÇÔöÇ Load GLB ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  async load(onProgress) {
    if (this.loading || this.ready) return;
    this.loading = true;

    const url   = this.modelUrl || AVATAR_CDN[this.gender];
    const cacheKey = `glb-${this.gender}-v${AVATAR_VERSION}`;

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
        // The superseded model is dead weight, and it is large.
        // Drop it once its replacement is safely stored.
        glbCacheDeleteLegacy(this.gender);
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

  // ÔöÇÔöÇ GLTF loaded handler ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  _onGLTFLoaded(gltf) {
    this.model = gltf.scene;

    // Scale and position ÔÇö Mixamo avatars are typically very tall
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

    // Build bone map ÔÇö match by name prefix (isBone unreliable in r128 GLTFLoader)
    const prefix = detectBonePrefix(this.model);
    if (prefix === null) {
      console.warn('[Signlytic 3D] No bone named *Hips found; this avatar cannot be posed.');
    }
    this.model.traverse(obj => {
      if (prefix === null || !obj.name || !obj.name.startsWith(prefix)) return;
      const shortName = obj.name.slice(prefix.length);
      for (const [key, bname] of Object.entries(BONE_NAMES)) {
        if (bname === shortName) {
          this.bones[key] = obj;
          this.restQ[key] = obj.quaternion.clone();
          break;
        }
      }
    });

    // Compute actual rest world directions from T-pose (Y-axis = bone pointing direction)
    // Ported from commit 4a6fa65: hardcoded rest axes do not match the Mixamo rig.
    // Rest directions are kept in full 3D: the landmark targets carry real depth
    // (see BODY_Z_SCALE), and rest space must match target space or every frame
    // picks up a spurious out-of-plane rotation.
    this.restDir = {};
    this.restWorldQ = {};
    for (const [key, bone] of Object.entries(this.bones)) {
      const wq = new THREE.Quaternion();
      bone.getWorldQuaternion(wq);
      // Bone's Y-axis in world space = the direction it points in T-pose
      this.restDir[key] = new THREE.Vector3(0, 1, 0).applyQuaternion(wq).normalize();
      // The full rest orientation, needed to place a bone whose parent has
      // itself been rotated this frame. See _driveSegment.
      this.restWorldQ[key] = wq.clone();
    }

    // Rig proportions, measured once from the T-pose. Segment lengths are fixed
    // by the bone hierarchy, and shoulder width and torso length are the
    // yardsticks the capture is mapped onto.
    {
      const wpos = (b) => { const v = new THREE.Vector3(); b.getWorldPosition(v); return v; };
      const B = this.bones;
      if (B.lArm && B.rArm && B.lForeArm && B.lHand && B.rForeArm && B.rHand) {
        const lS = wpos(B.lArm), rS = wpos(B.rArm);
        const shW = lS.distanceTo(rS);
        // Torso is measured shoulder joints to HIP JOINTS, matching MediaPipe's
        // hip landmarks 23 and 24. The Hips bone sits at the pelvis centre, a
        // little above the joints, and using it made the rig's torso read short,
        // which placed every hand slightly high.
        let torso = shW * 1.34;   // fallback if the leg bones are absent
        if (this.bones.lUpLeg && this.bones.rUpLeg) {
          const hipMidY = (wpos(this.bones.lUpLeg).y + wpos(this.bones.rUpLeg).y) / 2;
          torso = Math.abs(lS.clone().add(rS).multiplyScalar(0.5).y - hipMidY);
        } else if (this.bones.hips) {
          torso = Math.abs(lS.clone().add(rS).multiplyScalar(0.5).y - wpos(this.bones.hips).y);
        }
        this.rig = {
          shW, torso,
          lUpper: lS.distanceTo(wpos(B.lForeArm)),
          lFore:  wpos(B.lForeArm).distanceTo(wpos(B.lHand)),
          rUpper: rS.distanceTo(wpos(B.rForeArm)),
          rFore:  wpos(B.rForeArm).distanceTo(wpos(B.rHand)),
        };
      }
    }

    const boneCount = Object.keys(this.bones).length;
    console.log(`[Signlytic 3D] Loaded ${this.gender} avatar. Bones mapped: ${boneCount}/${Object.keys(BONE_NAMES).length}`);

    // -- Post-load: measure, scale, center, frame camera --
    const bbox = new THREE.Box3().setFromObject(this.model);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    console.log('[Signlytic 3D] Raw size - W:', size.x.toFixed(2), 'H:', size.y.toFixed(2), 'D:', size.z.toFixed(2));

    // Auto-scale to ~1.7m if needed
    if (size.y > 0.001 && size.y < 1.0) {
      const sf = 1.7 / size.y;
      this.model.scale.multiplyScalar(sf);
      this.model.updateMatrixWorld(true);
      console.log('[Signlytic 3D] Scaled by', sf.toFixed(1));
    }

    // Re-measure after scale
    const b2 = new THREE.Box3().setFromObject(this.model);
    const s2 = new THREE.Vector3();
    const c2 = new THREE.Vector3();
    b2.getSize(s2);
    b2.getCenter(c2);

    // Center model and put feet on ground
    this.model.position.x -= c2.x;
    this.model.position.z -= c2.z;
    this.model.position.y -= b2.min.y;

    // Frame camera: upper body (chest to head + hands visible)
    const targetY = s2.y * 0.72;
    const camZ = s2.y * 0.95;
    this.camera.position.set(0, targetY, camZ);
    this.camera.lookAt(0, targetY, 0);
    this.camera.updateProjectionMatrix();

    console.log('[Signlytic 3D] Final H:', s2.y.toFixed(2), 'W:', s2.x.toFixed(2));
    console.log('[Signlytic 3D] Camera at (0,', targetY.toFixed(2) + ',', camZ.toFixed(2) + ')');

    this.loading = false;
    this.ready   = true;
  }

  // ÔöÇÔöÇ Apply one pose frame to skeleton ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  // frame: { body: [[x,y,z]├ù33], lh: [[x,y,z]├ù21], rh: [[x,y,z]├ù21] }
  applyFrame(frame) {
    if (!this.ready || !frame) return;
    if (this._normaliser) frame = this._normaliser.normalise(frame);
    if (!frame) return;
    if (frame.body) this._drivePose(frame.body);
    if (frame.lh)   this._driveHand(frame.lh,  'r');  // mirror: left data -> right bone
    if (frame.rh)   this._driveHand(frame.rh,  'l');  // mirror: right data -> left bone
  }

  // ÔöÇÔöÇ Reset to T-pose ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  resetPose() {
    if (!this.ready) return;
    // Reset all bones to rest first
    for (const [key, bone] of Object.entries(this.bones)) {
      if (this.restQ[key]) bone.quaternion.copy(this.restQ[key]);
    }
    // Then drive arms to a natural resting position (slightly forward, at sides)
    // so the idle pose is arms-down rather than a raw T-pose. Ported from 4a6fa65.
    if (this.restDir) {
      const downL = new THREE.Vector3(0.15, -0.9, 0.1).normalize();
      const downR = new THREE.Vector3(-0.15, -0.9, 0.1).normalize();
      const idlePairs = [
        ['lArm', downL], ['lForeArm', downL],
        ['rArm', downR], ['rForeArm', downR],
      ];
      for (const [boneName, targetDir] of idlePairs) {
        const bone    = this.bones[boneName];
        const restDir = this.restDir[boneName];
        if (!bone || !restDir) continue;
        // Same local-space conversion as _driveSegment, but applied directly
        // instead of slerped: the idle pose should snap, not ease in.
        const deltaQ = new THREE.Quaternion().setFromUnitVectors(restDir.clone().normalize(), targetDir);
        const parentWorldQ = new THREE.Quaternion();
        if (bone.parent) bone.parent.getWorldQuaternion(parentWorldQ);
        const restWorldQ = (this.restWorldQ && this.restWorldQ[boneName])
          || parentWorldQ.clone().multiply(this.restQ[boneName] || new THREE.Quaternion());
        const localQ = parentWorldQ.clone().invert()
          .multiply(deltaQ)
          .multiply(restWorldQ);
        bone.quaternion.copy(localQ);
      }
    }
  }

  // ÔöÇÔöÇ Upper body pose driving ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  _drivePose(body) {
    // Helper: convert normalised MediaPipe [x,y,z] to THREE.Vector3
    // MediaPipe: x=right, y=down, z=depth. Three.js: x=right, y=up, z=toward camera
    const lm = (i) => {
      if (!body[i]) return null;
      // Landmark space must match the rest directions computed at load.
      return new THREE.Vector3(
        (body[i][0] * 2 - 1),   // X not negated: the arm bone swap below
                                // already mirrors, negating here double-mirrors
        -(body[i][1] * 2 - 1),  // flip Y
        -(body[i][2] || 0) * BODY_Z_SCALE  // MediaPipe -z is toward the camera
      );
    };

    const lShoulder = lm(MP.L_SHOULDER);
    const rShoulder = lm(MP.R_SHOULDER);
    const lElbow    = lm(MP.L_ELBOW);
    const rElbow    = lm(MP.R_ELBOW);
    const lWrist    = lm(MP.L_WRIST);
    const rWrist    = lm(MP.R_WRIST);

    // ÔöÇÔöÇ Arms. Rig LEFT is driven by RIGHT landmarks, mirroring face-to-face ÔöÇÔöÇ
    const solved = ARM_IK && this.rig
      && this._solveArm('l', body, MP.R_SHOULDER, MP.R_ELBOW, MP.R_WRIST)
      && this._solveArm('r', body, MP.L_SHOULDER, MP.L_ELBOW, MP.L_WRIST);

    if (!solved) {
      // Fallback: point the bones along the captured directions. Reaches only
      // as far as the rig's own arm allows, but never fails.
      if (rShoulder && rElbow && this.bones.lArm) {
        this._driveSegment('lArm', rShoulder, rElbow, new THREE.Vector3(-1, 0, 0));
      }
      if (rElbow && rWrist && this.bones.lForeArm) {
        this._driveSegment('lForeArm', rElbow, rWrist, new THREE.Vector3(-1, 0, 0));
      }
      if (lShoulder && lElbow && this.bones.rArm) {
        this._driveSegment('rArm', lShoulder, lElbow, new THREE.Vector3(1, 0, 0));
      }
      if (lElbow && lWrist && this.bones.rForeArm) {
        this._driveSegment('rForeArm', lElbow, lWrist, new THREE.Vector3(1, 0, 0));
      }
    }

    // ÔöÇÔöÇ Spine lean (from shoulder midpoint vs hip midpoint) ÔöÇÔöÇ
    const lHip = lm(MP.L_HIP);
    const rHip = lm(MP.R_HIP);
    if (lShoulder && rShoulder && lHip && rHip && this.bones.spine1) {
      const shoulderMid = lShoulder.clone().add(rShoulder).multiplyScalar(0.5);
      const hipMid      = lHip.clone().add(rHip).multiplyScalar(0.5);
      this._driveSegment('spine1', hipMid, shoulderMid, new THREE.Vector3(0, 1, 0));
    }
  }

  // ÔöÇÔöÇ Map a captured landmark into the rig's own proportions ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  // Returns a world position, or null when the frame lacks the anchors.
  //
  // Horizontal offsets are carried in shoulder-widths and vertical ones in
  // torso lengths, each measured on the body they came from, so a capture with
  // different proportions to the avatar still lands in the right place on it.
  _landmarkToRig(body, idx, shoulderMidWorld) {
    const b = body[idx];
    const lS = body[MP.L_SHOULDER], rS = body[MP.R_SHOULDER];
    const lH = body[MP.L_HIP], rH = body[MP.R_HIP];
    if (!b || !lS || !rS || !lH || !rH) return null;

    const shW = Math.hypot(lS[0] - rS[0], lS[1] - rS[1]);
    if (!(shW > 1e-6)) return null;
    const midX = (lS[0] + rS[0]) / 2;
    const midY = (lS[1] + rS[1]) / 2;
    const midZ = ((lS[2] || 0) + (rS[2] || 0)) / 2;
    const torso = Math.abs((lH[1] + rH[1]) / 2 - midY);
    if (!(torso > 1e-6)) return null;

    return new THREE.Vector3(
      // negated: the rig's left arm is driven by right-hand landmarks
      -((b[0] - midX) / shW) * this.rig.shW + shoulderMidWorld.x,
      // landmark y grows downward; torso fractions carry the vertical
      -((b[1] - midY) / torso) * this.rig.torso + shoulderMidWorld.y,
      // MediaPipe -z points toward the camera, which is +z here
      -(((b[2] || 0) - midZ) / shW) * this.rig.shW * IK_Z_GAIN + shoulderMidWorld.z
    );
  }

  // ÔöÇÔöÇ Two-bone IK: put the wrist on its target, elbow led by the capture ÔöÇÔöÇ
  // side: 'l' | 'r' rig side. Returns false if the frame cannot be solved, so
  // the caller can fall back rather than leave the arm frozen.
  _solveArm(side, body, shIdx, elIdx, wrIdx) {
    const armBone  = this.bones[side + 'Arm'];
    const foreBone = this.bones[side + 'ForeArm'];
    const handBone = this.bones[side + 'Hand'];
    if (!armBone || !foreBone || !handBone) return false;

    const lArm = this.bones.lArm, rArm = this.bones.rArm;
    if (!lArm || !rArm) return false;
    const lP = new THREE.Vector3(), rP = new THREE.Vector3();
    lArm.getWorldPosition(lP); rArm.getWorldPosition(rP);
    const shoulderMid = lP.clone().add(rP).multiplyScalar(0.5);

    const target = this._landmarkToRig(body, wrIdx, shoulderMid);
    const elbowT = this._landmarkToRig(body, elIdx, shoulderMid);
    if (!target || !elbowT) return false;

    const S = (side === 'l' ? lP : rP);
    const L1 = this.rig[side + 'Upper'];
    const L2 = this.rig[side + 'Fore'];
    if (!(L1 > 1e-6) || !(L2 > 1e-6)) return false;

    // Reach limit. Pull the target in rather than let the solver fail, and stop
    // just short of full extension so the elbow never locks dead straight.
    const toT = target.clone().sub(S);
    let d = toT.length();
    if (!(d > 1e-6)) return false;
    const dMax = (L1 + L2) * 0.995;
    const dMin = Math.abs(L1 - L2) + 1e-4;
    if (d > dMax) { toT.multiplyScalar(dMax / d); d = dMax; }
    else if (d < dMin) { toT.multiplyScalar(dMin / d); d = dMin; }
    const T = S.clone().add(toT);

    // Elbow sits on a circle around the shoulder-to-wrist axis. The captured
    // elbow chooses where on that circle, so the avatar bends its arm the way
    // the signer did instead of by a fixed rule.
    const axis = toT.clone().divideScalar(d);
    const cosA = Math.max(-1, Math.min(1, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)));
    const along = L1 * cosA;
    const radius = L1 * Math.sqrt(Math.max(0, 1 - cosA * cosA));

    let perp = elbowT.clone().sub(S);
    perp.sub(axis.clone().multiplyScalar(perp.dot(axis)));
    if (perp.lengthSq() < 1e-8) {
      // Captured elbow lies on the axis, so it cannot pick a direction. Fall
      // back to pointing the elbow down and slightly back, as a human arm does.
      perp = new THREE.Vector3(0, -1, -0.3);
      perp.sub(axis.clone().multiplyScalar(perp.dot(axis)));
      if (perp.lengthSq() < 1e-8) perp = new THREE.Vector3(0, 0, -1);
    }
    perp.normalize();

    const E = S.clone().add(axis.clone().multiplyScalar(along)).add(perp.multiplyScalar(radius));

    // |E-S| is L1 and |T-E| is L2 by construction, so pointing each bone along
    // these two directions lands the wrist exactly on T.
    this._driveSegment(side + 'Arm', S, E, null);
    this._driveSegment(side + 'ForeArm', E, T, null);
    return true;
  }

  // ÔöÇÔöÇ Drive a single bone segment toward a target direction ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  // boneName:  key in this.bones
  // from, to:  THREE.Vector3 world positions of parent/child joints
  // restDirOverride: fallback rest direction, used only if no auto-detected one exists
  _driveSegment(boneName, from, to, restDirOverride) {
    const bone = this.bones[boneName];
    if (!bone) return;

    // Target direction (world space)
    const targetDir = to.clone().sub(from).normalize();
    if (targetDir.length() < 0.001) return;

    // Auto-detected rest direction from T-pose (bone Y-axis in world space)
    const restDir = (this.restDir && this.restDir[boneName]) || restDirOverride;
    if (!restDir) return;

    // Delta rotation: world-space rotation from rest direction to target direction
    const deltaQ = new THREE.Quaternion().setFromUnitVectors(restDir.clone().normalize(), targetDir);

    // We want the bone to end up at  deltaQ * restWorldQ, so its Y axis lands
    // on targetDir. Converting that to a local rotation:
    //
    //   boneWorldQ = parentWorldQ_now * localQ = deltaQ * restWorldQ
    //   localQ     = parentWorldQ_now^-1 * deltaQ * restWorldQ
    //
    // restWorldQ is captured from the T-pose at load. The earlier form put the
    // CURRENT parent orientation where the rest one belongs and folded in restQ
    // instead. Those agree only while the parent is still at rest, which holds
    // for an upper arm, whose parent shoulder is never driven, and fails for a
    // forearm, whose parent was rotated moments earlier in the same frame.
    // Measured on the rig, that left the forearm 72 degrees out whenever the
    // upper arm moved, and the forearm is what decides where the hand lands.
    const parentWorldQ = new THREE.Quaternion();
    if (bone.parent) {
      bone.parent.getWorldQuaternion(parentWorldQ);
    }

    const restWorldQ = (this.restWorldQ && this.restWorldQ[boneName])
      || parentWorldQ.clone().multiply(this.restQ[boneName] || new THREE.Quaternion());

    const localQ = parentWorldQ.clone().invert()
      .multiply(deltaQ)
      .multiply(restWorldQ);

    // Smooth interpolation (slerp 0.6 for responsiveness without jitter)
    bone.quaternion.slerp(localQ, 0.6);
  }

  // ÔöÇÔöÇ Hand landmark ÔåÆ finger bone driving ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  // side: 'l' | 'r'
  // hand: [[x,y,z] ├ù 21] MediaPipe hand landmarks
  _driveHand(hand, side) {
    if (!hand || hand.length < 21) return;

    const FINGERS = {
      Index:  { key: `${side}Index`,  lms: MH.INDEX  },
      Middle: { key: `${side}Middle`, lms: MH.MIDDLE },
      Ring:   { key: `${side}Ring`,   lms: MH.RING   },
      Pinky:  { key: `${side}Pinky`,  lms: MH.PINKY  },
      Thumb:  { key: `${side}Thumb`,  lms: MH.THUMB  },
    };

    // Rest direction for each finger phalanx ÔÇö points along finger axis
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

  // ÔöÇÔöÇ Sign queue playback ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  // queue: [{ gloss, frames: [{ body, lh, rh }, ...] }, ...]
  playQueue(queue, onGlossChange, onDone) {
    this.stopQueue();
    // Fresh sequence: do not carry smoothing state over from the last one
    if (this._normaliser) this._normaliser.reset();
    this._queue         = queue;
    this._signIdx       = 0;
    this._frameIdx      = 0;
    this._paused        = false;
    this._onGlossChange = onGlossChange;
    this._onDone        = onDone;
    this._scheduleNextFrame();
  }

  stopQueue() {
    clearTimeout(this._playTimer);
    this._playTimer = null;
    this._paused = false;
    this._queue = [];
    this.resetPose();
  }

  pause() {
    if (!this._playTimer || this._paused) return;
    clearTimeout(this._playTimer);
    this._playTimer = null;
    this._paused = true;
  }

  resume() {
    if (!this._paused || !this._queue.length) return;
    this._paused = false;
    this._scheduleNextFrame();
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

  // ÔöÇÔöÇ Change gender (reloads GLB) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  async changeGender(gender, onProgress) {
    if (gender === this.gender && this.ready) return;
    this.gender = gender;
    this.ready  = false;
    this.bones  = {};
    this.restQ  = {};
    // Everything below is measured from the rig being replaced, so carrying it
    // over would pose the new body with the old one's proportions.
    this.restDir    = null;
    this.restWorldQ = null;
    this.rig        = null;

    if (this.model) {
      this.scene.remove(this.model);
      this.model = null;
    }

    return this.load(onProgress);
  }

  // ÔöÇÔöÇ Resize handler ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  resize(w, h) {
    if (!this.renderer || !this.camera) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ÔöÇÔöÇ Destroy ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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
// Shared so the 2D renderer normalises through the same implementation
window.PoseNormaliser = PoseNormaliser;
