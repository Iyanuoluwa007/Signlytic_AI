// overlay.js — Signlytic AI Extension Overlay Controller (v0.2.0)
// Depends on: avatar3d.js (ThreeAvatarRenderer), Three.js r128, GLTFLoader

// ─── DOM refs ────────────────────────────────────────────────────────────────
const panel            = document.getElementById('panel');
const logoDot          = document.getElementById('logo-dot');
const statusDot        = document.getElementById('status-dot');
const statusText       = document.getElementById('status-text');
const glossRow         = document.getElementById('gloss-row');
const canvas2d         = document.getElementById('canvas-2d');
const placeholder      = document.getElementById('skeleton-placeholder');
const canvas3dCont     = document.getElementById('canvas-3d-container');
const avatarLoading    = document.getElementById('avatar-loading');
const loadingLabel     = document.getElementById('loading-label');
const loadingBar       = document.getElementById('loading-bar');
const genderBadge      = document.getElementById('gender-badge');
const btnMinimise      = document.getElementById('btn-minimise');
const btnClose         = document.getElementById('btn-close');
const btn2d            = document.getElementById('btn-2d');
const btn3d            = document.getElementById('btn-3d');
const header           = document.getElementById('header');

const ctx2d = canvas2d.getContext('2d');

// ─── State ────────────────────────────────────────────────────────────────────
let settings       = {};
let renderMode     = '2d';
let signQueue2d    = [];
let current2dSign  = 0;
let current2dFrame = 0;
let anim2dTimer    = null;
let glossTokenEls  = [];
let activeGlossIdx = -1;
let isMinimised    = false;

// 3D renderer instance
let avatarRenderer = null;
let avatarLoaded   = false;
// Smoothing state (EMA for body extremities + hands)
let prevLH = null;
let prevRH = null;
let prevBody = null;
const HAND_SMOOTH = 0.4;   // 0 = no smoothing, 1 = frozen
const BODY_EXT_SMOOTH = 0.6;   // smoothing for body landmarks 15-22 (wrists/fingers)
const BODY_EXT_START = 15;     // first extremity landmark index
const BODY_EXT_END = 22;       // last extremity landmark index

// ─── Settings ────────────────────────────────────────────────────────────────
function applySettings(s) {
  settings = s;
  renderMode = s.renderMode || '2d';

  // Tell parent content_script to snap iframe to requested corner
  try {
    window.parent.postMessage({ type: 'SET_POSITION', position: s.position || 'bottom-right' }, '*');
  } catch (_) {}

  const sizeClasses = ['size-small','size-medium','size-large'];
  panel.classList.remove(...sizeClasses);
  panel.classList.add(`size-${s.size || 'medium'}`);

  const widthMap = { small: 300, medium: 400, large: 520 };
  const w = widthMap[s.size || 'medium'];
  canvas2d.width = w;

  applyMode(renderMode, false); // don't reload avatar on every settings update

  if (avatarRenderer) avatarRenderer.speed = s.signSpeed || 1.0;
}

function applyMode(mode, forceReload = true) {
  renderMode = mode;

  if (mode === '2d') {
    btn2d.classList.add('active');
    btn3d.classList.remove('active');
    canvas2d.style.display   = 'block';
    canvas3dCont.style.display = 'none';
    genderBadge.style.display  = 'none';
    stop2dAnimation();
  } else {
    btn3d.classList.add('active');
    btn2d.classList.remove('active');
    canvas2d.style.display    = 'none';
    canvas3dCont.style.display = 'block';
    placeholder.style.display  = 'none';
    genderBadge.style.display  = 'block';
    genderBadge.textContent    = settings.avatarGender || 'male';
    if (forceReload || !avatarLoaded) load3DAvatar();
  }
}

// ─── 3D Avatar loading ────────────────────────────────────────────────────────
async function load3DAvatar(forceGender) {
  const gender = forceGender || settings.avatarGender || 'male';

  // Already loaded? Skip.
  if (avatarRenderer && avatarLoaded && avatarRenderer.gender === gender) {
    setStatus('listening', 'avatar ready');
    return;
  }

  // Initialise renderer if needed
  if (!avatarRenderer) {
    const canvas3d = document.getElementById('canvas-3d');
    avatarRenderer = new ThreeAvatarRenderer(canvas3d, {
      gender,
      speed: settings.signSpeed || 1.0,
    });
    avatarRenderer.initScene();
  }

  // Show loading state
  avatarLoading.classList.add('visible');
  loadingLabel.textContent = `loading ${gender} avatar...`;
  loadingBar.style.width   = '0%';
  setStatus('loading', `loading ${gender} avatar...`);

  const success = await avatarRenderer.load((progress) => {
    loadingBar.style.width = Math.round(progress * 100) + '%';
    loadingLabel.textContent = `loading ${gender} avatar... ${Math.round(progress * 100)}%`;
  });

  avatarLoading.classList.remove('visible');

  if (success) {
    avatarLoaded = true;
    genderBadge.textContent = gender;
    setStatus('listening', 'avatar ready');
  } else {
    setStatus('error', 'avatar load failed — check GitHub CDN');
    showAvatarError();
  }
}

function showAvatarError() {
  avatarLoading.classList.add('visible');
  loadingLabel.textContent = 'load failed — check README';
  loadingBar.style.width = '0%';
}

// ─── Status helpers ───────────────────────────────────────────────────────────
function setStatus(state, message) {
  statusDot.className = '';
  statusDot.classList.add(state);
  statusText.textContent = message;
  const isActive = state !== 'idle' && state !== 'error';
  logoDot.classList.toggle('inactive', !isActive);
}

// ─── Gloss rendering ──────────────────────────────────────────────────────────
function renderGlosses(glosses, activeIdx = -1) {
  glossRow.innerHTML = '';
  glossTokenEls = [];

  if (!glosses || glosses.length === 0) {
    glossRow.innerHTML = '<span class="gloss-empty">no glosses yet</span>';
    return;
  }

  glosses.forEach((g, i) => {
    const span = document.createElement('span');
    span.className = 'gloss-token' + (i === activeIdx ? ' active' : '');
    span.textContent = g;
    glossRow.appendChild(span);
    glossTokenEls.push(span);
  });
  activeGlossIdx = activeIdx;
}

function setActiveGloss(idx) {
  if (idx === activeGlossIdx) return;
  glossTokenEls.forEach((el, i) => el.classList.toggle('active', i === idx));
  activeGlossIdx = idx;
}

// ─── 2D Skeleton renderer ─────────────────────────────────────────────────────
const BODY_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
];
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];

// ─── Skeleton normalisation ──────────────────────────────────────────────────
// Normalise body landmarks so the torso is always the same size/position
// Uses shoulder midpoint as anchor and shoulder width as scale reference
function normaliseFrame(frame) {
  if (!frame?.body) return frame;
  const body = frame.body;

  const lShoulder = body[11];
  const rShoulder = body[12];
  if (!lShoulder || !rShoulder) return frame;

  // Anchor: midpoint between shoulders
  const anchorX = (lShoulder[0] + rShoulder[0]) / 2;
  const anchorY = (lShoulder[1] + rShoulder[1]) / 2;

  // Scale: shoulder width normalised to a fixed reference width (0.25 of frame)
  const shoulderW = Math.abs(lShoulder[0] - rShoulder[0]);
  if (shoulderW < 0.01) return frame;
  const targetW = 0.18;
  const scale = targetW / shoulderW;

  // Target anchor position in canvas (upper-centre)
  const targetX = 0.5;
  const targetY = 0.35;

  function normLms(lms) {
    if (!lms) return null;
    return lms.map(lm => {
      if (!lm) return lm;
      return [
        targetX + (lm[0] - anchorX) * scale,
        targetY + (lm[1] - anchorY) * scale,
        lm[2] || 0,
      ];
    });
  }

  // Clamp outlier hand landmarks (points too far from hand centroid)
  function clampHand(lms) {
    if (!lms || lms.length < 5) return lms;
    // Centroid of wrist + MCP joints (indices 0,5,9,13,17)
    const anchors = [0,5,9,13,17].map(i => lms[i]).filter(Boolean);
    if (anchors.length < 3) return lms;
    const cx = anchors.reduce((s,p) => s+p[0], 0) / anchors.length;
    const cy = anchors.reduce((s,p) => s+p[1], 0) / anchors.length;
    // Max radius = 2.5x avg distance of anchors from centroid
    const avgR = anchors.reduce((s,p) => s + Math.hypot(p[0]-cx, p[1]-cy), 0) / anchors.length;
    const maxR = Math.max(avgR * 2.5, 0.02);
    return lms.map(lm => {
      if (!lm) return lm;
      const d = Math.hypot(lm[0]-cx, lm[1]-cy);
      if (d > maxR) {
        // Pull back toward centroid
        const ratio = maxR / d;
        return [cx + (lm[0]-cx)*ratio, cy + (lm[1]-cy)*ratio, lm[2]||0];
      }
      return lm;
    });
  }

  // EMA smooth hand landmarks across frames
  function smoothHand(cur, prev) {
    if (!cur) return cur;
    if (!prev || prev.length !== cur.length) return cur;
    const a = HAND_SMOOTH;
    return cur.map((lm, i) => {
      if (!lm || !prev[i]) return lm;
      return [
        lm[0] * (1-a) + prev[i][0] * a,
        lm[1] * (1-a) + prev[i][1] * a,
        lm[2] || 0,
      ];
    });
  }

  let normBody = normLms(frame.body);
  let normLH = clampHand(normLms(frame.lh));
  let normRH = clampHand(normLms(frame.rh));
  normLH = smoothHand(normLH, prevLH);
  normRH = smoothHand(normRH, prevRH);
  prevLH = normLH;
  prevRH = normRH;

  // Fill null extremity points from previous frame (prevents disappearing hands)
  if (normBody && prevBody) {
    for (let i = BODY_EXT_START; i <= BODY_EXT_END && i < normBody.length; i++) {
      if (!normBody[i] && prevBody[i]) normBody[i] = prevBody[i].slice();
    }
  }
  // Also fill null hand landmarks from previous
  if (normLH && prevLH) {
    normLH = normLH.map((lm, i) => (!lm && prevLH[i]) ? prevLH[i].slice() : lm);
  }
  if (normRH && prevRH) {
    normRH = normRH.map((lm, i) => (!lm && prevRH[i]) ? prevRH[i].slice() : lm);
  }
  // Clamp body extremity landmarks to reasonable range around torso
  if (normBody) {
    const shoulderMidX = (normBody[11]?.[0] + normBody[12]?.[0]) / 2 || 0.5;
    const shoulderMidY = (normBody[11]?.[1] + normBody[12]?.[1]) / 2 || 0.3;
    for (let i = BODY_EXT_START; i <= BODY_EXT_END && i < normBody.length; i++) {
      if (!normBody[i]) continue;
      const dx = normBody[i][0] - shoulderMidX;
      const dy = normBody[i][1] - shoulderMidY;
      const d = Math.hypot(dx, dy);
      const maxD = 0.35;  // max distance from shoulder midpoint
      if (d > maxD) {
        const ratio = maxD / d;
        normBody[i] = [shoulderMidX + dx * ratio, shoulderMidY + dy * ratio, normBody[i][2] || 0];
      }
    }
  }
  // Smooth body extremity landmarks (15-22: wrists, thumbs, index, pinky)
  if (normBody && prevBody && prevBody.length === normBody.length) {
    const a = BODY_EXT_SMOOTH;
    for (let i = BODY_EXT_START; i <= BODY_EXT_END && i < normBody.length; i++) {
      if (normBody[i] && prevBody[i]) {
        normBody[i] = [
          normBody[i][0] * (1-a) + prevBody[i][0] * a,
          normBody[i][1] * (1-a) + prevBody[i][1] * a,
          normBody[i][2] || 0,
        ];
      }
    }
  }
  prevBody = normBody;

  return { body: normBody, lh: normLH, rh: normRH };
}

function drawFingerspell(gloss) {
  const W = canvas2d.width, H = canvas2d.height;
  ctx2d.clearRect(0, 0, W, H);
  ctx2d.fillStyle = '#06080f';
  ctx2d.fillRect(0, 0, W, H);
  if (!gloss) return;

  const letters = gloss.toUpperCase().split('');
  const maxFontSize = 38;
  const padding = 8;
  const fontSize = Math.min(maxFontSize, Math.floor((W - padding * 2) / Math.max(letters.length, 1) * 0.85));
  const letterW = fontSize * 0.72;
  const totalW = letters.length * letterW + (letters.length - 1) * 4;
  let startX = (W - totalW) / 2 + letterW / 2;

  letters.forEach((letter, i) => {
    const x = startX + i * (letterW + 4);
    const boxX = x - letterW / 2;
    const boxY = H / 2 - fontSize * 0.65;
    const boxW = letterW;
    const boxH = fontSize * 1.3;

    // Card background
    ctx2d.fillStyle = 'rgba(14,124,107,0.2)';
    ctx2d.beginPath();
    ctx2d.roundRect(boxX, boxY, boxW, boxH, 5);
    ctx2d.fill();

    // Card border
    ctx2d.strokeStyle = 'rgba(94,234,212,0.4)';
    ctx2d.lineWidth = 1;
    ctx2d.stroke();

    // Letter
    ctx2d.fillStyle = '#5eead4';
    ctx2d.font = `600 ${fontSize}px JetBrains Mono, monospace`;
    ctx2d.textAlign = 'center';
    ctx2d.textBaseline = 'middle';
    ctx2d.fillText(letter, x, H / 2 + 1);
  });

  // Label
  ctx2d.font = '10px JetBrains Mono, monospace';
  ctx2d.fillStyle = '#3d4a5c';
  ctx2d.textAlign = 'center';
  ctx2d.fillText('fingerspell', W / 2, H - 8);
}

function getCanvasDimensions() {
  // offsetWidth is unreliable inside iframe -- use panel's computed width
  const panelW = panel.getBoundingClientRect().width || 400;
  const areaH  = document.getElementById('canvas-area')?.getBoundingClientRect().height || 220;
  return { w: Math.floor(panelW), h: Math.floor(areaH) };
}

function drawFrame2d(frame, currentGloss) {
  frame = normaliseFrame(frame);  // stabilise scale/position
  const { w: displayW, h: displayH } = getCanvasDimensions();
  if (displayW > 10 && canvas2d.width  !== displayW) canvas2d.width  = displayW;
  if (displayH > 10 && canvas2d.height !== displayH) canvas2d.height = displayH;
  const W = canvas2d.width  || 400;
  const H = canvas2d.height || 220;
  ctx2d.clearRect(0, 0, W, H);
  if (!frame) { drawFingerspell(currentGloss); return; }

  function pt(lms, i) {
    if (!lms?.[i]) return null;
    return { x: lms[i][0] * W, y: lms[i][1] * H };
  }
  function lines(lms, conns, color) {
    ctx2d.strokeStyle = color; ctx2d.lineWidth = 2;
    conns.forEach(([a,b]) => {
      const p1 = pt(lms,a), p2 = pt(lms,b);
      if (!p1||!p2) return;
      ctx2d.beginPath(); ctx2d.moveTo(p1.x,p1.y); ctx2d.lineTo(p2.x,p2.y); ctx2d.stroke();
    });
  }
  function dots(lms, color, r=2.5) {
    ctx2d.fillStyle = color;
    lms?.forEach(([x,y]) => {
      if (x==null) return;
      ctx2d.beginPath(); ctx2d.arc(x*W, y*H, r, 0, Math.PI*2); ctx2d.fill();
    });
  }

  ctx2d.lineCap = 'round'; ctx2d.lineJoin = 'round';

  // Draw head circle at nose landmark (index 0)
  const nose = pt(frame.body, 0);
  if (nose) {
    const lSh = pt(frame.body, 11);
    const rSh = pt(frame.body, 12);
    const headR = (lSh && rSh) ? Math.abs(lSh.x - rSh.x) * 0.22 : 10;
    ctx2d.strokeStyle = '#5eead4';
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    ctx2d.arc(nose.x, nose.y, Math.max(headR, 8), 0, Math.PI * 2);
    ctx2d.stroke();
  }

  lines(frame.body, BODY_CONNECTIONS, '#5eead4');
  dots(frame.body, '#5eead4');
  lines(frame.lh, HAND_CONNECTIONS, '#34d399');
  dots(frame.lh, '#34d399', 2);
  lines(frame.rh, HAND_CONNECTIONS, '#34d399');
  dots(frame.rh, '#34d399', 2);
}

function stop2dAnimation() {
  clearInterval(anim2dTimer);
  anim2dTimer = null;
  current2dSign = 0; current2dFrame = 0;
  signQueue2d = [];
  prevLH = null; prevRH = null; prevBody = null;
  ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);
  placeholder.style.display = 'flex';
}

function play2dQueue(queue) {
  stop2dAnimation();
  if (!queue?.length) return;
  signQueue2d = queue;
  placeholder.style.display = 'none';

  const FPS = 25 * (settings.signSpeed || 1.0);
  anim2dTimer = setInterval(() => {
    const sign = signQueue2d[current2dSign];
    if (!sign) { stop2dAnimation(); setStatus('idle','idle'); return; }

    setActiveGloss(current2dSign);
    drawFrame2d(sign.frames[current2dFrame], sign.gloss);

    current2dFrame++;
    if (current2dFrame >= sign.frames.length) {
      current2dFrame = 0;
      current2dSign++;
      if (current2dSign >= signQueue2d.length) {
        stop2dAnimation(); setStatus('idle','idle');
      }
    }
  }, 1000 / FPS);
}

// ─── Sign lookup ──────────────────────────────────────────────────────────────
// Auto-clear stale cache on version change
const _CACHE_VER = '0.3.5';
try {
  if (localStorage.getItem('slytic-cv') !== _CACHE_VER) {
    indexedDB.deleteDatabase('signlytic-signs');
    localStorage.setItem('slytic-cv', _CACHE_VER);
  }
} catch (_) {}

let _idb = null;
async function openIDB() {
  if (_idb) return _idb;
  return new Promise((res,rej) => {
    const req = indexedDB.open('signlytic-signs', 1);
    req.onupgradeneeded = e => e.target.result.createObjectStore('signs',{keyPath:'gloss'});
    req.onsuccess = e => { _idb = e.target.result; res(_idb); };
    req.onerror   = e => rej(e);
  });
}
async function idbGet(gloss) {
  try {
    const db = await openIDB();
    return new Promise(res => {
      const req = db.transaction('signs','readonly').objectStore('signs').get(gloss);
      req.onsuccess = () => res(req.result?.frames || null);
      req.onerror   = () => res(null);
    });
  } catch { return null; }
}
async function idbSet(gloss, frames) {
  try {
    const db = await openIDB();
    db.transaction('signs','readwrite').objectStore('signs').put({ gloss, frames });
  } catch {}
}

// Sign sources — checked in priority order:
//   1. IndexedDB cache (instant)
//   2. Bundled core/ (174 signs, offline)
//   3. Local FastAPI dashboard (http://localhost:8000) — full 5,203 signs
//   4. Fingerspell fallback (handled by caller)
const LOCAL_API = 'http://localhost:8000';

async function isLocalDashboardRunning() {
  try {
    const res = await fetch(`${LOCAL_API}/api/health`, { signal: AbortSignal.timeout(800) });
    return res.ok;
  } catch { return false; }
}

async function loadSignFrames(gloss) {
  const key = gloss.toUpperCase();

  // 1. IndexedDB cache -- only trust if non-empty array
  const cached = await idbGet(key);
  if (cached && Array.isArray(cached) && cached.length > 0) return cached;

  // 2. Bundled core (174 signs, always available offline)
  try {
    const url = chrome.runtime.getURL(`data/signs/core/${key}.json`);
    const res = await fetch(url);
    if (res.ok) {
      const f = await res.json();
      if (Array.isArray(f) && f.length > 0) {
        idbSet(key, f);
        return f;
      }
    }
  } catch (_) {} // expected for signs not in core

  // 3. Local FastAPI dashboard (private, full dictionary, no licence risk)
  try {
    const res = await fetch(
      `${LOCAL_API}/api/signs/${encodeURIComponent(key)}`,
      { signal: AbortSignal.timeout(800) }
    );
    if (res.ok) {
      const f = await res.json();
      if (Array.isArray(f) && f.length > 0) {
        idbSet(key, f);
        return f;
      }
    }
  } catch (_) {} // silent -- dashboard may not be running

  // Not found -- caller will fingerspell
  return null;
}

// ─── Main translate handler ───────────────────────────────────────────────────
async function handleTranslate(text, source) {
  setStatus('translating', `translating: "${text.substring(0,30)}"`);

  let converter;
  try {
    const modUrl = chrome.runtime.getURL('gloss/converter.js');
    converter = await import(modUrl);
  } catch (e) {
    setStatus('error', 'converter load failed'); return;
  }

  const glosses = converter.textToGloss(text);
  if (!glosses?.length) { setStatus('idle','no glosses produced'); return; }

  renderGlosses(glosses, -1);
  setStatus('signing', `signing ${glosses.length} gloss${glosses.length>1?'es':''}`);

  // Build frame queue
  // For signs with no pose data, hold fingerspell for 20 frames (~0.8s at 25fps)
  const FINGERSPELL_HOLD = 20;
  const queue = [];
  for (const gloss of glosses) {
    const frames = await loadSignFrames(gloss);
    if (frames?.length) {
      queue.push({ gloss, frames });
    } else {
      // Pad null so fingerspell shows for FINGERSPELL_HOLD frames
      queue.push({ gloss, frames: Array(FINGERSPELL_HOLD).fill(null) });
    }
  }

  if (renderMode === '3d' && avatarLoaded && avatarRenderer?.ready) {
    // 3D playback
    placeholder.style.display = 'none';
    const aw = document.getElementById('avatar-waiting');
    if (aw) aw.style.display = 'none';
    avatarRenderer.speed = settings.signSpeed || 1.0;
    avatarRenderer.playQueue(
      queue,
      (idx) => setActiveGloss(idx),                      // onGlossChange
      ()    => { setStatus('idle','idle'); avatarRenderer.resetPose(); const aw2=document.getElementById('avatar-waiting'); if(aw2)aw2.style.display='block'; } // onDone
    );
  } else {
    // 2D fallback
    play2dQueue(queue);
  }
}

// ─── Drag to reposition (moves iframe via parent postMessage) ─────────────────
let dragging = false;
let dragStartX = 0, dragStartY = 0;

header.addEventListener('mousedown', (e) => {
  if (e.target.closest('.ctrl-btn')) return;
  dragging = true;
  dragStartX = e.screenX;
  dragStartY = e.screenY;
  panel.style.transition = 'none';
  e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const dx = e.screenX - dragStartX;
  const dy = e.screenY - dragStartY;
  dragStartX = e.screenX;
  dragStartY = e.screenY;
  try {
    window.parent.postMessage({ type: 'DRAG_DELTA', dx, dy }, '*');
  } catch (_) {}
});

document.addEventListener('mouseup', () => {
  if (dragging) {
    dragging = false;
    panel.style.transition = '';
  }
});

// ─── Broadcast panel bounds to parent page (for pointer-events toggle) ─────────
function broadcastBounds() {
  const rect = panel.getBoundingClientRect();
  if (!rect.width) return;
  // Post to parent content_script via window.parent
  try {
    window.parent.postMessage({
      type: 'PANEL_BOUNDS',
      bounds: { x: rect.left, y: rect.top, w: rect.width, h: rect.height }
    }, '*');
  } catch (_) {}
}

// --- Caption toast ---
let toastTimer = null;
function showCaptionToast(source) {
  const toast = document.getElementById('caption-toast');
  if (!toast) return;
  const msg = toast.querySelector('.toast-msg');
  if (msg) msg.textContent = (source || 'Captions') + ' detected';
  toast.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 8000);
}
const toastSwitch = document.getElementById('toast-switch');
const toastDismiss = document.getElementById('toast-dismiss');
if (toastSwitch) toastSwitch.addEventListener('click', () => {
  document.getElementById('caption-toast').style.display = 'none';
  clearTimeout(toastTimer);
  window.parent.postMessage({ type: 'SWITCH_TO_CAPTIONS' }, '*');
});
if (toastDismiss) toastDismiss.addEventListener('click', () => {
  document.getElementById('caption-toast').style.display = 'none';
  clearTimeout(toastTimer);
});

// Broadcast on load, drag, resize, settings change
const _boundsObserver = new ResizeObserver(broadcastBounds);
_boundsObserver.observe(panel);

// Poll every 300ms to catch drag repositioning
setInterval(broadcastBounds, 300);

// ─── Resize handle logic ─────────────────────────────────────────────────────
let resizing = false;
let resizeStart = { x: 0, y: 0, w: 0, h: 0 };
let resizeDir = 'se'; // 'se' bottom-right, 'sw' bottom-left

function initResizeHandle(handleEl, dir) {
  handleEl.addEventListener('mousedown', (e) => {
    resizing = true;
    resizeDir = dir;
    const rect = panel.getBoundingClientRect();
    resizeStart = { x: e.clientX, y: e.clientY, w: rect.width, h: rect.height, l: rect.left };
    panel.style.transition = 'none';
    e.preventDefault();
    e.stopPropagation();
  });
}

const handleBR = document.getElementById('resize-handle-br');
const handleSW = document.getElementById('resize-handle');
if (handleBR) initResizeHandle(handleBR, 'se');
if (handleSW) initResizeHandle(handleSW, 'sw');

document.addEventListener('mousemove', (e) => {
  if (!resizing) return;
  const dx = e.clientX - resizeStart.x;
  const dy = e.clientY - resizeStart.y;
  const minW = 260, maxW = 700;
  const minH = 160, maxH = 500;

  let newW = resizeDir === 'se'
    ? Math.min(maxW, Math.max(minW, resizeStart.w + dx))
    : Math.min(maxW, Math.max(minW, resizeStart.w - dx));
  let newH = Math.min(maxH, Math.max(minH, resizeStart.h + dy));

  panel.style.width  = newW + 'px';
  panel.style.height = newH + 'px';  // explicit height overrides flexbox
  panel.style.removeProperty('right');

  if (resizeDir === 'sw') {
    panel.style.left = (resizeStart.l + (resizeStart.w - newW)) + 'px';
  }

  // Canvas area gets remaining height after all other elements
  const canvasArea = document.getElementById('canvas-area');
  const otherH = [header, document.getElementById('mode-toggle'), manualBar, glossRow, document.getElementById('status-bar')]
    .reduce((sum, el) => sum + (el && el.offsetHeight && el.style.display !== 'none' ? el.offsetHeight : 0), 0);
  const canvasH = Math.max(80, newH - Math.max(otherH, 100) - 4);
  if (canvasArea) canvasArea.style.height = canvasH + 'px';

  // Update canvas buffer
  canvas2d.width  = newW;
  canvas2d.height = canvasH;

  // Resize 3D renderer
  if (avatarRenderer) avatarRenderer.resize(newW, canvasH);
});

document.addEventListener('mouseup', () => {
  if (resizing) {
    resizing = false;
    panel.style.transition = '';
  }
});

// ─── Button handlers ──────────────────────────────────────────────────────────
// ─── Manual input wiring ─────────────────────────────────────────────────────
const manualBar   = document.getElementById('manual-input-bar');
const manualInput = document.getElementById('manual-input');
const manualSend  = document.getElementById('manual-send');

function showManualInput(show) {
  if (manualBar) manualBar.style.display = show ? 'block' : 'none';
  if (show && manualInput) manualInput.focus();
}

function submitManualText() {
  const text = manualInput?.value?.trim();
  if (!text) return;
  handleTranslate(text, 'manual');
  if (manualInput) manualInput.value = '';
}

if (manualSend) {
  manualSend.addEventListener('click', submitManualText);
}
if (manualInput) {
  manualInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submitManualText(); }
  });
  // Prevent panel drag when typing
  manualInput.addEventListener('mousedown', (e) => e.stopPropagation());
}

btnMinimise.addEventListener('click', () => {
  isMinimised = !isMinimised;
  panel.classList.toggle('minimised', isMinimised);
  btnMinimise.textContent = isMinimised ? '+' : '−';
});

btnClose.addEventListener('click', () => {
  panel.classList.add('hidden');
  window.parent.postMessage({ type: 'OVERLAY_CLOSED' }, '*');
});

// Exclude this site
const btnExclude = document.getElementById('btn-exclude');
if (btnExclude) {
  btnExclude.addEventListener('click', () => {
    window.parent.postMessage({ type: 'EXCLUDE_SITE' }, '*');
    panel.classList.add('hidden');
  });
}

btn2d.addEventListener('click', () => applyMode('2d'));
btn3d.addEventListener('click', () => {
  applyMode('3d');
  if (!avatarLoaded) load3DAvatar();
});

// ─── Message listener ─────────────────────────────────────────────────────────
window.addEventListener('message', (event) => {
  const msg = event.data;
  if (!msg?.type) return;

  switch (msg.type) {
    case 'INIT': {
      applySettings(msg.settings || {});
      panel.classList.remove('hidden');
      const initSrc = msg.settings?.captionSource || 'auto';
      if (initSrc === 'mic')         setStatus('listening', 'mic active — speak now');
      else if (initSrc === 'manual') setStatus('listening', 'manual mode — type below');
      else                           setStatus('listening', 'waiting for captions...');
      showManualInput(initSrc === 'manual');
      requestAnimationFrame(() => {
        const { w, h } = getCanvasDimensions();
        if (w > 10) { canvas2d.width = w; canvas2d.height = h; }
      });
      if ((msg.settings?.renderMode === '3d') && !avatarLoaded) {
        load3DAvatar();
      }
      break;
    }

    case 'SETTINGS_CHANGED': {
      const prev = settings.avatarGender;
      applySettings(msg.settings || {});
      showManualInput((msg.settings?.captionSource || 'auto') === 'manual');
      // Reload avatar if gender changed while in 3D mode
      if (renderMode === '3d' && msg.settings?.avatarGender !== prev && avatarRenderer) {
        avatarLoaded = false;
        avatarRenderer.changeGender(msg.settings.avatarGender, (p) => {
          loadingBar.style.width = Math.round(p*100)+'%';
        }).then(ok => {
          avatarLoading.classList.remove('visible');
          if (ok) { avatarLoaded=true; genderBadge.textContent=msg.settings.avatarGender; }
          else showAvatarError();
        });
        avatarLoading.classList.add('visible');
        genderBadge.textContent = msg.settings.avatarGender;
      }
      break;
    }

    case 'TRANSLATE':
      handleTranslate(msg.text, msg.source);
      break;

    case 'INTERIM_TEXT':
      setStatus('listening', `hearing: "${msg.text.substring(0,25)}..."`);
      break;

    case 'STATUS':
      setStatus(msg.status, msg.message || msg.status);
      break;
  }
});
