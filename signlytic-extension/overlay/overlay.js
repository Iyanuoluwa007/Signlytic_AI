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

// ─── Settings ────────────────────────────────────────────────────────────────
function applySettings(s) {
  settings = s;
  renderMode = s.renderMode || '2d';

  const posClasses = ['pos-bottom-right','pos-bottom-left','pos-top-right','pos-top-left'];
  panel.classList.remove(...posClasses);
  panel.classList.add(`pos-${s.position || 'bottom-right'}`);

  const sizeClasses = ['size-small','size-medium','size-large'];
  panel.classList.remove(...sizeClasses);
  panel.classList.add(`size-${s.size || 'medium'}`);

  const widthMap = { small: 260, medium: 320, large: 400 };
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
    genderBadge.style.display  = 'block';
    genderBadge.textContent    = settings.avatarGender || 'male';
    if (forceReload || !avatarLoaded) load3DAvatar();
  }
}

// ─── 3D Avatar loading ────────────────────────────────────────────────────────
async function load3DAvatar(forceGender) {
  const gender = forceGender || settings.avatarGender || 'male';

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

function drawFrame2d(frame) {
  const W = canvas2d.width, H = canvas2d.height;
  ctx2d.clearRect(0, 0, W, H);
  if (!frame) return;

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
    drawFrame2d(sign.frames[current2dFrame]);

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

  // 1. IndexedDB cache
  const cached = await idbGet(key);
  if (cached) return cached;

  // 2. Bundled core (174 signs, always available offline)
  try {
    const url = chrome.runtime.getURL(`data/signs/core/${key}.json`);
    const res = await fetch(url);
    if (res.ok) {
      const f = await res.json();
      idbSet(key, f);
      return f;
    }
  } catch {}

  // 3. Local FastAPI dashboard (private, full dictionary, no licence risk)
  try {
    const res = await fetch(
      `${LOCAL_API}/api/signs/${encodeURIComponent(key)}`,
      { signal: AbortSignal.timeout(3000) }
    );
    if (res.ok) {
      const f = await res.json();
      idbSet(key, f);
      return f;
    }
  } catch {}

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
  const queue = [];
  for (const gloss of glosses) {
    const frames = await loadSignFrames(gloss);
    queue.push({ gloss, frames: frames?.length ? frames : [null] });
  }

  if (renderMode === '3d' && avatarLoaded && avatarRenderer?.ready) {
    // 3D playback
    placeholder.style.display = 'none';
    avatarRenderer.speed = settings.signSpeed || 1.0;
    avatarRenderer.playQueue(
      queue,
      (idx) => setActiveGloss(idx),                      // onGlossChange
      ()    => { setStatus('idle','idle'); avatarRenderer.resetPose(); } // onDone
    );
  } else {
    // 2D fallback
    play2dQueue(queue);
  }
}

// ─── Drag to reposition ───────────────────────────────────────────────────────
let dragging = false, dragStart = { x:0, y:0, px:0, py:0 };

header.addEventListener('mousedown', (e) => {
  if (e.target.closest('.ctrl-btn')) return;
  dragging = true;
  const rect = panel.getBoundingClientRect();
  dragStart = { x: e.clientX, y: e.clientY, px: rect.left, py: rect.top };
  panel.style.transition = 'none';
  panel.style.left = rect.left+'px'; panel.style.top = rect.top+'px';
  panel.style.right = 'auto'; panel.style.bottom = 'auto';
  e.preventDefault();
});
document.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  panel.style.left = (dragStart.px + e.clientX - dragStart.x)+'px';
  panel.style.top  = (dragStart.py + e.clientY - dragStart.y)+'px';
});
document.addEventListener('mouseup', () => { if (dragging) { dragging=false; panel.style.transition=''; } });

// ─── Button handlers ──────────────────────────────────────────────────────────
btnMinimise.addEventListener('click', () => {
  isMinimised = !isMinimised;
  panel.classList.toggle('minimised', isMinimised);
  btnMinimise.textContent = isMinimised ? '+' : '−';
});

btnClose.addEventListener('click', () => {
  panel.classList.add('hidden');
  window.parent.postMessage({ type: 'OVERLAY_CLOSED' }, '*');
});

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
    case 'INIT':
      applySettings(msg.settings || {});
      panel.classList.remove('hidden');
      setStatus('listening','listening...');
      // If settings say 3D mode, pre-load avatar immediately
      if ((msg.settings?.renderMode === '3d') && !avatarLoaded) {
        load3DAvatar();
      }
      break;

    case 'SETTINGS_CHANGED': {
      const prev = settings.avatarGender;
      applySettings(msg.settings || {});
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
