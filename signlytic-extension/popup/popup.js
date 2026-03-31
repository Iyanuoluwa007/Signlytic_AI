// popup.js — Signlytic AI Extension Popup

const SOURCE_HINTS = {
  auto:     'Detects page captions, falls back to mic',
  captions: 'Only reads page captions (YouTube CC, BBC, Netflix…)',
  mic:      'Uses your microphone only',
  manual:   'Type text directly in the overlay panel',
};

// ─── DOM refs ────────────────────────────────────────────────────────────────
const toggleEnabled   = document.getElementById('toggle-enabled');
const statusIndicator = document.getElementById('status-indicator');
const statusMsg       = document.getElementById('status-msg');
const speedSlider     = document.getElementById('speed-slider');
const speedVal        = document.getElementById('speed-val');
const sourceHint      = document.getElementById('source-hint');
const avatarSection   = document.getElementById('avatar-section');

// ─── Load settings from storage ─────────────────────────────────────────────
chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (res) => {
  const s = res?.settings || {};
  applyToUI(s);
});

// ─── Apply settings to popup UI ──────────────────────────────────────────────
function applyToUI(s) {
  toggleEnabled.checked = !!s.enabled;
  updateStatusIndicator(s.enabled);

  setActiveOpt('captionSource', s.captionSource || 'auto');
  setActiveOpt('renderMode',    s.renderMode    || '2d');
  setActiveOpt('avatarGender',  s.avatarGender  || 'male');
  setActiveOpt('position',      s.position      || 'bottom-right');
  setActiveOpt('size',          s.size          || 'medium');

  const speed = s.signSpeed || 1.0;
  speedSlider.value = speed;
  speedVal.textContent = speed.toFixed(1) + 'x';

  updateSourceHint(s.captionSource || 'auto');
  updateAvatarSection(s.renderMode || '2d');
}

// ─── Option group helper ──────────────────────────────────────────────────────
function setActiveOpt(key, val) {
  const group = document.querySelector(`[data-key="${key}"]`);
  if (!group) return;
  group.querySelectorAll('.opt-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.val === val);
  });
}

// ─── Option group click wiring ────────────────────────────────────────────────
document.querySelectorAll('[data-key]').forEach(group => {
  group.addEventListener('click', (e) => {
    const btn = e.target.closest('.opt-btn');
    if (!btn) return;

    const key = group.dataset.key;
    const val = btn.dataset.val;

    setActiveOpt(key, val);
    sendPatch({ [key]: val });

    if (key === 'captionSource') updateSourceHint(val);
    if (key === 'renderMode')    updateAvatarSection(val);
  });
});

// ─── Power toggle ─────────────────────────────────────────────────────────────
toggleEnabled.addEventListener('change', () => {
  const enabled = toggleEnabled.checked;
  updateStatusIndicator(enabled);
  chrome.runtime.sendMessage({ type: 'TOGGLE_ENABLED', enabled });
});

// ─── Speed slider ─────────────────────────────────────────────────────────────
speedSlider.addEventListener('input', () => {
  const val = parseFloat(speedSlider.value);
  speedVal.textContent = val.toFixed(1) + 'x';
});
speedSlider.addEventListener('change', () => {
  sendPatch({ signSpeed: parseFloat(speedSlider.value) });
});

// ─── Helpers ──────────────────────────────────────────────────────────────────
function sendPatch(patch) {
  chrome.runtime.sendMessage({ type: 'SETTINGS_UPDATED', patch });
}

function updateStatusIndicator(enabled) {
  statusIndicator.classList.toggle('on', !!enabled);
  statusMsg.textContent = enabled ? 'active — overlay injected' : 'inactive';
}

function updateSourceHint(source) {
  if (sourceHint) sourceHint.textContent = SOURCE_HINTS[source] || '';
}

function updateAvatarSection(renderMode) {
  if (avatarSection) {
    avatarSection.style.display = renderMode === '3d' ? '' : 'none';
  }
}
