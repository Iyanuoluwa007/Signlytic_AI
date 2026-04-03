// content_script.js — Signlytic AI Extension
// Detects page captions or falls back to microphone, injects overlay iframe

// ─── Caption source selectors (site-specific) ──────────────────────────────
const CAPTION_SELECTORS = [
  // YouTube
  { host: 'youtube.com',  selector: '.ytp-caption-segment',                  name: 'YouTube' },
  // BBC iPlayer
  { host: 'bbc.co.uk',   selector: '.subtitles__subtitle-body',              name: 'BBC iPlayer' },
  { host: 'bbc.co.uk',   selector: '.sp_el',                                 name: 'BBC iPlayer (sp)' },
  // Netflix
  { host: 'netflix.com', selector: '.player-timedtext-text-container span',  name: 'Netflix' },
  // Amazon Prime
  { host: 'amazon.',     selector: '.timedTextBackground',                    name: 'Amazon Prime' },
  // All4 / Channel 4
  { host: 'channel4.com',selector: '[data-testid="subtitles"]',              name: 'All4' },
  // Disney+
  { host: 'disneyplus.', selector: '.subtitle-text',                         name: 'Disney+' },
  // Apple TV+
  { host: 'tv.apple.com',selector: '[data-testid="caption-container"]',      name: 'Apple TV+' },
  // Generic fallback — any element with common caption classes/roles
  { host: null,          selector: '[class*="caption-window"] span',         name: 'Generic (caption-window)' },
  { host: null,          selector: '[class*="subtitle"] span',               name: 'Generic (subtitle)' },
];

// ─── State ──────────────────────────────────────────────────────────────────
let overlayFrame = null;       // injected <iframe> element
let captionObserver = null;    // MutationObserver watching caption nodes
let speechRecognition = null;  // Web Speech API instance
let settings = {};
let activeSource = null;       // 'captions' | 'mic' | null
let lastSentText = '';         // debounce — avoid re-sending same sentence
let debounceTimer = null;

// ─── Initialise ─────────────────────────────────────────────────────────────
function safeMessage(msg, cb) {
  try { chrome.runtime.sendMessage(msg, cb); } catch (_) {}
}

try {
  chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (res) => {
    if (chrome.runtime.lastError) return;
    settings = res?.settings || {};
    // Check exclusion list before injecting
    chrome.storage.sync.get({ excludedSites: [] }, ({ excludedSites }) => {
      if ((excludedSites || []).includes(window.location.hostname)) return;
      injectOverlay();
    });
  });
} catch (_) {}

chrome.runtime.onMessage.addListener((msg) => {
  switch (msg.type) {
    case 'INJECT_OVERLAY':
      chrome.storage.sync.get({ excludedSites: [] }, ({ excludedSites }) => {
        if (!(excludedSites || []).includes(window.location.hostname)) injectOverlay();
      });
      break;
    case 'REMOVE_OVERLAY':    removeOverlay();    break;
    case 'RELAY_TEXT':        /* unused — direct post */ break;
    case 'SETTINGS_CHANGED': {
      const prevSource = settings.captionSource;
      settings = msg.settings;
      if (overlayFrame) postToOverlay({ type: 'SETTINGS_CHANGED', settings });
      // Restart detection if caption source changed
      if (msg.settings.captionSource !== prevSource) {
        stopCaptionDetection();
        stopMic();
        startCaptionDetection();
      }
      break;
    }
  }
});

// ─── Overlay injection ──────────────────────────────────────────────────────
// ─── Iframe position state (module-level) ────────────────────────────────────
const PAD = 16;
let iframeTop  = -1; // -1 = not yet initialised
let iframeLeft = -1;

function applyIframePos() {
  if (!overlayFrame) return;
  iframeTop  = Math.max(0, Math.min(window.innerHeight - 100, iframeTop));
  iframeLeft = Math.max(0, Math.min(window.innerWidth  - 100, iframeLeft));
  overlayFrame.style.top    = iframeTop  + 'px';
  overlayFrame.style.left   = iframeLeft + 'px';
  overlayFrame.style.bottom = 'auto';
  overlayFrame.style.right  = 'auto';
}

// Module-level message listener -- always active, single registration
window.addEventListener('message', (e) => {
  if (!e.data?.type || !overlayFrame) return;

  if (e.data.type === 'PANEL_BOUNDS') {
    const { w, h } = e.data.bounds;
    if (w > 50) {
      overlayFrame.style.width  = (w + PAD * 2) + 'px';
      overlayFrame.style.height = (h + PAD * 2) + 'px';
    }
  }

  if (e.data.type === 'DRAG_DELTA') {
    iframeTop  += e.data.dy;
    iframeLeft += e.data.dx;
    applyIframePos();
  }

  if (e.data.type === 'SET_POSITION') {
    const vw = window.innerWidth, vh = window.innerHeight;
    const iw = overlayFrame.offsetWidth  || 432;
    const ih = overlayFrame.offsetHeight || 432;
    switch (e.data.position) {
      case 'bottom-right': iframeTop = vh-ih-10; iframeLeft = vw-iw-10; break;
      case 'bottom-left':  iframeTop = vh-ih-10; iframeLeft = 10;       break;
      case 'top-right':    iframeTop = 10;        iframeLeft = vw-iw-10; break;
      case 'top-left':     iframeTop = 10;        iframeLeft = 10;       break;
    }
    applyIframePos();
  }

  if (e.data.type === 'OVERLAY_CLOSED') {
    removeOverlay();
  }
  if (e.data.type === 'EXCLUDE_SITE') {
    chrome.runtime.sendMessage({
      type: 'EXCLUDE_SITE',
      hostname: window.location.hostname
    });
    removeOverlay();
  }
});

function injectOverlay() {
  if (overlayFrame) return;
  if (document.getElementById('signlytic-overlay-frame')) {
    overlayFrame = document.getElementById('signlytic-overlay-frame');
    return;
  }

  const url = chrome.runtime.getURL('overlay/overlay.html');
  const iframe = document.createElement('iframe');
  iframe.id = 'signlytic-overlay-frame';
  iframe.src = url;

  // Default: bottom-right, 420x420
  const defW = 432, defH = 432;
  if (iframeTop  < 0) iframeTop  = window.innerHeight - defH - 10;
  if (iframeLeft < 0) iframeLeft = window.innerWidth  - defW - 10;

  Object.assign(iframe.style, {
    position:      'fixed',
    zIndex:        '2147483647',
    border:        'none',
    background:    'transparent',
    pointerEvents: 'all',
    width:         defW + 'px',
    height:        defH + 'px',
    top:           iframeTop  + 'px',
    left:          iframeLeft + 'px',
    bottom:        'auto',
    right:         'auto',
  });

  document.documentElement.appendChild(iframe);
  overlayFrame = iframe;

  iframe.addEventListener('load', () => {
    postToOverlay({ type: 'INIT', settings });
    safeMessage({ type: 'OVERLAY_READY' });
    // Small delay so page CC DOM (YouTube etc) has time to initialise
    setTimeout(() => startCaptionDetection(), 1500);
  });
}

function removeOverlay() {
  stopCaptionDetection();
  stopMic();
  if (overlayFrame) {
    overlayFrame.remove();
    overlayFrame = null;
    safeMessage({ type: 'OVERLAY_REMOVED' });
  }
}

// ─── Post message to overlay iframe ─────────────────────────────────────────
function postToOverlay(data) {
  if (!overlayFrame?.contentWindow) return;
  overlayFrame.contentWindow.postMessage(data, chrome.runtime.getURL('/'));
}

// ─── Caption detection (MutationObserver) ───────────────────────────────────
// Sites where mic fallback makes sense
const MIC_FRIENDLY_HOSTS = [
  'youtube.com', 'bbc.co.uk', 'netflix.com', 'amazon.',
  'channel4.com', 'disneyplus.', 'tv.apple.com', 'vimeo.com',
  'twitch.tv', 'dailymotion.com',
];

function isMicFriendlySite() {
  const host = window.location.hostname;
  return MIC_FRIENDLY_HOSTS.some(h => host.includes(h));
}

function resolveSelector() {
  const host = window.location.hostname;
  for (const s of CAPTION_SELECTORS) {
    if (s.host && !host.includes(s.host)) continue;
    if (document.querySelector(s.selector)) return s;
  }
  return null;
}

function startCaptionDetection() {
  const source = settings.captionSource || 'auto';

  if (source === 'mic') {
    activeSource = 'mic';
    startMic();
    return;
  }

  if (source === 'manual') {
    // Manual mode: overlay shows a text input, user types directly
    // Content script just signals overlay to show manual input UI
    activeSource = 'manual';
    postToOverlay({ type: 'STATUS', status: 'listening', message: 'manual mode — type in overlay' });
    return;
  }

  // 'auto' or 'captions'
  const selector = resolveSelector();
  if (selector) {
    console.log('[Signlytic] Caption source: ' + selector.name);
    activeSource = 'captions';
    watchCaptionNode(selector.selector);
  } else if (source === 'captions') {
    // Explicit captions mode but none found yet -- keep polling
    waitForCaptions();
  } else {
    // Auto mode -- poll silently
    waitForCaptions();
  }
}

let captionPollInterval = null;

function waitForCaptions() {
  // Clear any existing poll first
  if (captionPollInterval) { clearInterval(captionPollInterval); captionPollInterval = null; }

  postToOverlay({ type: 'STATUS', status: 'listening', message: 'waiting for captions...' });

  let attempts = 0;
  captionPollInterval = setInterval(() => {
    if (!overlayFrame) { clearInterval(captionPollInterval); return; }
    attempts++;
    const selector = resolveSelector();
    if (selector) {
      clearInterval(captionPollInterval);
      captionPollInterval = null;
      console.log('[Signlytic] Captions appeared: ' + selector.name);
      activeSource = 'captions';
      watchCaptionNode(selector.selector);
      postToOverlay({ type: 'STATUS', status: 'listening', message: 'captions detected — listening' });
    }
    if (attempts > 300) {
      clearInterval(captionPollInterval);
      captionPollInterval = null;
      postToOverlay({ type: 'STATUS', status: 'idle', message: 'no captions found — enable CC' });
    }
  }, 1000);
}

// Track when captions were last seen -- used to detect CC being turned off
let lastCaptionSeen = 0;
let captionWatchdogTimer = null;

function watchCaptionNode(selector) {
  // Watchdog: if no caption text seen for 4s, assume CC turned off -- go back to polling
  function resetWatchdog() {
    clearTimeout(captionWatchdogTimer);
    captionWatchdogTimer = setTimeout(() => {
      const nodes = document.querySelectorAll(selector);
      if (!nodes.length || !Array.from(nodes).some(n => n.textContent.trim())) {
        // Captions gone -- stop current observer and restart polling
        stopCaptionDetection();
        postToOverlay({ type: 'STATUS', status: 'listening', message: 'waiting for captions...' });
        waitForCaptions();
      } else {
        resetWatchdog();
      }
    }, 4000);
  }

  // Observe the document body for new caption text nodes matching selector
  captionObserver = new MutationObserver(() => {
    const nodes = document.querySelectorAll(selector);
    if (!nodes.length) return;

    const text = Array.from(nodes)
      .map(n => n.textContent.trim())
      .filter(Boolean)
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (text && text !== lastSentText) {
      lastCaptionSeen = Date.now();
      resetWatchdog();
      debouncedSend(text, 'captions');
    }
  });

  resetWatchdog(); // start watchdog immediately

  captionObserver.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });
}

function stopCaptionDetection() {
  clearTimeout(captionWatchdogTimer);
  captionWatchdogTimer = null;
  if (captionPollInterval) { clearInterval(captionPollInterval); captionPollInterval = null; }
  if (captionObserver) {
    captionObserver.disconnect();
    captionObserver = null;
  }
}

// ─── Microphone fallback (Web Speech API) ───────────────────────────────────
function startMic() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    console.warn('[Signlytic] Web Speech API not available in this browser.');
    postToOverlay({ type: 'STATUS', status: 'error', message: 'Speech recognition not available' });
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  speechRecognition = new SpeechRecognition();
  speechRecognition.lang = 'en-GB';
  speechRecognition.continuous = true;
  speechRecognition.interimResults = true;
  speechRecognition.maxAlternatives = 1;

  speechRecognition.onstart = () => {
    console.log('[Signlytic] Microphone started (en-GB).');
    postToOverlay({ type: 'STATUS', status: 'listening' });
  };

  let interimTimer = null;

  speechRecognition.onresult = (event) => {
    let finalText = '';
    let interimText = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript.trim();
      if (event.results[i].isFinal) {
        finalText += transcript + ' ';
      } else {
        interimText += transcript;
      }
    }

    if (finalText.trim()) {
      clearTimeout(interimTimer);
      debouncedSend(finalText.trim(), 'mic');
    } else if (interimText.trim()) {
      // Show live text in overlay immediately
      postToOverlay({ type: 'INTERIM_TEXT', text: interimText.trim() });
      // Also translate interim after 1.5s if no final result arrives
      // This handles continuous speech where isFinal is delayed
      clearTimeout(interimTimer);
      if (interimText.trim().split(' ').length >= 3) {
        interimTimer = setTimeout(() => {
          debouncedSend(interimText.trim(), 'mic');
        }, 1500);
      }
    }
  };

  speechRecognition.onerror = (e) => {
    console.warn('[Signlytic] Speech recognition error:', e.error);
    if (e.error === 'not-allowed') {
      postToOverlay({ type: 'STATUS', status: 'error', message: 'Microphone permission denied' });
    }
  };

  speechRecognition.onend = () => {
    try {
      // Only auto-restart if user explicitly set captionSource to 'mic'
      if (overlayFrame && settings.enabled && settings.captionSource === 'mic') {
        speechRecognition.start();
      }
    } catch (_) {}
  };

  speechRecognition.start();
}

function stopMic() {
  if (speechRecognition) {
    speechRecognition.onend = null; // prevent auto-restart
    speechRecognition.stop();
    speechRecognition = null;
  }
}

// ─── Debounced text send ─────────────────────────────────────────────────────
function debouncedSend(text, source) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    if (text === lastSentText) return;
    lastSentText = text;
    postToOverlay({ type: 'TRANSLATE', text, source });
  }, 300); // 300ms debounce — catches rapid caption updates
}
