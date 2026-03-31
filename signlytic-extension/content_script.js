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
chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (res) => {
  settings = res?.settings || {};
  if (settings.enabled) injectOverlay();
});

chrome.runtime.onMessage.addListener((msg) => {
  switch (msg.type) {
    case 'INJECT_OVERLAY':    injectOverlay();    break;
    case 'REMOVE_OVERLAY':    removeOverlay();    break;
    case 'RELAY_TEXT':        /* unused — direct post */ break;
    case 'SETTINGS_CHANGED':
      settings = msg.settings;
      if (overlayFrame) postToOverlay({ type: 'SETTINGS_CHANGED', settings });
      break;
  }
});

// ─── Overlay injection ──────────────────────────────────────────────────────
function injectOverlay() {
  if (overlayFrame) return; // already injected

  const url = chrome.runtime.getURL('overlay/overlay.html');
  const iframe = document.createElement('iframe');
  iframe.id = 'signlytic-overlay-frame';
  iframe.src = url;

  Object.assign(iframe.style, {
    position:   'fixed',
    zIndex:     '2147483647',
    border:     'none',
    background: 'transparent',
    pointerEvents: 'none',
    width:      '1px',
    height:     '1px',
    bottom:     '0',
    right:      '0',
  });

  document.documentElement.appendChild(iframe);
  overlayFrame = iframe;

  // Once iframe loads, send initial settings and start detection
  iframe.addEventListener('load', () => {
    postToOverlay({ type: 'INIT', settings });
    chrome.runtime.sendMessage({ type: 'OVERLAY_READY' });
    startCaptionDetection();
  });
}

function removeOverlay() {
  stopCaptionDetection();
  stopMic();
  if (overlayFrame) {
    overlayFrame.remove();
    overlayFrame = null;
    chrome.runtime.sendMessage({ type: 'OVERLAY_REMOVED' });
  }
}

// ─── Post message to overlay iframe ─────────────────────────────────────────
function postToOverlay(data) {
  if (!overlayFrame?.contentWindow) return;
  overlayFrame.contentWindow.postMessage(data, chrome.runtime.getURL('/'));
}

// ─── Caption detection (MutationObserver) ───────────────────────────────────
function startCaptionDetection() {
  const source = settings.captionSource || 'auto';
  if (source === 'mic') { startMic(); return; }

  const selector = resolveSelector();

  if (selector) {
    console.log(`[Signlytic] Caption source detected: ${selector.name}`);
    activeSource = 'captions';
    watchCaptionNode(selector.selector);
  } else if (source === 'auto') {
    // No captions found — fall back to mic
    console.log('[Signlytic] No caption node found — falling back to microphone.');
    activeSource = 'mic';
    startMic();
  }
}

function resolveSelector() {
  const host = window.location.hostname;
  for (const s of CAPTION_SELECTORS) {
    if (s.host && !host.includes(s.host)) continue;
    if (document.querySelector(s.selector)) return s;
  }
  return null;
}

function watchCaptionNode(selector) {
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
      debouncedSend(text, 'captions');
    }
  });

  captionObserver.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });
}

function stopCaptionDetection() {
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
      debouncedSend(finalText.trim(), 'mic');
    } else if (interimText.trim()) {
      // Send interim so overlay can show "listening" state with live text
      postToOverlay({ type: 'INTERIM_TEXT', text: interimText.trim() });
    }
  };

  speechRecognition.onerror = (e) => {
    console.warn('[Signlytic] Speech recognition error:', e.error);
    if (e.error === 'not-allowed') {
      postToOverlay({ type: 'STATUS', status: 'error', message: 'Microphone permission denied' });
    }
  };

  speechRecognition.onend = () => {
    // Auto-restart unless overlay was removed
    if (overlayFrame && settings.enabled) {
      speechRecognition.start();
    }
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
