// background.js: Signlytic AI Extension Service Worker
// Relay hub between content_script and overlay iframe

// Track which tabs have an active overlay
const activeTabs = new Map(); // tabId -> { overlayReady: bool, settings: {} }

// --- Default settings ---
const DEFAULT_SETTINGS = {
  enabled: true,
  renderMode: '2d',          // '2d' | '3d'
  position: 'bottom-right',  // 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left'
  size: 'medium',            // 'small' | 'medium' | 'large'
  captionSource: 'auto',     // 'auto' | 'captions' | 'mic' | 'manual'
  avatarGender: 'male',      // 'male' | 'female'
  overlayOpacity: 0.95,
  signSpeed: 1.0,
};

// --- Install / startup ---
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ settings: DEFAULT_SETTINGS });
  console.log('[Signlytic] Extension installed. Default settings written.');
});

// --- Message routing ---
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  const tabId = sender.tab?.id;

  switch (msg.type) {

    // Content script reports overlay iframe is mounted and ready
    case 'OVERLAY_READY':
      if (tabId) {
        activeTabs.set(tabId, { overlayReady: true });
        console.log(`[Signlytic] Overlay ready on tab ${tabId}`);
      }
      sendResponse({ ok: true });
      break;

    // Content script reports overlay was removed
    case 'OVERLAY_REMOVED':
      if (tabId) activeTabs.delete(tabId);
      sendResponse({ ok: true });
      break;

    // Content script captured text, relay to overlay on same tab
    case 'TEXT_CAPTURED':
      if (tabId) {
        chrome.tabs.sendMessage(tabId, {
          type: 'RELAY_TEXT',
          text: msg.text,
          source: msg.source, // 'captions' | 'mic'
        }).catch(() => {});
      }
      sendResponse({ ok: true });
      break;

    // Popup toggled extension on/off
    case 'TOGGLE_ENABLED':
      chrome.storage.sync.get('settings', ({ settings }) => {
        const updated = { ...settings, enabled: msg.enabled };
        chrome.storage.sync.set({ settings: updated }, () => {
          // Tell the active tab's content script
          chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]?.id) {
              chrome.tabs.sendMessage(tabs[0].id, {
                type: msg.enabled ? 'INJECT_OVERLAY' : 'REMOVE_OVERLAY',
              }).catch(() => {});
            }
          });
        });
      });
      sendResponse({ ok: true });
      break;

    // Settings changed from popup
    case 'SETTINGS_UPDATED':
      chrome.storage.sync.get('settings', ({ settings }) => {
        const updated = { ...settings, ...msg.patch };
        chrome.storage.sync.set({ settings: updated }, () => {
          // Notify all active tabs
          chrome.tabs.query({}, (tabs) => {
            tabs.forEach(tab => {
              if (tab.id) {
                chrome.tabs.sendMessage(tab.id, {
                  type: 'SETTINGS_CHANGED',
                  settings: updated,
                }).catch(() => {});
              }
            });
          });
          sendResponse({ ok: true, settings: updated });
        });
      });
      return true; // async

    // Popup requests current settings
    case 'GET_SETTINGS':
      chrome.storage.sync.get('settings', ({ settings }) => {
        sendResponse({ settings: settings || DEFAULT_SETTINGS });
      });
      return true; // async

    // Site exclusion
    case 'EXCLUDE_SITE':
      if (msg.hostname) {
        chrome.storage.sync.get({ excludedSites: [] }, ({ excludedSites }) => {
          if (!excludedSites.includes(msg.hostname)) {
            excludedSites.push(msg.hostname);
            chrome.storage.sync.set({ excludedSites });
            console.log('[Signlytic] Excluded site:', msg.hostname);
          }
        });
      }
      sendResponse({ ok: true });
      break;

    default:
      break;
  }

  return false;
});

// Clean up when a tab closes
chrome.tabs.onRemoved.addListener((tabId) => {
  activeTabs.delete(tabId);
});

// Auto-inject overlay on page load -- content script handles dedup
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete') return;
  if (!tab.url) return;
  // Never inject on chrome:// or extension pages
  if (tab.url.startsWith('chrome://') ||
      tab.url.startsWith('chrome-extension://') ||
      tab.url.startsWith('devtools://') ||
      tab.url.startsWith('about:')) return;

  chrome.storage.sync.get(['settings', 'excludedSites'], ({ settings, excludedSites }) => {
    const s = settings || DEFAULT_SETTINGS;
    if (!s.enabled) return;
    // Check site exclusion list
    try {
      const hostname = new URL(tab.url).hostname;
      if ((excludedSites || []).includes(hostname)) return;
    } catch (_) {}
    // Small delay so content script is ready
    setTimeout(() => {
      chrome.tabs.sendMessage(tabId, { type: 'INJECT_OVERLAY' }).catch(() => {});
    }, 500);
  });
});

// Remove overlay on tab navigation start (before new page loads)
chrome.webNavigation?.onBeforeNavigate?.addListener((details) => {
  if (details.frameId !== 0) return; // main frame only
  chrome.tabs.sendMessage(details.tabId, { type: 'REMOVE_OVERLAY' }).catch(() => {});
});
