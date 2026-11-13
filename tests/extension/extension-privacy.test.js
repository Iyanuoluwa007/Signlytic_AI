// What the Chrome extension does on a page, checked against its real source.
//
// These tests load signlytic-extension/content_script.js and background.js
// into a fake page and a fake browser: no Chrome, no network, no microphone.
// They exist because two privacy bugs shipped in 0.4.4 without anyone seeing
// them: the popup's off switch was ignored on newly opened pages, and Auto
// mode asked for the microphone on every kind of website.
//
// Time is a fake clock moved by hand, so nothing here waits. The fake browser
// answers asynchronously, as Chrome does; answering at once would run the
// script's callbacks before the script has finished loading.
//
// Run with: node --test tests/extension/extension-privacy.test.js

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const EXTENSION = path.join(__dirname, "..", "..", "signlytic-extension");
const CONTENT_SCRIPT = fs.readFileSync(path.join(EXTENSION, "content_script.js"), "utf8");
const BACKGROUND = fs.readFileSync(path.join(EXTENSION, "background.js"), "utf8");
const MANIFEST = JSON.parse(fs.readFileSync(path.join(EXTENSION, "manifest.json"), "utf8"));

function fakeClock() {
  let now = 0;
  let nextId = 1;
  const timers = [];
  const add = (fn, ms, repeat) => {
    const timer = { id: nextId++, fn, at: now + ms, ms, repeat, dead: false };
    timers.push(timer);
    return timer.id;
  };
  const cancel = (id) => {
    const timer = timers.find((t) => t.id === id);
    if (timer) timer.dead = true;
  };
  return {
    setTimeout: (fn, ms = 0) => add(fn, ms, false),
    setInterval: (fn, ms) => add(fn, ms, true),
    clearTimeout: cancel,
    clearInterval: cancel,
    now: () => now,
    advance(ms) {
      const end = now + ms;
      for (;;) {
        const due = timers.filter((t) => !t.dead && t.at <= end).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        now = due.at;
        if (due.repeat) due.at += due.ms;
        else due.dead = true;
        due.fn();
      }
      now = end;
    },
  };
}

// Loads the content script as a page on `host` would, with the given saved
// settings, and reports what it did.
function openPage({
  host = "example.org",
  settings = { enabled: true, captionSource: "auto" },
  excludedSites = [],
  captionsOnPage = false,
} = {}) {
  const clock = fakeClock();
  const page = { overlays: 0, micStarts: 0, micRunning: false, statuses: [] };
  const stored = { settings, excludedSites };
  const listeners = [];

  const frame = {
    style: {},
    offsetWidth: 432,
    offsetHeight: 432,
    contentWindow: {
      postMessage: (message) => {
        if (message.type === "STATUS" || message.type === "COUNTDOWN") page.statuses.push(message.message);
      },
    },
    addEventListener: (event, fn) => { if (event === "load") clock.setTimeout(fn, 0); },
    remove: () => { page.overlays -= 1; },
  };

  class FakeSpeechRecognition {
    start() { page.micStarts += 1; page.micRunning = true; }
    stop() { page.micRunning = false; }
  }

  const sandbox = {
    console: { log() {}, warn() {} },
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
    Date: { now: clock.now },
    MutationObserver: class { observe() {} disconnect() {} },
    chrome: {
      runtime: {
        lastError: null,
        sendMessage: (message, reply) => {
          if (message.type === "GET_SETTINGS" && reply) {
            clock.setTimeout(() => reply({ settings: stored.settings }), 0);
          }
        },
        onMessage: { addListener: (fn) => listeners.push(fn) },
        getURL: (file) => "chrome-extension://test/" + file,
      },
      storage: {
        sync: {
          get: (defaults, reply) => {
            const answer = {};
            for (const key of Object.keys(defaults)) {
              answer[key] = stored[key] === undefined ? defaults[key] : stored[key];
            }
            clock.setTimeout(() => reply(answer), 0);
          },
        },
      },
    },
    document: {
      getElementById: () => null,
      createElement: () => frame,
      documentElement: { appendChild: () => { page.overlays += 1; } },
      body: {},
      querySelector: () => (captionsOnPage ? {} : null),
      querySelectorAll: () => [],
    },
  };
  sandbox.window = {
    location: { hostname: host },
    innerHeight: 800,
    innerWidth: 1200,
    addEventListener() {},
    webkitSpeechRecognition: FakeSpeechRecognition,
  };

  vm.createContext(sandbox);
  vm.runInContext(CONTENT_SCRIPT, sandbox);

  return {
    page,
    stored,
    wait: (ms) => clock.advance(ms),
    // A message from the extension's background script or popup.
    receive: (message) => { listeners.forEach((fn) => fn(message)); clock.advance(1); },
  };
}

const ON_AUTO = { enabled: true, captionSource: "auto" };
const OFF_AUTO = { enabled: false, captionSource: "auto" };

// ---------------------------------------------------------------------------
// The off switch
// ---------------------------------------------------------------------------

test("switched off: a newly opened page gets no panel and no microphone", () => {
  for (const host of ["example.org", "www.youtube.com", "meet.example.com"]) {
    const { page, wait } = openPage({ host, settings: OFF_AUTO });
    wait(60_000);
    assert.equal(page.overlays, 0, `${host}: panel was added`);
    assert.equal(page.micStarts, 0, `${host}: microphone was started`);
    assert.deepEqual(page.statuses, [], `${host}: caption scanning started`);
  }
});

test("switched off: a page with captions is not read either", () => {
  const { page, wait } = openPage({ host: "www.youtube.com", settings: OFF_AUTO, captionsOnPage: true });
  wait(60_000);
  assert.equal(page.overlays, 0);
  assert.deepEqual(page.statuses, []);
});

test("switched off: an inject message does not bring the panel back", () => {
  const { page, receive } = openPage({ settings: OFF_AUTO });
  receive({ type: "INJECT_OVERLAY" });
  assert.equal(page.overlays, 0);
});

test("switched back on: the inject message adds the panel, using the saved settings", () => {
  const { page, stored, receive, wait } = openPage({ settings: OFF_AUTO });
  wait(1000);
  assert.equal(page.overlays, 0);
  stored.settings = ON_AUTO; // what the background script saves before it sends the message
  receive({ type: "INJECT_OVERLAY" });
  assert.equal(page.overlays, 1);
});

test("switched on while a panel is showing: a remove message takes it away and stops the microphone", () => {
  const { page, wait, receive } = openPage({ host: "www.youtube.com", settings: { enabled: true, captionSource: "mic" } });
  wait(3000);
  assert.equal(page.overlays, 1);
  assert.equal(page.micRunning, true);
  receive({ type: "REMOVE_OVERLAY" });
  assert.equal(page.overlays, 0);
  assert.equal(page.micRunning, false);
});

test("never set: with no saved on/off value the extension is on, as installed", () => {
  const { page, wait } = openPage({ settings: { captionSource: "captions" } });
  wait(1000);
  assert.equal(page.overlays, 1);
});

test("an excluded site gets no panel", () => {
  const { page, wait } = openPage({ host: "example.org", settings: ON_AUTO, excludedSites: ["example.org"] });
  wait(60_000);
  assert.equal(page.overlays, 0);
  assert.equal(page.micStarts, 0);
});

// ---------------------------------------------------------------------------
// When the microphone is used
// ---------------------------------------------------------------------------

test("Auto mode on an ordinary website never starts the microphone", () => {
  for (const host of ["example.org", "mail.example.com", "meet.example.com", "docs.example.net"]) {
    const { page, wait } = openPage({ host, settings: ON_AUTO });
    wait(10 * 60_000);
    assert.equal(page.overlays, 1, `${host}: panel missing`);
    assert.equal(page.micStarts, 0, `${host}: microphone was started`);
    assert.ok(!page.statuses.some((s) => /audio/i.test(s || "")), `${host}: audio countdown was shown`);
  }
});

test("Auto mode on a video site falls back to the microphone after the 5 second countdown", () => {
  for (const host of ["www.youtube.com", "www.bbc.co.uk", "www.netflix.com", "www.channel4.com"]) {
    const { page, wait } = openPage({ host, settings: ON_AUTO });
    wait(1500 + 4900); // panel load delay, then just short of 5 seconds
    assert.equal(page.micStarts, 0, `${host}: microphone started early`);
    wait(1000);
    assert.equal(page.micStarts, 1, `${host}: microphone did not start`);
  }
});

test("Auto mode on a video site with captions uses the captions, not the microphone", () => {
  const { page, wait } = openPage({ host: "www.youtube.com", settings: ON_AUTO, captionsOnPage: true });
  wait(60_000);
  assert.equal(page.overlays, 1);
  assert.equal(page.micStarts, 0);
});

test("Captions mode and Manual mode never start the microphone, on any site", () => {
  for (const captionSource of ["captions", "manual"]) {
    for (const host of ["example.org", "www.youtube.com"]) {
      const { page, wait } = openPage({ host, settings: { enabled: true, captionSource } });
      wait(10 * 60_000);
      assert.equal(page.micStarts, 0, `${captionSource} on ${host}`);
    }
  }
});

test("Microphone mode starts the microphone wherever the panel is showing", () => {
  const { page, wait } = openPage({ host: "example.org", settings: { enabled: true, captionSource: "mic" } });
  wait(3000);
  assert.equal(page.micStarts, 1);
});

// ---------------------------------------------------------------------------
// The background script's part of the off switch
// ---------------------------------------------------------------------------

function loadBackground({ tabs, settings }) {
  const sent = [];
  const stored = { settings };
  let onMessage;
  const sandbox = {
    console: { log() {} },
    setTimeout: () => 0,
    URL,
    chrome: {
      runtime: {
        onInstalled: { addListener() {} },
        onMessage: { addListener: (fn) => { onMessage = fn; } },
      },
      storage: {
        sync: {
          get: (key, reply) => reply({ settings: stored.settings }),
          set: (value, done) => { Object.assign(stored, value); if (done) done(); },
        },
      },
      tabs: {
        query: (filter, reply) => reply(filter.active ? tabs.filter((t) => t.active) : tabs),
        sendMessage: (tabId, message) => { sent.push({ tabId, type: message.type }); return Promise.resolve(); },
        onRemoved: { addListener() {} },
        onUpdated: { addListener() {} },
      },
      webNavigation: { onBeforeNavigate: { addListener() {} } },
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(BACKGROUND, sandbox);
  return { sent, stored, popupSends: (message) => onMessage(message, {}, () => {}) };
}

const THREE_TABS = [{ id: 1, active: false }, { id: 2, active: true }, { id: 3, active: false }];

test("turning off in the popup removes the panel from every open tab", () => {
  const { sent, stored, popupSends } = loadBackground({ tabs: THREE_TABS, settings: ON_AUTO });
  popupSends({ type: "TOGGLE_ENABLED", enabled: false });
  assert.equal(stored.settings.enabled, false);
  assert.deepEqual(sent, [
    { tabId: 1, type: "REMOVE_OVERLAY" },
    { tabId: 2, type: "REMOVE_OVERLAY" },
    { tabId: 3, type: "REMOVE_OVERLAY" },
  ]);
});

test("turning on in the popup adds the panel to the tab being looked at", () => {
  const { sent, stored, popupSends } = loadBackground({ tabs: THREE_TABS, settings: OFF_AUTO });
  popupSends({ type: "TOGGLE_ENABLED", enabled: true });
  assert.equal(stored.settings.enabled, true);
  assert.deepEqual(sent, [{ tabId: 2, type: "INJECT_OVERLAY" }]);
});

// ---------------------------------------------------------------------------
// Permissions
// ---------------------------------------------------------------------------

test("the manifest asks only for permissions the code uses", () => {
  const source = CONTENT_SCRIPT + BACKGROUND
    + fs.readFileSync(path.join(EXTENSION, "popup", "popup.js"), "utf8")
    + fs.readFileSync(path.join(EXTENSION, "overlay", "overlay.js"), "utf8");
  // Each permission and the browser API that needs it.
  const needs = {
    storage: /chrome\.storage\b/,
    tabs: /\btabs?\b[^\n]*\.url\b/,
    webNavigation: /chrome\.webNavigation\b/,
    scripting: /chrome\.scripting\b/,
    activeTab: /a^/, // grants nothing the extension's host access does not already give
  };
  assert.deepEqual([...MANIFEST.permissions].sort(), ["storage", "tabs", "webNavigation"]);
  for (const permission of MANIFEST.permissions) {
    assert.ok(needs[permission], `no reason recorded for ${permission}`);
    assert.match(source, needs[permission], `${permission} is requested but not used`);
  }
});
