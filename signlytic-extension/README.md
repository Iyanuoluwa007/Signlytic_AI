# Signlytic AI Chrome Extension

The BSL signing overlay for browser tabs: captions on the page in, a signing
panel out. This directory is the extension source. Version 0.4.1, Manifest V3.

## Install it

The built release, install instructions and usage notes live in the overlay
repository, which is where the extension is published from:

**[Download v0.4.1](https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases/download/v0.4.1/signlytic-extension.zip)**
&nbsp;|&nbsp;
[install instructions](https://github.com/Iyanuoluwa007/Signlytic-Overlay/tree/main/Extension)
&nbsp;|&nbsp;
[extension page](https://signlytic-ai-website.vercel.app/extension)

It is a manual install, not a Chrome Web Store listing yet: unzip, open
`chrome://extensions`, enable Developer Mode, and use Load Unpacked.

For signing across the whole desktop rather than only browser tabs, see
[signlytic-desktop/](../signlytic-desktop), which runs on Windows and macOS.

## Layout

    manifest.json          MV3 manifest
    background.js          Service worker: message hub, auto-inject
    content_script.js      Caption detection, microphone, iframe positioning
    gloss/converter.js     English to BSL gloss rules
    overlay/
      overlay.html         Panel UI
      overlay.js           Sign queue and fingerspell fallback
      skeleton2d.js        2D skeleton renderer
      avatar3d.js          3D avatar, Three.js GLB bone driver
      three.min.js         Bundled Three.js r128
      GLTFLoader.js        Bundled GLTFLoader
    popup/                 Settings popup
    data/signs/core/       192 bundled BSL sign pose JSONs
    icons/                 icon16/48/128.png

## The renderers are shared, and this is the copy they come from

`overlay/skeleton2d.js` and `overlay/avatar3d.js` are the source of truth for
both signing renderers. They are copied verbatim into the other two surfaces:

- `signlytic-desktop/renderer/vendor/`
- `signlytic-ai-website/public/bsl/`

Change them here and resync, rather than editing a copy. The copies are
byte-identical to this directory apart from line endings, and a change that
lands in only one place is the failure this arrangement is meant to prevent.

## Sign data

192 signs are bundled so the extension works offline. Anything outside that set
is fetched from the Signlytic API, and falls back to fingerspelling if it cannot
be fetched, so an unknown word is never silently dropped.
