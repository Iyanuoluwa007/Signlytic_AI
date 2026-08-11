# Signlytic AI Desktop

Turns speech into British Sign Language on the desktop: caption text in, a
signing avatar out.

    npm install
    npm start

## What works

Caption text reaches the renderer, is converted to BSL glosses, has its pose
frames fetched per gloss, and is played on the selected renderer. Sentences
are queued, so speech arriving faster than it can be signed is not cut off
mid-sign.

Two renderers, matching the website:

- **2D skeleton** (default) - no model download, and currently the more
  accurate of the two. Fingerspells any gloss with no pose data.
- **3D avatar** - loads Three.js and the Mixamo model on demand. The hands
  still sit lower than they should; that fix is outstanding.

## Platform support

The pipeline after the text arrives is identical everywhere. Only the caption
*source* is OS-specific.

| | Windows | macOS |
| --- | --- | --- |
| Manual text entry | yes | yes |
| System captions | yes, Windows 11 Live Captions | not yet |
| 2D and 3D signing | yes | yes |

`CaptionStream.capabilities()` is the single place that decides this. The
renderer asks it on start-up and disables the caption controls with a reason
where there is no source, so the app is usable on macOS today through the
text box.

**macOS captions are not wired up.** macOS has Live Captions from Ventura,
but it is read through the Accessibility API rather than UI Automation, needs
a signed helper and explicit Accessibility permission from the user, and is a
separate native implementation. It slots in behind the same interface when
built; nothing else has to change.

## Packaging

`npm run dist` builds with electron-builder (install it first; it is not a
dependency here). Targets are configured for Windows NSIS and macOS DMG. A
macOS build has to be produced on macOS for signing and notarisation.

## Layout

    main/main.js                 window, IPC, caption routing
    main/preload.js              context bridge
    main/captions/               see the README in that folder
    renderer/desktop.js          engines, playback queue, UI
    renderer/sign-source.js      text -> glosses -> pose frames
    renderer/converter-bridge.js exposes the extension's ES module converter
    renderer/vendor/             synced from the extension, do not edit

`renderer/vendor` is generated. The browser extension's `overlay/` folder is
the single source of truth for the avatar engine, the 2D renderer and the
gloss converter; `npm run sync-vendor` copies them and runs automatically
before `start`.

## Sign data

Glosses and pose frames come from the deployed website API, which already has
CDN and Redis caching in front of it. Bundling the 5,203-sign set for offline
use is a later decision.

## Dev switches

    SIGNLYTIC_CAPTIONS_AUTOSTART=1   start the caption reader without a click
    SIGNLYTIC_TEST_TEXT="..."        inject a sentence as if it were a caption
    SIGNLYTIC_SHOT=path.png          screenshot once the avatar is ready, then exit
