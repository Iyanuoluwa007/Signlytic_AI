# Signlytic AI Desktop

Turns speech into British Sign Language on the desktop: caption text in, a
signing avatar out.

Built releases, if you want to use it rather than work on it:
**[Windows](https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases/download/desktop-v0.3.8/Signlytic.AI.Setup.0.3.8.exe)**
&nbsp;|&nbsp;
**[macOS](https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases/download/desktop-v0.3.7/Signlytic.AI-0.3.7-universal.dmg)**
&nbsp;|&nbsp;
[release notes](https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases/tag/desktop-v0.3.8).
Neither is code-signed, so both need talking past the OS once. See
[Opening an unsigned build](#opening-an-unsigned-build) for the macOS side,
which is stricter than the Windows one.

To run from source:

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
| System captions | yes, Windows 11 Live Captions | yes, speech recognition |
| Caption audio | whatever Windows captions | microphone or system audio |
| 2D and 3D signing | yes | yes |

`CaptionStream.capabilities()` is the single place that decides this. The
renderer asks it on start-up and disables the caption controls with a reason
where there is no source.

The two sources produce the same thing: a cumulative caption buffer, revised
in place, as JSON lines on stdout. `caption-assembler.js` turns that into
finished sentences and has no platform branches in it. See
`main/captions/README.md` for why macOS recognises speech rather than reading
the system Live Captions window.

## macOS

Needs **macOS 13 or later**. Runs natively on Apple Silicon and Intel; the
caption helper is a universal binary.

### Permissions

Captions do not work until these are granted. macOS asks for the first two the
first time you press Start Captions.

| Permission | When | Why |
| --- | --- | --- |
| Speech Recognition | first start | turns speech into text, on device |
| Microphone | first start, microphone source | listens to the room |
| Screen Recording | system audio source only | the only supported way to capture system audio |

Screen Recording is the awkward one. macOS offers no separate "record system
audio" permission, so capturing what the machine is playing is done through
ScreenCaptureKit and counts as screen recording, even though no picture is
ever captured. It is not granted by a prompt you can accept in place: allow
**Signlytic AI** under System Settings, Privacy and Security, Screen and
System Audio Recording, then start captions again. Video with copy protection
will refuse to be captured.

Nothing is uploaded. Recognition uses the on-device model, which the app
requests explicitly.

### Opening an unsigned build

**The dmg is not signed and not notarised.** There is no Apple Developer ID
for this project, so `electron-builder` reports "skipped macOS application
code signing" and Gatekeeper treats the result as untrusted.

Copied from a mounted dmg on the machine that built it, it opens normally.
Downloaded from anywhere, macOS attaches a quarantine flag and refuses to open
it with a message about damaged or unverified software. To open it anyway:

- Right-click the app in Finder, choose **Open**, then **Open** again in the
  dialog. This is per-app and only needed once, or

- clear the quarantine flag:

      xattr -dr com.apple.quarantine "/Applications/Signlytic AI.app"

Signing and notarising properly needs a paid Apple Developer account. With
one, set `CSC_LINK` and `CSC_KEY_PASSWORD` and add a `notarize` block to the
`mac` section of `package.json`; nothing else about the build has to change.

### Building on macOS

The caption helper is Swift and has to be compiled. It needs the Command Line
Tools, which is a much smaller install than full Xcode:

    xcode-select --install
    npm install
    npm run build:mac-helper    # runs automatically as part of npm run dist
    npm run dist

Two things that will waste time otherwise:

- **npm 11 does not run install scripts by default**, so `npm install` leaves
  `node_modules/electron` with no binary and `npm start` fails. Run
  `node node_modules/electron/install.js` once, or approve the script.

- **Do not build onto an exFAT or FAT volume.** exFAT has no native extended
  attributes, so macOS writes `._` AppleDouble sidecars next to every file.
  electron-builder matches `._app.asar` with its own `*.asar` glob, tries to
  parse it as an archive, and dies with an out-of-range offset error. Working
  from such a drive is fine; only the output has to go elsewhere:

      npx electron-builder --mac dmg --config.directories.output=~/build

## Packaging

`npm run dist` builds with electron-builder (install it first; it is not a
dependency here). Targets are configured for Windows NSIS and macOS DMG. A
macOS build has to be produced on macOS, because the caption helper is
compiled with the Swift toolchain there.

## Layout

    main/main.js                 window, IPC, caption routing
    main/preload.js              context bridge
    main/captions/               see the README in that folder
    main/captions/mac/           the macOS caption helper, in Swift
    packaging/                   entitlements for the signed app
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
    SIGNLYTIC_CAPTION_LOG=path       macOS: copy the caption helper's JSON lines
                                     to a file. The helper is spawned by a
                                     windowed app, so its stdout goes nowhere,
                                     and this is the only way to see what it
                                     actually reported. For a build started
                                     with `open`, set it with
                                     `launchctl setenv` first.

`SIGNLYTIC_SHOT` only fires on the 3D renderer, which is the only one that
reports when it is ready. To check the 2D renderer, or anything else in a
packaged build, start the app with `--remote-debugging-port` and drive it over
the DevTools protocol.
