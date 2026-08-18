# Prompt: build the macOS version of the Signlytic desktop app

Paste everything below the line into a fresh Claude Code session, run from a
clone of this repository on the MacBook. It is written to be self-contained:
that session will have none of the context from the Windows work.

---

I want to add macOS support to the Signlytic AI desktop app in this repository.
The Windows version works and is shipped, and nothing about it should regress.

## What the app is

`signlytic-desktop/` is an Electron app that shows a floating, frameless,
always-on-top British Sign Language signer. It reads the system's live captions,
converts the text to BSL glosses, and animates an avatar signing them, so a Deaf
user can follow any audio playing on the machine rather than only what is in a
browser.

Read these first, in this order:

- `signlytic-desktop/README.md`
- `signlytic-desktop/main/main.js` for the window, positioning, IPC and startup
  guards
- `signlytic-desktop/main/captions/caption-stream.js` for the caption source
- `signlytic-desktop/main/captions/caption-assembler.js` for turning a rolling
  caption buffer into finalised sentences
- `signlytic-desktop/renderer/desktop.js` for the UI, settings panel and
  playback

## How captions work on Windows, and why that shapes the macOS design

Windows ships a Live Captions app. The Electron main process spawns a PowerShell
sidecar, `main/captions/live-captions.ps1`, which attaches to that app through
UI Automation and prints JSON lines on stdout. `caption-stream.js` spawns it,
parses those lines, and emits finalised sentences.

Two details cost a lot of time to discover, and macOS will have equivalents:

1. **The caption buffer is not a stream.** It is a cumulative rolling window
   revised in place after the fact. "Thank you." became "Thank you very much."
   two seconds later. Two sentences sometimes merge into one. Old lines scroll
   off. Speech from before you attach is already sitting in it.
   `caption-assembler.js` handles all of that: it swallows the buffer at attach,
   holds the newest sentence until it stops changing, compares on
   punctuation-and-case-stripped text, and skips sentences that merely extend
   one already released. An anchor-based diff was tried first and failed exactly
   on the merge case. **Reuse this assembler unchanged.** Feed it raw buffer
   text; do not do sentence splitting in the macOS layer.

2. **Packaged builds behave differently from `npm start`.** electron-builder
   packs `main/` into `app.asar`, which Electron can read but external processes
   cannot. A helper invoked by a path inside the asar fails instantly, with exit
   code 4294770688 on Windows. The fix is `asarUnpack` in `package.json` plus
   rewriting `app.asar` to `app.asar.unpacked` at runtime; see the `SIDECAR`
   constant in `caption-stream.js`. **Any macOS helper must do the same, and
   must be tested from a packaged build.** This bug shipped in three Windows
   releases because only `npm start` was ever tested.

## What to build

`CaptionStream.capabilities()` in `caption-stream.js` is the single place that
decides platform support. For darwin it currently returns:

```js
{ supported: false, reason: "System captions on macOS are not wired up yet; use the text box" }
```

Everything else is already platform-neutral. Manual text entry, the 2D and 3D
renderers, the settings panel, the speed control and window positioning should
all work on macOS today. **Verify that claim before writing any capture code**,
by running the app on the Mac, and report what actually works rather than
assuming it.

The job is to give macOS a real caption source, and to make the app installable.

### The caption source is the hard part

macOS has Live Captions from Ventura onward, but there is no public API to read
its output and it is not exposed the way the Windows one is. Investigate before
choosing. Options, roughly in order of how sanctioned they are:

- **Accessibility API (`AXUIElement`)** to read the on-screen text of the Live
  Captions window. This is the closest analogue to the Windows approach. It
  needs the user to grant Accessibility permission in System Settings, and needs
  a helper because Node cannot call these APIs directly. Check whether that
  window actually exposes its text through AX before committing to it. If it
  does not, that route is dead: say so plainly rather than building around it.
- **`SFSpeechRecognizer`** against microphone input. Fully supported, needs
  microphone and speech-recognition permission, and covers in-person
  conversation, which is arguably the more valuable case on a laptop.
- **ScreenCaptureKit audio capture** plus your own transcription. Heaviest, and
  apps serving DRM-protected video will refuse to be captured.

Whichever you choose, keep the same shape as Windows: a helper process printing
JSON lines, consumed by `caption-stream.js`, feeding the existing assembler. Do
not fork the pipeline.

### Packaging

`package.json` already declares a mac dmg target with category
`public.app-category.utilities`. It has never been built. Expect to handle:

- entitlements for the microphone, and for Accessibility if you go that way
- `Info.plist` usage descriptions, without which macOS terminates the app rather
  than prompting
- code signing and notarisation. If no Developer ID is available, say so and
  document what a user must do to open an unsigned build, rather than implying
  it is signed.

## How I want you to work

- **Verify rather than assume.** Run the packaged app, capture real output, and
  show it. If you claim something works, show the evidence. If you could not
  test something, say so explicitly rather than implying coverage.
- **Do not regress Windows.** `capabilities()` must keep its Windows behaviour
  exactly. Check that no shared file starts assuming darwin.
- **Keep the shared renderer shared.** `renderer/vendor/` is synced from
  `signlytic-extension/overlay/` by `scripts/sync-vendor.js`, and the extension
  is the source of truth. Edit the extension copy and run `npm run sync-vendor`.
  Never edit the vendor copy directly.
- **No emojis or icons in code files**, and no em-dashes anywhere in prose,
  comments or commit messages.
- Commit messages should explain why, including what was tried and rejected.
  Add specific paths; do not use `git add -A`.
- Ask before anything irreversible or outward-facing, such as publishing a
  release.

## Definition of done

1. The app runs on macOS and manual text entry drives the signing avatar.
2. A caption source works end to end, or, if you concluded none is viable, a
   written explanation of what you tested and what the evidence was.
3. It builds a dmg that launches on a clean machine, verified from the packaged
   build rather than from source.
4. `capabilities()` reports accurately on both platforms.
5. Windows behaviour is unchanged.
6. `signlytic-desktop/README.md` documents the macOS requirements, the
   permissions a user must grant, and how to open an unsigned build if it is not
   notarised.

Start by reading the files listed above and running the app on the Mac. Tell me
what works before changing anything.
