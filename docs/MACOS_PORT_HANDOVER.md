# macOS port: what was done, what is verified, what is not

Written on the MacBook at the end of the macOS session, to be read on the
Windows machine. Companion to `MACOS_PORT_PROMPT.md`, which was the brief.

Everything below was run on macOS 26.5.2, Apple Silicon, against the app in
`signlytic-desktop/`.

## Short version

macOS now has three working caption sources instead of none. Windows is
untouched and was checked rather than assumed. Two commits are on the branch
`macos-captions`; the third caption source is written, packaged and documented
but **not committed**, because one of its two verification steps is still
outstanding.

## Git state, and what is left to commit

Branch: `macos-captions`, off `main`. The repository lives on the shared
external drive, so both commits and the working tree travel with it.

Committed:

    23c8dfd  Give macOS a real caption source, by recognising speech
    63e6a3b  Correct the record: the Live Captions window can be read after all

Uncommitted, all of it the third caption source:

    main/captions/mac/caption-source.swift    LiveCaptionsSource, lazy recogniser
    main/captions/caption-stream.js           third source, exit code 7
    main/main.js                              three-value audio source pref
    renderer/index.html                       third button
    renderer/desktop.js                       third option and its note
    README.md                                 permissions table per source
    main/captions/README.md                   the accessibility findings

Nothing outside `signlytic-desktop/` was touched. Do not use `git add -A`: the
drive is exFAT and litters the tree with `._` AppleDouble files. They are now
gitignored, but only inside `signlytic-desktop/`.

The two commits are authored `Oke Iyanuoluwa E. <oke.iyanuoluwa12@gmail.com>`
to match the rest of the history. `user.name` and `user.email` were set
**repository-local** on the Mac, because none was configured and the first
commit picked up a machine-derived identity. Check the Windows side has its own
identity set before committing.

## How captions work on macOS

`CaptionStream.capabilities()` is still the only place that decides platform
support. On darwin it now returns
`{ supported: true, source: "macos-speech-recognition" }`.

One helper binary, three modes, selected by the Listen to setting:

| Setting | Helper argument | How it gets text | Permissions |
| --- | --- | --- | --- |
| Microphone (default) | `--source mic` | SFSpeechRecognizer on the mic | Microphone, Speech Recognition |
| System audio | `--source system` | ScreenCaptureKit into SFSpeechRecognizer | Screen Recording |
| Live Captions | `--source captions` | reads the macOS caption window over the Accessibility API | Accessibility |

The helper prints the same JSON lines the PowerShell sidecar prints, so
`caption-assembler.js` is **shared unchanged** and there are no platform
branches after the buffer. That was a requirement of the brief and it held.

## Verified, with evidence

Run from the packaged app installed to `/Applications` from the mounted dmg,
not from source.

Microphone and system audio, three sentences spoken aloud:

    caption in (livecaptions): The weather is good today.
    caption in (livecaptions): Thank you very much.
    caption in (livecaptions): I am going to work tomorrow.
    glosses: ["TOMORROW","I","GO","WORK"]

Gloss order is correct BSL, time marker first.

The shared assembler was replayed against the real captured macOS buffer
sequence: three sentences out, no duplicates, no fragments.

Windows was checked by forcing `process.platform` to `win32`:

- `capabilities()` returns `windows-live-captions`
- the spawn is still `powershell.exe`, same arguments, `-ParentPid`, `windowsHide`
- exit code 2 still reads "Live Captions window not found"
- exit code 3 does **not** mention permissions on Windows; the macOS codes are branched

## Not verified

**The Live Captions source has never been seen working end to end.** The
Accessibility permission it needs was not granted before the session ended, so
every run of it stops at:

    Signlytic AI needs Accessibility permission to read the Live Captions
    window. Allow it under System Settings, Privacy and Security,
    Accessibility, then start the app again.

That failure path is correct and was verified: the right exit code, the
helper's own message shown instead of a bare code, and no restart loop. What is
unproven is the success path inside the packaged app.

The reader itself was proven separately, before it was wired in, using
`main/captions/mac/tools/ax-probe.swift`:

    [3.7s] The weather is good today
    [5.6s] The weather is good today. | I am going to work tomorrow
    [7.7s] The weather is good today. | I am going to work tomorrow. | Thank you very much.

To finish it on a Mac: switch on System Settings, Accessibility, Live Captions;
grant Accessibility to Signlytic AI; pick Live Captions under Listen to; press
Start Captions. If it works, commit. If it does not, the fault is in
`LiveCaptionsSource` in `caption-source.swift`, not in the reading approach.

Also untested: Intel hardware. The build is universal and both slices are
present, but it has only ever run on Apple Silicon.

## Building

Windows is unchanged: `npm run dist` as before.

macOS needs the Swift toolchain from the Command Line Tools, which is much
smaller than full Xcode:

    xcode-select --install
    npm install
    npm run dist          # builds the helper, then packages

`npm run build:mac-helper` compiles `caption-source.swift` into a universal
binary, embeds an Info.plist into it and ad-hoc signs it. It is a no-op off
macOS, so it cannot break a Windows build.

## Four things that cost real time, so they do not cost it twice

1. **npm 11 does not run install scripts.** `npm install` leaves
   `node_modules/electron` with no binary and `npm start` fails with no useful
   error. Run `node node_modules/electron/install.js` once.

2. **Never build onto the exFAT drive.** exFAT has no native extended
   attributes, so macOS writes `._` sidecars beside every file.
   electron-builder's own `*.asar` glob matches `._app.asar`, tries to parse it
   as an archive, and dies with an out-of-range offset. Working from the drive
   is fine, only the output has to go elsewhere:

       npx electron-builder --mac dmg --universal --config.directories.output=~/signlytic-build

3. **macOS reads the usage description from the responsible process.** For a
   helper the app spawns, that is the app, not the helper. Get it wrong and the
   process is killed outright with
   `__TCC_CRASHING_DUE_TO_PRIVACY_VIOLATION__` and no prompt at all. The
   strings are in `extendInfo` in `package.json` and also compiled into the
   helper with `-sectcreate`, so it works standalone too.

4. **`app.asar.unpacked` only holds what `asarUnpack` lists.** `main.js` and the
   renderer load from inside `app.asar`. Copying updated files into the
   unpacked directory to test a change does nothing and silently runs the old
   code. This produced one false pass during the session. Rebuild, and confirm
   with `asar extract` if in doubt.

## Distribution status

`signlytic-desktop/dist/Signlytic AI-0.3.7-universal.dmg`, 217 MB, universal
(x86_64 and arm64). Mounts, installs by drag, launches, and all three sources
appear in the settings panel.

**It is not signed and not notarised.** There is no Apple Developer ID for this
project, so electron-builder reports "skipped macOS application code signing"
and the entitlements in `packaging/entitlements.mac.plist` are never applied.
Consequences a tester will actually hit:

- Downloaded from anywhere, macOS attaches a quarantine flag and refuses to
  open it, usually saying the app is damaged. Right-click, Open, then Open
  again, or `xattr -dr com.apple.quarantine "/Applications/Signlytic AI.app"`.
- Permissions granted to it may be dropped whenever the app is rebuilt, because
  ad-hoc signing gives it a new code identity each time. This looks like a bug
  and is not one.

So it is fine for testers who are told the above, and not fine for general
release. Fixing it needs a paid Apple Developer account, after which set
`CSC_LINK` and `CSC_KEY_PASSWORD` and add a `notarize` block to the `mac`
section of `package.json`. No other change is needed.

## Dev switches added

    SIGNLYTIC_CAPTION_LOG=path   copy the macOS helper's JSON lines to a file

The helper is spawned by a windowed app, so its stdout goes nowhere. For an app
started with `open`, set the variable with `launchctl setenv` first, or it will
not be inherited.

Note also that `SIGNLYTIC_SHOT` only ever fires on the 3D renderer, because
that is the only one that reports when it is ready. The 2D renderer and the
packaged build were checked instead by starting the app with
`--remote-debugging-port` and driving it over the DevTools protocol.
