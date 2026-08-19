# Caption input

Turns whatever the machine can hear into finished sentences for the avatar.
Manual text entry remains as a permanent fallback.

Windows reads the Live Captions window. macOS recognises speech. Both print
the same thing, so everything downstream is shared.

## Why PowerShell rather than a native sidecar

The original plan was a compiled C# or Rust sidecar. This machine has neither
a .NET SDK nor a Rust toolchain, and requiring one just to read a caption
string is a poor trade. Windows already ships the UI Automation assemblies,
and PowerShell can load them, so the reader is a plain script: no build step,
no native addon, nothing to install.

## Pieces

    live-captions.ps1        Windows: attaches to the Live Captions window
                             over UI Automation, one JSON line per change
    mac/caption-source.swift macOS: recognises speech and reports the same
                             kind of rolling buffer
    mac/tools/ax-probe.swift investigation only, not shipped
    caption-assembler.js     turns the raw buffer into finished sentences
    caption-stream.js        spawns the right helper and wires them together

The sidecar is deliberately dumb. It reports the raw caption buffer and does
no interpretation, because the hard part is much easier to reason about (and
to test) in one place in JavaScript.

## What the caption buffer actually does

This drove the whole design, and it is not a stream of new text:

- It is cumulative. The element holds the last few lines, not the latest words.
- It is revised in place, after the fact. In testing "Thank you." became
  "Thank you very much." two seconds later, and "yesterday. Thank you" was
  re-punctuated to "yesterday, thank you".
- Revisions can merge two sentences into one.
- Old lines scroll away once it is long enough.
- Whatever was said before attaching is already sitting there.

So `CaptionAssembler`:

1. Swallows the buffer present at attach, so earlier speech is not replayed.
2. Holds the newest sentence back until either more text arrives after it or
   it stops changing (`stableMs`), since it is still being corrected.
3. Compares sentences with punctuation and casing stripped, so a late fix does
   not read as new speech, and skips a sentence that merely extends one
   already released.

Verified against buffers actually recorded from Live Captions: no replay of
pre-attach history, no duplicates, no half sentences.

## Element identifiers

    LiveCaptionsDesktopWindow   window class
    CaptionsTextBlock           the caption text
    ReadyToCaptionTextBlock     shown while open but idle

## Notes and limits

- Live Captions transcribes system audio, so it captures any app's sound, not
  just a microphone. That is what makes this useful for video calls.
- Live Captions must be running. `captions-launch-app` starts it, but its
  first ever launch asks the user to accept terms and download a speech model,
  which cannot be automated.
- A sentence still in flight at attach time will be released once it
  completes, so the first sentence may predate pressing start.
- Set `SIGNLYTIC_CAPTIONS_AUTOSTART=1` to start the reader without a click,
  which is how the pipeline is tested headlessly.

## macOS

### The system Live Captions window, and what reading it actually does

**It can be read.** This was tested rather than assumed, and the first guess
was wrong, so the evidence is recorded here.

`mac/tools/ax-probe.swift` attaches to Live Captions and dumps its whole
Accessibility tree. Run it with Live Captions switched on **and audio playing**:

    swiftc -O main/captions/mac/tools/ax-probe.swift -o /tmp/ax-probe
    /tmp/ax-probe

The window is hidden until there is something to caption, so a probe run
against a silent machine reports `windows: 0` and looks like a dead end. With
audio playing it reports:

    AXWindow [AXSystemFloatingWindow] id=AXLiveCaptionsWindow
      AXTitle = "Live Captions"
      AXStaticText.AXValue = "The weather is good today."

The caption text sits in `AXStaticText` children of that window, one per line,
and reading them in order gives a rolling buffer with the same behaviour as the
Windows one: cumulative, revised in place, oldest lines scrolling away. It
arrives **already punctuated**, which is the part the recognition helper has to
reconstruct for itself, and it tracks speech closely, with text appearing
within a few seconds of it being spoken.

The window's own menu exposes `Computer Audio` and `Microphone`, so the source
can be switched the same way the Windows helper switches it.

Element identifiers, the macOS counterpart of the Windows ones below:

    com.apple.accessibility.LiveTranscriptionAgent   the process
    AXLiveCaptionsWindow                             the window identifier
    AXStaticText                                     the caption lines

### All three are offered

Reading that window is a real option, so it is one of the three sources the
helper supports, chosen with `--source captions`. It is not the default, for
reasons that are about cost to the user rather than about feasibility:

- It needs **Accessibility permission**, which is the most powerful permission
  on the machine, and grants the ability to read and control every other app.
  That is a lot to ask in order to caption a film.
- It needs the user to switch Live Captions on themselves, in System Settings,
  and its first run downloads a speech model. None of that can be automated,
  and the window only exists while it is running.
- It is an Apple internal app with no published API, so the tree could change
  in a point release.
- It captions what the Mac plays. Speech in the room is covered only by
  switching its own Microphone option on, which is another thing to drive.

`SFSpeechRecognizer` is public, runs on device for British English, needs no
Accessibility permission, and covers both audio sources directly, so the
microphone is what the app starts on.

The Live Captions source is the least code of the three. It needs none of the
reconstruction below, because the window is already a punctuated rolling
buffer, so it reports what it reads and stops there. That is the same job the
PowerShell sidecar does on Windows, which is the point: three sources, one
assembler, no branches after the buffer.

### What the recognised transcript actually does

Applies to the microphone and system audio sources only. The Live Captions
source has none of these problems.

Same shape as the Windows buffer, cumulative and revised in place, with three
differences that all cost time to find:

1. **It does not reliably end a sentence with a full stop.** "The weather is
   good today" comes back bare. The assembler splits on punctuation, so a bare
   tail reads as a sentence still being spoken and is held back for ever.
   Questions do come back with a question mark, which is exactly why this
   looked like it was working at first.

2. **It starts a new segment without saying so.** No final result, no error:
   the transcript simply stops being an extension of what it was, going from
   "the weather is good today" straight to "Thank". If that is not noticed the
   previous sentence is overwritten and never signed.

3. **A finished result ends the task, it does not pause it.** Recognition is
   dead from that point until a new request is opened, and a finished result
   arrives after every pause in speech.

So the helper keeps a `committed` prefix and a `live` tail and reports the two
joined, which is what makes the buffer continuous across all of that. A tail
is committed, with a full stop added, when it stops changing for a second.

**Silence is what closes a sentence, not divergence.** Divergence was tried
first and failed the same way the Windows anchor diff did: recognition revises
heavily in place, and a revision that rewrites the opening words is
indistinguishable from a new segment. It produced fragments like "Calligraphy
hair." out of one sentence that was still being corrected. Divergence is still
used, but only as a secondary signal and only when the text also got shorter,
which a revision does not do.

None of this is sentence splitting. The helper only reconstructs a continuous
buffer; `caption-assembler.js` still decides what a sentence is, and is shared
unchanged.

### Permissions, and the crash that is not a bug in this code

A process asking for the microphone or for speech recognition without a usage
description is **killed outright**, with
`__TCC_CRASHING_DUE_TO_PRIVACY_VIOLATION__` and no prompt. There is no polite
failure to handle.

The string has to be in the Info.plist of the process macOS holds
*responsible*, which for a helper the app spawns is the app, not the helper.
Both carry the strings anyway: the app's via `extendInfo` in `package.json`,
the helper's compiled into its own binary with `-sectcreate __TEXT
__info_plist` so it also works when run on its own.

### Exit codes

`caption-stream.js` turns these into something the user can act on, and does
not retry any of them, because no number of restarts will change a refused
permission.

    3   speech recognition permission refused
    4   microphone permission refused
    5   screen recording permission refused (system audio only)
    6   no recogniser for British English
    7   Accessibility permission refused (Live Captions source only)

Where the helper explained the problem itself, that message is shown instead of
the exit description: "code 7" says what happened, not what to do about it.
