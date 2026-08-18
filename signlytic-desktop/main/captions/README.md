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

### Why not read the system Live Captions window

macOS has had Live Captions since Ventura, and reading it would be the direct
analogue of the Windows approach. It was not chosen, for reasons that are
worth keeping:

- Apple publishes no API for reading it. Anything that worked would be reading
  the Accessibility tree of an Apple internal app
  (`com.apple.accessibility.LiveTranscriptionAgent`, which lives inside
  `AccessibilitySharedSupport.framework`), and could break in a point release.
- It would need the user to grant Accessibility permission, which is the most
  powerful permission on the machine, to caption a film.
- It only captions what the Mac plays. Speech in the room, which is the case a
  laptop user hits most, would still not be covered.

Speech recognition has none of those problems: `SFSpeechRecognizer` is public,
runs on device for British English, and can listen to either the microphone or
the system's own audio.

`mac/tools/ax-probe.swift` tests the Accessibility route rather than leaving it
to assumption. It attaches to Live Captions and dumps its whole Accessibility
tree, marking any readable text. Build and run it with Live Captions switched
on:

    swiftc -O main/captions/mac/tools/ax-probe.swift -o /tmp/ax-probe
    /tmp/ax-probe

### What the recognised transcript actually does

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
