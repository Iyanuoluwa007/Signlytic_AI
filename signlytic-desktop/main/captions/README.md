# Caption input

Reads Windows 11 Live Captions and turns it into finished sentences for the
avatar. Manual text entry remains as a permanent fallback.

## Why PowerShell rather than a native sidecar

The original plan was a compiled C# or Rust sidecar. This machine has neither
a .NET SDK nor a Rust toolchain, and requiring one just to read a caption
string is a poor trade. Windows already ships the UI Automation assemblies,
and PowerShell can load them, so the reader is a plain script: no build step,
no native addon, nothing to install.

## Pieces

    live-captions.ps1      attaches to the Live Captions window over UI
                           Automation and prints one JSON line per change
    caption-assembler.js   turns the raw buffer into finished sentences
    caption-stream.js      spawns the sidecar and wires the two together

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

## Not done yet

Caption text reaches the renderer and is displayed. Driving the avatar from it
(text to glosses to sign lookup to playQueue) is the next step.
