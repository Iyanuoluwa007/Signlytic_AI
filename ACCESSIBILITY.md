# Accessibility

Signlytic AI exists for Deaf and hard of hearing people, so accessibility is
part of whether a change works, not a finishing touch. This page says what we
aim for, what is known to fall short today, and how to tell us when something
does not work for you.

## What Signlytic is, and is not

Signlytic turns English into British Sign Language (BSL) signing, and BSL
glosses or video into English. It is a research prototype. Its signing has not
yet been reviewed in a structured way by Deaf BSL users, it has no facial
expressions or other non-manual features, and it signs word by word rather than
with full BSL grammar. **It is not a replacement for a qualified BSL/English
interpreter**, and must not be presented as one.

## What we aim for

Across the website, the Chrome extension, the desktop app and the local
dashboard:

- **Everything signed is also shown as text.** The English and the name of
  each sign appear alongside the signing, so nobody depends on the animation
  alone.
- **Everything works from the keyboard**, in a sensible order, with a focus
  outline you can see.
- **Controls have names a screen reader can announce**, and status changes
  (loading, signing, errors) are announced, not only drawn.
- **Text and controls have enough contrast** against their background, and
  colour is never the only way something is shown.
- **You control the signing**: play, pause, stop and speed, and a choice of 2D
  skeleton or 3D avatar.
- **Plain English** in labels, help and error messages.
- **Video and audio we publish have captions.**
- **Nothing flashes** more than three times a second.

Our target is [WCAG 2.2](https://www.w3.org/TR/WCAG22/) level AA.

## Known gaps

We would rather list these than claim more than is true:

- Keyboard and screen-reader support on the website is incomplete (#8).
- The website signing panel does not yet work well on phones (#11).
- Animations do not yet respect the operating system's reduce-motion setting.
- The avatar has no facial expressions or mouth patterns, which carry meaning
  in BSL.
- Unknown words are not yet fingerspelled with the BSL two-handed alphabet
  (#2).
- The desktop builds are not code-signed, so installing them means getting past
  an operating system warning, which is harder for screen-reader users.

Issue numbers refer to
[this repository's issues](https://github.com/Iyanuoluwa007/Signlytic_AI/issues).

## Tell us when something does not work

- **Open an issue** with the
  [accessibility template](https://github.com/Iyanuoluwa007/Signlytic_AI/issues/new/choose).
  Say what you were trying to do, what assistive technology you use (if any),
  and what happened.
- **Not on GitHub?** Use the
  [feedback form](https://forms.gle/oTy7Bi414fuThFc1A).
- **BSL users:** if a sign looks wrong, unclear or unnatural, the
  BSL signing feedback template asks for the sentence and what a fluent signer
  would expect. That feedback is the most valuable thing this project can get.

## For contributors

Every change to an interface has to keep the commitments above. Before opening
a pull request that changes what people see or do:

- Use it with the keyboard alone, and check focus is visible throughout.
- Check new controls have an accessible name and new status messages are
  announced (a screen reader such as NVDA, VoiceOver or Narrator is best).
- Check contrast of any new colours.
- Do not remove text that accompanies signing.
- Changes to how BSL is signed need review by a fluent BSL user, ideally Deaf
  (the `needs BSL reviewer` label).

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up and test locally.
