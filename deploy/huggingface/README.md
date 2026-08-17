---
title: Signlytic AI
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time English to British Sign Language translation
---

# Signlytic AI

English and live captions translated into British Sign Language signing, in
your browser, on any device.

Type a sentence and the avatar signs it. The signing is rendered on your own
device, so this works on a phone as well as a laptop.

## What is not here

Video recognition, speech transcription and cloned voice output need a local
GPU, so they are not part of this online demo. They return a message pointing
at the full download rather than timing out. To try the complete system,
including BSL video to English and voice output, clone and run the project
locally:

https://github.com/Iyanuoluwa007/Signlytic_AI

## How it works

Text is converted to BSL glosses, then each gloss is looked up as a sequence of
pose frames and animated. The sign data lives behind the project's own API, so
this Space carries no dataset and starts quickly.

## Attribution

Signlytic AI, by Oke Iyanuoluwa Enoch. If you use, fork or build on this work,
please keep the copyright notice and credit the project.

The 3D character rig is built on third-party assets licensed for use in this
project but not for redistribution, so no model files are included here.
