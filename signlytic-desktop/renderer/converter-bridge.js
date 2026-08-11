// The extension's gloss converter is an ES module, and the rest of the
// renderer is classic scripts. This bridges the two. It must be a real file
// rather than an inline module because the page's CSP forbids inline script.
//
// Only a fallback: the hosted API produces better BSL word order. This keeps
// the app usable offline.
import { textToGloss } from "./vendor/converter.js";

window.signlyticTextToGloss = textToGloss;
