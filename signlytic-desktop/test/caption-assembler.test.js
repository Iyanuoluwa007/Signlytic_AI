// Tests for CaptionAssembler, which turns the rolling Live Captions buffer
// into finished sentences. Run with `npm test` in signlytic-desktop (Node's
// built-in runner, no dependencies).
//
// Time is always passed in as the second argument to push(), so nothing here
// sleeps or uses a real timer. Every buffer is made-up English text typed by
// hand; no caption recordings and no sign data are involved.

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  CaptionAssembler,
  splitSentences,
  normalise,
} = require("../main/captions/caption-assembler");

// An assembler that has already swallowed its first buffer, which is how it
// spends its whole life after attaching to the caption window.
function primed(options, firstBuffer = "Earlier speech that was on screen.") {
  const a = new CaptionAssembler(options);
  assert.deepEqual(a.push(firstBuffer, 0), []);
  return { a, history: firstBuffer };
}

test("normalise ignores case, punctuation and spacing", () => {
  assert.equal(normalise("  Thank you,   VERY much!  "), "thank you very much");
  assert.equal(normalise("yesterday. Thank you"), normalise("Yesterday, thank you"));
  assert.equal(normalise("?!"), "");
});

test("splitSentences splits several sentences and keeps the unfinished tail apart", () => {
  assert.deepEqual(splitSentences("One two. Three four! Five six? seven"), {
    sentences: ["One two.", "Three four!", "Five six?"],
    fragment: "seven",
  });
});

test("splitSentences collapses line breaks and keeps closing quotes with their sentence", () => {
  assert.deepEqual(splitSentences('She said "go home."\nThen   she left.'), {
    sentences: ['She said "go home."', "Then she left."],
    fragment: "",
  });
});

test("splitSentences treats text with no sentence end as a fragment", () => {
  assert.deepEqual(splitSentences("still talking"), {
    sentences: [],
    fragment: "still talking",
  });
});

test("splitSentences does not break inside a decimal number or a web address", () => {
  assert.deepEqual(splitSentences("It costs 3.50 pounds. Visit example.com today."), {
    sentences: ["It costs 3.50 pounds.", "Visit example.com today."],
    fragment: "",
  });
});

test("the buffer already on screen when attaching is never replayed", () => {
  const a = new CaptionAssembler();
  const old = "Good morning everyone. Welcome to the meeting.";
  assert.deepEqual(a.push(old, 0), []);
  // Still there much later, and still there once new speech follows it.
  assert.deepEqual(a.push(old, 60_000), []);
  assert.deepEqual(a.push(old + " Let us begin now. And", 61_000), ["Let us begin now."]);
});

test("a fragment in flight when attaching is emitted once it completes", () => {
  const a = new CaptionAssembler();
  assert.deepEqual(a.push("so what I wanted", 0), []);
  assert.deepEqual(a.push("so what I wanted to say is this. And", 500), [
    "so what I wanted to say is this.",
  ]);
});

test("a trailing sentence is held until it has been stable for stableMs", () => {
  const { a, history } = primed({ stableMs: 1200 });
  const buffer = history + " Please take a seat.";
  assert.deepEqual(a.push(buffer, 1000), []);
  assert.deepEqual(a.push(buffer, 2199), []);
  assert.deepEqual(a.push(buffer, 2200), ["Please take a seat."]);
  // Released once only, however long the buffer then sits there.
  assert.deepEqual(a.push(buffer, 9000), []);
});

test("stableMs is configurable", () => {
  const { a, history } = primed({ stableMs: 300 });
  const buffer = history + " Please take a seat.";
  assert.deepEqual(a.push(buffer, 1000), []);
  assert.deepEqual(a.push(buffer, 1299), []);
  assert.deepEqual(a.push(buffer, 1300), ["Please take a seat."]);
});

test("a sentence with more text after it is released at once", () => {
  const { a, history } = primed();
  assert.deepEqual(a.push(history + " Please take a seat. We will", 1000), [
    "Please take a seat.",
  ]);
});

test("fragments shorter than minWords are not emitted", () => {
  const { a, history } = primed({ minWords: 2 });
  const buffer = history + " I. Okay.";
  assert.deepEqual(a.push(buffer, 1000), []);
  assert.deepEqual(a.push(buffer, 5000), []);
});

test("minWords is configurable", () => {
  const { a, history } = primed({ minWords: 4 });
  assert.deepEqual(a.push(history + " Thank you all. Thank you all very much. And", 1000), [
    "Thank you all very much.",
  ]);
});

test("a word corrected before the sentence is final gives one corrected sentence", () => {
  const { a, history } = primed({ stableMs: 1200 });
  assert.deepEqual(a.push(history + " I went to the shop.", 1000), []);
  // Corrected 500 ms later, which also restarts the stability clock.
  assert.deepEqual(a.push(history + " I went to the shops.", 1500), []);
  assert.deepEqual(a.push(history + " I went to the shops.", 2699), []);
  assert.deepEqual(a.push(history + " I went to the shops.", 2700), ["I went to the shops."]);
});

test("the same sentence scrolling past twice is emitted once", () => {
  const { a, history } = primed();
  assert.deepEqual(a.push(history + " The train leaves at nine. It", 1000), [
    "The train leaves at nine.",
  ]);
  // The window scrolls: the oldest line goes, the sentence is still on screen.
  assert.deepEqual(a.push("The train leaves at nine. It is on platform two. Do", 2000), [
    "It is on platform two.",
  ]);
  assert.deepEqual(a.push("It is on platform two. Do not be late.", 3000), []);
  assert.deepEqual(a.push("It is on platform two. Do not be late.", 4200), ["Do not be late."]);
});

test("a late punctuation or casing fix does not re-emit a sentence", () => {
  const { a, history } = primed();
  assert.deepEqual(a.push(history + " Thank you, everyone. Now", 1000), ["Thank you, everyone."]);
  assert.deepEqual(a.push(history + " thank you everyone! Now then", 1500), []);
});

test("several new sentences arriving in one buffer come out in order", () => {
  const { a, history } = primed();
  assert.deepEqual(
    a.push(history + " First we eat. Then we walk home. After that", 1000),
    ["First we eat.", "Then we walk home."]
  );
});

test("a rewrite that extends an already released sentence is not emitted again", () => {
  // Live Captions merges sentences when it revises punctuation: "Thank you."
  // can become "Thank you very much." after it was released.
  const { a, history } = primed({ stableMs: 1200 });
  assert.deepEqual(a.push(history + " Thank you.", 1000), []);
  assert.deepEqual(a.push(history + " Thank you.", 2200), ["Thank you."]);
  assert.deepEqual(a.push(history + " Thank you very much. Next", 3000), []);
});

test("empty and non-text buffers are ignored", () => {
  const a = new CaptionAssembler();
  assert.deepEqual(a.push("", 0), []);
  assert.deepEqual(a.push("   \n ", 0), []);
  assert.deepEqual(a.push(null, 0), []);
  assert.deepEqual(a.push(undefined, 0), []);
  // None of those counted as the first buffer.
  assert.equal(a.primed, false);
});

test("reset forgets history and swallows the next buffer again", () => {
  const { a, history } = primed();
  assert.deepEqual(a.push(history + " We are live now. And", 1000), ["We are live now."]);
  a.reset();
  assert.deepEqual(a.push("We are live now. And we continue.", 2000), []);
  assert.deepEqual(a.push("We are live now. And we continue. Good to see you. So", 3000), [
    "Good to see you.",
  ]);
});

test("history is bounded, so the oldest sentences are eventually forgotten", () => {
  const { a } = primed();
  for (let i = 0; i < 205; i++) {
    a.push(`Sentence number ${i} here. And`, 1000 + i);
  }
  assert.equal(a.seen.size, 200);
  assert.equal(a.seen.has(normalise("Sentence number 204 here.")), true);
  assert.equal(a.seen.has(normalise("Sentence number 0 here.")), false);
});
