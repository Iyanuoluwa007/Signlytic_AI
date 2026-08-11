// Signlytic 2D skeleton renderer.
//
// Shared by the browser extension overlay and the website BSL panel so both
// draw from one implementation. Depends on window.PoseNormaliser from
// avatar3d.js, which must be loaded first. Deliberately does NOT depend on
// Three.js: 2D mode must work without downloading the WebGL engine or the
// avatar model.
//
// Exposes the same playback surface as ThreeAvatarRenderer (playQueue /
// stopQueue / pause / resume / speed) so a host can swap between them.

const BODY_CONNECTIONS_2D = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
];
const HAND_CONNECTIONS_2D = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];

const SK2D = {
  BG:       '#06080f',
  BODY:     '#5eead4',
  HAND:     '#34d399',
  ACCENT:   'rgba(14,124,107,0.2)',
  ACCENT_B: 'rgba(94,234,212,0.4)',
  MUTED:    '#3d4a5c',
  FPS:      25,
};

class SkeletonRenderer2D {
  // options:
  //   speed         playback rate multiplier
  //   sizeProvider  () => ({ w, h }) when the host controls sizing
  //   normalise     false to skip pose repair (host already normalised)
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.speed  = options.speed || 1.0;
    this._sizeProvider = options.sizeProvider || null;
    this._normaliser = (options.normalise === false || !window.PoseNormaliser)
      ? null
      : new window.PoseNormaliser();

    this._queue    = [];
    this._signIdx  = 0;
    this._frameIdx = 0;
    this._timer    = null;
    this._paused   = false;
    this._onGlossChange = null;
    this._onDone        = null;

    this.ready = true;
  }

  // ── Sizing ────────────────────────────────────────────────────────────────
  _syncSize() {
    let w, h;
    if (this._sizeProvider) {
      const s = this._sizeProvider();
      w = s.w; h = s.h;
    } else {
      w = this.canvas.clientWidth  || this.canvas.width  || 400;
      h = this.canvas.clientHeight || this.canvas.height || 220;
    }
    w = Math.floor(w); h = Math.floor(h);
    if (w > 10 && this.canvas.width  !== w) this.canvas.width  = w;
    if (h > 10 && this.canvas.height !== h) this.canvas.height = h;
    return { W: this.canvas.width || 400, H: this.canvas.height || 220 };
  }

  clear() {
    const { W, H } = this._syncSize();
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = SK2D.BG;
    this.ctx.fillRect(0, 0, W, H);
  }

  // ── Fingerspelling fallback for glosses with no pose data ─────────────────
  drawFingerspell(gloss) {
    const ctx = this.ctx;
    const { W, H } = this._syncSize();
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = SK2D.BG;
    ctx.fillRect(0, 0, W, H);
    if (!gloss) return;

    const letters = String(gloss).toUpperCase().split('');
    const padding = 8;
    const fontSize = Math.min(38, Math.floor((W - padding * 2) / Math.max(letters.length, 1) * 0.85));
    const letterW = fontSize * 0.72;
    const totalW = letters.length * letterW + (letters.length - 1) * 4;
    const startX = (W - totalW) / 2 + letterW / 2;

    letters.forEach((letter, i) => {
      const x = startX + i * (letterW + 4);
      const boxX = x - letterW / 2;
      const boxY = H / 2 - fontSize * 0.65;
      ctx.fillStyle = SK2D.ACCENT;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(boxX, boxY, letterW, fontSize * 1.3, 5);
      else ctx.rect(boxX, boxY, letterW, fontSize * 1.3);
      ctx.fill();
      ctx.strokeStyle = SK2D.ACCENT_B;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = SK2D.BODY;
      ctx.font = `600 ${fontSize}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(letter, x, H / 2 + 1);
    });

    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = SK2D.MUTED;
    ctx.textAlign = 'center';
    ctx.fillText('fingerspell', W / 2, H - 8);
  }

  // ── Draw one pose frame ───────────────────────────────────────────────────
  drawFrame(frame, currentGloss) {
    if (this._normaliser) frame = this._normaliser.normalise(frame);
    const ctx = this.ctx;
    const { W, H } = this._syncSize();
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = SK2D.BG;
    ctx.fillRect(0, 0, W, H);
    if (!frame) { this.drawFingerspell(currentGloss); return; }

    const pt = (lms, i) => (lms && lms[i]) ? { x: lms[i][0] * W, y: lms[i][1] * H } : null;
    const lines = (lms, conns, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      conns.forEach(([a, b]) => {
        const p1 = pt(lms, a), p2 = pt(lms, b);
        if (!p1 || !p2) return;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      });
    };
    const dots = (lms, color, r = 2.5) => {
      ctx.fillStyle = color;
      if (!lms) return;
      lms.forEach((p) => {
        if (!p || p[0] == null) return;
        ctx.beginPath(); ctx.arc(p[0] * W, p[1] * H, r, 0, Math.PI * 2); ctx.fill();
      });
    };

    ctx.lineCap = 'round'; ctx.lineJoin = 'round';

    // Head circle at the nose landmark, sized from shoulder width
    const nose = pt(frame.body, 0);
    if (nose) {
      const lSh = pt(frame.body, 11), rSh = pt(frame.body, 12);
      const headR = (lSh && rSh) ? Math.abs(lSh.x - rSh.x) * 0.22 : 10;
      ctx.strokeStyle = SK2D.BODY;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(nose.x, nose.y, Math.max(headR, 8), 0, Math.PI * 2);
      ctx.stroke();
    }

    lines(frame.body, BODY_CONNECTIONS_2D, SK2D.BODY);
    dots(frame.body, SK2D.BODY);
    lines(frame.lh, HAND_CONNECTIONS_2D, SK2D.HAND);
    dots(frame.lh, SK2D.HAND, 2);
    lines(frame.rh, HAND_CONNECTIONS_2D, SK2D.HAND);
    dots(frame.rh, SK2D.HAND, 2);
  }

  // ── Playback, mirroring the ThreeAvatarRenderer surface ───────────────────
  // queue: [{ gloss, frames: [frame|null, ...] }, ...]
  playQueue(queue, onGlossChange, onDone) {
    this.stopQueue();
    if (this._normaliser) this._normaliser.reset();
    this._queue    = queue || [];
    this._signIdx  = 0;
    this._frameIdx = 0;
    this._paused   = false;
    this._onGlossChange = onGlossChange;
    this._onDone        = onDone;
    this._schedule();
  }

  stopQueue() {
    clearTimeout(this._timer);
    this._timer  = null;
    this._paused = false;
    this._queue  = [];
    this.clear();
  }

  pause() {
    if (!this._timer || this._paused) return;
    clearTimeout(this._timer);
    this._timer = null;
    this._paused = true;
  }

  resume() {
    if (!this._paused || !this._queue.length) return;
    this._paused = false;
    this._schedule();
  }

  _schedule() {
    const fps = SK2D.FPS * (this.speed || 1.0);
    this._timer = setTimeout(() => this._tick(), 1000 / fps);
  }

  _tick() {
    if (!this._queue.length) { if (this._onDone) this._onDone(); return; }
    const sign = this._queue[this._signIdx];
    if (!sign) { if (this._onDone) this._onDone(); return; }

    if (this._frameIdx === 0 && this._onGlossChange) this._onGlossChange(this._signIdx);

    this.drawFrame(sign.frames[this._frameIdx], sign.gloss);

    this._frameIdx++;
    if (this._frameIdx >= sign.frames.length) {
      this._frameIdx = 0;
      this._signIdx++;
      if (this._signIdx >= this._queue.length) {
        if (this._onDone) this._onDone();
        return;
      }
    }
    this._schedule();
  }

  resize() { this._syncSize(); }

  destroy() {
    this.stopQueue();
    this.ctx = null;
  }
}

window.SkeletonRenderer2D = SkeletonRenderer2D;
window.BODY_CONNECTIONS_2D = BODY_CONNECTIONS_2D;
window.HAND_CONNECTIONS_2D = HAND_CONNECTIONS_2D;
