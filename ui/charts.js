// Telemetry charts drawn on canvas, styled for the black cards.
// Outcome colors are categorical slots validated for color-blind safety on black;
// crashes use a recessive gray so successes stand out as the cloud turns blue.

import { OUTCOME_LABEL } from "./nn.js";

export const OUTCOME_COLORS = ["#3987e5", "#d95926", "#199e70", "#6e6d68", "#6e6d68"];
const GROUPS = [
  { label: "Landed on pad", codes: [0], color: OUTCOME_COLORS[0] },
  { label: "Landed off pad", codes: [1], color: OUTCOME_COLORS[1] },
  { label: "Timed out", codes: [2], color: OUTCOME_COLORS[2] },
  { label: "Crashed or flew off", codes: [3, 4], color: OUTCOME_COLORS[3] },
];
const AXIS_TEXT = "rgba(255,255,255,0.45)";
const GRID = "rgba(255,255,255,0.08)";
const BASELINE = "rgba(255,255,255,0.25)";
const MONO = '10px "JetBrains Mono", ui-monospace, monospace';
const LABEL = '500 10px "Inter", system-ui, sans-serif';
const START_HEIGHT = 10; // meters: flights from higher up score more, since scores count the distance covered
const startHeight = (g) => g.config?.conditions?.start_height ?? START_HEIGHT;
/** Room above the standard scale (−400 to 330) for pilots that start high and score more. */
const scoreTop = (scores) => Math.max(330, Math.ceil((scores.reduce((a, b) => Math.max(a, b), -Infinity) + 30) / 100) * 100);
const solvedLabel = (generations) => (generations.some((g) => startHeight(g) !== START_HEIGHT) ? "200 = solved from 10 m" : "200 = solved");
const fmt = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}

/** Tooltip rows built with textContent only. rows: [{value, label, color, muted}] */
function fillTip(tip, title, rows) {
  tip.replaceChildren();
  const head = document.createElement("div");
  head.className = "muted";
  head.textContent = title;
  tip.append(head);
  for (const row of rows) {
    const line = document.createElement("div");
    if (row.color) {
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = row.color;
      line.append(key);
    }
    if (row.value !== undefined) {
      const value = document.createElement("span");
      value.className = row.big ? "value" : "mono";
      value.textContent = row.value;
      line.append(value, " ");
    }
    const label = document.createElement("span");
    label.className = row.big ? "" : "muted";
    label.textContent = row.label || "";
    line.append(label);
    tip.append(line);
  }
}

class Chart {
  constructor(container, margin) {
    this.el = container;
    this.margin = margin;
    this.canvas = document.createElement("canvas");
    this.tip = document.createElement("div");
    this.tip.className = "tooltip";
    this.empty = document.createElement("div");
    this.empty.className = "empty-note";
    container.append(this.canvas, this.tip, this.empty);
    this.ctx = this.canvas.getContext("2d");
    this.hoverAt = null;
    new ResizeObserver(() => this.resize()).observe(container);
    this.canvas.addEventListener("pointermove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hoverAt = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.request();
    });
    this.canvas.addEventListener("pointerleave", () => { this.hoverAt = null; this.request(); });
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.w = this.el.clientWidth;
    this.h = this.el.clientHeight;
    this.canvas.width = this.w * dpr;
    this.canvas.height = this.h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.request();
  }

  request() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      if (!this.w) return;
      this.ctx.clearRect(0, 0, this.w, this.h);
      this.draw();
    });
  }

  get plot() {
    const m = this.margin;
    return { left: m.left, top: m.top, right: this.w - m.right, bottom: this.h - m.bottom,
      width: this.w - m.left - m.right, height: this.h - m.top - m.bottom };
  }

  setEmpty(text) {
    this.empty.textContent = text || "";
    this.empty.hidden = !text;
  }

  showTip(x, y, title, rows) {
    fillTip(this.tip, title, rows);
    this.tip.classList.add("show");
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    const left = x + 14 + tw > this.w ? x - 14 - tw : x + 14;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${Math.min(Math.max(0, y - th / 2), this.h - th)}px`;
  }

  hideTip() { this.tip.classList.remove("show"); }

  yAxis(ticks, sy, format = (v) => fmt.format(v)) {
    const { ctx } = this;
    const p = this.plot;
    ctx.font = MONO;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of ticks) {
      const y = Math.round(sy(t)) + 0.5;
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p.left, y); ctx.lineTo(p.right, y); ctx.stroke();
      ctx.fillStyle = AXIS_TEXT;
      ctx.fillText(format(t), p.left - 8, y);
    }
  }

  xAxis(ticks, sx, format) {
    const { ctx } = this;
    const p = this.plot;
    ctx.strokeStyle = BASELINE;
    ctx.beginPath(); ctx.moveTo(p.left, p.bottom + 0.5); ctx.lineTo(p.right, p.bottom + 0.5); ctx.stroke();
    ctx.font = MONO;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = AXIS_TEXT;
    for (const t of ticks) ctx.fillText(format(t), sx(t), p.bottom + 8);
  }

  genMarks(marks, sx) {
    const { ctx } = this;
    const p = this.plot;
    ctx.font = LABEL;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    for (const { x, gen, height } of marks) {
      const px = Math.round(sx(x)) + 0.5;
      if (px < p.left || px > p.right) continue;
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.beginPath(); ctx.moveTo(px, p.top - 4); ctx.lineTo(px, p.bottom); ctx.stroke();
      ctx.fillStyle = AXIS_TEXT;
      ctx.fillText(`GEN ${gen}${height === START_HEIGHT ? "" : ` · ${height} m`}`, px + 5, p.top - 4);
    }
  }
}


export class ScoreChart extends Chart {
  constructor(el) {
    super(el, { left: 48, right: 56, top: 22, bottom: 30 });
    this.episodes = [];
    this.marks = [];
  }

  set(episodes, generations) {
    this.episodes = episodes;
    // Mark where each generation's attempts begin in this list (deleted or filtered-out generations leave gaps).
    const first = new Map();
    episodes.forEach((e, i) => { if (!first.has(e[1])) first.set(e[1], i); });
    this.marks = generations.filter((g) => first.get(g.gen) > 0)
      .map((g) => ({ x: first.get(g.gen), gen: g.gen, height: startHeight(g) }));
    this.solved = solvedLabel(generations);
    this.hi = episodes.length ? scoreTop(episodes.map((e) => e[2])) : 330;
    this.rolling = new Float32Array(episodes.length);
    let sum = 0;
    for (let i = 0; i < episodes.length; i++) {
      sum += episodes[i][2];
      if (i >= 100) sum -= episodes[i - 100][2];
      this.rolling[i] = sum / Math.min(i + 1, 100);
    }
    this.request();
  }

  draw() {
    const { ctx, episodes } = this;
    this.setEmpty(episodes.length ? "" : "No attempts yet. Train a pilot to watch this fill up.");
    if (!episodes.length) return this.hideTip();
    const p = this.plot;
    const n = Math.min(episodes.length, this.rolling.length); // the list grows in place while training
    if (!n) return this.hideTip();
    const lo = -400, hi = this.hi;
    const sx = (i) => p.left + (i / Math.max(n - 1, 1)) * p.width;
    const sy = (v) => p.top + (1 - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * p.height;
    this.yAxis([-400, -200, 0, 200, 400, 600, 800, 1000].filter((t) => t < hi), sy);
    this.xAxis(niceTicks(0, n, 6).filter((t) => t <= n), (t) => sx(Math.max(t - 1, 0)), (t) => fmt.format(t));
    this.genMarks(this.marks, sx);

    const solved = Math.round(sy(200)) + 0.5;
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.beginPath(); ctx.moveTo(p.left, solved); ctx.lineTo(p.right, solved); ctx.stroke();
    ctx.font = MONO; ctx.fillStyle = AXIS_TEXT; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText(this.solved, p.left + 6, solved - 4);

    const size = n > 6000 ? 1.6 : n > 2000 ? 2 : 2.6;
    ctx.globalAlpha = n > 6000 ? 0.55 : 0.75;
    for (let i = 0; i < n; i++) {
      const e = episodes[i];
      ctx.fillStyle = OUTCOME_COLORS[e[3]];
      ctx.fillRect(sx(i) - size / 2, sy(e[2]) - size / 2, size, size);
    }
    ctx.globalAlpha = 1;

    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    const step = Math.max(1, Math.floor(n / 1500));
    for (let i = 0; i < n; i += step) (i ? ctx.lineTo : ctx.moveTo).call(ctx, sx(i), sy(this.rolling[i]));
    ctx.lineTo(sx(n - 1), sy(this.rolling[n - 1]));
    ctx.stroke();
    ctx.lineWidth = 1;
    const endY = sy(this.rolling[n - 1]);
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(sx(n - 1), endY, 4, 0, Math.PI * 2); ctx.fill();
    ctx.font = '600 12px "JetBrains Mono", monospace';
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(this.rolling[n - 1].toFixed(0).replace("-", "−"), sx(n - 1) + 9, endY);

    if (!this.hoverAt) return this.hideTip();
    const i = Math.round(((this.hoverAt.x - p.left) / p.width) * (n - 1));
    if (i < 0 || i >= n) return this.hideTip();
    const e = episodes[i];
    const x = Math.round(sx(i)) + 0.5;
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.beginPath(); ctx.moveTo(x, p.top); ctx.lineTo(x, p.bottom); ctx.stroke();
    ctx.fillStyle = OUTCOME_COLORS[e[3]];
    ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx(i), sy(e[2]), 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.lineWidth = 1;
    this.showTip(x, sy(e[2]), `Attempt ${fmt.format(e[0])} · gen ${e[1]}`, [
      { value: e[2].toFixed(1), label: "score", big: true },
      { color: OUTCOME_COLORS[e[3]], label: OUTCOME_LABEL[e[3]] },
      { value: this.rolling[i].toFixed(1), label: "average of last 100", color: "#fff" },
      { value: `${(e[4] / 50).toFixed(1)} s`, label: "flight time" },
    ]);
  }
}

export class OutcomeChart extends Chart {
  constructor(el) {
    super(el, { left: 44, right: 12, top: 14, bottom: 30 });
    this.episodes = [];
  }

  set(episodes) {
    this.episodes = episodes;
    this.request();
  }

  get block() {
    return 100 * Math.max(1, Math.ceil(this.episodes.length / 6000));
  }

  draw() {
    const { ctx, episodes } = this;
    this.setEmpty(episodes.length ? "" : "Waiting for attempts.");
    if (!episodes.length) return this.hideTip();
    const p = this.plot;
    const block = this.block;
    const blocks = [];
    for (let start = 0; start < episodes.length; start += block) {
      const slice = episodes.slice(start, start + block);
      if (slice.length < Math.min(20, block) && blocks.length) break;
      blocks.push({ start, size: slice.length, shares: GROUPS.map((g) => slice.filter((e) => g.codes.includes(e[3])).length / slice.length) });
    }
    const slot = p.width / Math.max(blocks.length, 10);
    const barW = Math.min(24, slot * 0.72);
    const sy = (v) => p.top + (1 - v) * p.height;
    this.yAxis([0, 0.25, 0.5, 0.75, 1], sy, (v) => `${v * 100}%`);
    const sx = (i) => p.left + slot * (i + 0.5);
    const labelEvery = Math.ceil(blocks.length / 6);
    this.xAxis(blocks.map((_, i) => i).filter((i) => i % labelEvery === 0), sx, (i) => fmt.format(blocks[i].start + blocks[i].size));

    let hovered = null;
    blocks.forEach((b, i) => {
      let y = p.bottom;
      const x = sx(i) - barW / 2;
      const isHover = this.hoverAt && Math.abs(this.hoverAt.x - sx(i)) <= slot / 2;
      if (isHover) hovered = { b, i };
      GROUPS.forEach((g, k) => {
        const h = b.shares[k] * p.height;
        if (h <= 0) return;
        ctx.fillStyle = g.color;
        ctx.globalAlpha = isHover || !this.hoverAt ? 1 : 0.55;
        ctx.fillRect(x, y - h + (y < p.bottom ? 2 : 0), barW, h - (y < p.bottom ? 2 : 0));
        y -= h;
      });
    });
    ctx.globalAlpha = 1;
    if (!hovered) return this.hideTip();
    const { b, i } = hovered;
    this.showTip(sx(i), p.top + p.height / 3, `Attempts ${fmt.format(b.start + 1)}–${fmt.format(b.start + b.size)}`,
      GROUPS.map((g, k) => ({ value: `${Math.round(b.shares[k] * 100)}%`, label: g.label, color: g.color })));
  }
}

export class TouchdownChart extends Chart {
  constructor(el) {
    super(el, { left: 56, right: 12, top: 18, bottom: 30 });
    this.points = [];
  }

  set(episodes) {
    this.points = episodes.filter((e) => e[5] !== null && e[5] !== undefined);
    this.request();
  }

  draw() {
    const { ctx, points } = this;
    this.setEmpty(points.length ? "" : "Touchdown positions are recorded for pilots trained here.");
    if (!points.length) return this.hideTip();
    const p = this.plot;
    const first = points[0][0], last = points[points.length - 1][0];
    const sx = (x) => p.left + ((Math.max(-1, Math.min(1, x)) + 1) / 2) * p.width;
    const sy = (n) => p.bottom - ((n - first) / Math.max(last - first, 1)) * p.height;

    ctx.fillStyle = "rgba(255,255,255,0.07)";
    ctx.fillRect(sx(-0.2), p.top, sx(0.2) - sx(-0.2), p.height);
    ctx.font = LABEL; ctx.fillStyle = AXIS_TEXT; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("PAD", sx(0), p.top - 3);
    this.yAxis(niceTicks(first, last, 4).filter((t) => t >= first && t <= last), sy);
    this.xAxis([-1, -0.5, 0, 0.5, 1], sx, (v) => (v === 0 ? "0" : v > 0 ? `+${v}` : `${v}`.replace("-", "−")));

    const size = points.length > 5000 ? 1.8 : 2.4;
    ctx.globalAlpha = 0.75;
    for (const e of points) {
      ctx.fillStyle = OUTCOME_COLORS[e[3]];
      ctx.fillRect(sx(e[5]) - size / 2, sy(e[0]) - size / 2, size, size);
    }
    ctx.globalAlpha = 1;
    if (!this.hoverAt) return this.hideTip();
    let best = null, bestD = 16 * 16;
    for (const e of points) {
      const dx = sx(e[5]) - this.hoverAt.x, dy = sy(e[0]) - this.hoverAt.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = e; }
    }
    if (!best) return this.hideTip();
    ctx.fillStyle = OUTCOME_COLORS[best[3]];
    ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx(best[5]), sy(best[0]), 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.lineWidth = 1;
    this.showTip(sx(best[5]), sy(best[0]), `Attempt ${fmt.format(best[0])} · gen ${best[1]}`, [
      { value: `${best[5] >= 0 ? "+" : ""}${best[5].toFixed(2)}`, label: "from pad center (pad is ±0.20)", big: true },
      { color: OUTCOME_COLORS[best[3]], label: OUTCOME_LABEL[best[3]] },
      { value: `${best[6].toFixed(1)} m/s`, label: "speed just before the end" },
    ]);
  }
}

export class TestChart extends Chart {
  constructor(el) {
    super(el, { left: 48, right: 24, top: 22, bottom: 30 });
    this.tests = [];
    this.marks = [];
  }

  set(tests, generations) {
    this.tests = tests;
    this.marks = generations.filter((g) => g.gen > 1 && g.frames_start)
      .map((g) => ({ x: g.frames_start, gen: g.gen, height: startHeight(g) }));
    this.heights = new Map(generations.map((g) => [g.gen, startHeight(g)]));
    this.request();
  }

  draw() {
    const { ctx, tests } = this;
    this.setEmpty(tests.length ? "" : "The first test runs after 100k frames of practice.");
    if (!tests.length) return this.hideTip();
    const p = this.plot;
    const maxX = Math.max(...tests.map((t) => t[0]), 1);
    const lo = Math.min(-200, ...tests.map((t) => t[2])), hi = scoreTop(tests.map((t) => t[2]));
    const sx = (x) => p.left + (x / maxX) * p.width;
    const sy = (v) => p.top + (1 - (v - lo) / (hi - lo)) * p.height;
    this.yAxis(niceTicks(lo, hi, 5), sy);
    this.xAxis(niceTicks(0, maxX, 6), sx, (v) => compact.format(v));
    this.genMarks(this.marks, sx);
    const solved = Math.round(sy(200)) + 0.5;
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.beginPath(); ctx.moveTo(p.left, solved); ctx.lineTo(p.right, solved); ctx.stroke();

    ctx.strokeStyle = OUTCOME_COLORS[0];
    ctx.lineWidth = 2;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    tests.forEach((t, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, sx(t[0]), sy(t[2])));
    ctx.stroke();
    ctx.lineWidth = 2;
    let hovered = null, bestD = Infinity;
    for (const t of tests) {
      ctx.fillStyle = OUTCOME_COLORS[0];
      ctx.strokeStyle = "#000";
      ctx.beginPath(); ctx.arc(sx(t[0]), sy(t[2]), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (this.hoverAt) {
        const d = Math.abs(sx(t[0]) - this.hoverAt.x);
        if (d < bestD) { bestD = d; hovered = t; }
      }
    }
    ctx.lineWidth = 1;
    if (!hovered || bestD > 40) return this.hideTip();
    const x = Math.round(sx(hovered[0])) + 0.5;
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.beginPath(); ctx.moveTo(x, p.top); ctx.lineTo(x, p.bottom); ctx.stroke();
    const height = this.heights.get(hovered[1]) ?? START_HEIGHT;
    this.showTip(x, sy(hovered[2]), `After ${fmt.format(hovered[0])} frames · gen ${hovered[1]}${height === START_HEIGHT ? "" : ` · from ${height} m`}`, [
      { value: hovered[2].toFixed(1), label: "average test score", big: true },
      { value: `${hovered[3]} of ${hovered[4]}`, label: "landed on the pad", color: OUTCOME_COLORS[0] },
    ]);
  }
}

export function legend(el) {
  el.replaceChildren(...GROUPS.map((g) => {
    const item = document.createElement("span");
    const dot = document.createElement("i");
    dot.style.background = g.color;
    item.append(dot, g.label);
    return item;
  }));
}
