// PerfectLand mission control: wires the live stream, the worlds, the flight deck,
// the brain view, the charts and the controls together.

import GUI from "lil-gui";
import { createBrain } from "./brain.js";
import { legend, OUTCOME_COLORS, OutcomeChart, ScoreChart, TestChart, TouchdownChart } from "./charts.js";
import {
  ACTIONS, analyzeFlight, F, forward, INPUTS, OUTCOME_LABEL, OUTCOMES, potential, prepare, rewardBreakdown, THROTTLE_OUTPUTS,
} from "./nn.js";
import { createStage } from "./stage.js";
import { createGlobe } from "./worlds.js";

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 });
const FPS = 50;
const PHYSICS = ["gravity", "enable_wind", "wind_power", "turbulence_power", "atmosphere", "engine_power"];
const START_HEIGHT = 10; // meters above the pad where the simulator starts every flight
const WORLD_ORDER = ["classic", "mercury", "venus", "earth", "moon", "mars", "jupiter", "europa", "saturn", "titan", "uranus", "neptune", "pluto"];
const WORLD_DOT = {
  classic: "#8d8d88", mercury: "#9a8e82", venus: "#e2c27f", earth: "#2f72b8", moon: "#a8a6a0", mars: "#c0663a",
  jupiter: "#c9a27a", europa: "#d8cbb6", saturn: "#e0c98f", titan: "#d89a46", uranus: "#93cfd6", neptune: "#3f63c9", pluto: "#cdb89b",
};
const signed = (v, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const minus = (text) => String(text).replace("-", "−");

const app = {
  defaults: null,
  planets: {},
  pilots: [],
  pilot: null, // the selected pilot's generations, episodes, tests and trails
  training: null, // the training job in progress, if any
  weights: new Map(), // "pilot/gen/model" -> prepared weights
  world: "classic", // the world picked in the hero and flight deck
  genWorld: "all", // the world filter above the generations table
  chartWorld: "all", // ...and above the charts
  conditions: null, // its physics, possibly edited by hand
  startHeight: START_HEIGHT, // how high above the pad flights start, on every world
};

async function api(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function toast(message, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 9000 : 5000);
}

function storageGet(key) {
  try { return localStorage.getItem(`perfectland:${key}`); } catch { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(`perfectland:${key}`, value); } catch { /* private mode */ }
}

function worldName(conditions) {
  const id = conditions?.planet;
  return app.planets[id]?.name || (id === "custom" ? "Custom world" : "Classic");
}

/** " from 30 m" when flights don't start at the standard 10 m. */
function fromHeight(conditions) {
  const h = conditions?.start_height ?? START_HEIGHT;
  return h === START_HEIGHT ? "" : ` from ${h} m`;
}

/** Seconds before a flight times out: 20 s from 10 m, growing with the square root of the height. */
function timeLimit(height) {
  return Math.round(1000 * Math.sqrt(Math.max(height, START_HEIGHT) / START_HEIGHT)) / FPS;
}

function describeConditions(c) {
  const bits = [`${minus(c.gravity)} m/s²`];
  if (c.enable_wind) bits.push(`wind ${c.wind_power}`);
  if (c.atmosphere) bits.push(`air ${c.atmosphere}`);
  if (c.engine_power !== 1) bits.push(`engine ${c.engine_power}×`);
  return `${worldName(c)}${fromHeight(c)} · ${bits.join(" · ")}`;
}

/** "landed 0.31 m from the center at 0.42 m/s", for a flight or a batch of test flights. */
function precisionText(r) {
  return r?.miss == null ? "" : `${r.miss.toFixed(2)} m from the center, touching down at ${r.touchdown.toFixed(2)} m/s`;
}

// Views ------------------------------------------------------------------------

const stage = createStage($("stage"));
const globe = createGlobe($("globe"));
const brain = createBrain({ canvas: $("brain-canvas"), labels: $("brain-labels"), view: $("brain-view") });
const charts = {
  score: new ScoreChart($("chart-score")),
  outcomes: new OutcomeChart($("chart-outcomes")),
  touchdown: new TouchdownChart($("chart-touchdown")),
  tests: new TestChart($("chart-tests")),
};
legend($("legend"));

const visible = { hero: true, stage: false, brain: false, cards: false };
const watch = (el, key) => new IntersectionObserver(([e]) => { visible[key] = e.isIntersecting; }).observe(el);
watch($("mission"), "hero");
watch($("stage"), "stage");
watch($("brain-view"), "brain");
watch(document.querySelector(".cards"), "cards");

// Flight player: one clock drives the stage, the HUD, the trace, the brain and the cards.

let selected = null; // the neuron picked in the brain view

const player = {
  flight: null,
  weights: null,
  analysis: null,
  reward: null, // reward weights used to explain each frame
  cumulative: null,
  caption: "",
  source: "auto",
  t: 0,
  hold: 0,
  speed: 1,
  paused: false,
  autoplay: true,
  liveNext: null,
  waiting: false,
  loading: false,
  lastFrame: -1,
  manual: null, // inputs set by hand in "What if"

  get frames() { return this.flight?.frames || []; },
  get index() { return Math.min(Math.floor(this.t), this.frames.length - 1); },

  play(flight, { weights = null, caption = "", source = "auto", reward = null } = {}) {
    const shape = (w) => (w ? `${w.actor.map((l) => l.w.length).join("x")}:${w.control}` : "");
    if (shape(weights) !== shape(this.weights)) {
      selected = null;
      brain.clearSelection();
    }
    this.waiting = false;
    this.flight = flight;
    this.weights = weights;
    this.caption = caption;
    this.source = source;
    this.reward = reward || app.defaults.reward;
    this.t = 0;
    this.hold = 0;
    this.lastFrame = -1;
    this.outcomeCode = OUTCOMES.indexOf(flight.outcome);
    let sum = 0;
    this.cumulative = flight.frames.map((f) => (sum += f[F.reward]) - f[F.reward]);
    this.analysis = weights ? analyzeFlight(weights, flight.frames, F.obs) : null;
    stage.setFlight(flight);
    if (weights) brain.setNetwork(weights);
    $("frame-slider").max = flight.frames.length - 1;
    $("flight-caption").textContent = caption;
    $("outcome-badge").classList.remove("show");
    renderHealth();
    renderNeuronDetail();
    drawTrace();
  },

  tick(dt) {
    if (!this.flight) return;
    const last = this.frames.length - 1;
    if (this.paused || this.manual) return;
    if (this.t < last) this.t = Math.min(this.t + dt * FPS * this.speed, last);
    else {
      this.hold += dt;
      if (this.hold > 1.8) this.next();
    }
  },

  next() {
    this.hold = -1e9; // hold the last frame until a new flight starts
    if (this.liveNext) {
      const flight = this.liveNext;
      this.liveNext = null;
      return this.play(flight, liveOptions(flight));
    }
    this.waiting = true;
    if (liveHere()) return; // the next live flight is on its way
    if (this.autoplay) autopilot();
  },
};

function liveHere() {
  return !!(app.training && app.pilot && app.training.pilot === app.pilot.name);
}

function liveOptions(flight) {
  return {
    weights: flight.weights,
    source: "live",
    reward: app.training?.reward,
    caption: `Live · gen ${flight.gen} · ${worldName(flight.conditions)}${fromHeight(flight.conditions)} · after ${fmt.format(flight.frames_trained)} frames`,
  };
}

async function weightsFor(pilot, gen, model = "best") {
  const key = `${pilot}/${gen}/${model}`;
  if (!app.weights.has(key)) {
    app.weights.set(key, prepare(await api(`/api/weights?pilot=${encodeURIComponent(pilot)}&gen=${gen}&model=${model}`)));
  }
  return app.weights.get(key);
}

function generation(gen) {
  return app.pilot?.generations.find((g) => g.gen === gen);
}

const finished = (g) => g.status !== "failed" && g.models && Object.keys(g.models).length > 0;
const worldOf = (g) => g.world || g.config?.conditions?.planet || "classic";

/** The newest generation with a saved network; with world, the newest trained on that world. */
function latestGen(world) {
  const gens = app.pilot?.generations.filter((g) => finished(g) && (!world || worldOf(g) === world)) || [];
  return gens[gens.length - 1] || null;
}

/** The pilot that should fly a world: its own specialist, or else the newest of any world. */
function specialist(world) {
  return latestGen(world) || latestGen();
}

/** "gen 12 · Mars #2": the generation, its world, and its place in that world's line. */
function genLabel(g) {
  const world = worldOf(g);
  const index = app.pilot.generations.filter((x) => worldOf(x) === world && x.gen <= g.gen).length;
  return `gen ${g.gen} · ${world === "custom" ? "Custom" : app.planets[world]?.name || world} #${index}`;
}

const DURATIONS = [[250_000, "Quick"], [500_000, "Short"], [1_000_000, "Standard"], [2_000_000, "Long"], [5_000_000, "Extra long"]];
/** Rough training time on this computer, from the speed measured in earlier runs. */
function estimate(frames) {
  const seconds = (frames / (app.fps || 12_000)) * 1.1 + 4;
  return seconds < 90 ? `~${Math.max(5, Math.round(seconds / 5) * 5)} s` : `~${Math.round(seconds / 30) / 2} min`;
}

async function autopilot() {
  if (player.loading) return;
  player.loading = true;
  const conditions = { ...app.conditions };
  try {
    const pilot = specialist(app.world);
    if (!pilot) {
      const flight = await api("/api/fly", { random: true, conditions });
      player.play(flight, { caption: `Untrained · random buttons · ${worldName(conditions)}${fromHeight(conditions)}`, source: "auto" });
    } else {
      const [flight, weights] = await Promise.all([
        api("/api/fly", { pilot: app.pilot.name, gen: pilot.gen, conditions }),
        weightsFor(app.pilot.name, pilot.gen),
      ]);
      player.play(flight, {
        weights, source: "auto", reward: pilot.config.reward,
        caption: `Autopilot · ${app.pilot.name} ${genLabel(pilot)} · flying ${worldName(conditions)}${fromHeight(conditions)} · terrain #${flight.seed % 100000}`,
      });
    }
  } catch (error) {
    toast(`Couldn't fly: ${error.message}`, "error");
    setTimeout(() => { player.hold = 0; player.waiting = false; }, 3000);
  } finally {
    player.loading = false;
  }
}

// Worlds -----------------------------------------------------------------------------

function presetConditions(id) {
  const p = app.planets[id] || app.planets.classic;
  return { planet: id, ...Object.fromEntries(PHYSICS.map((k) => [k, p[k]])), start_height: app.startHeight };
}

function selectWorld(id, { fly = true } = {}) {
  app.world = id;
  app.conditions = presetConditions(id);
  storageSet("world", id);
  globe.setWorld(id);
  renderWorldInfo();
  syncConditionInputs();
  if (app.pilot) renderLabGenerations(); // offer this world's own specialist first
  updateTrainingUi();
  if (!liveHere()) {
    stage.setWorld(id);
    if (fly) {
      player.liveNext = null;
      player.next();
    }
  }
}

function stepWorld(delta) {
  const i = WORLD_ORDER.indexOf(app.world);
  selectWorld(WORLD_ORDER[(i + delta + WORLD_ORDER.length) % WORLD_ORDER.length]);
}

function renderWorldPicker() {
  $("world-picker").replaceChildren(...WORLD_ORDER.map((id) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "world-chip";
    chip.setAttribute("role", "tab");
    chip.dataset.world = id;
    const dot = document.createElement("i");
    dot.style.background = WORLD_DOT[id];
    chip.append(dot, app.planets[id].name);
    chip.addEventListener("click", () => selectWorld(id));
    return chip;
  }));
}

function renderWorldInfo() {
  const p = app.planets[app.world];
  const custom = app.conditions.planet === "custom";
  $("world-name").textContent = p.name + (custom ? "*" : "");
  $("world-fact").textContent = custom ? `Custom settings, based on ${p.name}. Press Reset in the flight deck to go back.` : p.fact;
  const c = app.conditions;
  const stats = [
    ["Gravity", `${minus(Math.abs(c.gravity).toFixed(2).replace(/\.?0+$/, ""))} m/s²`],
    ["Air", c.atmosphere ? `drag ${c.atmosphere}` : "none"],
    ["Wind", c.enable_wind ? `${c.wind_power}` : "none"],
    ["Engine", `${c.engine_power}×`],
  ];
  $("world-stats").replaceChildren(...stats.map(([label, value]) => {
    const div = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    div.append(dt, dd);
    return div;
  }));
  document.querySelectorAll(".world-chip").forEach((chip) => {
    const on = chip.dataset.world === app.world;
    chip.setAttribute("aria-selected", String(on));
    chip.classList.toggle("custom", on && custom);
  });
}

const CONDITION_INPUTS = [
  ["lab-gravity", "gravity", (v) => `${minus(Number(v).toFixed(1))}`],
  ["lab-wind-power", "wind_power", (v) => (app.conditions.enable_wind ? Number(v).toFixed(1) : "off")],
  ["lab-turbulence", "turbulence_power", (v) => Number(v).toFixed(1)],
  ["lab-atmosphere", "atmosphere", (v) => (Number(v) ? Number(v).toFixed(2) : "none")],
  ["lab-engine", "engine_power", (v) => `${Number(v).toFixed(2)}×`],
  ["lab-height", "start_height", (v) => `${v} m`],
];

function syncConditionInputs() {
  const c = app.conditions;
  $("lab-wind").checked = c.enable_wind;
  for (const [id, key, label] of CONDITION_INPUTS) {
    $(id).value = c[key];
    $(`${id}-label`).textContent = label(c[key]);
  }
  $("lab-seed-label").textContent = $("lab-seed").value === "" ? "new" : `#${$("lab-seed").value}`;
}

function bindConditions() {
  const changed = () => {
    const preset = presetConditions(app.world);
    const same = PHYSICS.every((k) => String(preset[k]) === String(app.conditions[k]));
    app.conditions.planet = same ? app.world : "custom";
    syncConditionInputs();
    renderWorldInfo();
  };
  for (const [id, key] of CONDITION_INPUTS) {
    $(id).addEventListener("input", (e) => { app.conditions[key] = Number(e.target.value); changed(); });
  }
  $("lab-wind").addEventListener("change", (e) => { app.conditions.enable_wind = e.target.checked; changed(); });
  $("lab-height").addEventListener("input", (e) => setStartHeight(Number(e.target.value)));
  $("lab-seed").addEventListener("input", syncConditionInputs);
  $("lab-reset").addEventListener("click", () => { setStartHeight(START_HEIGHT); selectWorld(app.world, { fly: false }); });
}

/** Flights start this high above the pad on every world, until it's changed again. */
function setStartHeight(height) {
  app.startHeight = height;
  app.conditions.start_height = height;
  storageSet("startHeight", String(height));
  syncConditionInputs();
}

// HUD, trace, decision panel and mini cards ------------------------------------------

function currentDecision() {
  if (!player.weights || !player.flight) return null;
  if (player.manual) return forward(player.weights, player.manual);
  const result = player.analysis?.results[player.index];
  const pressed = player.frames[player.index][F.action];
  // Weights arrive rounded, so on a near-tie trust the button the pilot actually pressed.
  return result && pressed >= 0 ? { ...result, action: pressed } : result || null;
}

function sideText(side) {
  return side ? `${side < 0 ? "left" : "right"} ${Math.round(Math.abs(side) * 100)}%` : "off";
}

function updateHud() {
  const frames = player.frames;
  if (!frames.length) return;
  const i = player.index;
  const f = frames[i];
  const o = f.slice(F.obs, F.obs + 8);
  const end = i === frames.length - 1;
  $("t-alt").textContent = `${(o[1] * 6.667).toFixed(2)} m`;
  $("t-speed").textContent = `${Math.hypot(o[2] * 5, o[3] * 7.5).toFixed(2)} m/s`;
  $("t-tilt").textContent = `${((f[F.angle] * 180) / Math.PI).toFixed(1)}°`;
  const main = end ? 0 : f[F.main], side = end ? 0 : f[F.side];
  $("t-main").textContent = main ? `${Math.round(main * 100)}%` : "off";
  $("g-main").style.width = `${main * 100}%`;
  $("t-side").textContent = sideText(side);
  const gauge = $("g-side");
  gauge.style.left = `${50 - Math.max(-side, 0) * 50}%`;
  gauge.style.width = `${Math.abs(side) * 50}%`;
  $("t-score").textContent = player.cumulative[i].toFixed(1);
  if (!end) {
    const outcome = i + 1 === frames.length - 1 ? player.outcomeCode : null;
    const r = rewardBreakdown(o, frames[i + 1].slice(F.obs, F.obs + 8), outcome, player.reward, main, side);
    $("t-reward").textContent = signed(r.total);
    $("reward-breakdown").textContent = `Your reward this frame: ${r.parts.map(([k, v]) => `${k} ${signed(v)}`).join(" · ")}`;
    player.lastReward = r;
  }
  const badge = $("outcome-badge");
  if (end) {
    badge.textContent = `${OUTCOME_LABEL[player.outcomeCode]} · ${player.flight.score.toFixed(1)}`;
    badge.classList.add("show");
  } else badge.classList.remove("show");
  $("frame-slider").value = i;
  $("frame-label").textContent = `${i} / ${frames.length - 1}`;
}

function sizeCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

// The flight trace: altitude, speed and engine power over the whole flight, as three
// small charts sharing one time axis. Click or drag to jump to a moment.
let traceCache = null;
function drawTrace() {
  const canvas = $("trace");
  const { ctx, w, h } = sizeCanvas(canvas);
  const frames = player.frames;
  if (!frames.length || !w) return;
  const left = 112, right = 16, n = frames.length - 1;
  const rows = [
    { label: "Altitude", unit: "m", values: frames.map((f) => f[F.obs + 1] * 6.667), color: "#ffffff" },
    { label: "Speed", unit: "m/s", values: frames.map((f) => Math.hypot(f[F.obs + 2] * 5, f[F.obs + 3] * 7.5)), color: "#ffffff" },
    { label: "Main engine", unit: "%", values: frames.map((f) => f[F.main] * 100), color: "#eb6834", bars: true, fixed: 100 },
  ];
  const rowH = (h - 18) / rows.length;
  const sx = (i) => left + (i / Math.max(n, 1)) * (w - left - right);
  ctx.font = '500 10px "Inter", system-ui, sans-serif';
  rows.forEach((row, r) => {
    const top = 8 + r * rowH, bottom = top + rowH - 10;
    const hi = row.fixed || Math.max(...row.values, 1e-6);
    const sy = (v) => bottom - (v / hi) * (bottom - top);
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText(row.label.toUpperCase(), 16, top + (bottom - top) / 2 - 6);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(`0–${row.fixed || hi.toFixed(1)} ${row.unit}`, 16, top + (bottom - top) / 2 + 8);
    ctx.font = '500 10px "Inter", system-ui, sans-serif';
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.beginPath(); ctx.moveTo(left, bottom + 0.5); ctx.lineTo(w - right, bottom + 0.5); ctx.stroke();
    if (row.bars) {
      ctx.fillStyle = row.color;
      const bw = Math.max((w - left - right) / Math.max(n, 1), 1);
      row.values.forEach((v, i) => { if (v > 0) ctx.fillRect(sx(i), sy(v), bw, bottom - sy(v)); });
      ctx.fillStyle = "rgba(57,135,229,0.85)"; // side engine firing, as ticks under the bars
      frames.forEach((f, i) => { if (f[F.side]) ctx.fillRect(sx(i), bottom + 2, bw, 3); });
    } else {
      ctx.strokeStyle = row.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      row.values.forEach((v, i) => (i ? ctx.lineTo(sx(i), sy(v)) : ctx.moveTo(sx(i), sy(v))));
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  });
  traceCache = { left, right, n, w };
  const x = Math.round(sx(player.index)) + 0.5;
  ctx.strokeStyle = "#ffffff";
  ctx.beginPath(); ctx.moveTo(x, 4); ctx.lineTo(x, h - 6); ctx.stroke();
}

function bindTrace() {
  const canvas = $("trace");
  let dragging = false;
  const seek = (e) => {
    if (!traceCache) return;
    const r = canvas.getBoundingClientRect();
    const t = (e.clientX - r.left - traceCache.left) / (traceCache.w - traceCache.left - traceCache.right);
    player.t = Math.round(Math.min(Math.max(t, 0), 1) * traceCache.n);
    player.hold = 0;
    setPaused(true);
  };
  canvas.addEventListener("pointerdown", (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); seek(e); });
  canvas.addEventListener("pointermove", (e) => { if (dragging) seek(e); });
  canvas.addEventListener("pointerup", () => { dragging = false; });
}

function setPaused(paused) {
  player.paused = paused;
  for (const id of ["play-btn", "deck-play"]) {
    $(id).textContent = paused ? "▶" : "❚❚";
    $(id).setAttribute("aria-label", paused ? "Play" : "Pause");
  }
}

function note(text) {
  const p = document.createElement("p");
  p.className = "hint";
  p.textContent = text;
  return p;
}

function bold(text) {
  const b = document.createElement("b");
  b.textContent = text;
  return b;
}

function barRow(label, fraction, valueText, chosen, centered = false) {
  const row = document.createElement("div");
  row.className = `prob-row${chosen ? " chosen" : ""}`;
  const name = document.createElement("span");
  name.textContent = label;
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("span");
  if (centered) { // -1..1 around a middle line
    fill.style.left = `${50 + Math.min(fraction, 0) * 50}%`;
    fill.style.width = `${Math.abs(fraction) * 50}%`;
  } else {
    fill.style.width = `${fraction * 100}%`;
  }
  bar.append(fill);
  const value = document.createElement("span");
  value.className = "mono";
  value.textContent = valueText;
  row.append(name, bar, value);
  return row;
}

function updateBrain() {
  const decision = currentDecision();
  if (!decision) {
    $("decision").replaceChildren(note("This flight has no network: it picked random actions."));
    $("critic").textContent = "";
    return;
  }
  brain.update(decision);
  if (player.weights.control === "throttle") {
    const [main, side] = decision.outputs;
    $("decision").replaceChildren(
      barRow("main throttle", main, `${signed(main)} → ${decision.engines.main ? `${Math.round(decision.engines.main * 100)}%` : "off"}`, true, true),
      barRow("side thrust", side, `${signed(side)} → ${sideText(decision.engines.side)}`, !!decision.engines.side, true),
      note("A throttle pilot outputs two numbers. Main above 0 fires the main engine at 50–100%; side beyond ±0.5 fires a side engine."),
    );
  } else {
    $("decision").replaceChildren(...ACTIONS.map((name, k) =>
      barRow(name, decision.probs[k], `${Math.round(decision.probs[k] * 100)}%`, k === decision.action)));
  }
  $("critic").replaceChildren("The critic expects ", bold(signed(decision.value, 1)), " more points from here (in training-reward units).");
  const caption = player.weights ? `${player.caption.split(" · ").slice(0, 3).join(" · ")} · frame ${player.index}` : "";
  $("brain-caption").textContent = player.manual ? "What if · inputs set by hand" : caption;
  updateNeuronValue(decision);
}

/** The patch of world a mini card shows: the 20 m-wide stage, tall enough for flights starting at top meters. */
function worldBox(top = START_HEIGHT) {
  return { x0: -11, x1: 11, y0: -2, y1: Math.max(11.5, Math.ceil(top) + 1.5) };
}
const startOf = (frames) => frames?.[0]?.[F.y] ?? 0;
function toCanvas(w, h, x, y, box = worldBox()) {
  const s = Math.min(w / (box.x1 - box.x0), h / (box.y1 - box.y0));
  return [w / 2 + x * s, h - -box.y0 * s - 6 - y * s, s];
}

function drawPractice() {
  const { ctx, w, h } = sizeCanvas($("mini-practice"));
  const terrain = player.flight?.terrain;
  const trails = app.pilot?.trails.slice(-70) || [];
  const box = worldBox(Math.max(startOf(player.frames), ...trails.map(([, , path]) => path[0]?.[1] ?? 0)));
  if (terrain) {
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    for (let x = -11; x <= 11; x += 0.22) {
      let y = terrain[terrain.length - 1][1];
      for (let k = 1; k < terrain.length; k++) {
        if (x <= terrain[k][0]) {
          const [x0, y0] = terrain[k - 1], [x1, y1] = terrain[k];
          y = y0 + ((y1 - y0) * (Math.max(x, x0) - x0)) / (x1 - x0);
          break;
        }
      }
      const [px, py] = toCanvas(w, h, x, y, box);
      ctx.fillRect(px - 0.8, py - 0.8, 1.6, 1.6);
    }
  }
  const [p0x, padY] = toCanvas(w, h, -2, 0, box);
  const [p1x] = toCanvas(w, h, 2, 0, box);
  ctx.fillStyle = "#fff";
  ctx.fillRect(p0x, padY - 1, p1x - p0x, 2);
  trails.forEach(([, code, path], rank) => {
    ctx.globalAlpha = 0.15 + 0.85 * ((rank + 1) / trails.length);
    ctx.fillStyle = OUTCOME_COLORS[code];
    for (const [x, y] of path) {
      const [px, py] = toCanvas(w, h, x, y, box);
      ctx.fillRect(px - 0.9, py - 0.9, 1.8, 1.8);
    }
  });
  ctx.globalAlpha = 1;
  if (!trails.length) {
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = '12px "Inter", system-ui, sans-serif';
    ctx.textAlign = "center";
    ctx.fillText("Practice paths appear here while a pilot trains.", w / 2, h * 0.38);
  }
  const episodes = app.pilot?.episodes || [];
  const recent = episodes.slice(-100);
  const landed = recent.filter((e) => e[3] === 0).length;
  $("mini-practice-stat").textContent = episodes.length
    ? `${fmt.format(episodes.length)} attempts · ${Math.round((landed / recent.length) * 100)}% of the last ${recent.length} landed on the pad`
    : "No practice yet";
}

let rewardField = null;
function drawReward() {
  const { ctx, w, h } = sizeCanvas($("mini-reward"));
  const weights = player.reward || app.defaults?.reward;
  if (!weights || !w) return;
  const box = worldBox(startOf(player.frames));
  const key = `${w}x${h}x${box.y1}:${JSON.stringify(weights)}`;
  if (!rewardField || rewardField.key !== key) {
    const off = document.createElement("canvas");
    const dpr = Math.min(window.devicePixelRatio, 2);
    off.width = w * dpr;
    off.height = h * dpr;
    const g = off.getContext("2d");
    g.scale(dpr, dpr);
    const [cx, cy, s] = toCanvas(w, h, 0, 0, box);
    const dots = [];
    let lo = Infinity, hi = -Infinity;
    for (let px = 6; px < w; px += 9) {
      for (let py = 6; py < h; py += 9) {
        const p = potential([(px - cx) / s / 10, (cy - py) / s / 6.667, 0, 0, 0, 0, 0, 0], weights);
        dots.push([px, py, p]);
        lo = Math.min(lo, p);
        hi = Math.max(hi, p);
      }
    }
    for (const [px, py, p] of dots) {
      const v = (p - lo) / (hi - lo || 1);
      g.fillStyle = `rgba(255,255,255,${0.12 + 0.8 * v * v})`;
      g.beginPath(); g.arc(px, py, 0.6 + 2.2 * v * v, 0, Math.PI * 2); g.fill();
    }
    const [p0x, padY] = toCanvas(w, h, -2, 0, box);
    const [p1x] = toCanvas(w, h, 2, 0, box);
    g.fillStyle = "#fff";
    g.fillRect(p0x, padY - 1, p1x - p0x, 2);
    rewardField = { key, canvas: off };
  }
  ctx.drawImage(rewardField.canvas, 0, 0, w, h);
  const frames = player.frames;
  if (!frames.length) return;
  const i = player.index;
  ctx.fillStyle = "#eb6834";
  for (let k = Math.max(0, i - 60); k <= i; k += 3) {
    const [px, py] = toCanvas(w, h, frames[k][F.x], frames[k][F.y], box);
    ctx.globalAlpha = 0.15 + 0.85 * ((k - i + 60) / 60);
    ctx.fillRect(px - 1, py - 1, 2, 2);
  }
  ctx.globalAlpha = 1;
  const [px, py] = toCanvas(w, h, frames[i][F.x], frames[i][F.y], box);
  ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.fill();
  const r = player.lastReward;
  $("mini-reward-stat").textContent = `Potential ${potential(frames[i].slice(F.obs, F.obs + 8), weights).toFixed(1)}` +
    (r ? ` · this frame ${signed(r.total)}` : "");
}

function drawMiniBrain() {
  const { ctx, w, h } = sizeCanvas($("mini-brain"));
  const decision = currentDecision();
  if (!decision) {
    $("mini-brain-stat").textContent = "Waiting for a trained pilot";
    return;
  }
  const hidden = decision.layers.slice(1, -1);
  const relu = player.weights.activation === "relu";
  const panelW = (w - 90) / hidden.length;
  hidden.forEach((acts, l) => {
    const cols = Math.ceil(Math.sqrt(acts.length));
    const rows = Math.ceil(acts.length / cols);
    const gap = Math.min((panelW - 20) / cols, (h - 30) / rows);
    const ox = 10 + l * panelW + (panelW - gap * cols) / 2;
    const oy = (h - gap * rows) / 2;
    const scale = relu ? Math.max(...acts, 1e-6) : 1;
    acts.forEach((a, k) => {
      const v = Math.max(-1, Math.min(1, a / scale));
      ctx.globalAlpha = 0.25 + 0.75 * Math.abs(v);
      ctx.fillStyle = Math.abs(v) < 0.08 ? "#3a3a37" : v > 0 ? OUTCOME_COLORS[1] : OUTCOME_COLORS[0];
      ctx.beginPath();
      ctx.arc(ox + (k % cols) * gap + gap / 2, oy + Math.floor(k / cols) * gap + gap / 2, 1.2 + Math.abs(v) * gap * 0.32, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  ctx.globalAlpha = 1;
  const bx = w - 72, bh = 12;
  const throttle = player.weights.control === "throttle";
  const outputs = throttle ? decision.outputs : decision.probs;
  outputs.forEach((v, k) => {
    const y = h / 2 - (outputs.length / 2) * (bh + 12) + k * (bh + 12);
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    ctx.fillRect(bx, y, 60, bh);
    ctx.fillStyle = throttle || k === decision.action ? "#fff" : "rgba(255,255,255,0.5)";
    if (throttle) ctx.fillRect(bx + 30 + Math.min(v, 0) * 30, y, Math.abs(v) * 30, bh);
    else ctx.fillRect(bx, y, 60 * v, bh);
  });
  $("mini-brain-stat").textContent = throttle
    ? `Main engine ${decision.engines.main ? `${Math.round(decision.engines.main * 100)}%` : "off"} · side ${sideText(decision.engines.side)}`
    : `Pressing ${ACTIONS[decision.action]} (${Math.round(decision.probs[decision.action] * 100)}% sure)`;
}

// Selected neuron, what-if, brain health -----------------------------------------------

brain.onSelect((id) => { selected = id; renderNeuronDetail(); });

function outputNames() {
  return player.weights?.control === "throttle" ? THROTTLE_OUTPUTS : ACTIONS;
}

function neuronName({ layer, index }) {
  const L = player.weights.actor.length;
  if (layer === 0) return `Sensor · ${INPUTS[index]}`;
  if (layer === L) return `Output · ${outputNames()[index]}`;
  return `Hidden ${layer} · neuron ${index + 1}`;
}

function renderNeuronDetail() {
  const el = $("neuron-detail");
  if (!selected || !player.weights) {
    el.replaceChildren(note("Click a neuron in the 3D view to see what it listens to and how it fires during the flight."));
    return;
  }
  const { layer, index } = selected;
  const actor = player.weights.actor;
  const L = actor.length;
  const head = document.createElement("div");
  head.className = "head";
  const name = document.createElement("span");
  name.textContent = neuronName(selected);
  const value = document.createElement("span");
  value.className = "mono";
  value.id = "neuron-value";
  head.append(name, value);
  const spark = document.createElement("canvas");
  spark.id = "neuron-spark";
  const parts = [head, spark];
  const weightRows = (title, items) => {
    const h = note(title);
    h.style.margin = "12px 0 6px";
    parts.push(h);
    const max = Math.max(...items.map((it) => Math.abs(it.w)), 1e-6);
    for (const it of items) {
      const row = document.createElement("div");
      row.className = "weight-row";
      const label = document.createElement("span");
      label.textContent = it.name;
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.className = it.w >= 0 ? "pos" : "neg";
      fill.style.width = `${(Math.abs(it.w) / max) * 100}%`;
      bar.append(fill);
      const v = document.createElement("span");
      v.className = "mono";
      v.textContent = signed(it.w);
      row.append(label, bar, v);
      parts.push(row);
    }
  };
  if (layer > 0) {
    const items = Array.from(actor[layer - 1].w[index], (w, i) => ({ w, name: layer === 1 ? INPUTS[i] : `H${layer - 1} #${i + 1}` }))
      .sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 6);
    weightRows("Listens most to (incoming weights)", items);
  }
  if (layer < L) {
    const last = layer + 1 === L;
    const items = actor[layer].w.map((row, j) => ({ w: row[index], name: last ? outputNames()[j] : `H${layer + 1} #${j + 1}` }))
      .sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, last ? 4 : 5);
    weightRows(last ? "Pushes these outputs (outgoing weights)" : "Feeds most into (outgoing weights)", items);
  }
  parts.push(note("Orange is a positive weight, blue a negative one. The trace shows this neuron across the whole flight; the white line is the current frame."));
  el.replaceChildren(...parts);
  updateNeuronValue(currentDecision());
}

function updateNeuronValue(decision) {
  if (!selected || !decision || !$("neuron-value")) return;
  const L = player.weights.actor.length;
  const { layer, index } = selected;
  const throttle = player.weights.control === "throttle";
  const read = (r) => (layer === L ? r.outputs[index] : r.layers[layer][index]);
  const value = read(decision);
  $("neuron-value").textContent = layer === L && !throttle ? `${Math.round(value * 100)}%` : signed(value);
  const { ctx, w, h } = sizeCanvas($("neuron-spark"));
  const results = player.analysis?.results;
  if (!results) return;
  const series = results.map(read);
  const lo = Math.min(...series, layer === L && !throttle ? 0 : -1), hi = Math.max(...series, 1);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  const zero = h - ((0 - lo) / (hi - lo)) * h;
  ctx.beginPath(); ctx.moveTo(0, zero); ctx.lineTo(w, zero); ctx.stroke();
  ctx.strokeStyle = OUTCOME_COLORS[1];
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  series.forEach((v, k) => {
    const x = (k / (series.length - 1)) * w, y = h - ((v - lo) / (hi - lo)) * h;
    k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  const x = (player.index / (series.length - 1)) * w;
  ctx.strokeStyle = "#fff";
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
}

const WHATIF = [[-1, 1, 0.01], [-0.2, 6, 0.01], [-1, 1, 0.01], [-1.5, 0.5, 0.01], [-1, 1, 0.01], [-1, 1, 0.01], [0, 1, 1], [0, 1, 1]];
function buildWhatIf() {
  const grid = $("whatif");
  const nodes = [];
  INPUTS.forEach((name, k) => {
    const label = document.createElement("span");
    label.textContent = name;
    const input = document.createElement("input");
    input.type = "range";
    [input.min, input.max, input.step] = WHATIF[k];
    input.disabled = true;
    const value = document.createElement("span");
    value.className = "mono";
    value.id = `whatif-${k}`;
    input.addEventListener("input", () => {
      if (!player.manual) return;
      player.manual[k] = Number(input.value);
      value.textContent = Number(input.value).toFixed(2);
      updateBrain();
      drawMiniBrain();
    });
    nodes.push(label, input, value);
  });
  grid.replaceChildren(...nodes);
  $("whatif-toggle").addEventListener("change", (e) => {
    const inputs = grid.querySelectorAll("input");
    if (e.target.checked && player.flight) {
      player.manual = player.frames[player.index].slice(F.obs, F.obs + 8);
      inputs.forEach((input, k) => {
        input.disabled = false;
        input.value = player.manual[k];
        $(`whatif-${k}`).textContent = player.manual[k].toFixed(2);
      });
    } else {
      player.manual = null;
      inputs.forEach((input) => { input.disabled = true; });
      e.target.checked = false;
    }
    updateBrain();
  });
}

function renderHealth() {
  const el = $("health");
  const health = player.analysis?.health;
  if (!health) {
    el.replaceChildren(note("Shown for flights by a trained pilot."));
    $("health-hint").textContent = "";
    return;
  }
  el.replaceChildren(...health.map((h) => {
    const row = document.createElement("div");
    row.className = "health-row";
    const label = document.createElement("span");
    label.textContent = `Hidden ${h.layer}`;
    const bar = document.createElement("div");
    bar.className = "health-bar";
    for (const [k, color] of [["active", "#fff"], ["quiet", "#3a3a37"], ["saturated", OUTCOME_COLORS[1]]]) {
      const s = document.createElement("span");
      s.style.flex = String(h[k] || 0.0001);
      s.style.background = color;
      bar.append(s);
    }
    const text = document.createElement("span");
    text.style.gridColumn = "2";
    text.className = "mono";
    text.style.fontSize = "11px";
    text.textContent = `${h.active} active · ${h.quiet} quiet · ${h.saturated} saturated`;
    row.append(label, bar, document.createElement("span"), text);
    return row;
  }));
  const total = health.reduce((s, h) => s + h.size, 0);
  const quiet = health.reduce((s, h) => s + h.quiet, 0) / total;
  const saturated = health.reduce((s, h) => s + h.saturated, 0) / total;
  $("health-hint").textContent = saturated > 0.25
    ? "Many neurons are stuck fully on or off. Try a lower learning rate, or ReLU for a new pilot."
    : quiet > 0.35
      ? "Many neurons barely react in this flight. A smaller network may learn as well, or harder conditions (wind) would put them to work."
      : "Most neurons react to this flight, so the network is using its capacity. Active = changes during the flight; quiet = barely moves; saturated = stuck at the limit.";
}

// Charts, KPIs and the generations table ---------------------------------------------

let chartTimer = null;
function refreshCharts(now = false) {
  if (chartTimer && !now) return;
  const run = () => {
    chartTimer = null;
    const p = app.pilot;
    if (!p) return;
    if (app.chartWorld !== "all" && !p.generations.some((g) => worldOf(g) === app.chartWorld)) app.chartWorld = "all";
    renderWorldFilter($("chart-worlds"), app.chartWorld, (id) => { app.chartWorld = id; refreshCharts(true); });
    const shown = new Set(p.generations.filter((g) => app.chartWorld === "all" || worldOf(g) === app.chartWorld).map((g) => g.gen));
    const all = app.chartWorld === "all";
    const episodes = all ? p.episodes : p.episodes.filter((e) => shown.has(e[1]));
    const tests = all ? p.tests : p.tests.filter((t) => shown.has(t[1]));
    const generations = p.generations.filter((g) => shown.has(g.gen));
    charts.score.set(episodes, generations);
    charts.outcomes.set(episodes);
    charts.touchdown.set(episodes);
    charts.tests.set(tests, generations);
    $("outcome-sub").textContent = `Share of each block of ${fmt.format(charts.outcomes.block)} attempts.`;
    const recent = episodes.slice(-100);
    $("score-now").textContent = recent.length ? `last 100: ${(recent.reduce((s, e) => s + e[2], 0) / recent.length).toFixed(1)}` : "";
    const t = tests[tests.length - 1];
    $("test-now").textContent = t ? `latest: ${t[2].toFixed(1)} · ${t[3]}/${t[4]} landed` : "";
    if (visible.cards) drawPractice();
  };
  if (now) run();
  else chartTimer = setTimeout(run, 700);
}

function updateKpis() {
  const p = app.pilot;
  $("strip-pilot").textContent = p ? `Pilot ${p.name}` : "No pilot yet";
  $("footer-pilot").textContent = p?.name || "pilot";
  $("csv-link").href = p ? `/api/episodes.csv?pilot=${encodeURIComponent(p.name)}` : "#";
  const training = liveHere() ? app.training : null;
  const latest = latestGen();
  $("k-gen").textContent = training ? `${training.gen}` : latest ? `${latest.gen}` : "–";
  const frames = training?.progress?.frames ?? latest?.frames_end;
  $("k-frames").textContent = frames != null ? compact.format(frames) : "–";
  $("k-attempts").textContent = p ? fmt.format(p.episodes.length) : "–";
  const recent = p?.episodes.slice(-100) || [];
  $("k-landed").textContent = recent.length ? `${Math.round((recent.filter((e) => e[3] === 0).length / recent.length) * 100)}%` : "–";
  const best = p?.tests.length ? Math.max(...p.tests.map((t) => t[2])) : null;
  $("k-best").textContent = best != null ? best.toFixed(1) : "–";
}

function settingsSummary(config) {
  if (!config) return "";
  const d = app.defaults;
  const c = config.conditions || {};
  const bits = c.planet === "custom" ? [`custom: ${minus(c.gravity)} m/s²${c.enable_wind ? `, wind ${c.wind_power}` : ""}, engine ${c.engine_power}×`] : [];
  if (fromHeight(c)) bits.unshift(`starts at ${c.start_height} m`);
  const ppo = config.ppo || {};
  if (ppo.control === "throttle") bits.push("throttle");
  for (const [k, v] of Object.entries(config.reward || {})) {
    if (d.reward[k] !== v) bits.push(`${k.toLowerCase().replace(/_/g, " ")} ${v}`);
  }
  if (ppo.net_arch && ppo.net_arch.join("x") !== d.ppo.net_arch.join("x")) bits.push(`network ${ppo.net_arch.join("×")}`);
  if (ppo.activation && ppo.activation !== d.ppo.activation) bits.push(ppo.activation);
  if (ppo.learning_rate !== d.ppo.learning_rate) bits.push(`learning rate ${ppo.learning_rate}`);
  if (ppo.ent_coef !== d.ppo.ent_coef) bits.push(`exploration ${ppo.ent_coef}`);
  if (ppo.gamma !== d.ppo.gamma) bits.push(`gamma ${ppo.gamma}`);
  return bits.length ? bits.join(" · ") : "default settings";
}

/** A row of world chips: "All worlds" plus every world this pilot has trained on. */
function renderWorldFilter(el, current, pick) {
  const gens = app.pilot?.generations || [];
  const worlds = [...new Set(gens.map(worldOf))].sort((a, b) => WORLD_ORDER.indexOf(a) - WORLD_ORDER.indexOf(b));
  el.hidden = worlds.length < 2;
  const chip = (id, label, count) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "world-chip";
    button.setAttribute("aria-selected", String(current === id));
    if (id !== "all") {
      const dot = document.createElement("i");
      dot.style.background = WORLD_DOT[id] || "#898781";
      button.append(dot);
    }
    button.append(label, Object.assign(document.createElement("small"), { textContent: ` ${count}` }));
    button.addEventListener("click", () => pick(id));
    return button;
  };
  el.replaceChildren(chip("all", "All worlds", gens.length),
    ...worlds.map((id) => chip(id, id === "custom" ? "Custom" : app.planets[id]?.name || id, gens.filter((g) => worldOf(g) === id).length)));
}

function renderGenerations() {
  const table = $("gen-table");
  const p = app.pilot;
  table.replaceChildren();
  if (p && app.genWorld !== "all" && !p.generations.some((g) => worldOf(g) === app.genWorld)) app.genWorld = "all";
  renderWorldFilter($("gen-worlds"), app.genWorld, (id) => { app.genWorld = id; renderGenerations(); });
  if (!p || !p.generations.length) {
    const cell = table.insertRow().insertCell();
    cell.className = "empty-gens";
    cell.textContent = "No generations yet. Train a new pilot to start generation 1.";
    return;
  }
  const best = Math.max(...p.generations.map((g) => g.summary?.best_test ?? -1e9), 1);
  const rows = p.generations.filter((g) => app.genWorld === "all" || worldOf(g) === app.genWorld);
  const precise = rows.some((g) => g.summary?.last_miss != null); // measured by tests since start altitudes arrived
  const hr = table.createTHead().insertRow();
  for (const t of ["Gen", "World", "Started from", "Frames", "Attempts", "Test score", "Landed", ...(precise ? ["Precision"] : []), "Settings", ""]) {
    const th = document.createElement("th");
    th.textContent = t;
    if (t) th.className = `col-${t.toLowerCase().replace(/ /g, "-")}`;
    if (t === "Precision") th.title = "In the latest test: how far from the pad center the landings stopped, and how fast the legs touched down.";
    hr.append(th);
  }
  const body = document.createElement("tbody");
  [...rows].reverse().forEach((g) => {
    const tr = body.insertRow();
    const cell = (content, cls) => {
      const td = tr.insertCell();
      if (cls) td.className = cls;
      if (content instanceof Node) td.append(content); else td.textContent = content;
    };
    const id = document.createElement("span");
    id.className = "gen-id";
    id.textContent = String(g.gen).padStart(2, "0");
    cell(id);
    const world = document.createElement("span");
    world.className = "world-cell";
    const dot = document.createElement("i");
    dot.style.background = WORLD_DOT[worldOf(g)] || "#898781";
    world.append(dot, genLabel(g).split(" · ")[1]);
    cell(world);
    const status = document.createElement("span");
    status.className = `status ${g.status}`;
    const parent = g.parent ? generation(g.parent) : null;
    status.textContent = g.status === "training" ? "training…"
      : g.mode === "cli" ? "train.py"
        : g.mode === "fresh" ? "blank network"
          : parent ? (worldOf(parent) === worldOf(g) ? `gen ${g.parent}` : `gen ${g.parent} · ${genLabel(parent).split(" · ")[1]}`)
            : "new pilot";
    if (g.status === "failed" || g.status === "stopped") status.textContent += ` · ${g.status}`;
    cell(status);
    const frames = g.frames_end != null && g.frames_start != null ? g.frames_end - g.frames_start : g.frames;
    cell(frames ? compact.format(frames) : "–", "num");
    cell(g.attempts_end != null ? fmt.format(g.attempts_end - g.attempts_start) : "–", "num");
    const score = g.summary?.best_test;
    const sc = document.createElement("div");
    sc.className = "score-cell";
    const sv = document.createElement("span");
    sv.className = "mono";
    sv.textContent = score != null ? score.toFixed(1) : "–";
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, (score ?? 0) / best) * 100}%`;
    bar.append(fill);
    sc.append(sv, bar);
    cell(sc);
    cell(g.summary?.last_landed != null ? `${g.summary.last_landed}/20` : "–", "num");
    if (precise) {
      const precision = document.createElement("span"); // from the pad center, and touchdown speed
      if (g.summary?.last_miss != null) {
        precision.append(`${g.summary.last_miss.toFixed(2)} m`,
          Object.assign(document.createElement("small"), { textContent: `${g.summary.last_touchdown.toFixed(2)} m/s` }));
      } else precision.textContent = "–";
      cell(precision, "num precision");
    }
    cell(settingsSummary(g.config), "settings");
    const actions = document.createElement("div");
    actions.className = "row-actions";
    if (finished(g)) {
      const fly = document.createElement("button");
      fly.textContent = "Fly";
      fly.addEventListener("click", () => flyGeneration(g.gen));
      const inspect = document.createElement("button");
      inspect.textContent = "Brain";
      inspect.addEventListener("click", async () => { await flyGeneration(g.gen, false); $("neurons").scrollIntoView(); });
      actions.append(fly, inspect);
      if (latestGen(worldOf(g)) === g && !app.training) { // the head of this world's line
        const train = document.createElement("button");
        train.textContent = "Train ↗";
        train.addEventListener("click", () => openDrawer("continue", { world: worldOf(g), conditions: g.config.conditions, fromGen: g.gen }));
        actions.append(train);
      }
    }
    cell(actions);
  });
  table.append(body);
}

async function flyGeneration(gen, scroll = true) {
  try {
    const g = generation(gen);
    const conditions = g?.config?.conditions || app.conditions; // where this generation trained
    const [flight, weights] = await Promise.all([
      api("/api/fly", { pilot: app.pilot.name, gen, conditions }),
      weightsFor(app.pilot.name, gen),
    ]);
    player.play(flight, { weights, source: "lab", reward: g?.config?.reward,
      caption: `Replay · ${app.pilot.name} gen ${gen} · ${worldName(conditions)}${fromHeight(conditions)} · terrain #${flight.seed % 100000}` });
    if (scroll) $("flight-deck").scrollIntoView();
  } catch (error) {
    toast(error.message, "error");
  }
}

// Pilot loading and the live stream -----------------------------------------------------

async function loadPilot(name) {
  app.pilot = name ? await api(`/api/pilot?name=${encodeURIComponent(name)}`) : null;
  if (name) storageSet("pilot", name);
  $("pilot-select").value = name || "";
  stage.setTrails(app.pilot?.trails || []);
  refreshCharts(true);
  updateKpis();
  renderGenerations();
  renderLabGenerations();
  updateTrainingUi();
  if (!liveHere()) {
    player.liveNext = null;
    if (player.source !== "lab") { player.hold = 0; player.t = player.frames.length; player.next(); }
  }
}

function renderPilotSelect() {
  const select = $("pilot-select");
  select.replaceChildren(...app.pilots.map((p) => {
    const option = document.createElement("option");
    option.value = p.name;
    option.textContent = `${p.name}${p.latest_gen ? ` · gen ${p.latest_gen}` : ""}`;
    return option;
  }));
  if (!app.pilots.length) select.append(Object.assign(document.createElement("option"), { value: "", textContent: "none yet" }));
  if (app.pilot) select.value = app.pilot.name;
}

function updateTrainingUi() {
  const job = app.training;
  const mine = liveHere();
  const status = $("nav-status");
  status.classList.toggle("live", !!job);
  status.querySelector("span").textContent = job ? `Training ${job.pilot} · gen ${job.gen}` : "Idle";
  $("btn-stop").hidden = !job;
  const latest = latestGen();
  const here = app.planets[app.world]?.name || "this world";
  $("btn-retrain").disabled = !!job;
  $("btn-retrain").textContent = latest ? `Train on ${here}` : "Train a pilot";
  $("btn-retrain-2").disabled = !!job || !latest;
  $("btn-retrain-2").textContent = "Train this pilot";
  $("btn-new").disabled = !!job;
  $("nav-train").textContent = job ? "Stop" : "Train";
  $("lab-train").disabled = !!job || !latest;
  const heroStatus = $("hero-status");
  heroStatus.hidden = !job;
  if (job) heroStatus.textContent = `Training ${job.pilot} gen ${job.gen} on ${worldName(job.conditions)}${fromHeight(job.conditions)}. Watch it live ↓`;
  const panel = $("training-panel");
  panel.hidden = !mine;
  if (mine) {
    const progress = job.progress || {};
    const pct = Math.min(100, ((progress.done || 0) / job.frames) * 100);
    $("tp-title").textContent = `Training gen ${job.gen} on ${worldName(job.conditions)}${fromHeight(job.conditions)}`;
    $("tp-pct").textContent = `${pct.toFixed(0)}%`;
    $("tp-bar").style.width = `${pct}%`;
    const eta = progress.fps ? Math.max(0, (job.frames - progress.done) / progress.fps) : null;
    $("tp-meta").textContent = `${fmt.format(progress.done || 0)} / ${fmt.format(job.frames)} frames` +
      (progress.fps ? ` · ${fmt.format(progress.fps)}/s · ${eta < 90 ? `${Math.round(eta)} s` : `${Math.round(eta / 60)} min`} left` : "") +
      (job.lastTest ? ` · test ${job.lastTest}` : "") + (job.auto ? " · auto-retrain on" : "");
  }
}

function handleEvent(event) {
  const p = app.pilot;
  const mine = p && event.pilot === p.name;
  switch (event.type) {
    case "job":
      app.training = event.training;
      app.pilots = event.pilots;
      renderPilotSelect();
      if (!p || event.training.pilot !== p.name) loadPilot(event.training.pilot);
      else api(`/api/pilot?name=${encodeURIComponent(p.name)}`).then((data) => { app.pilot = data; renderGenerations(); updateKpis(); updateTrainingUi(); });
      updateTrainingUi();
      break;
    case "round":
      if (app.training && app.training.pilot === event.pilot && app.training.gen === event.gen) {
        app.training.progress = { done: event.done, fps: event.fps, elapsed: event.elapsed, frames: event.frames };
        if (event.done > 50_000 && event.fps) { // remember this computer's speed for duration estimates
          app.fps = app.fps ? app.fps * 0.7 + event.fps * 0.3 : event.fps;
          storageSet("fps", String(Math.round(app.fps)));
        }
      }
      if (mine) {
        p.episodes.push(...event.episodes);
        p.trails.push(...event.trails);
        if (p.trails.length > 400) p.trails.splice(0, p.trails.length - 400);
        stage.setTrails(p.trails);
        refreshCharts();
        updateKpis();
      }
      updateTrainingUi();
      break;
    case "flight":
      if (mine) {
        event.flight.weights = prepare(event.flight.weights);
        const busy = (player.source === "live" || player.source === "lab") && !player.waiting;
        if (busy) player.liveNext = event.flight;
        else player.play(event.flight, liveOptions(event.flight));
      }
      break;
    case "test":
      if (mine) {
        p.tests.push(event.row);
        const g = generation(event.gen);
        if (g) {
          g.summary = g.summary || {};
          g.summary.last_test = event.row[2];
          g.summary.last_landed = event.row[3];
          g.summary.best_test = Math.max(g.summary.best_test ?? -1e9, event.row[2]);
        }
        refreshCharts();
        updateKpis();
        renderGenerations();
      }
      if (app.training?.pilot === event.pilot) {
        app.training.lastTest = `${event.row[3]}/20 landed, ${event.row[2].toFixed(1)}`;
        updateTrainingUi();
      }
      break;
    case "stopping":
      $("nav-status").querySelector("span").textContent = "Stopping after a final test…";
      break;
    case "done":
    case "error": {
      app.training = null;
      app.pilots = event.pilots || app.pilots;
      app.weights.clear();
      renderPilotSelect();
      if (event.type === "done") {
        const best = Number.isFinite(event.best_test) ? event.best_test.toFixed(1) : "–";
        toast(`Gen ${event.gen} of ${event.pilot} ${event.stopped ? "stopped" : "finished"}: best test score ${best}.`);
      } else {
        toast(`Training failed: ${event.message.split("\n").filter(Boolean).slice(-1)[0]}`, "error");
      }
      if (mine) {
        api(`/api/pilot?name=${encodeURIComponent(p.name)}`).then((data) => {
          app.pilot = data;
          refreshCharts(true);
          updateKpis();
          renderGenerations();
          renderLabGenerations();
          updateTrainingUi();
        });
      }
      updateTrainingUi();
      break;
    }
    case "notice":
      toast(event.message);
      break;
  }
}

function connect() {
  const source = new EventSource("/api/events");
  let opened = false;
  source.onmessage = (e) => handleEvent(JSON.parse(e.data));
  source.onerror = () => { $("nav-status").querySelector("span").textContent = "Reconnecting…"; };
  source.onopen = async () => {
    const state = await api("/api/state");
    app.training = state.training;
    app.pilots = state.pilots;
    renderPilotSelect();
    updateTrainingUi();
    if (opened && app.pilot) { // reconnected: catch up on anything missed
      const stillThere = state.pilots.some((p) => p.name === app.pilot.name);
      if (!stillThere) await loadPilot(state.pilots[0]?.name || null); // its folder was removed meanwhile
      else {
        app.pilot = await api(`/api/pilot?name=${encodeURIComponent(app.pilot.name)}`);
        stage.setTrails(app.pilot.trails);
        refreshCharts(true);
        updateKpis();
        renderGenerations();
      }
    }
    opened = true;
  };
}

// Train drawer ---------------------------------------------------------------------------

const REWARD_HELP = [
  ["DISTANCE_WEIGHT", "Distance", "Points for each step closer to the pad, taken back when it drifts away."],
  ["SPEED_WEIGHT", "Speed", "Points for slowing down."],
  ["TILT_WEIGHT", "Tilt", "Points for straightening up."],
  ["LEG_CONTACT_BONUS", "Leg contact", "Points per leg touching the ground."],
  ["MAIN_ENGINE_COST", "Main engine fuel", "Cost of each frame the main engine fires, times its power."],
  ["SIDE_ENGINE_COST", "Side engine fuel", "Cost of each frame a side engine fires, times its power."],
  ["TIME_COST", "Time", "Cost of every frame, so hovering isn't free."],
  ["GROUND_ENGINE_COST", "Firing while parked", "Cost of firing with both legs down and the rocket stopped. Without it, button pilots fidget on the pad."],
  ["LANDING_BONUS", "Landing bonus", "Given once when the rocket comes to rest on its legs."],
  ["CRASH_PENALTY", "Crash penalty", "Given once for a crash or flying off screen."],
];
let drawerMode = "new";

function knob({ name, label, help, value, type = "number", step, options, def }) {
  const row = document.createElement("div");
  row.className = "knob";
  const l = document.createElement("label");
  l.textContent = label;
  l.htmlFor = `k-${name}`;
  let input;
  if (options) {
    input = document.createElement("select");
    for (const [v, text] of options) input.append(Object.assign(document.createElement("option"), { value: v, textContent: text }));
  } else {
    input = document.createElement("input");
    input.type = type;
    if (step) input.step = step;
  }
  input.id = `k-${name}`;
  input.name = name;
  input.value = value;
  const mark = () => row.classList.toggle("changed", def !== undefined && String(input.value) !== String(def));
  input.addEventListener("input", mark);
  mark();
  row.append(l, input);
  if (help) row.append(Object.assign(document.createElement("small"), { textContent: help }));
  return row;
}

function fieldset(title, hint, children) {
  const fs = document.createElement("fieldset");
  fs.append(Object.assign(document.createElement("legend"), { textContent: title }));
  if (hint) fs.append(Object.assign(document.createElement("p"), { className: "hint", textContent: hint }));
  fs.append(...children);
  return fs;
}

function check(name, label, help, checked) {
  const wrap = document.createElement("label");
  wrap.className = "check";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.name = name;
  input.checked = !!checked;
  const text = document.createElement("span");
  text.textContent = label;
  text.append(Object.assign(document.createElement("small"), { textContent: help }));
  wrap.append(input, text);
  return wrap;
}

function openDrawer(mode, overrides = {}) {
  drawerMode = mode;
  const d = app.defaults;
  const latest = latestGen();
  if (mode === "continue" && !latest) return;
  const base = mode === "continue" ? latest.config : { reward: d.reward, ppo: d.ppo };
  const reward = { ...d.reward, ...base.reward };
  const ppo = { ...d.ppo, ...base.ppo };
  const world = overrides.world || app.world;
  const conditions = overrides.conditions?.planet === world || overrides.conditions?.planet === "custom"
    ? { ...overrides.conditions } : world === app.world ? { ...app.conditions } : presetConditions(world);
  $("drawer-eyebrow").textContent = mode === "continue" ? `Pilot ${app.pilot.name}` : "New pilot";
  $("drawer-title").textContent = mode === "continue" ? `Train ${app.pilot.name}` : "Train a new pilot";
  $("drawer-sub").textContent = mode === "continue"
    ? "Each world has its own line of generations. Pick where to train and for how long; it continues that world's latest generation."
    : "A new pilot starts from a blank network that presses random buttons.";

  // Where and how long ------------------------------------------------------------------
  const head = [];
  let n = 1;
  while (app.pilots.some((p) => p.name === `pilot-${n}`)) n++;
  if (mode === "new") head.push(knob({ name: "pilot", label: "Pilot name", type: "text", value: `pilot-${n}` }));
  const worlds = document.createElement("div");
  worlds.className = "choice-grid worlds";
  const worldIds = conditions.planet === "custom" ? [...WORLD_ORDER, "custom"] : WORLD_ORDER;
  for (const id of worldIds) {
    const count = mode === "continue" ? app.pilot.generations.filter((g) => finished(g) && worldOf(g) === id).length : 0;
    const label = document.createElement("label");
    label.className = "world-chip";
    const input = Object.assign(document.createElement("input"), { type: "radio", name: "world", value: id, checked: id === conditions.planet });
    const dot = document.createElement("i");
    dot.style.background = WORLD_DOT[id] || "#898781";
    label.append(input, dot, id === "custom" ? "Custom" : app.planets[id].name);
    if (count) label.append(Object.assign(document.createElement("small"), { textContent: ` ${count}` }));
    worlds.append(label);
  }
  head.push(choiceRow("World", "Where it practices. The number is how many generations this pilot already has there.", worlds));
  const altitude = document.createElement("div");
  altitude.className = "altitude-pick";
  const height = Object.assign(document.createElement("input"), {
    type: "range", name: "conditions.start_height", min: 5, max: 40, step: 1, value: conditions.start_height ?? app.startHeight,
  });
  height.setAttribute("aria-label", "Start altitude, meters above the pad");
  const heightValue = Object.assign(document.createElement("b"), { className: "mono" });
  const heightNote = Object.assign(document.createElement("span"), { className: "mono" });
  const showHeight = () => {
    const h = Number(height.value);
    heightValue.textContent = `${h} m`;
    heightNote.textContent = `${h === START_HEIGHT ? "standard · " : ""}${Math.round(timeLimit(h))} s to land`;
  };
  height.addEventListener("input", showHeight);
  showHeight();
  const scratchTip = Object.assign(document.createElement("p"), { className: "altitude-tip", hidden: true,
    textContent: "From high up, a blank network usually spends a while hovering just above the pad before it dares to touch down. Give it 2M frames (Long): in tests from 30 m it got past that by about 1.3M." });
  altitude.append(height, heightValue, heightNote, scratchTip);
  head.push(choiceRow("Start altitude", "How high above the pad every practice flight starts; the standard is 10 m. From higher up, the pilot has more time to steady itself and line up, and the clock gets longer. Scores grow too, since they count the distance covered.", altitude));
  const durations = document.createElement("div");
  durations.className = "choice-grid durations";
  const frames = overrides.frames || 1_000_000;
  for (const [value, name] of DURATIONS) {
    const label = document.createElement("label");
    label.className = "duration";
    const input = Object.assign(document.createElement("input"), { type: "radio", name: "frames", value, checked: value === frames });
    const title = Object.assign(document.createElement("b"), { textContent: name });
    const sub = Object.assign(document.createElement("span"), { textContent: `${compact.format(value)} frames · ${estimate(value)}` });
    label.append(input, title, sub);
    durations.append(label);
  }
  head.push(choiceRow("Duration", "One frame is 1/50 s of one rocket's flight. Low-gravity worlds need longer.", durations));
  if (mode === "continue") {
    const start = knob({ name: "start", label: "Start from", value: "", options: [],
      help: "Continue this world's latest generation, bring another world's skills here, or start over." });
    start.classList.add("wide");
    head.push(start);
    head.push(knob({ name: "from_model", label: "Which version", value: "final", options: [["final", "as its training ended"], ["best", "its best test"]],
      help: "Continue from the network as its training ended, or from its best-scoring checkpoint." }));
  }
  head.push(check("auto", "Keep retraining automatically", "When this generation ends, start the next one on the same world, until you press Stop.", false));
  const sections = [fieldset("Where and how long", null, head)];

  // The details, folded away ---------------------------------------------------------------
  const physics = [
    knob({ name: "conditions.gravity", label: "Gravity", value: conditions.gravity, step: 0.01, help: "m/s², negative = down. Earth is −9.8, the Moon −1.6, Jupiter −23.1." }),
    check("conditions.enable_wind", "Wind", "Gusts push the rocket sideways.", conditions.enable_wind),
    knob({ name: "conditions.wind_power", label: "Wind power", value: conditions.wind_power, step: 0.5, help: "0 to 20." }),
    knob({ name: "conditions.turbulence_power", label: "Turbulence", value: conditions.turbulence_power, step: 0.1, help: "Random spin, 0 to 2. Only with wind." }),
    knob({ name: "conditions.atmosphere", label: "Air drag", value: conditions.atmosphere, step: 0.01, help: "0 is a vacuum; Venus's thick air is 1.2." }),
    knob({ name: "conditions.engine_power", label: "Engine power", value: conditions.engine_power, step: 0.01,
      help: "Main engine thrust, times the standard 17.8 m/s². Real landers carry about 1.8 times local gravity." }),
  ];
  sections.push(folded("World details", "Filled in from the world you pick. Gravity is real; air and wind are simplified. Edit them to make a custom world.", physics));
  if (mode === "new") {
    sections.push(folded("Controls", null, [
      knob({ name: "ppo.control", label: "Engines", value: ppo.control, def: d.ppo.control,
        options: [["buttons", "Buttons · on or off"], ["throttle", "Throttle · the network picks the power"]],
        help: "With a throttle, the network outputs how hard to fire each engine on every frame, so it learns where more power pays off. Fixed once a pilot exists." }),
    ], true));
  }
  sections.push(folded("Reward", "What the pilot is scored on. This is reward shaping: change it and the flying style changes.",
    REWARD_HELP.map(([k, label, help]) => knob({ name: `reward.${k}`, label, help, value: reward[k], step: "any", def: d.reward[k] }))));
  const learning = [
    knob({ name: "ppo.learning_rate", label: "Learning rate", value: ppo.learning_rate, def: d.ppo.learning_rate,
      options: [[0.0001, "0.0001 · careful"], [0.0003, "0.0003 · default"], [0.001, "0.001 · fast"], [0.003, "0.003 · risky"]],
      help: "How big each nudge to the weights is. Too big and learning becomes unstable." }),
    knob({ name: "ppo.ent_coef", label: "Exploration", value: ppo.ent_coef, step: 0.001, def: d.ppo.ent_coef,
      help: "Bonus for keeping options open. Higher = more random practice; lower = settles on a style sooner." }),
    knob({ name: "ppo.gamma", label: "Foresight (gamma)", value: ppo.gamma, def: d.ppo.gamma,
      options: [[0.99, "0.99 · ~100 frames"], [0.995, "0.995"], [0.999, "0.999 · ~1000 frames"], [0.9995, "0.9995"]],
      help: "How far ahead future points count." }),
    knob({ name: "ppo.n_envs", label: "Simulators", value: ppo.n_envs, def: d.ppo.n_envs, options: [[4, "4"], [8, "8"], [16, "16"], [32, "32"]],
      help: "How many rockets practice at once." }),
  ];
  if (mode === "new") {
    learning.push(
      knob({ name: "ppo.net_arch", label: "Network size", value: ppo.net_arch.join(","), def: d.ppo.net_arch.join(","),
        options: [["16,16", "16 × 16"], ["32,32", "32 × 32"], ["64,64", "64 × 64 · default"], ["128,128", "128 × 128"], ["64,64,64", "64 × 64 × 64"], ["256,256", "256 × 256"]],
        help: "Neurons per hidden layer. Fixed once a pilot exists." }),
      knob({ name: "ppo.activation", label: "Neuron type", value: ppo.activation, def: d.ppo.activation,
        options: [["tanh", "tanh · smooth, −1 to 1"], ["relu", "ReLU · off or on"]], help: "How a neuron turns its input into an output." }),
      knob({ name: "ppo.seed", label: "Random seed", value: ppo.seed, step: 1, help: "Same seed and settings, same pilot." }),
    );
  }
  const note = mode === "continue" ? `This pilot uses ${ppo.control === "throttle" ? "a throttle" : "buttons"}; its network's shape is fixed, so these knobs change how it keeps learning.` : null;
  sections.push(folded("Learning", note, learning));
  const form = $("train-form");
  form.replaceChildren(...sections);

  const sync = () => { // keep the start options, physics and button label in step with the choices
    const id = form.elements.world.value;
    if (mode === "continue") {
      const select = form.elements.start;
      const keep = select.value;
      select.replaceChildren(...startOptions(id).map(([v, t]) => Object.assign(document.createElement("option"), { value: v, textContent: t })));
      if (overrides.fromGen && !sync.done) select.value = `gen:${overrides.fromGen}`;
      else if ([...select.options].some((o) => o.value === keep)) select.value = keep;
      sync.done = true;
    }
    const name = id === "custom" ? "a custom world" : app.planets[id].name;
    const from = fromHeight({ start_height: Number(form.elements["conditions.start_height"].value) });
    const scratch = mode === "new" || form.elements.start?.value === "fresh";
    scratchTip.hidden = !(scratch && Number(form.elements["conditions.start_height"].value) > START_HEIGHT);
    $("drawer-start").textContent = `Train on ${name}${from} · ${estimate(Number(form.elements.frames.value))}`;
  };
  form.querySelectorAll('input[name="world"]').forEach((input) => input.addEventListener("change", () => {
    if (app.planets[input.value]) {
      const preset = presetConditions(input.value);
      for (const key of PHYSICS) {
        const el = form.elements[`conditions.${key}`];
        if (el.type === "checkbox") el.checked = preset[key]; else el.value = preset[key];
      }
    }
    sync.done = true;
    sync();
  }));
  form.querySelectorAll('input[name="frames"]').forEach((input) => input.addEventListener("change", sync));
  height.addEventListener("input", sync);
  form.elements.start?.addEventListener("change", sync);
  sync();
  $("drawer-note").textContent = mode === "continue" ? `Becomes gen ${Math.max(...app.pilot.generations.map((g) => g.gen)) + 1} of ${app.pilot.name}` : "";
  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  $("scrim").classList.add("open");
}

function choiceRow(title, help, control) {
  const row = document.createElement("div");
  row.className = "choice";
  row.append(Object.assign(document.createElement("p"), { className: "choice-title", textContent: title }), control,
    Object.assign(document.createElement("small"), { textContent: help }));
  return row;
}

function folded(title, hint, children, open = false) {
  const details = document.createElement("details");
  details.open = open;
  details.append(Object.assign(document.createElement("summary"), { textContent: title }));
  const fs = fieldset(title, hint, children);
  fs.querySelector("legend").remove();
  details.append(fs);
  return details;
}

/** Where a new generation on this world can start: [value, label] pairs, best guess first. */
function startOptions(world) {
  const gens = app.pilot.generations.filter(finished);
  const line = gens.filter((g) => worldOf(g) === world);
  const name = world === "custom" ? "the custom world" : app.planets[world].name;
  const options = [];
  if (line.length) options.push([`gen:${line.at(-1).gen}`, `${genLabel(line.at(-1))} · the latest on ${name}`]);
  const newest = gens.at(-1);
  if (newest && newest !== line.at(-1)) options.push([`gen:${newest.gen}`, `${genLabel(newest)} · bring its skills to ${name}`]);
  for (const g of [...gens].reverse()) {
    if (!options.some(([v]) => v === `gen:${g.gen}`)) options.push([`gen:${g.gen}`, genLabel(g)]);
  }
  options.push(["fresh", `A blank network · start ${name} from scratch`]);
  return options;
}

function closeDrawer() {
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("open");
}

async function startTraining() {
  const form = $("train-form");
  const body = { mode: drawerMode, reward: {}, conditions: {}, ppo: {} };
  const text = new Set(["activation", "control", "pilot", "from_model"]);
  for (const el of form.elements) {
    if (!el.name || el.type === "radio" || el.name === "start") continue;
    const value = el.type === "checkbox" ? el.checked : el.value;
    const [group, key] = el.name.split(".");
    const field = key || group;
    const parsed = typeof value === "boolean" || text.has(field) ? value
      : field === "net_arch" ? value.split(",").map(Number) : Number(value);
    if (key) body[group][key] = parsed;
    else body[group] = parsed;
  }
  body.frames = Number(form.elements.frames.value);
  const world = form.elements.world.value;
  const preset = app.planets[world] && presetConditions(world);
  const edited = !preset || PHYSICS.some((k) => String(preset[k]) !== String(body.conditions[k]));
  body.conditions.planet = edited ? "custom" : world;
  if (drawerMode === "continue") {
    body.pilot = app.pilot.name;
    const start = form.elements.start.value;
    if (start === "fresh") body.fresh = true;
    else body.from_gen = Number(start.split(":")[1]);
  }
  try {
    $("drawer-start").disabled = true;
    await api("/api/train", body);
    closeDrawer();
    $("flight-deck").scrollIntoView();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("drawer-start").disabled = false;
  }
}

// Flight deck actions ------------------------------------------------------------------

function renderLabGenerations() {
  const gens = (app.pilot?.generations || []).filter(finished).reverse();
  const options = gens.map((g) => [String(g.gen), `${genLabel(g)}${g.config?.ppo?.control === "throttle" ? " · throttle" : ""}`]);
  options.push(["random", "Untrained (random buttons)"]);
  $("lab-gen").replaceChildren(...options.map(([v, t]) => Object.assign(document.createElement("option"), { value: v, textContent: t })));
  const pick = specialist(app.world);
  if (pick) $("lab-gen").value = String(pick.gen);
}

function labPilot() {
  const choice = $("lab-gen").value;
  return choice === "random" ? { random: true } : { pilot: app.pilot.name, gen: Number(choice) };
}

async function labFly() {
  const seed = $("lab-seed").value === "" ? null : Number($("lab-seed").value);
  const conditions = { ...app.conditions };
  const who = labPilot();
  try {
    $("lab-fly").disabled = true;
    const [flight, weights] = await Promise.all([
      api("/api/fly", { ...who, seed, conditions }),
      who.random ? null : weightsFor(who.pilot, who.gen),
    ]);
    const name = who.random ? "Untrained pilot" : `${who.pilot} gen ${who.gen}`;
    const caption = `Flight lab · ${name} · ${worldName(conditions)}${fromHeight(conditions)} · terrain #${flight.seed % 100000}`;
    const reward = who.random ? null : generation(who.gen)?.config?.reward;
    player.play(flight, { weights, source: "lab", reward, caption });
    logFlight(flight, name, conditions, () => player.play(flight, { weights, source: "lab", reward, caption }));
    setPaused(false);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("lab-fly").disabled = false;
  }
}

function logFlight(flight, who, conditions, replay) {
  const list = $("flight-log");
  list.querySelector(".placeholder")?.remove();
  const li = document.createElement("li");
  const title = document.createElement("span");
  title.textContent = `${OUTCOME_LABEL[OUTCOMES.indexOf(flight.outcome)]} · ${flight.score.toFixed(1)}`;
  const button = document.createElement("button");
  button.textContent = "Replay";
  button.addEventListener("click", replay);
  const meta = document.createElement("span");
  meta.className = "muted";
  meta.textContent = `${who} · ${describeConditions(conditions)} · ${(flight.steps / FPS).toFixed(1)} s`
    + (flight.miss != null ? ` · ${flight.miss.toFixed(2)} m from the center` : "");
  li.append(title, button, meta);
  list.prepend(li);
  while (list.children.length > 8) list.lastChild.remove();
}

function outcomeBars(result) {
  const bars = document.createElement("div");
  bars.className = "outcome-bars";
  for (const [name, count] of Object.entries(result.outcomes)) {
    const row = document.createElement("div");
    const label = document.createElement("span");
    label.textContent = OUTCOME_LABEL[OUTCOMES.indexOf(name)];
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${(count / result.flights) * 100}%`;
    bar.append(fill);
    const value = Object.assign(document.createElement("span"), { className: "mono", textContent: String(count) });
    row.append(label, bar, value);
    bars.append(row);
  }
  return bars;
}

async function labTest() {
  const who = labPilot();
  const conditions = { ...app.conditions };
  const box = $("test-summary");
  try {
    $("lab-test").disabled = true;
    $("lab-test").textContent = "Flying…";
    const result = await api("/api/test", { ...who, conditions });
    const name = who.random ? "Untrained pilot" : `${who.pilot} gen ${who.gen}`;
    const big = Object.assign(document.createElement("div"), { className: "big", textContent: `${result.landed} of ${result.flights} landed` });
    const sub = Object.assign(document.createElement("p"), { className: "sub",
      textContent: `${name} · ${describeConditions(conditions)} · average score ${result.score.toFixed(1)}` });
    box.replaceChildren(big, sub, outcomeBars(result));
    if (result.miss != null) {
      box.append(Object.assign(document.createElement("p"), { className: "sub precision",
        textContent: `Landings averaged ${precisionText(result)}.` }));
    }
    box.hidden = false;
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("lab-test").disabled = false;
    $("lab-test").textContent = "Run 20 test flights";
  }
}

async function labTour() {
  const who = labPilot();
  const box = $("test-summary");
  const name = who.random ? "Untrained pilot" : `${who.pilot} gen ${who.gen}`;
  const big = Object.assign(document.createElement("div"), { className: "big", textContent: "Touring the solar system…" });
  const sub = Object.assign(document.createElement("p"), { className: "sub",
    textContent: `${name} · 20 test flights on every world${fromHeight({ start_height: app.startHeight })}, with its own gravity, air and engine` });
  const bars = document.createElement("div");
  bars.className = "outcome-bars";
  box.replaceChildren(big, sub, bars);
  box.hidden = false;
  $("lab-tour").disabled = true;
  let total = 0;
  try {
    for (const id of WORLD_ORDER) {
      const result = await api("/api/test", { ...who, conditions: presetConditions(id) });
      total += result.landed;
      const row = document.createElement("div");
      const label = document.createElement("button");
      label.type = "button";
      label.textContent = app.planets[id].name;
      label.addEventListener("click", () => selectWorld(id));
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${(result.landed / result.flights) * 100}%`;
      bar.append(fill);
      row.append(label, bar, Object.assign(document.createElement("span"), { className: "mono", textContent: `${result.landed}/20` }));
      bars.append(row);
      big.textContent = `Touring… ${bars.children.length}/${WORLD_ORDER.length}`;
    }
    big.textContent = `${total} of ${WORLD_ORDER.length * 20} landed`;
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("lab-tour").disabled = false;
  }
}

// Controls -------------------------------------------------------------------------------

function bindControls() {
  $("btn-new").addEventListener("click", () => openDrawer("new"));
  $("btn-retrain").addEventListener("click", () => openDrawer(latestGen() ? "continue" : "new"));
  $("btn-retrain-2").addEventListener("click", () => openDrawer("continue"));
  $("btn-stop").addEventListener("click", stopTraining);
  $("nav-train").addEventListener("click", () => (app.training ? stopTraining() : openDrawer(latestGen() ? "continue" : "new")));
  $("mini-reward-link").addEventListener("click", () => openDrawer(latestGen() ? "continue" : "new"));
  $("drawer-cancel").addEventListener("click", closeDrawer);
  $("scrim").addEventListener("click", closeDrawer);
  $("drawer-start").addEventListener("click", startTraining);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  $("pilot-select").addEventListener("change", (e) => loadPilot(e.target.value));
  $("world-prev").addEventListener("click", () => stepWorld(-1));
  $("world-next").addEventListener("click", () => stepWorld(1));
  $("lab-fly").addEventListener("click", labFly);
  $("lab-test").addEventListener("click", labTest);
  $("lab-tour").addEventListener("click", labTour);
  $("lab-train").addEventListener("click", () => openDrawer(latestGen() ? "continue" : "new", { world: app.world, conditions: app.conditions }));
  for (const id of ["play-btn", "deck-play"]) $(id).addEventListener("click", () => setPaused(!player.paused));
  $("deck-next").addEventListener("click", () => { setPaused(false); player.next(); });
  $("frame-slider").addEventListener("input", (e) => {
    player.t = Number(e.target.value);
    player.hold = 0;
    setPaused(true);
  });
  document.querySelectorAll("#brain-mode button").forEach((btn) => btn.addEventListener("click", () => {
    document.querySelectorAll("#brain-mode button").forEach((b) => b.classList.toggle("on", b === btn));
    brain.setMode(btn.dataset.mode);
  }));
  const gui = new GUI({ container: $("gui-host"), title: "View" });
  gui.add(stage.options, "cameraMode", { Follow: "follow", Wide: "wide", Free: "free" }).name("camera");
  gui.add(stage.options, "distance", 14, 46, 1).name("distance");
  gui.add(player, "speed", { "0.25×": 0.25, "0.5×": 0.5, "1×": 1, "2×": 2, "4×": 4 }).name("speed");
  gui.add(stage.options, "showTrails").name("practice trails");
  gui.add(player, "autoplay").name("autoplay");
  gui.add({ reset: () => stage.resetView() }, "reset").name("reset view");
  gui.close();
}

async function stopTraining() {
  try {
    await api("/api/stop", {});
  } catch (error) {
    toast(error.message, "error");
  }
}

// Main loop ------------------------------------------------------------------------------

let lastTime = performance.now();
let lastHud = 0;
function loop(now) {
  const dt = Math.min((now - lastTime) / 1000, 0.05);
  lastTime = now;
  player.tick(dt);
  const frames = player.frames;
  if (frames.length) {
    const i = player.index;
    const a = frames[i], b = frames[Math.min(i + 1, frames.length - 1)];
    stage.setPose(a, b, player.t - i, i === frames.length - 1);
    if (i !== player.lastFrame || now - lastHud > 250) {
      player.lastFrame = i;
      lastHud = now;
      updateHud();
      if (visible.stage) drawTrace();
      if (visible.brain || visible.cards) updateBrain();
      if (visible.cards) { drawReward(); drawMiniBrain(); }
    }
  }
  if (visible.stage) stage.render(dt);
  if (visible.hero) globe.render();
  if (visible.brain && brain.ready) brain.render();
  requestAnimationFrame(loop);
}

async function init() {
  $("footer-host").textContent = location.host;
  const state = await api("/api/state");
  app.defaults = state.defaults;
  app.planets = state.planets;
  app.fps = Number(storageGet("fps")) || null;
  app.pilots = state.pilots;
  app.training = state.training;
  bindControls();
  bindConditions();
  bindTrace();
  buildWhatIf();
  renderWorldPicker();
  renderPilotSelect();
  const savedHeight = Number(storageGet("startHeight"));
  if (savedHeight >= 5 && savedHeight <= 40) app.startHeight = savedHeight;
  const savedWorld = storageGet("world");
  selectWorld(app.planets[savedWorld] ? savedWorld : "classic", { fly: false });
  const saved = storageGet("pilot");
  const names = app.pilots.map((p) => p.name);
  const start = app.training?.pilot || (names.includes(saved) ? saved : names.includes("main") ? "main" : names[names.length - 1]);
  connect();
  requestAnimationFrame(loop);
  await loadPilot(start || null);
  drawPractice();
  new ResizeObserver(() => { rewardField = null; drawPractice(); drawTrace(); }).observe(document.querySelector(".cards"));
}

init().catch((error) => toast(`Couldn't start: ${error.message}`, "error"));
