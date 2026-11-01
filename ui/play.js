// The game: land the rocket yourself, racing a trained network on the same terrain.
// Everything runs in the browser: the simulator (sim.js), the network (nn.js) and the 3D
// stage (stage.js). Served by app.py it can also race the pilots you trained there.

import { createAnnunciator } from "./annunciator.js";
import { myAIs } from "./my-ais.js";
import { setupTrainer } from "./trainer-ui.js";
import { setupNav } from "./nav.js";
import { ACTIONS, F, OUTCOME_LABEL, forward, prepare } from "./nn.js";
import { FPS, Flight, OUTCOMES } from "./sim.js";
import { createSound } from "./sound.js";
import { specs } from "./specs.js";
import { createStage } from "./stage.js";
import { createGlobe, WORLD_DOT } from "./worlds.js";

const $ = (id) => document.getElementById(id);
const STEP = 1 / FPS;
const REPO = "https://github.com/maheessh/PerfectLand";
const GAME_URL = "https://maheessh.github.io/PerfectLand/"; // the game on GitHub Pages
const PHYSICS = ["gravity", "enable_wind", "wind_power", "turbulence_power", "atmosphere", "engine_power"];
const WORLD_ORDER = ["classic", "mercury", "venus", "earth", "moon", "mars", "jupiter", "europa", "saturn", "titan", "uranus", "neptune", "pluto"];

/** One line on how to fly, for the keys this device has. */
const HOW_TO = () => (matchMedia("(pointer: coarse)").matches
  ? "Hold ▲ to fire the engine and slow down, tap ◀ ▶ to steer, and touch down gently between the flags."
  : "Hold ↑ to fire the engine and slow down, tap ← → to steer, and touch down gently between the flags.");

const game = {
  worlds: {},
  world: "classic",
  mode: "race", // race | solo | watch
  phase: "idle", // idle | countdown | flying | done
  slow: false,
  opponents: [],
  opponent: null, // { id, title, weights }
  you: null, // your Flight
  ai: null, // the AI's Flight
  attract: null, // the AI flying behind the start card
  attractWait: 0,
  acc: 0,
  frame: 0,
  held: new Set(),
  server: false,
  challenge: null, // a terrain seed from a shared link: every round flies that same terrain
};

const stage = createStage($("stage"));
const globe = createGlobe($("globe")); // the world you picked, spinning at the top of the page
const lamps = createAnnunciator($("annunciator"), $("callout"));
const sound = createSound({ fallback: true });
stage.options.showTrails = false;
stage.options.distance = 21; // a little closer than the flight deck: you're flying it

// Storage, which can be unavailable (private windows, blocked site data).
const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(`perfectland.${key}`)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`perfectland.${key}`, JSON.stringify(value)); } catch { /* fine without */ }
  },
};
let records = store.get("records", {});
let tally = store.get("tally", { rounds: 0, wins: 0 });

function toast(text, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = text;
  $("toasts").append(el);
  setTimeout(() => el.remove(), 3600);
}

const fmt = {
  score: (v) => (v == null ? "–" : v.toFixed(1)),
  meters: (v) => (v == null ? "–" : `${v.toFixed(2)} m`),
  speed: (v) => (v == null ? "–" : `${v.toFixed(2)} m/s`),
  seconds: (frames) => `${(frames / FPS).toFixed(1)} s`,
};

function conditions(id) {
  const w = game.worlds[id];
  return { planet: id, ...Object.fromEntries(PHYSICS.map((k) => [k, w[k]])), start_height: 10 };
}

// Worlds ----------------------------------------------------------------------------

function renderWorldPicker() {
  const picker = $("world-picker");
  picker.replaceChildren();
  for (const id of WORLD_ORDER) {
    const w = game.worlds[id];
    if (!w) continue;
    const chip = document.createElement("button");
    chip.className = "world-chip";
    chip.setAttribute("role", "tab");
    chip.dataset.world = id;
    const dot = document.createElement("i");
    dot.style.background = WORLD_DOT[id] || "#888";
    const done = records[id]?.landed;
    chip.append(dot, w.name);
    if (done) {
      const tick = document.createElement("small");
      tick.textContent = "✓";
      chip.append(tick);
    }
    chip.setAttribute("aria-selected", String(id === game.world));
    chip.addEventListener("click", () => selectWorld(id));
    picker.append(chip);
  }
  revealChip();
}

/** On phones the chips scroll sideways: keep the selected world in view. */
function revealChip() {
  const picker = $("world-picker");
  const chip = picker.querySelector('[aria-selected="true"]');
  if (chip && picker.scrollWidth > picker.clientWidth) {
    picker.scrollLeft = chip.offsetLeft - picker.offsetLeft - (picker.clientWidth - chip.offsetWidth) / 2;
  }
}

function selectWorld(id) {
  if (!game.worlds[id]) return;
  if (id !== game.world) game.challenge = null; // a shared terrain belongs to its world
  game.world = id;
  const url = new URL(location.href);
  url.searchParams.set("world", id);
  if (game.challenge === null) url.searchParams.delete("seed");
  history.replaceState(null, "", url);
  for (const chip of $("world-picker").children) chip.setAttribute("aria-selected", String(chip.dataset.world === id));
  revealChip();
  stage.setWorld(id);
  showGlobe();
  renderSpecs();
  toIdle();
}

function stepWorld(delta) {
  const i = WORLD_ORDER.indexOf(game.world);
  selectWorld(WORLD_ORDER[(i + delta + WORLD_ORDER.length) % WORLD_ORDER.length]);
}

/** The globe at the top: the world's name, a fact about it, and what it's like to land there. */
function showGlobe() {
  const w = game.worlds[game.world];
  globe.setWorld(game.world);
  $("world-name").textContent = w.name;
  $("world-fact").textContent = w.fact || "";
  const stats = [
    ["Gravity", `${Math.abs(w.gravity).toFixed(2).replace(/\.?0+$/, "")} m/s²`],
    ["Air", !w.atmosphere ? "none" : w.atmosphere < 0.5 ? "thin" : "thick"],
    ["Wind", w.enable_wind ? `${w.wind_power}` : "none"],
    ["Engine", `${w.engine_power}×`],
  ];
  $("world-stats").replaceChildren(...stats.map(([label, value]) => {
    const div = document.createElement("div");
    div.innerHTML = "<dt></dt><dd></dd>";
    div.firstChild.textContent = label;
    div.lastChild.textContent = value;
    return div;
  }));
}

// The hand-drawn figures live in their own pages; each tells us its height so its frame fits it.
addEventListener("message", (e) => {
  if (!e.data?.figure || !e.data.height) return;
  for (const frame of document.querySelectorAll(".figure-frame")) if (frame.contentWindow === e.source) frame.style.height = `${Math.ceil(e.data.height)}px`;
});

function renderSpecs() {
  const w = game.worlds[game.world];
  $("spec-world").textContent = w.name;
  $("spec-fact").textContent = w.fact;
  $("specs").replaceChildren(...specs(w).flatMap(([label, value, note]) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.innerHTML = "<b></b><small></small>";
    dd.firstChild.textContent = value;
    dd.lastChild.textContent = note;
    const row = document.createElement("div");
    row.append(dt, dd);
    return [row];
  }));
}

function worldLine(id) {
  const w = game.worlds[id];
  const bits = [`${Math.abs(w.gravity)} m/s²`];
  if (w.atmosphere) bits.push(w.atmosphere >= 1 ? "thick air" : "air");
  if (w.enable_wind) bits.push(w.wind_power >= 10 ? "strong wind" : "wind");
  return `${w.name} · ${bits.join(" · ")}`;
}

// Opponents -------------------------------------------------------------------------

async function loadOpponents(prefer = null) {
  const shipped = await fetch("data/pilots.json").then((r) => r.json());
  game.opponents = shipped.map((p) => ({ ...p, source: "shipped" }));
  // pilots trained in this browser
  for (const ai of await myAIs.list()) {
    if (!ai.weights) continue;
    const b = ai.best;
    game.opponents.push({ id: `mine/${ai.id}`, aiId: ai.id, title: `${ai.name}${b ? ` (${b.landed}/${b.flights})` : ""}`, source: "mine",
      control: "buttons", world: ai.world, note: `Trained in this browser on ${game.worlds[ai.world]?.name || ai.world}.` });
  }
  if (game.server) {
    try {
      const state = await fetch("/api/state").then((r) => r.json());
      for (const pilot of state.pilots || []) {
        const data = await fetch(`/api/pilot?name=${encodeURIComponent(pilot.name)}`).then((r) => r.json());
        for (const g of [...(data.generations || [])].reverse()) {
          if (g.status === "failed" || !g.models) continue;
          const world = g.config?.conditions?.planet;
          const where = world === "universal" ? "every world" : game.worlds[world]?.name || "custom";
          game.opponents.push({
            id: `${pilot.name}/${g.gen}`, title: `${pilot.name} · gen ${g.gen} (${where})`,
            note: "One of your pilots, from mission control.", source: "local", pilot: pilot.name, gen: g.gen,
            control: g.config?.ppo?.control || "buttons",
          });
        }
      }
    } catch { /* the shipped pilot is enough */ }
  }
  const select = $("opponent");
  select.replaceChildren();
  const groups = { mine: "My AIs (this browser)", shipped: "Ships with the game", local: "From mission control" };
  for (const [source, label] of Object.entries(groups)) {
    const list = game.opponents.filter((o) => o.source === source);
    if (!list.length) continue;
    const group = document.createElement("optgroup");
    group.label = label;
    for (const o of list) group.append(new Option(o.title, o.id));
    select.append(group);
  }
  const params = new URLSearchParams(location.search);
  const wanted = params.get("pilot") ? `${params.get("pilot")}/${params.get("gen") || ""}` : null;
  const pick = game.opponents.find((o) => o.id === prefer) || game.opponents.find((o) => o.id === wanted)
    || game.opponents.find((o) => o.id === game.opponent?.id) || game.opponents.find((o) => o.source === "local" && wanted?.startsWith(`${o.pilot}/`)) || game.opponents[0];
  select.value = pick.id;
  // never swap the opponent under a round that's flying (training can finish mid-round)
  if (game.phase === "countdown" || game.phase === "flying") { select.value = game.opponent?.id ?? pick.id; return; }
  if (!prefer && game.opponent?.id === pick.id) return;
  await setOpponent(pick.id);
}

async function setOpponent(id) {
  const o = game.opponents.find((p) => p.id === id);
  if (!o) return;
  const raw = o.source === "mine" ? (await myAIs.get(o.aiId))?.weights
    : await fetch(o.source === "shipped" ? `data/${o.file}` : `/api/weights?pilot=${encodeURIComponent(o.pilot)}&gen=${o.gen}`).then((r) => r.json());
  if (!raw) return;
  const weights = prepare(raw);
  game.opponent = { ...o, weights };
  $("opponent-note").textContent = o.control === "throttle" ? "throttle" : "buttons";
  $("opponent").title = o.note || "";
  toIdle();
}

function aiAction(flight) {
  const out = forward(game.opponent.weights, flight.obs);
  return game.opponent.weights.control === "throttle" ? out.outputs : out.action;
}

// Your engines ------------------------------------------------------------------------

const KEYS = {
  ArrowUp: "main", KeyW: "main", Space: "main",
  ArrowLeft: "left", KeyA: "left",
  ArrowRight: "right", KeyD: "right",
};

/** Your button this frame. The simulator fires one engine per frame, so holding two takes turns. */
function yourAction() {
  const on = [];
  if (game.held.has("main")) on.push(2);
  const left = game.held.has("left"), right = game.held.has("right");
  if (left && !right) on.push(1);
  if (right && !left) on.push(3);
  return on.length ? on[game.frame % on.length] : 0;
}

// Rounds ---------------------------------------------------------------------------------

function newFlights(seed) {
  const c = conditions(game.world);
  const throttle = game.opponent?.weights.control === "throttle";
  game.you = game.mode === "watch" ? null : new Flight(c, { seed });
  game.ai = game.mode === "solo" ? null : new Flight(c, { seed, continuous: throttle });
  const shown = game.you || game.ai;
  stage.setFlight({ conditions: c, terrain: shown.terrain, frames: [liveFrame(shown)] });
}

function toIdle() {
  game.phase = "idle";
  game.you = game.ai = null;
  game.held.clear();
  startAttract();
  const w = game.worlds[game.world];
  $("overlay-eyebrow").textContent = worldLine(game.world);
  $("overlay-title").textContent = game.mode === "watch" ? `Watch the AI land on ${w.name}.` : `Land on ${w.name}.`;
  $("overlay-text").textContent = game.mode === "watch"
    ? "It sees the same 8 numbers you see below the view and picks an engine 50 times a second."
    : `${game.challenge !== null ? `A shared challenge: terrain #${game.challenge}, the same for everyone. ` : ""}${HOW_TO()}`;
  $("launch").innerHTML = game.mode === "watch" ? "Watch it fly <kbd>Space</kbd>" : "Launch <kbd>Space</kbd>";
  $("overlay").hidden = false;
  $("nudge").hidden = true;
  $("countdown").hidden = true;
  updateCaption();
  updateHud();
}

/** Behind the start card the AI keeps landing, so the page is never still. */
function startAttract() {
  if (!game.opponent) return;
  const throttle = game.opponent.weights.control === "throttle";
  game.attract = new Flight(conditions(game.world), { continuous: throttle });
  game.attractWait = 0;
  stage.setFlight({ conditions: conditions(game.world), terrain: game.attract.terrain, frames: [liveFrame(game.attract)] });
}

function launch() {
  if (!game.opponent) return;
  if (game.phase === "countdown" || game.phase === "flying") return;
  sound.wake(); // launching is a click or key press, so audio may start
  lamps.reset();
  const view = $("play-view").getBoundingClientRect();
  if (view.top < 0 || view.bottom > innerHeight) $("play-view").scrollIntoView({ behavior: "smooth", block: "center" });
  game.attract = null;
  newFlights(game.replay ?? game.challenge ?? Math.floor(Math.random() * 2 ** 31));
  game.replay = null;
  $("overlay").hidden = true;
  $("result-block").hidden = true;
  game.acc = 0;
  game.frame = 0;
  if (game.mode === "watch") {
    game.phase = "flying";
    return;
  }
  game.phase = "countdown";
  const el = $("countdown");
  el.hidden = false;
  let n = 3;
  const tick = () => {
    if (game.phase !== "countdown") return;
    if (n === 0) {
      el.hidden = true;
      game.phase = "flying";
      return;
    }
    el.textContent = n;
    el.classList.remove("pop");
    void el.offsetWidth;
    el.classList.add("pop");
    n -= 1;
    setTimeout(tick, 420);
  };
  tick();
}

function finished() {
  return (!game.you || game.you.done) && (!game.ai || game.ai.done);
}

function stepRound() {
  if (game.you && !game.you.done) game.you.step(yourAction());
  if (game.ai && !game.ai.done) game.ai.step(aiAction(game.ai));
  game.frame += 1;
  if (finished()) endRound();
}

function endRound() {
  game.phase = "done";
  game.held.clear();
  const you = game.you?.record(), ai = game.ai?.record();
  const mine = game.mode === "watch" ? ai : you;
  if (mine?.outcome === "crashed" || mine?.outcome === "flew off screen") {
    sound.crash();
    lamps.say(mine.outcome === "crashed" ? "Hull contact" : "Lost signal", "red");
  } else if (mine?.outcome === "landed on pad") lamps.say("Touchdown");
  if (you) {
    const r = records[game.world] || { best: null, aiBest: null, rounds: 0, landed: false };
    r.rounds += 1;
    r.best = r.best == null ? you.score : Math.max(r.best, you.score);
    if (you.outcome === "landed on pad") r.landed = true;
    if (ai) r.aiBest = r.aiBest == null ? ai.score : Math.max(r.aiBest, ai.score);
    records[game.world] = r;
    store.set("records", records);
    tally.rounds += 1;
    if (ai && you.score > ai.score) tally.wins += 1;
    store.set("tally", tally);
    renderRecords();
    renderWorldPicker();
  }
  showResult(you, ai);
}

function verdict(you, ai) {
  if (!you) return { title: ai.outcome === "landed on pad" ? "The AI landed it." : "Even the AI misses sometimes.", text: `${OUTCOME_LABEL[OUTCOMES.indexOf(ai.outcome)]} with ${ai.score.toFixed(1)} points.` };
  const landed = you.outcome === "landed on pad";
  if (!ai) return {
    title: landed ? "Touchdown." : you.outcome === "crashed" ? "Crashed." : `${OUTCOME_LABEL[OUTCOMES.indexOf(you.outcome)]}.`,
    text: landed ? `${you.score.toFixed(1)} points, ${you.miss.toFixed(2)} m from the center.` : "Slow down before you touch, and keep it upright.",
  };
  const gap = you.score - ai.score;
  if (Math.abs(gap) < 0.05) return { title: "A dead heat.", text: `You both scored ${you.score.toFixed(1)}. Try a harder world: Jupiter or Neptune.` };
  if (gap > 0) return { title: "You beat the AI.", text: `By ${gap.toFixed(1)} points. Now try a harder world: Jupiter or Neptune.` };
  if (landed) return { title: "Landed, but the AI did better.", text: `It won by ${(-gap).toFixed(1)} points: it stopped closer to the center or burned less fuel.` };
  return { title: you.outcome === "crashed" ? "Crashed. The AI didn't." : `${OUTCOME_LABEL[OUTCOMES.indexOf(you.outcome)]}.`, text: "Brake early with the main engine, and use the side engines in short taps." };
}

/** After a loss: you don't have to land it yourself. Train an AI that can. */
function showNudge(you, ai) {
  const lost = you && (you.outcome !== "landed on pad" || (ai && ai.score - you.score >= 0.05));
  $("nudge").hidden = !lost;
  if (!lost) return;
  const world = game.worlds[game.world].name;
  $("nudge-text").textContent = you.outcome === "landed on pad"
    ? `So close! Could your own AI beat this one? Train it on ${world}, then race it right here.`
    : `Crashing is how the AI learned too: its first thousand landings were crashes. Train your own AI and it will land ${world} for you.`;
  game.lastSeed = you.seed;
}

let trainer = null;

/** Race one of your browser-trained AIs, on the world it trained for. */
async function raceMine(id) {
  await loadOpponents(`mine/${id}`);
  const o = game.opponents.find((x) => x.id === `mine/${id}`);
  if (o && o.world && o.world !== game.world) selectWorld(o.world);
  if (game.mode !== "race") $("mode").querySelector('[data-mode="race"]').click();
  $("play-view").scrollIntoView({ behavior: "smooth", block: "center" });
}

function watchReplay() {
  game.replay = game.lastSeed ?? null; // the same terrain you just flew
  $("mode").querySelector('[data-mode="watch"]').click();
  launch();
}

function showResult(you, ai) {
  const v = verdict(you, ai);
  $("overlay-eyebrow").textContent = worldLine(game.world);
  $("overlay-title").textContent = v.title;
  $("overlay-text").textContent = v.text;
  $("launch").innerHTML = "Fly again <kbd>R</kbd>";
  $("overlay").hidden = false;
  showNudge(you, ai);

  const rows = [
    ["Result", (r) => OUTCOME_LABEL[OUTCOMES.indexOf(r.outcome)]],
    ["Score", (r) => fmt.score(r.score)],
    ["From the center", (r) => fmt.meters(r.miss)],
    ["Touchdown", (r) => fmt.speed(r.touchdown)],
    ["Time", (r) => fmt.seconds(r.steps)],
  ];
  const table = $("versus");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const label of ["", you ? "You" : null, ai ? "AI" : null].filter((x) => x !== null)) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  const body = table.createTBody();
  for (const [label, get] of rows) {
    const tr = body.insertRow();
    tr.insertCell().textContent = label;
    for (const r of [you, ai].filter(Boolean)) {
      const td = tr.insertCell();
      td.textContent = get(r);
      td.className = "num";
    }
  }
  if (you && ai && Math.abs(you.score - ai.score) >= 0.05) {
    const winner = you.score > ai.score ? 1 : 2;
    head.cells[winner].classList.add("win");
    body.rows[1].cells[winner].classList.add("win");
  }
  $("share").hidden = !you;
  $("result-block").hidden = false;
}

// Drawing ------------------------------------------------------------------------------

function liveFrame(f) {
  if (f.done) return f.frames[f.frames.length - 1];
  return [...f.pose(), 0, 0, -1, ...f.obs, 0];
}

/** The two frames to blend between for a flight in progress, as setPose wants them. */
function poseOf(f, t) {
  if (f.done) {
    const last = f.frames[f.frames.length - 1];
    return [last, last, 0, true];
  }
  const b = liveFrame(f);
  const a = f.frames.length ? f.frames[f.frames.length - 1] : b;
  return [a, b, t, false];
}

function updateCaption() {
  const name = game.worlds[game.world]?.name || "";
  const mode = { race: "race the AI", solo: "fly solo", watch: "watching the AI" }[game.mode];
  const who = game.opponent ? ` · ${game.opponent.title}` : "";
  $("caption").textContent = `${name} · ${mode}${game.mode === "solo" ? "" : who}`;
}

function updateHud() {
  const f = game.you || game.ai || game.attract;
  const set = (id, text, warn = false) => {
    $(id).textContent = text;
    $(id).classList.toggle("warn", warn);
  };
  if (!f || game.phase === "idle") {
    for (const id of ["h-alt", "h-vy", "h-vx", "h-tilt", "h-time", "h-score", "h-ai"]) set(id, "–");
    return;
  }
  const o = f.obs;
  const alt = o[1] * 6.667, vy = -o[3] * 7.5, vx = o[2] * 5;
  set("h-alt", `${alt.toFixed(1)} m`);
  set("h-vy", `${vy.toFixed(1)} m/s`, alt < 3 && vy > 2.5 && !f.done);
  set("h-vx", `${vx.toFixed(1)} m/s`, Math.abs(vx) > 2 && alt < 3 && !f.done);
  set("h-tilt", `${((o[4] * 180) / Math.PI).toFixed(0)}°`, Math.abs(o[4]) > 0.4 && !f.done);
  set("h-time", fmt.seconds(Math.max(f.limit - f.steps, 0)));
  set("h-score", game.you ? fmt.score(game.you.score) : "–");
  set("h-ai", game.ai ? fmt.score(game.ai.score) : "–");
}

function updateDecision() {
  const f = game.mode === "watch" ? game.ai || game.attract : null;
  $("decision-block").hidden = game.mode !== "watch";
  if (!f || !game.opponent) return;
  const out = forward(game.opponent.weights, f.obs);
  const box = $("decision");
  box.replaceChildren();
  const rows = game.opponent.weights.control === "throttle"
    ? [["main throttle", (out.outputs[0] + 1) / 2, out.outputs[0].toFixed(2)], ["side thrust", Math.abs(out.outputs[1]), out.outputs[1].toFixed(2)]]
    : ACTIONS.map((name, i) => [name, out.probs[i], `${Math.round(out.probs[i] * 100)}%`, i === out.action]);
  for (const [name, share, text, chosen] of rows) {
    const row = document.createElement("div");
    row.className = `prob-row${chosen ? " chosen" : ""}`;
    row.innerHTML = `<span></span><span class="bar"><span></span></span><span class="mono"></span>`;
    row.children[0].textContent = name;
    row.children[1].firstChild.style.width = `${Math.max(0, Math.min(share, 1)) * 100}%`;
    row.children[2].textContent = text;
    box.append(row);
  }
  $("critic").innerHTML = `The network expects <b>${out.value >= 0 ? "+" : ""}${out.value.toFixed(0)}</b> more points from here.`;
}

function renderRecords() {
  const table = $("records");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const label of ["World", "Gravity", "Your best", "AI's best", "Rounds", ""]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  const body = table.createTBody();
  for (const id of WORLD_ORDER) {
    const w = game.worlds[id];
    if (!w) continue;
    const r = records[id] || {};
    const tr = body.insertRow();
    const name = tr.insertCell();
    name.innerHTML = `<span class="world-cell"><span class="world-num"></span><i></i></span>`;
    name.querySelector(".world-num").textContent = String(WORLD_ORDER.indexOf(id) + 1).padStart(2, "0");
    name.querySelector("i").style.background = WORLD_DOT[id] || "#888";
    name.firstChild.append(w.name);
    const cells = [`${Math.abs(w.gravity)} m/s²`, fmt.score(r.best), fmt.score(r.aiBest), String(r.rounds || 0)];
    for (const text of cells) {
      const td = tr.insertCell();
      td.className = "num";
      td.textContent = text;
    }
    const status = tr.insertCell();
    status.className = "row-actions";
    const go = document.createElement("button");
    go.textContent = r.landed ? "Landed ✓ · fly again" : "Fly here";
    go.addEventListener("click", () => {
      selectWorld(id);
      $("play-view").scrollIntoView({ behavior: "smooth", block: "center" });
    });
    status.append(go);
  }
  const landed = Object.values(records).filter((r) => r.landed).length;
  $("tally-rounds").textContent = tally.rounds;
  $("tally-wins").textContent = tally.wins;
  $("tally-worlds").textContent = `${landed} / ${WORLD_ORDER.length}`;
}

// Input --------------------------------------------------------------------------------

function bindInput() {
  const active = () => game.phase === "countdown" || game.phase === "flying";
  const viewVisible = () => {
    const r = $("play-view").getBoundingClientRect();
    return r.bottom > 80 && r.top < innerHeight - 80;
  };
  addEventListener("keydown", (e) => {
    if (e.target.closest?.("select, input, textarea")) return;
    const engine = KEYS[e.code];
    if (engine && active()) {
      e.preventDefault();
      game.held.add(engine);
      return;
    }
    if ((e.code === "Space" || e.code === "Enter") && !active() && viewVisible() && e.target === document.body) {
      e.preventDefault();
      launch();
    } else if (e.code === "KeyR" && !e.metaKey && !e.ctrlKey && !active()) {
      launch();
    } else if (e.code === "KeyN" && !e.metaKey && !e.ctrlKey && !active()) {
      nextWorld();
    } else if (e.code === "Escape" && active()) {
      toIdle();
    } else if (engine && viewVisible() && e.code.startsWith("Arrow")) {
      e.preventDefault(); // arrows fly the rocket here, not the page
    }
  });
  addEventListener("keyup", (e) => {
    const engine = KEYS[e.code];
    if (engine) game.held.delete(engine);
  });
  addEventListener("blur", () => game.held.clear());

  for (const button of $("touch-pad").querySelectorAll("button")) {
    const engine = button.dataset.engine;
    const down = (e) => {
      e.preventDefault();
      if (!active()) launch();
      game.held.add(engine);
      button.classList.add("on");
    };
    const up = () => {
      game.held.delete(engine);
      button.classList.remove("on");
    };
    button.addEventListener("pointerdown", down);
    for (const kind of ["pointerup", "pointerleave", "pointercancel"]) button.addEventListener(kind, up);
    button.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  $("launch").addEventListener("click", launch);
  $("again").addEventListener("click", launch);
  $("next").addEventListener("click", nextWorld);
  $("share").addEventListener("click", share);
  $("nudge-watch").addEventListener("click", watchReplay);
  $("nudge-train").addEventListener("click", () => trainer?.openFor(game.world));
  $("slow").addEventListener("change", (e) => { game.slow = e.target.checked; });
  $("sound").checked = sound.enabled;
  $("sound").addEventListener("change", (e) => {
    sound.enable(e.target.checked);
    if (!e.target.checked) sound.silence();
  });
  $("opponent").addEventListener("change", (e) => setOpponent(e.target.value));
  for (const button of $("mode").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      game.mode = button.dataset.mode;
      for (const b of $("mode").querySelectorAll("button")) {
        b.classList.toggle("on", b === button);
        b.setAttribute("aria-checked", String(b === button));
      }
      $("controls-block").hidden = game.mode === "watch";
      $("result-block").hidden = true;
      toIdle();
    });
  }
  $("reset-records").addEventListener("click", () => {
    if (!confirm("Clear your records and tally in this browser?")) return;
    records = {};
    tally = { rounds: 0, wins: 0 };
    store.set("records", records);
    store.set("tally", tally);
    renderRecords();
    renderWorldPicker();
  });
}

function nextWorld() {
  const i = WORLD_ORDER.indexOf(game.world);
  selectWorld(WORLD_ORDER[(i + 1) % WORLD_ORDER.length]);
}

async function share() {
  const you = game.you?.record(), ai = game.ai?.record();
  if (!you) return;
  const name = game.worlds[game.world].name;
  const base = location.hostname.endsWith("github.io") ? `${location.origin}${location.pathname}` : GAME_URL;
  const link = `${base}?world=${game.world}&seed=${game.you.seed}`; // the same terrain, gusts and push
  const how = you.outcome === "landed on pad" ? `landed on ${name} with ${you.score.toFixed(0)} points` : `tried to land on ${name} (${you.outcome}, ${you.score.toFixed(0)} points)`;
  const versus = ai ? ` The neural network scored ${ai.score.toFixed(0)}.` : "";
  const text = `I ${how} in PerfectLand.${versus} Can you beat it on the same terrain? ${link}`;
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied. Paste it anywhere.");
  } catch {
    prompt("Copy your result:", text);
  }
}

// Main loop ----------------------------------------------------------------------------------

let last = performance.now();
let lastHud = 0;
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  const speed = game.slow && game.mode !== "watch" ? 0.5 : 1;

  if (game.phase === "flying") {
    game.acc += dt * speed;
    while (game.acc >= STEP && game.phase === "flying") {
      stepRound();
      game.acc -= STEP;
    }
  } else if (game.phase === "idle" && game.attract) {
    if (!game.attract.done) {
      game.acc += dt;
      while (game.acc >= STEP && !game.attract.done) {
        game.attract.step(aiAction(game.attract));
        game.acc -= STEP;
      }
    } else if ((game.attractWait += dt) > 1.6) startAttract();
  }

  const t = Math.min(game.acc / STEP, 1);
  const main = game.phase === "idle" ? game.attract : game.mode === "watch" ? game.ai : game.you;
  if (main) {
    stage.setPose(...poseOf(main, t));
    const last = !main.done && main.frames.length ? main.frames[main.frames.length - 1] : null;
    const engine = last ? last[F.main] : 0, side = last ? last[F.side] : 0;
    const happened = lamps.update(engine, side, [main.obs[6], main.obs[7]]);
    if (game.phase === "flying" || game.phase === "done") {
      sound.update(engine, side, game.worlds[game.world].atmosphere);
      if (happened.contact) { sound.ping(); sound.touchdown(Math.min(1, 0.4 + (main.touched ?? 0.6) / 3)); }
    } else sound.silence();
  }
  const ghost = game.phase !== "idle" && game.mode === "race" ? game.ai : null;
  if (ghost) stage.setGhost(...poseOf(ghost, t));
  else stage.setGhost(null);

  if (now - lastHud > 80) {
    lastHud = now;
    updateHud();
    updateDecision();
  }
  stage.render(dt);
  if (globeVisible) globe.render();
  requestAnimationFrame(loop);
}

async function detectServer() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 1500);
  try {
    const r = await fetch("/api/state", { signal: ctrl.signal, cache: "no-store" });
    return r.ok && (r.headers.get("content-type") || "").includes("json");
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

let globeVisible = true;
new IntersectionObserver(([e]) => { globeVisible = e.isIntersecting; }).observe($("globe"));

async function init() {
  game.worlds = await fetch("data/worlds.json").then((r) => r.json());
  game.server = location.protocol.startsWith("http") && !location.hostname.endsWith("github.io") && (await detectServer());
  // Mission control runs on your own computer (python app.py), so the published game has no link to it.
  if (location.hostname.endsWith("github.io")) $("panel-mc").hidden = true;
  setupNav();
  const params = new URLSearchParams(location.search);
  const mode = params.get("mode");
  if (["race", "solo", "watch"].includes(mode)) game.mode = mode;
  const seed = Number(params.get("seed"));
  if (Number.isInteger(seed) && seed > 0) game.challenge = seed;
  const world = params.get("world");
  game.world = game.worlds[world] ? world : store.get("world", "classic");
  if (!game.worlds[game.world]) game.world = "classic";
  renderWorldPicker();
  renderRecords();
  bindInput();
  if (game.mode !== "race") $("mode").querySelector(`[data-mode="${game.mode}"]`).click();
  stage.setWorld(game.world);
  showGlobe();
  $("world-prev").addEventListener("click", () => stepWorld(-1));
  $("world-next").addEventListener("click", () => stepWorld(1));
  renderSpecs();
  await loadOpponents();
  trainer = setupTrainer({ worlds: game.worlds, order: WORLD_ORDER, getWorld: () => game.world, onRace: raceMine,
    onSaved: () => loadOpponents() });
  addEventListener("pagehide", () => store.set("world", game.world));
  requestAnimationFrame(loop);
}

init().catch((error) => {
  console.error(error);
  toast(`Couldn't start the game: ${error.message}`, "error");
});

export { game }; // for scripted tests and recordings
