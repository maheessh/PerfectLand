// "Train your own AI", in the game page: pick a world and how long, watch your pilot learn in a
// background thread, then race it. Pilots are saved in this browser (my-ais.js), so they're yours.

import { createFlightScreen } from "./flight-screen.js";
import { myAIs, profile } from "./my-ais.js";
import { WORLD_DOT } from "./worlds.js";

const $ = (id) => document.getElementById(id);
const PHYSICS = ["gravity", "enable_wind", "wind_power", "turbulence_power", "atmosphere", "engine_power"];
const LENGTHS = [ // measured: ~7,500 frames a second on a laptop; first landings around 500k frames
  { id: "quick", label: "Quick", frames: 300_000, time: "45 sec", note: "learns to hover" },
  { id: "standard", label: "Normal", frames: 700_000, time: "1½ min", note: "usually lands" },
  { id: "long", label: "Long", frames: 1_500_000, time: "3½ min", note: "lands well" },
];
const COORDS = [ // the 8 numbers the AI sees, in order
  ["Side", (v) => v.toFixed(2)], ["Height", (v) => v.toFixed(2)], ["Side speed", (v) => v.toFixed(2)], ["Up speed", (v) => v.toFixed(2)],
  ["Tilt", (v) => v.toFixed(2)], ["Spin", (v) => v.toFixed(2)], ["Left leg", (v) => (v ? "down" : "up")], ["Right leg", (v) => (v ? "down" : "up")],
];
const OUTCOME_COLOR = { "landed on pad": "#3987e5", "landed off pad": "#d95926", "timed out": "#199e70", crashed: "#6e6d68", "flew off screen": "#6e6d68" };

export function setupTrainer({ worlds, order, onRace, onSaved, getWorld }) {
  const state = { worker: null, pilot: null, length: "standard", tests: [], dots: [], last: null, framesBefore: 0 };
  const el = {
    setup: $("tr-setup"), live: $("tr-live"), name: $("tr-player"), aiName: $("tr-ai-name"), world: $("tr-world"),
    lengths: $("tr-lengths"), dialog: $("tr-dialog"), form: $("tr-form"), stop: $("tr-stop"), bar: $("tr-bar"), status: $("tr-status"),
    chart: $("tr-chart"), list: $("tr-list"), done: $("tr-done"), import: $("tr-import"), file: $("tr-file"), title: $("tr-title"),
  };

  // The flight screen, its planet picker and the 8 numbers the AI sees ----------------------------------
  el.coords = $("tr-coords");
  el.coords.replaceChildren(...COORDS.map(([label]) => {
    const d = document.createElement("div");
    d.innerHTML = "<dt></dt><dd class=\"mono\">–</dd>";
    d.firstChild.textContent = label;
    return d;
  }));
  let lastCoords = 0;
  const screen = createFlightScreen($("tr-stage"), {
    worlds,
    onFrame(f) {
      const now = performance.now();
      if (now - lastCoords < 100) return;
      lastCoords = now;
      COORDS.forEach(([, show], i) => { el.coords.children[i].lastChild.textContent = show(f.obs[i]); });
    },
  });
  const tag = (text) => { $("tr-view-tag").textContent = text; };
  const IDLE_TAG = "A brand-new AI · pressing buttons at random";
  const planets = $("tr-planets");
  planets.replaceChildren(...order.filter((id) => worlds[id]).map((id) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "world-chip";
    chip.setAttribute("role", "tab");
    chip.dataset.world = id;
    const dot = document.createElement("i");
    dot.style.background = WORLD_DOT[id] || "#888";
    chip.append(dot, worlds[id].name);
    chip.addEventListener("click", () => pickWorld(id));
    return chip;
  }));
  /** Show a planet on the screen and make it the one to practise on. */
  function pickWorld(id) {
    if (!worlds[id]) return;
    el.world.value = id;
    for (const c of planets.children) c.setAttribute("aria-selected", String(c.dataset.world === id));
    if (!state.worker) screen.setWorld(id);
    el.aiName.placeholder = suggestName();
  }

  // Setup form ----------------------------------------------------------------------------------
  el.name.value = profile.name;
  for (const id of order) if (worlds[id]) el.world.append(new Option(worlds[id].name, id));
  const suggestName = () => `${profile.name ? `${profile.name}'s` : "My"} ${worlds[el.world.value].name} pilot`;
  pickWorld(getWorld());
  el.name.addEventListener("input", () => { profile.name = el.name.value; el.aiName.placeholder = suggestName(); });
  el.world.addEventListener("change", () => pickWorld(el.world.value));
  el.lengths.replaceChildren(...LENGTHS.map((l) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.length = l.id;
    b.innerHTML = `<b>${l.label}</b><span>${l.time}</span><span>${l.note}</span>`;
    b.addEventListener("click", () => { state.length = l.id; paintLengths(); });
    return b;
  }));
  const paintLengths = () => { for (const b of el.lengths.children) b.classList.toggle("on", b.dataset.length === state.length); };
  paintLengths();

  el.form.addEventListener("submit", (e) => { e.preventDefault(); el.dialog.close(); start(); });
  for (const id of ["tr-cancel", "tr-cancel-x"]) $(id).addEventListener("click", () => el.dialog.close());
  el.dialog.addEventListener("click", (e) => { if (e.target === el.dialog) el.dialog.close(); }); // click outside
  $("tr-open").addEventListener("click", () => openDialog());
  el.stop.addEventListener("click", () => state.worker?.postMessage({ type: "stop" }));
  el.import.addEventListener("click", () => el.file.click());
  el.file.addEventListener("change", async () => {
    const file = el.file.files[0];
    el.file.value = "";
    if (!file) return;
    try { const p = await myAIs.upload(file); await renderList(); onSaved(p); flash(`Imported ${p.name}.`); }
    catch (e) { flash(e.message || "Couldn't read that file."); }
  });

  /** Start training: a new pilot, or more training for an existing one (keepTraining). */
  async function start(existing = null) {
    if (state.worker) return;
    if (typeof Worker === "undefined") return flash("This browser can't train in the background.");
    const length = LENGTHS.find((l) => l.id === state.length);
    const pilot = existing || myAIs.create({ name: el.aiName.value.trim() || suggestName(), world: el.world.value });
    const w = worlds[pilot.world];
    const conditions = { ...Object.fromEntries(PHYSICS.map((k) => [k, w[k]])), start_height: 10 };
    state.pilot = pilot;
    screen.setWorld(pilot.world);
    screen.setPilot(existing?.weights || null);
    tag(existing ? `${pilot.name} · its best so far` : `${pilot.name} · random button presses at first`);
    state.tests = pilot.history.map(([frames, landed, flights, score]) => ({ frames, landed, flights, score }));
    state.framesBefore = pilot.frames;
    state.dots = [];
    state.last = null;
    el.title.textContent = existing ? `Training ${pilot.name} more` : `Training ${pilot.name}`;
    el.setup.hidden = true;
    el.live.hidden = false;
    el.done.hidden = true;
    el.stop.hidden = false;
    el.status.textContent = "Starting… the first flights are random button presses.";
    const worker = new Worker(new URL("./train-worker.js", import.meta.url), { type: "module" });
    state.worker = worker;
    worker.onmessage = ({ data }) => onMessage(data);
    worker.onerror = (e) => { flash(`Training stopped: ${e.message || "an error"}`); finish(null); };
    worker.postMessage({ type: "start", conditions, frames: length.frames, weights: existing?.latest || null });
  }

  function onMessage(m) {
    const p = state.pilot;
    if (m.type === "progress") {
      state.last = m;
      for (const [score, outcome] of m.flights) state.dots.push([p.frames + m.frames, score, outcome]);
      if (state.dots.length > 4000) state.dots.splice(0, state.dots.length - 4000);
      el.bar.style.width = `${(100 * m.frames) / m.goal}%`;
      const left = m.frames > 0 ? (m.seconds / m.frames) * (m.goal - m.frames) : 0;
      el.status.textContent = `${(m.frames / 1000).toFixed(0)}k of ${(m.goal / 1000).toFixed(0)}k frames · `
        + `${left > 1 ? `about ${fmtTime(left)} left` : "finishing"} · keep this tab open`;
      draw();
    } else if (m.type === "test") {
      state.tests.push({ frames: state.framesBefore + m.frames, landed: m.landed, flights: m.flights, score: m.score });
      if (m.snapshot) {
        screen.setPilot(m.snapshot);
        tag(`${p.name} after ${((state.framesBefore + m.frames) / 1000).toFixed(0)}k frames · landed ${m.landed} of ${m.flights} tests`);
      }
      if (m.weights && (!p.best || m.landed > p.best.landed || (m.landed === p.best.landed && m.score > p.best.score))) {
        p.best = { landed: m.landed, flights: m.flights, score: Math.round(m.score * 10) / 10, frames: state.framesBefore + m.frames };
        p.weights = m.weights;
      }
      draw();
    } else if (m.type === "done") {
      p.frames = state.framesBefore + m.frames;
      p.latest = m.weights;
      if (!p.weights) p.weights = m.weights;
      p.sessions += 1;
      p.history = state.tests.map((t) => [t.frames, t.landed, t.flights, Math.round(t.score)]);
      finish(p, m.stopped);
    }
  }

  async function finish(p, stopped = false) {
    state.worker?.terminate();
    state.worker = null;
    el.stop.hidden = true;
    if (!p) { toIdle(); return; }
    await myAIs.save(p);
    await renderList();
    onSaved(p);
    const b = p.best;
    screen.setPilot(p.weights);
    tag(`${p.name} · its best version${b ? ` (${b.landed} of ${b.flights} landed)` : ""}`);
    el.status.textContent = `${stopped ? "Stopped" : "Done"} after ${(p.frames / 1000).toFixed(0)}k frames of practice. Saved in this browser.`;
    el.done.hidden = false;
    $("tr-done-text").textContent = !b ? "It didn't get a test flight in yet. Keep training it."
      : b.landed >= 8 ? `${p.name} landed ${b.landed} of ${b.flights} test flights. Race it and see who's better!`
      : b.landed >= 3 ? `${p.name} landed ${b.landed} of ${b.flights} test flights. It's getting there: race it, or train it more.`
      : `${p.name} landed ${b.landed} of ${b.flights} test flights. Every pilot starts by crashing. Train it more and it will learn.`;
    $("tr-race").onclick = () => onRace(p.id);
    $("tr-more").onclick = () => start(p);
    $("tr-new").onclick = () => { toIdle(); openDialog(); };
  }

  // The live chart: every practice flight as a dot, test landings as bars --------------------------------
  function draw() {
    const c = el.chart, dpr = Math.min(devicePixelRatio || 1, 2), w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const total = state.framesBefore + (state.last?.goal || 1), left = 34, right = 8, top = 10, bottom = 20;
    const X = (f) => left + ((w - left - right) * f) / Math.max(total, 1), Y = (s) => top + ((h - top - bottom) * (300 - Math.max(-400, Math.min(300, s)))) / 700;
    g.font = "10px JetBrains Mono, monospace";
    g.fillStyle = "rgba(255,255,255,0.45)";
    g.strokeStyle = "rgba(255,255,255,0.12)";
    for (const s of [200, 0, -200]) {
      g.beginPath(); g.moveTo(left, Y(s)); g.lineTo(w - right, Y(s)); g.stroke();
      g.fillText(String(s), 2, Y(s) + 3);
    }
    g.fillText("200 = solved", left + 4, Y(200) - 4);
    // test landings: bars from the bottom, height = share landed
    for (const t of state.tests) {
      const bh = ((h - top - bottom) * t.landed) / t.flights;
      g.fillStyle = "rgba(57,135,229,0.22)";
      g.fillRect(X(t.frames) - 3, h - bottom - bh, 6, bh);
    }
    for (const [f, s, outcome] of state.dots) {
      g.fillStyle = OUTCOME_COLOR[outcome] || "#888";
      g.fillRect(X(f) - 1, Y(s) - 1, 2, 2);
    }
    // the average of the last 100 practice flights, as a line
    if (state.dots.length > 20) {
      g.strokeStyle = "#fff";
      g.lineWidth = 1.5;
      g.beginPath();
      let sum = 0;
      state.dots.forEach(([f, s], i) => {
        sum += s;
        if (i >= 100) sum -= state.dots[i - 100][1];
        if (i % 10 === 0) { const y = Y(sum / Math.min(i + 1, 100)); if (i === 0) g.moveTo(X(f), y); else g.lineTo(X(f), y); }
      });
      g.stroke();
      g.lineWidth = 1;
    }
    const lastTest = state.tests.at(-1);
    $("tr-test").textContent = lastTest ? `${lastTest.landed}/${lastTest.flights}` : "–";
    $("tr-avg").textContent = state.last?.avgScore != null ? Math.round(state.last.avgScore) : "–";
    $("tr-frames").textContent = `${(((state.framesBefore || 0) + (state.last?.frames || 0)) / 1000).toFixed(0)}k`;
  }
  new ResizeObserver(draw).observe(el.chart);

  // My AIs ---------------------------------------------------------------------------------------------
  async function renderList() {
    const all = await myAIs.list();
    if (!all.length) {
      el.list.innerHTML = `<li class="empty">No AIs yet. Train one: it starts knowing nothing and learns to land in a few minutes.</li>`;
      return all;
    }
    el.list.replaceChildren(...all.map((p) => {
      const li = document.createElement("li");
      const b = p.best;
      li.innerHTML = `<div class="ai-head"><strong></strong><span class="mono"></span></div><p class="ai-meta"></p>
        <div class="ai-actions"><button data-do="race">Race it</button><button data-do="train">Train more</button><button data-do="export">Export</button><button data-do="delete" class="danger">Delete</button></div>`;
      li.querySelector("strong").textContent = p.name;
      li.querySelector(".mono").textContent = b ? `${b.landed}/${b.flights}` : "–";
      li.querySelector(".ai-meta").textContent = `${worlds[p.world]?.name || p.world} · ${(p.frames / 1000).toFixed(0)}k frames of practice · best test ${b ? `${b.landed} of ${b.flights} landed` : "not yet"}`;
      li.addEventListener("click", async (e) => {
        const what = e.target.closest("button")?.dataset.do;
        if (!what) return;
        if (what === "race") onRace(p.id);
        if (what === "train") { if (state.worker) return flash("One at a time: wait for this training to finish."); el.setup.hidden = true; start(p); $("train").scrollIntoView({ behavior: "smooth" }); }
        if (what === "export") myAIs.download(p);
        if (what === "delete") {
          if (!confirm(`Delete ${p.name}? It's only saved in this browser, so it will be gone.`)) return;
          await myAIs.remove(p.id);
          await renderList();
          onSaved(null);
        }
      });
      return li;
    }));
    return all;
  }

  function flash(text) {
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = text;
    $("toasts").append(t);
    setTimeout(() => t.remove(), 3600);
  }

  function toIdle() {
    el.setup.hidden = false;
    el.live.hidden = true;
    screen.setPilot(null);
    tag(IDLE_TAG);
    pickWorld(el.world.value);
  }

  /** The "Train a new AI" pop-up: name, world and how long. */
  function openDialog(world) {
    if (state.worker) { $("train").scrollIntoView({ behavior: "smooth" }); return flash("One at a time: wait for this training to finish."); }
    if (worlds[world]) pickWorld(world);
    el.aiName.value = "";
    el.aiName.placeholder = suggestName();
    el.dialog.showModal();
    (profile.name ? el.aiName : el.name).focus();
  }

  renderList();
  return {
    renderList,
    /** Open the trainer on a world (from the "Train my AI" nudge). */
    openFor(world) {
      if (!state.worker) toIdle();
      openDialog(world);
    },
  };
}

const fmtTime = (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);
