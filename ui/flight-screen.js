// The screen in "Teach your own pilot": a rocket flying over the world you picked, in the same
// 3D stage as the game. Before training it flies at random (an AI that knows nothing); while
// training it flies the network as it is right now (a new snapshot every few seconds), so you watch
// it get better.

import { forward, prepare } from "./nn.js";
import { FPS, Flight } from "./sim.js";
import { createStage } from "./stage.js";

const STEP = 1 / FPS;
const PHYSICS = ["gravity", "enable_wind", "wind_power", "turbulence_power", "atmosphere", "engine_power"];

export function createFlightScreen(canvas, { worlds, onFrame, watch = [] }) {
  const stage = createStage(canvas);
  stage.options.showTrails = false;
  stage.options.distance = 24;
  const state = { world: null, weights: null, flight: null, acc: 0, wait: 0, visible: false, last: 0 };

  const conditions = () => ({ planet: state.world, ...Object.fromEntries(PHYSICS.map((k) => [k, worlds[state.world][k]])), start_height: 10 });
  const liveFrame = (f) => (f.done ? f.frames.at(-1) : [...f.pose(), 0, 0, -1, ...f.obs, 0]);

  function newFlight() {
    state.flight = new Flight(conditions());
    state.wait = 0;
    stage.setFlight({ conditions: conditions(), terrain: state.flight.terrain, frames: [liveFrame(state.flight)] });
  }

  /** No weights: random button presses, like a brand-new AI. */
  const action = (f) => (state.weights ? forward(state.weights, f.obs).action : Math.floor(Math.random() * 4));

  function loop(now) {
    if (!state.visible) { state.raf = 0; return; }
    const dt = Math.min((now - state.last) / 1000, 0.05);
    state.last = now;
    const f = state.flight;
    if (!f.done) {
      state.acc += dt;
      while (state.acc >= STEP && !f.done) { f.step(action(f)); state.acc -= STEP; }
    } else if ((state.wait += dt) > 1.4) newFlight();
    const b = liveFrame(state.flight), a = state.flight.frames.at(-1) || b;
    stage.setPose(a, b, state.flight.done ? 0 : Math.min(state.acc / STEP, 1), state.flight.done);
    stage.render(dt);
    onFrame?.(state.flight, state.weights ? forward(state.weights, state.flight.obs) : null);
    state.raf = requestAnimationFrame(loop);
  }

  // Only fly while the screen (or something watching it, like the brain) is on screen.
  const seen = new Set();
  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) e.isIntersecting ? seen.add(e.target) : seen.delete(e.target);
    state.visible = seen.size > 0;
    if (state.visible && !state.raf && state.world) { state.last = performance.now(); state.raf = requestAnimationFrame(loop); }
  });
  for (const el of [canvas, ...watch]) observer.observe(el);

  return {
    setWorld(id) {
      if (!worlds[id] || id === state.world) return;
      const first = !state.world;
      state.world = id;
      stage.setWorld(id);
      newFlight();
      if (first && state.visible && !state.raf) { state.last = performance.now(); state.raf = requestAnimationFrame(loop); }
    },
    /** Fly these weights from this frame on (null: fly at random). */
    setPilot(weights) { state.weights = weights ? prepare({ ...weights }) : null; },
  };
}
