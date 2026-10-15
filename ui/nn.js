// The pilot's network, run in the browser: same weights and math as the PyTorch model,
// so every frame's neuron activity can be shown (and "what if" inputs tried) instantly.

export const INPUTS = ["x position", "altitude", "x speed", "y speed", "tilt", "spin", "left leg", "right leg"];
export const ACTIONS = ["nothing", "left engine", "main engine", "right engine"];
export const THROTTLE_OUTPUTS = ["main throttle", "side thrust"];
// Where things live in a recorded flight frame (see FRAME_FIELDS in lab.py).
export const F = { x: 0, y: 1, angle: 2, leg1: 3, leg2: 6, main: 9, side: 10, action: 11, obs: 12, reward: 20 };

/** How the simulator turns a throttle pilot's two outputs into engine power. */
export function throttleEngines(main, side) {
  return {
    main: main > 0 ? (Math.min(main, 1) + 1) / 2 : 0,
    side: Math.abs(side) > 0.5 ? Math.sign(side) * Math.min(Math.abs(side), 1) : 0,
  };
}
export const OUTCOMES = ["landed on pad", "landed off pad", "timed out", "crashed", "flew off screen"];
export const OUTCOME_LABEL = ["Landed on pad", "Landed off pad", "Timed out", "Crashed", "Flew off screen"];

/** Convert the server's nested lists into typed arrays once. */
export function prepare(weights) {
  if (!weights || weights.prepared) return weights;
  const layers = (list) => list.map(({ w, b }) => ({
    w: w.map((row) => Float32Array.from(row)),
    b: Float32Array.from(b),
  }));
  return {
    prepared: true, activation: weights.activation, control: weights.control,
    actor: layers(weights.actor), critic: layers(weights.critic),
  };
}

function run(layers, input, activation) {
  const act = activation === "relu" ? (v) => (v > 0 ? v : 0) : Math.tanh;
  const outputs = [Float32Array.from(input)];
  let x = outputs[0];
  layers.forEach(({ w, b }, l) => {
    const out = new Float32Array(w.length);
    const last = l === layers.length - 1;
    for (let j = 0; j < w.length; j++) {
      const row = w[j];
      let s = b[j];
      for (let i = 0; i < row.length; i++) s += row[i] * x[i];
      out[j] = last ? s : act(s);
    }
    outputs.push(out);
    x = out;
  });
  return outputs;
}

/**
 * One decision: activations of every layer and the critic's estimate, plus either
 * button probabilities (button pilots) or the two throttle outputs (throttle pilots).
 */
export function forward(weights, obs) {
  const layers = run(weights.actor, obs, weights.activation);
  const out = layers[layers.length - 1];
  const critic = run(weights.critic, obs, weights.activation);
  const value = critic[critic.length - 1][0];
  if (weights.control === "throttle") {
    const outputs = Array.from(out, (v) => Math.max(-1, Math.min(1, v))); // the simulator clips to -1..1
    return { layers, outputs, engines: throttleEngines(outputs[0], outputs[1]), action: -1, value };
  }
  const max = Math.max(...out);
  const exps = Array.from(out, (v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / total);
  return { layers, probs, outputs: probs, action: probs.indexOf(Math.max(...probs)), value };
}

/** Activations for every frame of a flight, plus per-neuron statistics. */
export function analyzeFlight(weights, frames, obsIndex) {
  const results = frames.map((f) => forward(weights, f.slice(obsIndex, obsIndex + 8)));
  const hidden = weights.actor.length - 1;
  const health = [];
  for (let l = 1; l <= hidden; l++) {
    const n = results[0].layers[l].length;
    const counts = { active: 0, quiet: 0, saturated: 0 };
    for (let j = 0; j < n; j++) {
      let sum = 0, sq = 0, absSum = 0;
      for (const r of results) {
        const v = r.layers[l][j];
        sum += v; sq += v * v; absSum += Math.abs(v);
      }
      const mean = sum / results.length;
      const std = Math.sqrt(Math.max(sq / results.length - mean * mean, 0));
      const meanAbs = absSum / results.length;
      if (std < 0.05 && meanAbs > 0.9) counts.saturated++;
      else if (std < 0.05) counts.quiet++;
      else counts.active++;
    }
    health.push({ layer: l, size: n, ...counts });
  }
  return { results, health };
}

// The reward function from reward.py, for showing how training would score a frame.
export function potential(o, w) {
  return (
    -w.DISTANCE_WEIGHT * Math.hypot(o[0], o[1]) -
    w.SPEED_WEIGHT * Math.hypot(o[2], o[3]) -
    w.TILT_WEIGHT * Math.abs(o[4]) +
    w.LEG_CONTACT_BONUS * (o[6] + o[7])
  );
}

/** How reward.py scores one frame, given the engine power that was applied (main 0..1, side -1..1). */
export function rewardBreakdown(prev, obs, outcome, w, main, side) {
  if (outcome === 0 || outcome === 1) return { total: w.LANDING_BONUS, parts: [["landing", w.LANDING_BONUS]] };
  if (outcome === 3 || outcome === 4) return { total: w.CRASH_PENALTY, parts: [["crash", w.CRASH_PENALTY]] };
  const parts = [["progress", potential(obs, w) - potential(prev, w)]];
  const fuel = w.MAIN_ENGINE_COST * main + w.SIDE_ENGINE_COST * Math.abs(side);
  if (fuel) parts.push(["fuel", -fuel]);
  parts.push(["time", -w.TIME_COST]);
  const parked = prev[6] && prev[7] && Math.hypot(prev[2], prev[3]) < 0.05;
  if (parked && (main || side)) parts.push(["firing while parked", -w.GROUND_ENGINE_COST]);
  return { total: parts.reduce((s, [, v]) => s + v, 0), parts };
}
