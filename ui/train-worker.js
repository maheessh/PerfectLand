// Trains a pilot in a background thread, so the page stays smooth while it learns.
// In:  { type: "start", conditions, frames, weights?, seed }  |  { type: "stop" }
// Out: progress after every batch, a test every ~20k frames (with the weights when it's the best
//      so far), and "done" with the final and best weights.

import { createTrainer, exportWeights, importWeights, newAgent, test } from "./ppo.js";

let stopped = false;

self.onmessage = async ({ data }) => {
  if (data.type === "stop") { stopped = true; return; }
  if (data.type !== "start") return;
  stopped = false;
  const { conditions, frames: goal, weights, seed = Math.floor(Math.random() * 1e9) } = data;
  const agent = weights ? importWeights(weights) : newAgent(seed);
  const trainer = createTrainer({ conditions, agent, seed });
  const start = performance.now();
  let recent = [], best = null, lastTest = 0, done = 0;
  const report = () => {
    const t = test(agent, conditions, 10);
    const better = !best || t.landed > best.landed || (t.landed === best.landed && t.score > best.score);
    if (better) best = { ...t, frames: done, weights: exportWeights(agent) };
    self.postMessage({ type: "test", frames: done, ...t, best: better, weights: better ? best.weights : null });
  };
  while (done < goal && !stopped) {
    const r = trainer.step();
    done = r.frames;
    recent.push(...r.flights);
    recent = recent.slice(-100);
    const landed = recent.filter(([, outcome]) => outcome === "landed on pad").length;
    self.postMessage({
      type: "progress", frames: done, goal, seconds: (performance.now() - start) / 1000,
      avgScore: recent.length ? recent.reduce((s, [score]) => s + score, 0) / recent.length : null,
      landRate: recent.length ? landed / recent.length : 0, attempts: r.flights.length,
      flights: r.flights.map(([score, outcome]) => [Math.round(score), outcome]),
    });
    if (done - lastTest >= 20000) { lastTest = done; report(); }
    await new Promise((resolve) => setTimeout(resolve, 0)); // let "stop" arrive
  }
  if (done > lastTest) report();
  self.postMessage({ type: "done", frames: done, stopped, weights: exportWeights(agent), best });
};
