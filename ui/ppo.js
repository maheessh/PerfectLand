// PPO in plain JavaScript, so anyone can train a pilot in their own browser: the same recipe as
// Stable-Baselines3's PPO that mission control uses (separate 64x64 tanh networks for the policy and
// the value, orthogonal init, GAE, clipped objective, entropy bonus, Adam, gradient clipping).
// No libraries: the networks are small enough that hand-written backpropagation is plenty fast.
//
// The trained weights come out in the same format as export_web.py's, so the game, the neuron
// view and the flight computer figure can use a browser-trained pilot directly.

import { Flight } from "./sim.js";

export const DEFAULTS = {
  envs: 8, steps: 256, epochs: 4, minibatch: 256, gamma: 0.999, lambda: 0.98,
  clip: 0.2, lr: 3e-4, entropy: 0.01, valueCoef: 0.5, maxGrad: 0.5, hidden: [64, 64],
};

// Random numbers ----------------------------------------------------------------------------

export function rng(seed = 1) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  next.gauss = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    const u = Math.max(next(), 1e-12), v = next(), r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return next;
}

// Networks ----------------------------------------------------------------------------------

/** An orthogonal (rows x cols) matrix times gain, like PyTorch's orthogonal_ init. */
function orthogonal(rows, cols, gain, rand) {
  const n = Math.max(rows, cols), m = Math.min(rows, cols);
  const q = Array.from({ length: m }, () => Float64Array.from({ length: n }, () => rand.gauss()));
  for (let i = 0; i < m; i++) { // Gram-Schmidt
    for (let j = 0; j < i; j++) {
      let d = 0;
      for (let k = 0; k < n; k++) d += q[i][k] * q[j][k];
      for (let k = 0; k < n; k++) q[i][k] -= d * q[j][k];
    }
    let len = 0;
    for (let k = 0; k < n; k++) len += q[i][k] ** 2;
    len = Math.sqrt(len) || 1;
    for (let k = 0; k < n; k++) q[i][k] /= len;
  }
  const W = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) W[r * cols + c] = gain * (rows <= cols ? q[r][c] : q[c][r]);
  return W;
}

/** A tanh MLP: layers of { W (out x in, row-major), b }. The last layer is linear. */
export function makeNet(sizes, outGain, rand) {
  return sizes.slice(1).map((out, l) => ({
    nin: sizes[l], nout: out,
    W: orthogonal(out, sizes[l], l === sizes.length - 2 ? outGain : Math.SQRT2, rand),
    b: new Float64Array(out),
  }));
}

/** Forward pass keeping every layer's output (hidden ones after tanh). */
export function forward(net, x) {
  const outs = [];
  let a = x;
  for (let l = 0; l < net.length; l++) {
    const { W, b, nin, nout } = net[l], o = new Float64Array(nout), last = l === net.length - 1;
    for (let j = 0; j < nout; j++) {
      let s = b[j];
      const row = j * nin;
      for (let i = 0; i < nin; i++) s += W[row + i] * a[i];
      o[j] = last ? s : Math.tanh(s);
    }
    outs.push(o);
    a = o;
  }
  return outs;
}

/** Add dLoss/dParams to grads, given the input, the stored outputs and dLoss/dLastOutput. */
function backward(net, x, outs, gLast, grads) {
  let g = gLast;
  for (let l = net.length - 1; l >= 0; l--) {
    const { W, nin, nout } = net[l], a = l === 0 ? x : outs[l - 1], gW = grads[l].W, gb = grads[l].b;
    for (let j = 0; j < nout; j++) {
      const gj = g[j];
      if (gj === 0) continue;
      gb[j] += gj;
      const row = j * nin;
      for (let i = 0; i < nin; i++) gW[row + i] += gj * a[i];
    }
    if (l === 0) break;
    const ga = new Float64Array(nin);
    for (let j = 0; j < nout; j++) {
      const gj = g[j];
      if (gj === 0) continue;
      const row = j * nin;
      for (let i = 0; i < nin; i++) ga[i] += W[row + i] * gj;
    }
    for (let i = 0; i < nin; i++) ga[i] *= 1 - a[i] * a[i]; // tanh'
    g = ga;
  }
}

const zerosLike = (net) => net.map((L) => ({ W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) }));

function softmax(z) {
  let m = -Infinity;
  for (const v of z) m = Math.max(m, v);
  const p = Array.from(z, (v) => Math.exp(v - m));
  const s = p.reduce((a, b) => a + b, 0);
  return p.map((v) => v / s);
}

/** Weights as the game reads them (export_web.py's format). */
export function exportWeights(agent, digits = 6) {
  const layers = (net) => net.map(({ W, b, nin, nout }) => ({
    w: Array.from({ length: nout }, (_, j) => Array.from(W.subarray(j * nin, (j + 1) * nin), (v) => +v.toFixed(digits))),
    b: Array.from(b, (v) => +v.toFixed(digits)),
  }));
  return { activation: "tanh", control: "buttons", actor: layers(agent.actor), critic: layers(agent.critic) };
}

/** A pilot from exported weights, to keep training it. */
export function importWeights(w) {
  const layers = (list) => list.map(({ w: rows, b }) => ({
    nin: rows[0].length, nout: rows.length, W: Float64Array.from(rows.flat()), b: Float64Array.from(b),
  }));
  return { actor: layers(w.actor), critic: layers(w.critic) };
}

export function newAgent(seed = 1, hidden = DEFAULTS.hidden) {
  const rand = rng(seed);
  return { actor: makeNet([8, ...hidden, 4], 0.01, rand), critic: makeNet([8, ...hidden, 1], 1, rand) };
}

/** The best button for an observation (what the game does with a trained pilot). */
export function act(agent, obs) {
  const z = forward(agent.actor, obs).at(-1);
  let best = 0;
  for (let i = 1; i < 4; i++) if (z[i] > z[best]) best = i;
  return best;
}

// Adam ----------------------------------------------------------------------------------------

function adam(params, lr) {
  const m = params.map((p) => new Float64Array(p.length)), v = params.map((p) => new Float64Array(p.length));
  let t = 0;
  return (grads) => {
    t += 1;
    const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
    params.forEach((p, k) => {
      const g = grads[k], mk = m[k], vk = v[k];
      for (let i = 0; i < p.length; i++) {
        mk[i] = 0.9 * mk[i] + 0.1 * g[i];
        vk[i] = 0.999 * vk[i] + 0.001 * g[i] * g[i];
        p[i] -= (lr * (mk[i] / c1)) / (Math.sqrt(vk[i] / c2) + 1e-5);
      }
    });
  };
}
const flat = (net) => net.flatMap((L) => [L.W, L.b]);

// Training --------------------------------------------------------------------------------------

/**
 * A trainer for one world. step() collects one batch of practice (envs x steps frames), learns from
 * it, and returns what happened; call it until you have enough frames.
 */
export function createTrainer({ conditions, agent = newAgent(), seed = 1, ...options }) {
  const o = { ...DEFAULTS, ...options };
  const rand = rng(seed + 7919);
  const nextSeed = () => Math.floor(rand() * 2 ** 31);
  const envs = Array.from({ length: o.envs }, () => ({ flight: new Flight(conditions, { seed: nextSeed() }), score: 0 }));
  const stepActor = adam(flat(agent.actor), o.lr), stepCritic = adam(flat(agent.critic), o.lr);
  const N = o.envs * o.steps;
  const buf = {
    obs: new Float64Array(N * 8), act: new Int8Array(N), logp: new Float64Array(N), val: new Float64Array(N),
    rew: new Float64Array(N), done: new Uint8Array(N), adv: new Float64Array(N), ret: new Float64Array(N),
  };
  let frames = 0;
  const finished = []; // [score, outcome] of practice flights since the last report

  function collect() {
    for (let t = 0; t < o.steps; t++) {
      envs.forEach((e, k) => {
        const i = t * o.envs + k, obs = e.flight.obs;
        for (let d = 0; d < 8; d++) buf.obs[i * 8 + d] = obs[d];
        const p = softmax(forward(agent.actor, obs).at(-1));
        let r = rand(), a = 0;
        while (a < 3 && r > p[a]) r -= p[a++];
        buf.act[i] = a;
        buf.logp[i] = Math.log(Math.max(p[a], 1e-12));
        buf.val[i] = forward(agent.critic, obs).at(-1)[0];
        const before = e.flight.score;
        e.flight.step(a);
        buf.rew[i] = e.flight.score - before;
        buf.done[i] = e.flight.done ? 1 : 0;
        if (e.flight.done) {
          finished.push([e.flight.score, e.flight.outcome]);
          e.flight = new Flight(conditions, { seed: nextSeed() });
        }
      });
    }
    frames += N;
    // Generalized advantage estimation, per simulator, bootstrapping from the value of where it is now.
    envs.forEach((e, k) => {
      let next = forward(agent.critic, e.flight.obs).at(-1)[0], gae = 0;
      for (let t = o.steps - 1; t >= 0; t--) {
        const i = t * o.envs + k, alive = 1 - buf.done[i];
        const delta = buf.rew[i] + o.gamma * next * alive - buf.val[i];
        gae = delta + o.gamma * o.lambda * alive * gae;
        buf.adv[i] = gae;
        buf.ret[i] = gae + buf.val[i];
        next = buf.val[i];
      }
    });
  }

  function learn() {
    const idx = Int32Array.from({ length: N }, (_, i) => i);
    let clipped = 0, seen = 0;
    for (let epoch = 0; epoch < o.epochs; epoch++) {
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      for (let s = 0; s < N; s += o.minibatch) {
        const batch = idx.subarray(s, Math.min(s + o.minibatch, N)), B = batch.length;
        let mean = 0, sq = 0;
        for (const i of batch) { mean += buf.adv[i]; sq += buf.adv[i] ** 2; }
        mean /= B;
        const std = Math.sqrt(Math.max(sq / B - mean * mean, 0)) + 1e-8;
        const gA = zerosLike(agent.actor), gC = zerosLike(agent.critic);
        for (const i of batch) {
          const x = buf.obs.subarray(i * 8, i * 8 + 8), a = buf.act[i], adv = (buf.adv[i] - mean) / std;
          // policy: clipped surrogate and an entropy bonus
          const outs = forward(agent.actor, x), p = softmax(outs.at(-1));
          const ratio = Math.exp(Math.log(Math.max(p[a], 1e-12)) - buf.logp[i]);
          const clip = adv >= 0 ? ratio > 1 + o.clip : ratio < 1 - o.clip;
          if (clip) clipped++;
          seen++;
          let H = 0;
          for (const q of p) H -= q * Math.log(Math.max(q, 1e-12));
          const g = new Float64Array(4);
          for (let j = 0; j < 4; j++) {
            const lp = Math.log(Math.max(p[j], 1e-12));
            const surrogate = clip ? 0 : -adv * ratio * ((j === a ? 1 : 0) - p[j]);
            const entropy = -o.entropy * (-p[j] * (lp + H));
            g[j] = (surrogate + entropy) / B;
          }
          backward(agent.actor, x, outs, g, gA);
          // value: squared error to the return
          const vOuts = forward(agent.critic, x);
          backward(agent.critic, x, vOuts, [(2 * o.valueCoef * (vOuts.at(-1)[0] - buf.ret[i])) / B], gC);
        }
        const fa = flat(gA), fc = flat(gC);
        let norm = 0;
        for (const arr of [...fa, ...fc]) for (const v of arr) norm += v * v;
        norm = Math.sqrt(norm);
        if (norm > o.maxGrad) for (const arr of [...fa, ...fc]) for (let k = 0; k < arr.length; k++) arr[k] *= o.maxGrad / norm;
        stepActor(fa);
        stepCritic(fc);
      }
    }
    return clipped / Math.max(seen, 1);
  }

  return {
    agent,
    get frames() { return frames; },
    /** One batch: practise, then learn. Returns the practice flights that ended and how much it changed. */
    step() {
      collect();
      const clipFraction = learn();
      return { frames, flights: finished.splice(0), clipFraction };
    },
  };
}

/** Test flights: the same terrains every time, no randomness in the decisions. */
export function test(agent, conditions, flights = 10, seed = 9000) {
  let landed = 0, score = 0;
  for (let k = 0; k < flights; k++) {
    const f = new Flight(conditions, { seed: seed + k });
    while (!f.done) f.step(act(agent, f.obs));
    landed += f.outcome === "landed on pad" ? 1 : 0;
    score += f.score;
  }
  return { landed, flights, score: score / flights };
}
