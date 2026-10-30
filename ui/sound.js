// The lander's sounds, generated with Web Audio (no files): the main engine's roar, the side
// thrusters' hiss, touchdown and crash. Browsers only allow audio after a click or key press,
// so the audio graph is built by the first enable() that follows one.

const KEY = "perfectland.sound";

function noise(ctx, brown) {
  const buffer = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  let last = 0;
  for (let i = 0; i < data.length; i++) {
    const white = Math.random() * 2 - 1;
    if (brown) {
      last = (last + 0.02 * white) / 1.02; // integrated noise: a deep rumble
      data[i] = last * 3.5;
    } else data[i] = white;
  }
  return buffer;
}

function loop(ctx, buffer) {
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  source.start();
  return source;
}

/** fallback: whether sound starts on when this browser has no saved choice. */
export function createSound({ fallback = false } = {}) {
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch { /* fine without */ }
  let on = saved === null ? fallback : saved === "on";
  let ctx = null, out = null, roar = null, roarFilter = null, rumble = null, hiss = null, brown = null;

  function build() {
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) return false;
    ctx = new Context();
    out = ctx.createGain();
    out.gain.value = 0.55;
    out.connect(ctx.destination);
    brown = noise(ctx, true);
    // Main engine: brown noise through a lowpass that opens with the throttle, plus a low hum.
    roarFilter = ctx.createBiquadFilter();
    roarFilter.type = "lowpass";
    roarFilter.frequency.value = 200;
    roar = ctx.createGain();
    roar.gain.value = 0;
    loop(ctx, brown).connect(roarFilter).connect(roar).connect(out);
    const hum = ctx.createOscillator();
    hum.frequency.value = 47;
    rumble = ctx.createGain();
    rumble.gain.value = 0;
    hum.connect(rumble).connect(out);
    hum.start();
    // Side thrusters: a band of white noise, like gas through a small nozzle.
    const band = ctx.createBiquadFilter();
    band.type = "bandpass";
    band.frequency.value = 3400;
    band.Q.value = 0.8;
    hiss = ctx.createGain();
    hiss.gain.value = 0;
    loop(ctx, noise(ctx, false)).connect(band).connect(hiss).connect(out);
    return true;
  }

  /** A short decaying sound: a low thump, optionally with a burst of noise. */
  function hit({ from = 90, to = 32, length = 0.35, level = 0.8, crunch = 0 }) {
    if (!ctx || !on) return;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.frequency.setValueAtTime(from, t);
    osc.frequency.exponentialRampToValueAtTime(to, t + length);
    const g = ctx.createGain();
    g.gain.setValueAtTime(level, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + length);
    osc.connect(g).connect(out);
    osc.start(t);
    osc.stop(t + length + 0.05);
    if (crunch) {
      const src = ctx.createBufferSource();
      src.buffer = brown;
      const lp = ctx.createBiquadFilter();
      lp.type = "lowpass";
      lp.frequency.value = 900;
      const ng = ctx.createGain();
      ng.gain.setValueAtTime(crunch, t);
      ng.gain.exponentialRampToValueAtTime(0.001, t + length * 2.5);
      src.connect(lp).connect(ng).connect(out);
      src.start(t);
      src.stop(t + length * 2.5 + 0.05);
    }
  }

  return {
    get enabled() { return on; },
    /** Turn sound on or off; call from a click or key handler the first time. */
    enable(value = true) {
      on = value;
      try { localStorage.setItem(KEY, on ? "on" : "off"); } catch { /* fine without */ }
      if (on && !ctx && !build()) on = false;
      if (ctx) {
        if (on) ctx.resume();
        out.gain.setTargetAtTime(on ? 0.55 : 0, ctx.currentTime, 0.05);
      }
      return on;
    },
    /** Wake the audio after a user gesture, if sound is on. */
    wake() {
      if (on && !ctx) build();
      if (ctx && on && ctx.state === "suspended") ctx.resume();
    },
    /** Every frame: main 0..1, side -1..1, air (0 = vacuum: heard only through the structure). */
    update(main, side, air) {
      if (!ctx || !on) return;
      const t = ctx.currentTime;
      const level = main > 0 ? 0.35 + 0.65 * main : 0;
      const muffle = air < 0.005 ? 0.6 : 1;
      roar.gain.setTargetAtTime(level * 0.9 * muffle, t, 0.05);
      roarFilter.frequency.setTargetAtTime(main > 0 ? (air < 0.005 ? 160 : 260) + 520 * main : 160, t, 0.06);
      rumble.gain.setTargetAtTime(level * 0.28, t, 0.06);
      hiss.gain.setTargetAtTime(Math.abs(side) * 0.14, t, 0.02);
    },
    silence() {
      if (!ctx) return;
      const t = ctx.currentTime;
      for (const g of [roar, rumble, hiss]) g.gain.setTargetAtTime(0, t, 0.05);
    },
    touchdown(strength = 1) { hit({ level: 0.5 * strength, crunch: 0.25 * strength }); },
    crash() { hit({ from: 70, to: 24, length: 0.6, level: 0.9, crunch: 0.9 }); },
    /** A soft two-tone ping, like a cockpit's contact light. */
    ping() {
      if (!ctx || !on) return;
      const t = ctx.currentTime;
      [1318, 1760].forEach((f, i) => {
        const osc = ctx.createOscillator();
        osc.type = "sine";
        osc.frequency.value = f;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0, t + i * 0.09);
        g.gain.linearRampToValueAtTime(0.09, t + i * 0.09 + 0.01);
        g.gain.exponentialRampToValueAtTime(0.001, t + i * 0.09 + 0.25);
        osc.connect(g).connect(out);
        osc.start(t + i * 0.09);
        osc.stop(t + i * 0.09 + 0.3);
      });
    },
  };
}
