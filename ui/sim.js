// Lunar Lander, running in the browser. A line-by-line port of Gymnasium's LunarLander-v3
// (gymnasium/envs/box2d/lunar_lander.py) onto planck.js, a JavaScript rewrite of the same
// Box2D physics engine, plus PerfectLand's world wrappers from lander.py: gravity of any
// strength, air drag, engine power, sturdier legs on heavy worlds, the start altitude and
// the throttle's touchdown cutoff. It needs no server, so you can fly it yourself, and a
// trained network can fly it too, from weights exported to JSON.
//
// Frames come out in the same layout as the server's recorded flights (FRAME_FIELDS in
// lab.py), so the 3D stage and the neuron view play them without knowing the difference.

import { Box, Edge, Polygon, RevoluteJoint, Vec2, World } from "./vendor/planck.mjs"; // planck.js 1.5.0, MIT

export const FPS = 50;
const SCALE = 30;
const MAIN_ENGINE_POWER = 13;
const SIDE_ENGINE_POWER = 0.6;
const INITIAL_RANDOM = 1000;
const LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]];
const LEG_AWAY = 20, LEG_DOWN = 18, LEG_W = 2, LEG_H = 8, LEG_SPRING_TORQUE = 40;
const SIDE_ENGINE_HEIGHT = 14, SIDE_ENGINE_AWAY = 12, MAIN_ENGINE_Y_LOCATION = 4;
const W = 600 / SCALE, H = 400 / SCALE;
export const START_HEIGHT = 10;
const PAD_HALF_WIDTH = 0.2;
export const OUTCOMES = ["landed on pad", "landed off pad", "timed out", "crashed", "flew off screen"];

/** Frames before a flight times out: 20 s from the standard 10 m, growing with √height. */
export function timeLimit(startHeight) {
  return Math.round(1000 * Math.sqrt(Math.max(startHeight, START_HEIGHT) / START_HEIGHT));
}

/** A small seeded random generator (mulberry32), so a seed always gives the same flight. */
export function random(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    integer: (lo, hi) => lo + Math.floor(next() * (hi - lo)),
  };
}

const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const round = (v, digits = 4) => Math.round(v * 10 ** digits) / 10 ** digits;

export class Flight {
  /**
   * conditions: { gravity, enable_wind, wind_power, turbulence_power, atmosphere,
   * engine_power, start_height } as in lab.PLANETS. continuous: a throttle instead of
   * four buttons. The seed fixes the terrain, the starting push and the engines' jitter.
   */
  constructor(conditions, { seed = Math.floor(Math.random() * 2 ** 31), continuous = false } = {}) {
    this.conditions = { start_height: START_HEIGHT, ...conditions };
    this.seed = seed;
    this.continuous = continuous;
    this.rng = random(seed);
    this.limit = timeLimit(this.conditions.start_height);
    this.frames = [];
    this.score = 0;
    this.steps = 0;
    this.done = false;
    this.outcome = null;
    this.miss = null;
    this.touchdown = null;
    this._build();
  }

  _build() {
    const c = this.conditions, rng = this.rng;
    const world = new World({ gravity: Vec2(0, c.gravity) });
    this.world = world;
    this.gameOver = false;
    this.prevShaping = null;

    // Terrain: eleven chunks of random height, flattened around the pad, then smoothed.
    const CHUNKS = 11;
    const height = Array.from({ length: CHUNKS + 1 }, () => rng.uniform(0, H / 2));
    const chunkX = Array.from({ length: CHUNKS }, (_, i) => (W / (CHUNKS - 1)) * i);
    this.padX1 = chunkX[CHUNKS / 2 - 1 | 0];
    this.padX2 = chunkX[(CHUNKS / 2 | 0) + 1];
    this.padY = H / 4;
    for (let k = -2; k <= 2; k++) height[(CHUNKS / 2 | 0) + k] = this.padY;
    const smooth = chunkX.map((_, i) => 0.33 * (height[(i - 1 + height.length) % height.length] + height[i] + height[i + 1]));
    const moon = world.createBody();
    moon.createFixture(Edge(Vec2(0, 0), Vec2(W, 0)));
    this.ground = [];
    for (let i = 0; i < CHUNKS - 1; i++) {
      const p1 = Vec2(chunkX[i], smooth[i]), p2 = Vec2(chunkX[i + 1], smooth[i + 1]);
      moon.createFixture(Edge(p1, p2), { density: 0, friction: 0.1 });
      this.ground.push([p1.x, p1.y]);
    }
    this.ground.push([chunkX[CHUNKS - 1], smooth[CHUNKS - 1]]);

    // The rocket, with its random starting push.
    const x0 = W / 2, y0 = H;
    this.lander = world.createBody({ type: "dynamic", position: Vec2(x0, y0), angle: 0 });
    this.lander.createFixture(Polygon(LANDER_POLY.map(([x, y]) => Vec2(x / SCALE, y / SCALE))), {
      density: 5, friction: 0.1, restitution: 0, filterCategoryBits: 0x0010, filterMaskBits: 0x001,
    });
    this.lander.applyForceToCenter(Vec2(rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), true);
    if (c.enable_wind) {
      this.windIdx = rng.integer(-9999, 9999);
      this.torqueIdx = rng.integer(-9999, 9999);
    }

    // Two spring-loaded legs.
    this.legs = [-1, 1].map((i) => {
      const leg = world.createBody({ type: "dynamic", position: Vec2(x0 - (i * LEG_AWAY) / SCALE, y0), angle: i * 0.05 });
      leg.createFixture(Box(LEG_W / SCALE, LEG_H / SCALE), {
        density: 1, restitution: 0, filterCategoryBits: 0x0020, filterMaskBits: 0x001,
      });
      leg.groundContact = false;
      leg.joint = world.createJoint(new RevoluteJoint({
        localAnchorA: Vec2(0, 0), localAnchorB: Vec2((i * LEG_AWAY) / SCALE, LEG_DOWN / SCALE),
        referenceAngle: 0, // Box2D's default; planck would otherwise measure it from the bodies
        enableMotor: true, enableLimit: true, maxMotorTorque: LEG_SPRING_TORQUE, motorSpeed: 0.3 * i,
        lowerAngle: i === -1 ? 0.9 - 0.5 : -0.9, upperAngle: i === -1 ? 0.9 : -0.9 + 0.5,
      }, this.lander, leg));
      return leg;
    });

    world.on("begin-contact", (contact) => this._contact(contact, true));
    world.on("end-contact", (contact) => this._contact(contact, false));

    this.obs = this._physicsStep(this._noop()).obs;

    // PerfectLand's worlds (lander.py): air drag and landing gear rated for this gravity.
    for (const body of [this.lander, ...this.legs]) body.setLinearDamping(c.atmosphere || 0);
    for (const leg of this.legs) leg.joint.setMaxMotorTorque(LEG_SPRING_TORQUE * Math.max(1, Math.abs(c.gravity) / 10));
    // The start altitude: lift the rocket, keeping its push, and score progress from there.
    const lift = c.start_height - START_HEIGHT;
    if (lift) {
      for (const body of [this.lander, ...this.legs]) {
        const p = body.getPosition();
        body.setPosition(Vec2(p.x, p.y + lift));
      }
      this.prevShaping = null;
      this.obs = this._physicsStep(this._noop()).obs;
    }
    this.terrain = this.ground.map(([x, y]) => [round(x - 10, 3), round(y - this.padY, 3)]);
    this.pad = [round(this.padX1 - 10, 3), round(this.padX2 - 10, 3)];
    this.speed = 0;
    this.touched = null;
  }

  _noop() {
    return this.continuous ? [0, 0] : 0;
  }

  _contact(contact, begin) {
    const a = contact.getFixtureA().getBody(), b = contact.getFixtureB().getBody();
    if (begin && (a === this.lander || b === this.lander)) this.gameOver = true;
    for (const leg of this.legs) if (leg === a || leg === b) leg.groundContact = begin;
  }

  /** One frame of the simulator itself: wind, engines, physics, observation and score. */
  _physicsStep(action) {
    const c = this.conditions, lander = this.lander, rng = this.rng;
    if (c.enable_wind && !(this.legs[0].groundContact || this.legs[1].groundContact)) {
      const gust = (idx) => Math.tanh(Math.sin(0.02 * idx) + Math.sin(Math.PI * 0.01 * idx));
      lander.applyForceToCenter(Vec2(gust(this.windIdx) * c.wind_power, 0), true);
      this.windIdx += 1;
      lander.applyTorque(gust(this.torqueIdx) * c.turbulence_power, true);
      this.torqueIdx += 1;
    }
    if (this.continuous) action = [clip(action[0], -1, 1), clip(action[1], -1, 1)];

    const angle = lander.getAngle(), pos = lander.getPosition();
    const tip = [Math.sin(angle), Math.cos(angle)];
    const side = [-tip[1], tip[0]];
    const dispersion = [rng.uniform(-1, 1) / SCALE, rng.uniform(-1, 1) / SCALE];
    const mainPower = MAIN_ENGINE_POWER * (c.engine_power ?? 1);

    let mPower = 0;
    if ((this.continuous && action[0] > 0) || (!this.continuous && action === 2)) {
      mPower = this.continuous ? (clip(action[0], 0, 1) + 1) * 0.5 : 1;
      const ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1];
      const oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1];
      lander.applyLinearImpulse(Vec2(-ox * mainPower * mPower, -oy * mainPower * mPower), Vec2(pos.x + ox, pos.y + oy), true);
    }
    let sPower = 0, direction = 0;
    if ((this.continuous && Math.abs(action[1]) > 0.5) || (!this.continuous && (action === 1 || action === 3))) {
      direction = this.continuous ? Math.sign(action[1]) : action - 2;
      sPower = this.continuous ? clip(Math.abs(action[1]), 0.5, 1) : 1;
      const ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
      const oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
      const at = Vec2(pos.x + ox - (tip[0] * 17) / SCALE, pos.y + oy + (tip[1] * SIDE_ENGINE_HEIGHT) / SCALE);
      lander.applyLinearImpulse(Vec2(-ox * SIDE_ENGINE_POWER * sPower, -oy * SIDE_ENGINE_POWER * sPower), at, true);
    }

    this.world.step(1 / FPS, 6 * 30, 2 * 30);

    const p = lander.getPosition(), v = lander.getLinearVelocity();
    const obs = [
      (p.x - W / 2) / (W / 2),
      (p.y - (this.padY + LEG_DOWN / SCALE)) / (H / 2),
      (v.x * (W / 2)) / FPS,
      (v.y * (H / 2)) / FPS,
      lander.getAngle(),
      (20 * lander.getAngularVelocity()) / FPS,
      this.legs[0].groundContact ? 1 : 0,
      this.legs[1].groundContact ? 1 : 0,
    ];
    const shaping = -100 * Math.hypot(obs[0], obs[1]) - 100 * Math.hypot(obs[2], obs[3])
      - 100 * Math.abs(obs[4]) + 10 * obs[6] + 10 * obs[7];
    let reward = this.prevShaping === null ? 0 : shaping - this.prevShaping;
    this.prevShaping = shaping;
    reward -= mPower * 0.3;
    reward -= sPower * 0.03;
    let terminated = false;
    if (this.gameOver || Math.abs(obs[0]) >= 1) {
      terminated = true;
      reward = -100;
    }
    if (!lander.isAwake()) {
      terminated = true;
      reward = 100;
    }
    return { obs, reward, terminated, main: mPower, side: sPower * direction };
  }

  /** The rocket and its legs, in world units with the pad center at (0, 0). */
  pose() {
    const out = [];
    for (const body of [this.lander, ...this.legs]) {
      const p = body.getPosition();
      out.push(round(p.x - 10), round(p.y - this.padY), round(body.getAngle()));
    }
    return out;
  }

  /**
   * Fly one frame. action: 0 nothing, 1 left engine, 2 main engine, 3 right engine;
   * or [main, side] from -1 to 1 for a throttle. Returns the new observation.
   */
  step(action) {
    if (this.done) return this.obs;
    const seen = this.obs, pose = this.pose();
    if (this.continuous) {
      action = [Number(action[0]) || 0, Number(action[1]) || 0];
      // The throttle's touchdown cutoff (lander.py): engines stop once parked on both legs.
      if (seen[6] && seen[7] && Math.hypot(seen[2], seen[3]) < 0.05) action = [-1, 0];
    }
    const r = this._physicsStep(action);
    this.steps += 1;
    this.score += r.reward;
    this.frames.push([...pose, round(r.main, 3), round(r.side, 3), this.continuous ? -1 : action,
      ...seen.map((v) => round(v)), round(r.reward, 3)]);
    this.obs = r.obs;

    if (this.touched === null && (r.obs[6] || r.obs[7])) this.touched = this.speed;
    const v = this.lander.getLinearVelocity();
    this.speed = Math.hypot(v.x, v.y);
    const truncated = this.steps >= this.limit;
    if (r.terminated || truncated) {
      this.done = true;
      if (!this.lander.isAwake()) {
        this.outcome = Math.abs(r.obs[0]) <= PAD_HALF_WIDTH ? "landed on pad" : "landed off pad";
        this.miss = Math.abs(this.lander.getPosition().x - (this.padX1 + this.padX2) / 2);
        this.touchdown = this.touched;
      } else if (this.gameOver) this.outcome = "crashed";
      else if (Math.abs(r.obs[0]) >= 1) this.outcome = "flew off screen";
      else this.outcome = "timed out";
      this.frames.push([...this.pose(), 0, 0, -1, ...r.obs.map((v) => round(v)), 0]);
    }
    return this.obs;
  }

  /** The flight so far, shaped like the server's recorded flights. */
  record() {
    return {
      seed: this.seed,
      conditions: this.conditions,
      control: this.continuous ? "throttle" : "buttons",
      outcome: this.outcome,
      score: round(this.score, 1),
      steps: this.steps,
      terrain: this.terrain,
      pad: this.pad,
      miss: this.miss === null ? null : round(this.miss, 2),
      touchdown: this.touchdown === null ? null : round(this.touchdown, 2),
      frames: this.frames,
    };
  }
}
