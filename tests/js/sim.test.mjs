// The browser's copy of the simulator (ui/sim.js) must fly like the real one, or the game's
// AI and your scores mean nothing. Run with: npm install --no-save planck@1.5.0 && node --test tests/js/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { forward, prepare } from "../../ui/nn.js";
import { Flight, OUTCOMES, timeLimit } from "../../ui/sim.js";

const data = (path) => JSON.parse(readFileSync(new URL(`../../ui/data/${path}`, import.meta.url)));
const worlds = data("worlds.json");
const pilot = prepare(data("pilots/universal.json"));
const PHYSICS = ["gravity", "enable_wind", "wind_power", "turbulence_power", "atmosphere", "engine_power"];
const conditions = (id) => Object.fromEntries(PHYSICS.map((k) => [k, worlds[id][k]]));

function fly(id, seed) {
  const flight = new Flight(conditions(id), { seed });
  while (!flight.done) flight.step(forward(pilot, flight.obs).action);
  return flight;
}

test("a seed always gives the same flight", () => {
  const a = fly("earth", 42), b = fly("earth", 42);
  assert.equal(a.steps, b.steps);
  assert.deepEqual(a.frames.at(-1), b.frames.at(-1));
  assert.deepEqual(a.terrain, b.terrain);
});

test("frames have the server's layout", () => {
  const flight = fly("moon", 1);
  assert.equal(flight.frames.length, flight.steps + 1);
  for (const frame of flight.frames) assert.equal(frame.length, 21); // lab.FRAME_FIELDS
  assert.ok(OUTCOMES.includes(flight.outcome));
});

test("the clock grows with the start altitude", () => {
  assert.equal(timeLimit(10), 1000);
  assert.equal(timeLimit(40), 2000);
  const standard = new Flight(conditions("classic"), { seed: 3 });
  const lifted = new Flight({ ...conditions("classic"), start_height: 40 }, { seed: 3 });
  assert.ok(Math.abs((lifted.obs[1] - standard.obs[1]) * 6.667 - 30) < 0.3);
});

test("the trained Universal pilot lands on every world, as it does in Python", () => {
  let landed = 0, flights = 0;
  for (const id of Object.keys(worlds)) {
    for (let seed = 0; seed < 10; seed++) {
      landed += fly(id, 1000 + seed).outcome === "landed on pad" ? 1 : 0;
      flights += 1;
    }
  }
  // Python (Box2D) measured 227/260 = 87% for this network; the port should be close.
  assert.ok(landed / flights > 0.8, `${landed}/${flights} landed`);
});

test("random buttons rarely land", () => {
  let landed = 0;
  for (let seed = 0; seed < 20; seed++) {
    const flight = new Flight(conditions("classic"), { seed });
    let n = seed;
    while (!flight.done) flight.step((n = (n * 1103515245 + 12345) % 2 ** 31) % 4);
    landed += flight.outcome === "landed on pad" ? 1 : 0;
  }
  assert.ok(landed <= 2);
});
