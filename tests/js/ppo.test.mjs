// Browser training (ui/ppo.js): saved pilots load back exactly and fly like the game reads them.
import assert from "node:assert/strict";
import { test } from "node:test";

import { forward, prepare } from "../../ui/nn.js";
import { act, createTrainer, exportWeights, importWeights, newAgent } from "../../ui/ppo.js";

const classic = { gravity: -10, enable_wind: false, wind_power: 15, turbulence_power: 1.5, atmosphere: 0, engine_power: 1 };

test("exported weights load back and pick the same buttons as the game", () => {
  const agent = newAgent(3);
  const again = importWeights(JSON.parse(JSON.stringify(exportWeights(agent, 12))));
  const game = prepare(exportWeights(agent));
  for (let k = 0; k < 50; k++) {
    const obs = Array.from({ length: 8 }, (_, i) => (i < 6 ? Math.sin(k * 7 + i) * 0.8 : k % 2));
    assert.equal(act(again, obs), act(agent, obs));
    assert.equal(forward(game, obs).action, act(agent, obs));
  }
});

test("one batch of training runs and changes the policy", () => {
  const trainer = createTrainer({ conditions: classic, seed: 2, envs: 4, steps: 64 });
  const before = JSON.stringify(exportWeights(trainer.agent));
  const r = trainer.step();
  assert.equal(r.frames, 256);
  assert.notEqual(JSON.stringify(exportWeights(trainer.agent)), before);
});
