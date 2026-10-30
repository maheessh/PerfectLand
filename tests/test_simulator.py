"""The simulator wrappers: worlds, start altitude, outcomes, and training plumbing."""

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import lab
from lander import OUTCOMES, START_HEIGHT, make_flight_env, make_training_env, time_limit


@pytest.mark.parametrize("world", sorted(lab.PLANETS))
def test_every_world_flies_to_an_outcome(world):
    env = make_flight_env(**lab.world_conditions(world))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (8,)
    env.action_space.seed(1)
    for _ in range(time_limit(START_HEIGHT) + 1):
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert info["outcome"] in OUTCOMES
    env.close()


def test_gravity_past_the_simulators_limit():
    """Gymnasium refuses gravity beyond 12 m/s²; Jupiter's 23.1 goes through the wrapper."""
    env = make_flight_env(**lab.world_conditions("jupiter"))
    env.reset(seed=0)
    assert env.unwrapped.world.gravity[1] == pytest.approx(-23.1)
    env.close()


def test_start_altitude_lifts_the_rocket_and_extends_the_clock():
    low = make_flight_env(start_height=10.0)
    high = make_flight_env(start_height=40.0)
    y_low = low.reset(seed=3)[0][1]
    y_high = high.reset(seed=3)[0][1]
    assert y_high - y_low == pytest.approx(30.0 / 6.667, abs=0.05)  # observation y is in units of 6.67 m
    assert time_limit(40.0) == 2 * time_limit(10.0)
    low.close()
    high.close()


def test_a_landing_reports_its_precision():
    """Fly the shipped pre-trained pilot once and check the landing measurements."""
    model = lab.load_model(lab.STARTER)
    env = make_flight_env()
    obs, _ = env.reset(seed=10_000)
    while True:
        obs, _, terminated, truncated, info = env.step(int(model.predict(obs, deterministic=True)[0]))
        if terminated or truncated:
            break
    assert info["outcome"] == "landed on pad"
    assert 0 <= env.miss < 0.5
    assert 0 < env.touchdown_speed < 3
    env.close()


def test_ppo_trains_on_the_shaped_reward():
    """A few hundred frames of PPO on two simulators: the training pipeline runs end to end."""
    envs = DummyVecEnv([lambda: make_training_env(**lab.world_conditions("mars"))] * 2)
    model = PPO("MlpPolicy", envs, n_steps=64, batch_size=64, n_epochs=1, seed=0, device="cpu")
    model.learn(256)
    assert model.num_timesteps >= 256
    action, _ = model.predict(np.zeros(8, dtype=np.float32), deterministic=True)
    assert int(action) in range(4)
