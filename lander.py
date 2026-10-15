"""The simulator, and the plumbing that plugs our reward function into it."""

import math
import warnings
from pathlib import Path

import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward import compute_reward

ENV_ID = "LunarLander-v3"
MAIN_ENGINE_POWER = lunar_lander.MAIN_ENGINE_POWER  # the simulator's standard thrust
FPS = 50  # simulator frames per second
PAD_HALF_WIDTH = 0.2  # the pad (between the flags) spans x = -0.2 .. 0.2
START_HEIGHT = 10.0  # the simulator drops the rocket 10 m above the pad
TIME_LIMIT = 1000  # frames (20 s) to land from the standard height
OUTCOMES = ("landed on pad", "landed off pad", "timed out", "crashed", "flew off screen")
RUNS = Path("runs")

# Lunar Lander declares rocket heights up to about 17 m above the pad. Higher starts go past
# that, which the network takes in its stride, so skip gymnasium's warning about it.
warnings.filterwarnings("ignore", message=".*is not within the observation space")


class LandingOutcome(gym.Wrapper):
    """Adds info["outcome"] on the last frame of every episode: how did it end?

    Landings also say how precise they were: self.miss is the distance from the pad
    center in meters, and self.touchdown_speed the speed (m/s) just before the first
    leg touched the ground. Both are None until the rocket has landed.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.speed, self.touched, self.miss, self.touchdown_speed = 0.0, None, None, None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sim = self.env.unwrapped
        if self.touched is None and (obs[6] or obs[7]):
            self.touched = self.speed  # the speed on the frame before the legs touched
        self.speed = float(sim.lander.linearVelocity.length)
        if terminated or truncated:
            if not sim.lander.awake:  # came to rest: the simulator's definition of landed
                on_pad = abs(obs[0]) <= PAD_HALF_WIDTH
                info["outcome"] = "landed on pad" if on_pad else "landed off pad"
                self.miss = abs(sim.lander.position.x - (sim.helipad_x1 + sim.helipad_x2) / 2)
                self.touchdown_speed = self.touched
            elif sim.game_over:  # the hull touched the ground
                info["outcome"] = "crashed"
            elif abs(obs[0]) >= 1.0:
                info["outcome"] = "flew off screen"
            else:  # the time limit passed without coming to rest
                info["outcome"] = "timed out"
        return obs, reward, terminated, truncated, info


class ShapedReward(gym.Wrapper):
    """Replaces the simulator's built-in reward with compute_reward() from reward.py."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        obs, _official_reward, terminated, truncated, info = self.env.step(action)
        applied = info.get("applied_action", action)  # after any touchdown cutoff
        reward = compute_reward(self.prev_obs, applied, obs, info.get("outcome"))
        self.prev_obs = obs
        return obs, reward, terminated, truncated, info


def set_engine_power(multiplier):
    """Scale the main engine's thrust for every simulator in this Python process.

    The simulator reads its engine strength from a module constant, so this is
    process-wide. Each training run gets its own process.
    """
    lunar_lander.MAIN_ENGINE_POWER = MAIN_ENGINE_POWER * multiplier


class WorldPhysics(gym.Wrapper):
    """Gravity of any strength (the simulator alone allows up to 12 m/s²) and air drag."""

    def __init__(self, env, gravity, atmosphere):
        super().__init__(env)
        env.unwrapped.gravity = gravity  # the physics world is rebuilt with it on every reset
        self.gravity, self.atmosphere = gravity, atmosphere

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sim = self.env.unwrapped
        for body in (sim.lander, *sim.legs):
            body.linearDamping = self.atmosphere  # thicker air slows every movement
        # The legs' springs are sized for 10 m/s²; above that they would fold under the
        # rocket's own weight. Give heavier worlds landing gear rated for their gravity.
        for leg in sim.legs:
            leg.joint.maxMotorTorque = lunar_lander.LEG_SPRING_TORQUE * max(1.0, abs(self.gravity) / 10)
        return obs, info


def time_limit(start_height):
    """Frames before a flight times out. Falling 4 times as far takes twice as long, so
    the clock grows with the square root of the height: 20 s from 10 m, 40 s from 40 m."""
    return round(TIME_LIMIT * math.sqrt(max(start_height, START_HEIGHT) / START_HEIGHT))


class StartHeight(gym.Wrapper):
    """Starts every flight at start_height meters above the pad (the simulator's is 10 m).

    The simulator builds each flight at its standard height and gives the rocket a random
    push. This lifts the rocket (body and legs together, so the push is kept) and runs one
    frame without engines, so the observation and the simulator's score start from there.
    """

    def __init__(self, env, start_height):
        super().__init__(env)
        self.lift = start_height - START_HEIGHT

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.lift:
            sim = self.env.unwrapped
            for body in (sim.lander, *sim.legs):
                body.position = (body.position.x, body.position.y + self.lift)
            sim.prev_shaping = None  # the simulator scores progress from the new height
            obs = sim.step(np.array([0.0, 0.0]) if sim.continuous else 0)[0]
        return obs, info


class TouchdownCutoff(gym.Wrapper):
    """Throttle pilots: cut the engines once the rocket is parked on both legs.

    A throttle pilot explores by adding random noise to every engine setting, and that
    noise doesn't go away on the ground, so it would keep nudging the engines and never
    come to rest. Like Apollo's lunar module ("contact light, engine stop"), the engines
    stop once both legs are down and the rocket has almost stopped. The action actually
    applied is reported in info["applied_action"].
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = obs
        return obs, info

    def step(self, action):
        o = self.obs
        if o[6] and o[7] and np.hypot(o[2], o[3]) < 0.05:
            action = np.array([-1.0, 0.0], dtype=np.float32)
        obs, reward_, terminated, truncated, info = self.env.step(action)
        info["applied_action"] = action
        self.obs = obs
        return obs, reward_, terminated, truncated, info


def make_flight_env(render_mode=None, gravity=-10.0, enable_wind=False, wind_power=15.0,
                    turbulence_power=1.5, atmosphere=0.0, engine_power=1.0, continuous=False,
                    start_height=START_HEIGHT, planet=None):
    """The simulator as-is. Its rewards add up to the official Lunar Lander score.

    gravity is in m/s² (negative = down). atmosphere is air drag (0 = vacuum).
    engine_power scales the main engine's thrust. continuous=True gives the pilot
    a throttle instead of on/off buttons. start_height is how far above the pad each
    flight starts, in meters; higher starts get more time. planet is only a label.
    """
    set_engine_power(engine_power)
    env = gym.make(ENV_ID, render_mode=render_mode, continuous=continuous, enable_wind=enable_wind,
                   wind_power=wind_power, turbulence_power=turbulence_power,
                   max_episode_steps=time_limit(start_height))
    env = StartHeight(WorldPhysics(env, gravity, atmosphere), start_height)
    if continuous:
        env = TouchdownCutoff(env)
    return LandingOutcome(env)


def make_training_env(log_file=None, **conditions):
    """The simulator as the agent experiences it: scored by our reward function.

    Monitor sits *inside* ShapedReward, so it logs the official score instead of
    ours. That keeps a fixed yardstick while you experiment with the reward.
    """
    env = make_flight_env(**conditions)
    env = Monitor(env, log_file, info_keywords=("outcome",))
    return ShapedReward(env)


def load_pilot(run):
    """The trained network saved in runs/<run>/, preferring the best-scoring one."""
    for name in ("best_model.zip", "final_model.zip"):
        path = RUNS / run / name
        if path.exists():
            print(f"Pilot: {path}")
            return PPO.load(path, device="cpu")
    raise SystemExit(f"No trained pilot in {RUNS / run}/. Run `python train.py` first.")


def fly(env, pilot, seed=None, frames=None):
    """Fly one episode and return (outcome, official score, frames flown).

    pilot=None presses random buttons. Pass a list as frames to collect images
    (the env must use render_mode="rgb_array").
    """
    obs, _ = env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)  # makes the random pilot repeatable too
    score, steps = 0.0, 0
    while True:
        if frames is not None:
            frames.append(env.render())
        if pilot is None:
            action = env.action_space.sample()
        else:
            action, _ = pilot.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        steps += 1
        if terminated or truncated:
            if frames is not None:
                frames.append(env.render())
            return info["outcome"], score, steps
