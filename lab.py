"""The training lab behind the web UI (app.py): pilots, generations and flights.

A pilot is one neural network that keeps learning across generations. Generation 1
starts from scratch; every retrain continues from the latest generation's network.
Everything lives in runs/<pilot>/:

    lab.json       every generation: its settings, frames and results
    episodes.csv   every practice attempt
    tests.csv      every 20-flight test
    trails.json    flight paths of the most recent practice attempts
    gen-N/         final_model.zip and best_model.zip of generation N

Deleted generations and pilots move to runs/.trash/, in case you change your mind.
"""

import ast
import csv
import json
import math
import random
import shutil
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import reward
from lander import RUNS, START_HEIGHT, fly, make_flight_env, make_training_env

OUTCOMES = ["landed on pad", "landed off pad", "timed out", "crashed", "flew off screen"]
CODE = {name: i for i, name in enumerate(OUTCOMES)}
REWARD_KEYS = [
    "DISTANCE_WEIGHT", "SPEED_WEIGHT", "TILT_WEIGHT", "LEG_CONTACT_BONUS",
    "MAIN_ENGINE_COST", "SIDE_ENGINE_COST", "TIME_COST", "GROUND_ENGINE_COST",
    "LANDING_BONUS", "CRASH_PENALTY",
]
# Worlds to land on. Gravity is real, in m/s² (NASA Planetary Fact Sheet; Titan and
# Europa from their mass and radius). Wind, turbulence and air drag are simplified to
# fit the simulator. Gas giants have no surface, so the pad floats in the cloud tops.
# Each world's engine is sized like a real lander's: about 1.8 times local gravity,
# and at least 6 m/s² so it can cancel the random push every flight starts with.
PLANETS = {
    "classic": {"name": "Classic", "gravity": -10.0, "enable_wind": False, "wind_power": 15.0, "turbulence_power": 1.5,
                "atmosphere": 0.0, "engine_power": 1.0,
                "fact": "The original Lunar Lander test site: 10 m/s², no air, no wind."},
    "mercury": {"name": "Mercury", "gravity": -3.7, "enable_wind": False, "wind_power": 0.0, "turbulence_power": 0.0,
                "atmosphere": 0.0, "engine_power": 0.37,
                "fact": "No air at all. One day lasts 176 Earth days; 167 °C on average."},
    "venus": {"name": "Venus", "gravity": -8.9, "enable_wind": True, "wind_power": 3.0, "turbulence_power": 0.5,
              "atmosphere": 1.2, "engine_power": 1.0,
              "fact": "Air 92 times thicker than Earth's slows every move; 464 °C."},
    "earth": {"name": "Earth", "gravity": -9.8, "enable_wind": True, "wind_power": 5.0, "turbulence_power": 0.5,
              "atmosphere": 0.1, "engine_power": 1.0,
              "fact": "One bar of air, a breeze, 15 °C on average."},
    "moon": {"name": "Moon", "gravity": -1.6, "enable_wind": False, "wind_power": 0.0, "turbulence_power": 0.0,
             "atmosphere": 0.0, "engine_power": 0.34,
             "fact": "No air, no wind, and a sixth of Earth's gravity."},
    "mars": {"name": "Mars", "gravity": -3.7, "enable_wind": True, "wind_power": 2.0, "turbulence_power": 0.3,
             "atmosphere": 0.02, "engine_power": 0.37,
             "fact": "Thin CO₂ air, 1% of Earth's, with dusty gusts; −65 °C."},
    "jupiter": {"name": "Jupiter", "gravity": -23.1, "enable_wind": True, "wind_power": 12.0, "turbulence_power": 1.2,
                "atmosphere": 0.3, "engine_power": 2.3,
                "fact": "No surface: a pad floats in the cloud tops. Needs a 2.3× engine and sturdy legs."},
    "europa": {"name": "Europa", "gravity": -1.31, "enable_wind": False, "wind_power": 0.0, "turbulence_power": 0.0,
               "atmosphere": 0.0, "engine_power": 0.34,
               "fact": "Jupiter's icy moon. No air, about −160 °C, Jupiter fills the sky."},
    "saturn": {"name": "Saturn", "gravity": -9.0, "enable_wind": True, "wind_power": 12.0, "turbulence_power": 1.2,
               "atmosphere": 0.25, "engine_power": 1.0,
               "fact": "No surface: a pad floats above fast, freezing clouds; −140 °C."},
    "titan": {"name": "Titan", "gravity": -1.35, "enable_wind": True, "wind_power": 2.0, "turbulence_power": 0.3,
              "atmosphere": 0.6, "engine_power": 0.34,
              "fact": "Saturn's moon: air 1.5 times thicker than Earth's, gravity weaker than the Moon's."},
    "uranus": {"name": "Uranus", "gravity": -8.7, "enable_wind": True, "wind_power": 10.0, "turbulence_power": 1.0,
               "atmosphere": 0.25, "engine_power": 1.0,
               "fact": "An ice giant tipped on its side by 98°; −195 °C."},
    "neptune": {"name": "Neptune", "gravity": -11.0, "enable_wind": True, "wind_power": 16.0, "turbulence_power": 1.6,
                "atmosphere": 0.25, "engine_power": 1.1,
                "fact": "The fastest winds in the solar system; −200 °C."},
    "pluto": {"name": "Pluto", "gravity": -0.7, "enable_wind": False, "wind_power": 0.0, "turbulence_power": 0.0,
              "atmosphere": 0.0, "engine_power": 0.34,
              "fact": "A whisper of air, 1/100,000 of Earth's; −225 °C."},
}
PHYSICS_KEYS = ["gravity", "enable_wind", "wind_power", "turbulence_power", "atmosphere", "engine_power"]
# How far above the pad flights start, in meters. Not part of a world: Mars from 30 m is
# still Mars, so its generations stay in Mars's line.
START_HEIGHTS = (5.0, 40.0)
DEFAULT_CONDITIONS = {"planet": "classic", **{k: PLANETS["classic"][k] for k in PHYSICS_KEYS},
                      "start_height": START_HEIGHT}
DEFAULT_PPO = {
    "learning_rate": 3e-4, "ent_coef": 0.01, "gamma": 0.999, "control": "buttons",
    "net_arch": [64, 64], "activation": "tanh", "n_envs": 16, "seed": 0,
}
TEST_FLIGHTS = 20
TEST_SEED = 10_000  # the same 20 terrains for every test, so scores are comparable
TEST_EVERY = 100_000  # frames between tests
TRAIL_EVERY = 3  # keep every 3rd frame of practice flight paths
TRAILS_KEPT = 400

# One recorded frame, as sent to the browser.
FRAME_FIELDS = [
    "x", "y", "angle",  # lander body, in world units with the pad center at (0, 0)
    "leg1_x", "leg1_y", "leg1_angle", "leg2_x", "leg2_y", "leg2_angle",
    "main",  # main engine power on this frame, 0 to 1
    "side",  # side engine power, -1 to 1: negative = left engine, positive = right engine
    "action",  # the button pressed (-1 for throttle pilots, and on the final frame)
    "o_x", "o_y", "o_vx", "o_vy", "o_angle", "o_spin", "o_left_leg", "o_right_leg",  # what the pilot saw
    "reward",  # official reward received for this frame
]
FLIGHTS = threading.Lock()  # engine power is process-wide, so flights here take turns


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_reward():
    return {key: float(getattr(reward, key)) for key in REWARD_KEYS}


def clean_conditions(raw):
    raw = dict(raw or {})
    planet = raw.get("planet")
    if planet not in PLANETS and planet != "custom":  # older settings: find the matching world
        planet = next((name for name, p in PLANETS.items()
                       if all(p[k] == raw[k] for k in PHYSICS_KEYS if k in raw)), "custom")
    base = PLANETS.get(planet, PLANETS["classic"])
    c = {**{k: base[k] for k in PHYSICS_KEYS}, **raw}
    return {
        "planet": planet,
        "gravity": min(max(float(c["gravity"]), -30.0), -0.1),
        "enable_wind": bool(c["enable_wind"]),
        "wind_power": min(max(float(c["wind_power"]), 0.0), 20.0),
        "turbulence_power": min(max(float(c["turbulence_power"]), 0.0), 2.0),
        "atmosphere": min(max(float(c["atmosphere"]), 0.0), 3.0),
        "engine_power": min(max(float(c["engine_power"]), 0.3), 4.0),
        "start_height": min(max(float(c.get("start_height", START_HEIGHT)), START_HEIGHTS[0]), START_HEIGHTS[1]),
    }


def clean_reward(raw):
    weights = {**default_reward(), **(raw or {})}
    return {key: float(weights[key]) for key in REWARD_KEYS}


def clean_ppo(raw):
    p = {**DEFAULT_PPO, **(raw or {})}
    arch = [min(max(int(n), 2), 256) for n in p["net_arch"]][:3] or [64, 64]
    return {
        "learning_rate": min(max(float(p["learning_rate"]), 1e-5), 1e-2),
        "ent_coef": min(max(float(p["ent_coef"]), 0.0), 0.2),
        "gamma": min(max(float(p["gamma"]), 0.9), 0.9999),
        "net_arch": arch,
        "activation": p["activation"] if p["activation"] in ("tanh", "relu") else "tanh",
        "control": p["control"] if p["control"] in ("buttons", "throttle") else "buttons",
        "n_envs": min(max(int(p["n_envs"]), 1), 64),
        "seed": int(p["seed"]),
    }


# Flights ---------------------------------------------------------------------

def _pose(sim):
    """Lander body and leg poses, in world units with the pad center at (0, 0)."""
    cx, ground = 10.0, sim.helipad_y  # the pad sits at the middle of a 20-unit-wide world
    values = []
    for body in (sim.lander, *sim.legs):
        values += [body.position.x - cx, body.position.y - ground, body.angle]
    return [round(float(v), 4) for v in values]


def _terrain(sim):
    points = [poly[0] for poly in sim.sky_polys] + [sim.sky_polys[-1][1]]
    return [[round(x - 10.0, 3), round(y - sim.helipad_y, 3)] for x, y in points]


def is_throttle(model):
    return isinstance(model.action_space, gym.spaces.Box)


def _engines(action, continuous):
    """Main engine power (0..1) and side engine power (-1..1, negative = left engine)."""
    main, side = reward.throttle(action)
    if side:
        side *= float(np.sign(np.clip(action, -1, 1)[1])) if continuous else action - 2
    return [round(main, 3), round(side, 3)]


def record_flight(model, seed=None, conditions=None, control="buttons"):
    """Fly one episode and record every frame, for the 3D replay and the neuron view.

    model=None picks random actions, with buttons or a throttle as control says.
    """
    conditions = clean_conditions(conditions)
    continuous = is_throttle(model) if model is not None else control == "throttle"
    seed = random.randrange(2**31) if seed is None else int(seed)
    with FLIGHTS:
        env = make_flight_env(**conditions, continuous=continuous)
        try:
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            sim = env.unwrapped
            frames, score = [], 0.0
            while True:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action = model.predict(obs, deterministic=True)[0]
                if not continuous:
                    action = int(action)
                pose, seen = _pose(sim), [round(float(v), 4) for v in obs]
                obs, r, terminated, truncated, info = env.step(action)
                applied = info.get("applied_action", action)  # after any touchdown cutoff
                score += r
                frames.append(pose + _engines(applied, continuous) + [-1 if continuous else action]
                              + seen + [round(float(r), 3)])
                if terminated or truncated:
                    break
            frames.append(_pose(sim) + [0, 0, -1] + [round(float(v), 4) for v in obs] + [0.0])
            return {
                "seed": seed,
                "conditions": conditions,
                "control": "throttle" if continuous else "buttons",
                "outcome": info["outcome"],
                "score": round(score, 1),
                "steps": len(frames) - 1,
                "terrain": _terrain(sim),
                "pad": [round(sim.helipad_x1 - 10.0, 3), round(sim.helipad_x2 - 10.0, 3)],
                "miss": _rounded(env.miss), "touchdown": _rounded(env.touchdown_speed),
                "frames": frames,
            }
        finally:
            env.close()


def _rounded(value, digits=2):
    return None if value is None else round(float(value), digits)


def network_weights(model):
    """The actor (picks actions) and critic (predicts points) networks, layer by layer.

    A button pilot's actor ends in 4 scores, one per button. A throttle pilot's actor
    ends in 2 numbers: main engine and side engine settings, each -1 to 1.
    """
    policy = model.policy

    def linear_layers(modules):
        return [
            {
                "w": np.round(m.weight.detach().cpu().numpy(), 4).tolist(),
                "b": np.round(m.bias.detach().cpu().numpy(), 4).tolist(),
            }
            for m in modules if isinstance(m, torch.nn.Linear)
        ]

    relu = any(isinstance(m, torch.nn.ReLU) for m in policy.mlp_extractor.policy_net)
    return {
        "activation": "relu" if relu else "tanh",
        "control": "throttle" if is_throttle(model) else "buttons",
        "actor": linear_layers(policy.mlp_extractor.policy_net) + linear_layers([policy.action_net]),
        "critic": linear_layers(policy.mlp_extractor.value_net) + linear_layers([policy.value_net]),
    }


_models, _models_lock = {}, threading.Lock()


def load_model(path):
    """PPO.load with a small cache keyed by file and modification time."""
    key = (str(path), path.stat().st_mtime)
    with _models_lock:
        if key not in _models:
            if len(_models) > 8:
                _models.clear()
            _models[key] = PPO.load(path, device="cpu")
        return _models[key]


def test_flights(model, conditions=None, flights=TEST_FLIGHTS, seed=TEST_SEED, control="buttons"):
    """Fly a batch of flights with no randomness and sum them up."""
    continuous = is_throttle(model) if model is not None else control == "throttle"
    with FLIGHTS:
        env = make_flight_env(**clean_conditions(conditions), continuous=continuous)
        results, landings = [], []
        for i in range(flights):
            results.append(fly(env, model, seed=seed + i))
            if env.miss is not None:  # it landed: how close to the center, and how softly
                landings.append((env.miss, env.touchdown_speed))
        env.close()
    scores = [score for _, score, _ in results]
    counts = {name: 0 for name in OUTCOMES}
    for outcome, _, _ in results:
        counts[outcome] += 1
    return {
        "flights": flights,
        "score": round(float(np.mean(scores)), 1),
        "worst": round(float(np.min(scores)), 1),
        "best": round(float(np.max(scores)), 1),
        "landed": counts["landed on pad"],
        "outcomes": counts,
        "seconds": round(float(np.mean([steps for _, _, steps in results])) / 50, 1),
        # Average over the landings: meters from the pad center, and m/s as the legs touched.
        "miss": _rounded(np.mean([m for m, _ in landings])) if landings else None,
        "touchdown": _rounded(np.mean([s for _, s in landings])) if landings else None,
    }


# Storage ---------------------------------------------------------------------

EPISODE_COLUMNS = ["n", "gen", "score", "outcome", "frames", "end_x", "impact"]
TEST_COLUMNS = ["frames", "gen", "score", "landed", "flights"]
TRASH = RUNS / ".trash"  # deleted generations and pilots, kept in case you change your mind


def _write_csv(path, columns, rows):
    """Replace a CSV file in one step, so a crash can't leave it half written."""
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    tmp.replace(path)


def _trash_slot(*parts):
    """A fresh folder under runs/.trash/, named with the time so nothing is overwritten."""
    base = TRASH.joinpath(*parts[:-1], f"{datetime.now():%Y-%m-%d-%H%M%S}-{parts[-1]}")
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def delete_pilot(name):
    """Move a whole pilot, every generation included, to runs/.trash/."""
    trash = _trash_slot(name)
    shutil.move(str(RUNS / name), str(trash))
    return trash


def _number(text):
    if text in ("", None):
        return None
    value = float(text)
    return int(value) if value.is_integer() and "." not in text else value


class Pilot:
    """One lineage of generations, stored in runs/<name>/."""

    def __init__(self, name):
        self.name = name
        self.dir = RUNS / name
        self.created = now_iso()
        self.generations = []
        self.episodes = []  # [n, gen, score, outcome code, frames, end x, impact speed]
        self.tests = []  # [frames trained, gen, avg score, landed, flights]
        self.trails = deque(maxlen=TRAILS_KEPT)  # [n, outcome code, [[x, y], ...]]
        if (self.dir / "lab.json").exists():
            self._load()

    def _load(self):
        lab = json.loads((self.dir / "lab.json").read_text())
        self.created = lab.get("created", self.created)
        self.generations = lab["generations"]
        self.episodes = self._read_csv("episodes.csv")
        self.tests = self._read_csv("tests.csv")
        if (self.dir / "trails.json").exists():
            self.trails.extend(json.loads((self.dir / "trails.json").read_text()))
        for g in self.generations:
            if g["status"] == "training":  # a training run that never finished (server stopped)
                g["status"] = "stopped"
            g["config"]["conditions"] = clean_conditions(g["config"].get("conditions"))
            g["config"]["ppo"] = clean_ppo(g["config"].get("ppo"))
            g["world"] = g["config"]["conditions"]["planet"]  # each world has its own line of generations

    def _read_csv(self, name):
        path = self.dir / name
        if not path.exists():
            return []
        with open(path) as f:
            return [[_number(v) for v in row] for row in list(csv.reader(f))[1:]]

    def _append_csv(self, name, columns, rows):
        path = self.dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(columns)
            writer.writerows(rows)

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        lab = {"pilot": self.name, "created": self.created, "generations": self.generations}
        tmp = self.dir / "lab.json.tmp"
        tmp.write_text(json.dumps(lab, indent=2))
        tmp.replace(self.dir / "lab.json")
        (self.dir / "trails.json").write_text(json.dumps(list(self.trails)))

    def add_episodes(self, rows, trails):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episodes.extend(rows)
        self.trails.extend(trails)
        self._append_csv("episodes.csv", EPISODE_COLUMNS, rows)

    def add_test(self, row):
        self.tests.append(row)
        self._append_csv("tests.csv", TEST_COLUMNS, [row])

    def generation(self, gen):
        return next(g for g in self.generations if g["gen"] == gen)

    def delete_generations(self, gens):
        """Remove generations with their networks, practice attempts, tests and trails.

        Everything removed is moved to runs/.trash/<pilot>/. Later generations keep their
        own networks, so they still fly; their "started from" just names a deleted one.
        """
        gens = {int(g) for g in gens}
        doomed = [g for g in self.generations if g["gen"] in gens]
        if not doomed or len(doomed) != len(gens):
            raise ValueError("No such generation.")
        if any(g["status"] == "training" for g in doomed):
            raise RuntimeError("That generation is still training. Stop it first.")
        names = [str(g["gen"]) for g in doomed]
        trash = _trash_slot(self.name, "gen-" + "-".join(names) if len(names) <= 4 else f"{len(names)}-generations")
        trash.mkdir()
        gone = [e for e in self.episodes if e[1] in gens]
        attempts_gone = {e[0] for e in gone}
        (trash / "generations.json").write_text(json.dumps(doomed, indent=2))
        _write_csv(trash / "episodes.csv", EPISODE_COLUMNS, gone)
        _write_csv(trash / "tests.csv", TEST_COLUMNS, [t for t in self.tests if t[1] in gens])
        (trash / "trails.json").write_text(json.dumps([t for t in self.trails if t[0] in attempts_gone]))
        for g in doomed:
            folder = self.dir / f"gen-{g['gen']}"
            if folder.exists():
                shutil.move(str(folder), str(trash / folder.name))
            for model in (g.get("models") or {}).values():  # a train.py import keeps its networks at the top level
                path = self.dir / model
                if path.exists():
                    shutil.move(str(path), str(trash / path.name))
        self.generations = [g for g in self.generations if g["gen"] not in gens]
        self.episodes = [e for e in self.episodes if e[1] not in gens]
        self.tests = [t for t in self.tests if t[1] not in gens]
        self.trails = deque((t for t in self.trails if t[0] not in attempts_gone), maxlen=TRAILS_KEPT)
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_csv(self.dir / "episodes.csv", EPISODE_COLUMNS, self.episodes)
        _write_csv(self.dir / "tests.csv", TEST_COLUMNS, self.tests)
        self.save()
        return trash

    def latest(self, world=None):
        """The newest generation with a saved network; with world, the newest trained there."""
        finished = [g for g in self.generations if g["status"] != "failed" and g.get("models")
                    and (world is None or g["world"] == world)]
        return finished[-1] if finished else None

    def model_path(self, gen=None, which="best"):
        g = self.generation(gen) if gen else self.latest()
        if g is None:
            return None
        for name in (which, "final", "best"):
            path = self.dir / g["models"].get(name, "")
            if g["models"].get(name) and path.exists():
                return path
        return None

    def summary(self):
        latest = self.latest()
        return {
            "name": self.name,
            "generations": len(self.generations),
            "attempts": len(self.episodes),
            "best_test": max((t[2] for t in self.tests), default=None),
            "latest_gen": latest["gen"] if latest else None,
        }

    def payload(self):
        return {
            "name": self.name,
            "generations": json.loads(json.dumps(self.generations)),
            "episodes": list(self.episodes),
            "tests": list(self.tests),
            "trails": list(self.trails),
            "outcomes": OUTCOMES,
        }


def _reward_snapshot(path):
    """Read the reward constants from a saved reward.py without running it."""
    weights = default_reward()
    if path.exists():
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in REWARD_KEYS:
                    try:
                        weights[target.id] = float(ast.literal_eval(node.value))
                    except ValueError:
                        pass
    return weights


def import_cli_run(name):
    """Turn a run made by train.py (runs/<name>/) into a one-generation pilot."""
    run = RUNS / name
    rows = []
    for path in (run / "monitor").glob("*.monitor.csv"):
        with open(path) as f:
            t_start = json.loads(f.readline()[1:])["t_start"]
            for row in csv.DictReader(f):
                rows.append((t_start + float(row["t"]), float(row["r"]), int(row["l"]), row["outcome"]))
    rows.sort()
    model_name = "best_model.zip" if (run / "best_model.zip").exists() else "final_model.zip"
    model = load_model(run / "final_model.zip")
    pilot = Pilot(name)
    pilot.generations = [{
        "gen": 1, "mode": "cli", "status": "done", "world": "classic", "started": now_iso(),
        "duration": None, "frames": int(model.num_timesteps),
        "frames_start": 0, "frames_end": int(model.num_timesteps),
        "attempts_start": 0, "attempts_end": len(rows),
        "config": {
            "reward": _reward_snapshot(run / "reward.py"),
            "conditions": dict(DEFAULT_CONDITIONS),
            "ppo": dict(DEFAULT_PPO),
            "frames": int(model.num_timesteps),
        },
        "models": {"final": "final_model.zip", "best": model_name},
        "summary": {},
    }]
    pilot.add_episodes(
        [[i + 1, 1, round(r, 2), CODE[outcome], length, None, None]
         for i, (_, r, length, outcome) in enumerate(rows)],
        [],
    )
    test = test_flights(load_model(run / model_name))
    pilot.add_test([int(model.num_timesteps), 1, test["score"], test["landed"], test["flights"]])
    pilot.generations[0]["summary"] = {"best_test": test["score"], "last_test": test["score"],
                                       "last_landed": test["landed"]}
    pilot.save()
    return pilot


def discover_pilots():
    """Names of every pilot under runs/, importing train.py runs on first sight."""
    names = []
    if not RUNS.exists():
        return names
    for run in sorted(RUNS.iterdir()):
        if (run / "lab.json").exists():
            names.append(run.name)
        elif (run / "final_model.zip").exists() and (run / "monitor").is_dir():
            print(f"Importing {run}/ as a pilot (one-time, runs 20 test flights)...")
            import_cli_run(run.name)
            names.append(run.name)
    return names


# Training (runs in its own process) ----------------------------------------------

class FlightRecorder(gym.Wrapper):
    """On the last frame of an episode, adds its flight path and touchdown to info."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame, self.path, self.prev_obs = 0, [self._position()], obs
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self.frame += 1
        if self.frame % TRAIL_EVERY == 0 or terminated or truncated:
            self.path.append(self._position())
        if terminated or truncated:
            info["trail"] = self.path
            info["end_x"] = round(float(obs[0]), 3)
            # Observation velocities are scaled; x5 and x7.5 turn them back into world units/s.
            info["impact"] = round(float(np.hypot(self.prev_obs[2] * 5, self.prev_obs[3] * 7.5)), 2)
        self.prev_obs = obs
        return obs, r, terminated, truncated, info

    def _position(self):
        sim = self.env.unwrapped
        return [round(sim.lander.position.x - 10.0, 2), round(sim.lander.position.y - sim.helipad_y, 2)]


def make_practice_env(env_kwargs):
    return FlightRecorder(make_training_env(**env_kwargs))


class LabCallback(BaseCallback):
    """Streams practice attempts, test flights and network weights to the UI."""

    def __init__(self, job, events, stop, gen_dir):
        super().__init__()
        self.job, self.events, self.stop, self.gen_dir = job, events, stop, gen_dir
        self.attempt = job["attempt_offset"]
        self.pending, self.trails = [], []
        self.best = -math.inf

    def _on_training_start(self):
        self.start_frames = self.num_timesteps
        self.start_time = self.round_time = time.time()
        self.round_frames = self.num_timesteps
        self.next_test = self.start_frames + TEST_EVERY
        self.send("started", frames=self.num_timesteps)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.attempt += 1
                code = CODE[info["outcome"]]
                ep = info["episode"]
                self.pending.append([self.attempt, self.job["gen"], round(float(ep["r"]), 2), code,
                                     int(ep["l"]), info["end_x"], info["impact"]])
                self.trails.append([self.attempt, code, info["trail"]])
        return not self.stop.is_set()

    def _on_rollout_end(self):
        self.flush()
        flight = record_flight(self.model, conditions=self.job["conditions"])
        flight.update(weights=network_weights(self.model), gen=self.job["gen"], frames_trained=self.num_timesteps)
        self.send("flight", flight=flight)
        if self.num_timesteps >= self.next_test:
            self.next_test = self.num_timesteps + TEST_EVERY
            self.test()

    def _on_training_end(self):
        self.flush()
        self.test()  # always test the final pilot

    def flush(self):
        if not self.pending and self.num_timesteps == self.round_frames:
            return
        now = time.time()
        fps = (self.num_timesteps - self.round_frames) / max(now - self.round_time, 1e-6)
        self.round_time, self.round_frames = now, self.num_timesteps
        self.send("round", frames=self.num_timesteps, done=self.num_timesteps - self.start_frames,
                  target=self.job["frames"], elapsed=round(now - self.start_time, 1), fps=round(fps),
                  episodes=self.pending, trails=self.trails)
        self.pending, self.trails = [], []

    def test(self):
        result = test_flights(self.model, self.job["conditions"])
        is_best = result["score"] > self.best
        if is_best:
            self.best = result["score"]
            self.model.save(self.gen_dir / "best_model")
        row = [int(self.num_timesteps), self.job["gen"], result["score"], result["landed"], result["flights"]]
        self.send("test", row=row, best=is_best, result=result)

    def send(self, kind, **fields):
        self.events.put({"type": kind, "pilot": self.job["pilot"], "gen": self.job["gen"], **fields})


def train_worker(job, events, stop):
    """Entry point of the training process that app.py starts."""
    try:
        _train(job, events, stop)
    except Exception:
        events.put({"type": "error", "pilot": job["pilot"], "gen": job["gen"],
                    "message": traceback.format_exc()})


def _train(job, events, stop):
    for key, value in job["reward"].items():  # this process trains with the UI's reward settings
        setattr(reward, key, value)
    ppo = job["ppo"]
    env_kwargs = {**job["conditions"], "continuous": ppo["control"] == "throttle"}
    env = DummyVecEnv([partial(make_practice_env, env_kwargs) for _ in range(ppo["n_envs"])])
    if job["mode"] == "continue":
        model = PPO.load(job["from_path"], env=env, device="cpu", learning_rate=ppo["learning_rate"],
                         ent_coef=ppo["ent_coef"], gamma=ppo["gamma"])
    else:
        activation = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[ppo["activation"]]
        model = PPO(
            "MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, gae_lambda=0.98,
            gamma=ppo["gamma"], ent_coef=ppo["ent_coef"], learning_rate=ppo["learning_rate"],
            policy_kwargs={"net_arch": {"pi": ppo["net_arch"], "vf": ppo["net_arch"]},
                           "activation_fn": activation},
            seed=ppo["seed"], device="cpu",
        )
    gen_dir = RUNS / job["pilot"] / f"gen-{job['gen']}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    callback = LabCallback(job, events, stop, gen_dir)
    start = time.time()
    model.learn(total_timesteps=job["frames"], reset_num_timesteps=job["mode"] == "new", callback=callback)
    model.save(gen_dir / "final_model")
    env.close()
    events.put({
        "type": "done", "pilot": job["pilot"], "gen": job["gen"], "stopped": stop.is_set(),
        "frames_end": int(model.num_timesteps), "attempts_end": callback.attempt,
        "duration": round(time.time() - start, 1), "best_test": callback.best,
    })
