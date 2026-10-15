"""The reward function: the only place we tell the rocket what we want.

We never tell the agent *how* to fly. Every frame (50 per second) this file turns
what just happened into a score, and training searches for the behavior that
collects the most points. Change a number below, retrain, and the flying changes.

Each observation is 8 numbers, normalized by the simulator:
    x, y                  position relative to the pad (the pad spans x = -0.2 .. 0.2)
    vx, vy                velocity
    angle, spin           tilt in radians (0 = upright) and how fast it is turning
    left_leg, right_leg   1.0 when that leg is touching the ground, else 0.0
"""

import numpy as np

# The four buttons the agent can press each frame.
DO_NOTHING, LEFT_ENGINE, MAIN_ENGINE, RIGHT_ENGINE = 0, 1, 2, 3

# Progress: points for moving toward a better state, every frame.
DISTANCE_WEIGHT = 100.0  # getting closer to the pad
SPEED_WEIGHT = 100.0  # slowing down
TILT_WEIGHT = 100.0  # staying upright
LEG_CONTACT_BONUS = 10.0  # per leg touching the ground

# Costs: points off, every frame.
MAIN_ENGINE_COST = 0.30  # fuel burned by the main engine
SIDE_ENGINE_COST = 0.03  # fuel burned by a side engine
TIME_COST = 0.05  # taking too long
GROUND_ENGINE_COST = 0.30  # firing any engine while parked on the ground

# The verdict: given once, on the frame the episode ends.
LANDING_BONUS = 100.0  # came to rest on its legs
CRASH_PENALTY = -100.0  # hull hit the ground, or flew off screen


def potential(obs):
    """How good is this moment? Closer, slower, upright, legs down = higher."""
    x, y, vx, vy, angle, _spin, left_leg, right_leg = obs
    return (
        -DISTANCE_WEIGHT * np.hypot(x, y)
        - SPEED_WEIGHT * np.hypot(vx, vy)
        - TILT_WEIGHT * abs(angle)
        + LEG_CONTACT_BONUS * (left_leg + right_leg)
    )


def throttle(action):
    """How hard the main and side engines fire this frame, each from 0 to 1.

    A button pilot presses one of the four buttons, so an engine is fully on or off.
    A throttle pilot outputs two numbers from -1 to 1, read the way the simulator
    reads them: main above 0 fires the main engine at 50-100% power, and side
    beyond ±0.5 fires a side engine at 50-100%.
    """
    if np.ndim(action) == 0:
        return float(action == MAIN_ENGINE), float(action in (LEFT_ENGINE, RIGHT_ENGINE))
    main, side = np.clip(action, -1, 1)
    return (float(main + 1) / 2 if main > 0 else 0.0), (float(abs(side)) if abs(side) > 0.5 else 0.0)


def compute_reward(prev_obs, action, obs, outcome):
    """Score one frame: the agent was in prev_obs, pressed action, and ended up in obs.

    outcome is None while the rocket is still flying; on the last frame it says how
    the episode ended: "landed on pad", "landed off pad", "crashed",
    "flew off screen" or "timed out".
    """
    if outcome in ("landed on pad", "landed off pad"):
        return LANDING_BONUS
    if outcome in ("crashed", "flew off screen"):
        return CRASH_PENALTY

    # Reward the *change* in potential, not the potential itself. Points earned by
    # approaching the pad are taken back if the rocket drifts away again, so the
    # only way to collect them for good is to actually finish the landing.
    reward = potential(obs) - potential(prev_obs)

    main, side = throttle(action)
    reward -= MAIN_ENGINE_COST * main + SIDE_ENGINE_COST * side
    reward -= TIME_COST

    # Firing while parked: without this cost the agent touches down and then keeps
    # tapping the (cheap) side engines, so the simulator never sees it come to rest and
    # the landing never counts. Touching down itself stays free, so the agent can still
    # brake all the way to the ground.
    parked = prev_obs[6] and prev_obs[7] and np.hypot(prev_obs[2], prev_obs[3]) < 0.05
    if parked and (main or side):
        reward -= GROUND_ENGINE_COST
    return float(reward)
