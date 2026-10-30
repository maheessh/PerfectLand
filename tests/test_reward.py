"""The reward function: what earns and costs points on one frame."""

import numpy as np
import pytest

import reward

HOVER = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_landing_and_crash_end_the_flight_with_fixed_points():
    assert reward.compute_reward(HOVER, 0, HOVER, "landed on pad") == reward.LANDING_BONUS
    assert reward.compute_reward(HOVER, 0, HOVER, "landed off pad") == reward.LANDING_BONUS
    assert reward.compute_reward(HOVER, 0, HOVER, "crashed") == reward.CRASH_PENALTY
    assert reward.compute_reward(HOVER, 0, HOVER, "flew off screen") == reward.CRASH_PENALTY


def test_getting_closer_earns_points_and_drifting_away_costs_them():
    closer = HOVER.copy()
    closer[1] = 0.4
    gained = reward.compute_reward(HOVER, 0, closer, None)
    lost = reward.compute_reward(closer, 0, HOVER, None)
    assert gained > 0 > lost
    # Potential-based shaping: a round trip costs only the time it took.
    assert gained + lost == pytest.approx(-2 * reward.TIME_COST)


def test_engines_cost_fuel():
    idle = reward.compute_reward(HOVER, 0, HOVER, None)
    main = reward.compute_reward(HOVER, reward.MAIN_ENGINE, HOVER, None)
    side = reward.compute_reward(HOVER, reward.LEFT_ENGINE, HOVER, None)
    assert main == pytest.approx(idle - reward.MAIN_ENGINE_COST)
    assert side == pytest.approx(idle - reward.SIDE_ENGINE_COST)


def test_throttle_reads_actions_like_the_simulator():
    assert reward.throttle(2) == (1.0, 0.0)
    assert reward.throttle(3) == (0.0, 1.0)
    assert reward.throttle(np.array([-0.5, 0.2])) == (0.0, 0.0)  # main off, side inside the dead zone
    main, side = reward.throttle(np.array([1.0, -0.8]))
    assert main == 1.0 and side == pytest.approx(0.8)


def test_firing_while_parked_costs_extra():
    parked = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    quiet = reward.compute_reward(parked, 0, parked, None)
    tapping = reward.compute_reward(parked, 1, parked, None)
    assert tapping == pytest.approx(quiet - reward.SIDE_ENGINE_COST - reward.GROUND_ENGINE_COST)
