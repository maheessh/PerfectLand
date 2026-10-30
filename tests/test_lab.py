"""The lab: conditions, recorded flights, pilots on disk, and the browser game's data."""

import json
from pathlib import Path

import pytest

import lab

UI_DATA = Path(__file__).resolve().parents[1] / "ui" / "data"


def test_conditions_are_clamped_to_what_the_simulator_handles():
    c = lab.clean_conditions({"planet": "custom", "gravity": -99, "start_height": 500, "engine_power": 0})
    assert c["gravity"] == -30.0
    assert c["start_height"] == lab.START_HEIGHTS[1]
    assert c["engine_power"] == 0.3
    assert lab.clean_conditions({"planet": "universal"})["planet"] == "universal"


def test_old_settings_find_their_world():
    mars = {k: lab.PLANETS["mars"][k] for k in lab.PHYSICS_KEYS}
    assert lab.clean_conditions(mars)["planet"] == "mars"


def test_recorded_flight_has_one_row_per_frame():
    flight = lab.record_flight(None, seed=7, conditions=lab.world_conditions("moon"))
    assert flight["outcome"] in lab.OUTCOMES
    assert flight["steps"] == len(flight["frames"]) - 1
    assert all(len(frame) == len(lab.FRAME_FIELDS) for frame in flight["frames"])
    assert flight["pad"][0] < 0 < flight["pad"][1]


def test_network_weights_match_the_policy():
    weights = lab.network_weights(lab.load_model(lab.STARTER))
    assert weights["control"] == "buttons"
    assert [len(layer["w"]) for layer in weights["actor"]] == [64, 64, 4]
    assert [len(layer["w"]) for layer in weights["critic"]] == [64, 64, 1]


def test_starter_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "RUNS", tmp_path)
    pilot = lab.load_starter("starter")
    g = pilot.generations[0]
    assert g["world"] == lab.UNIVERSAL and g["mode"] == "pretrained"
    assert g["summary"]["last_flights"] == 3 * len(lab.PLANETS)
    assert g["summary"]["last_landed"] >= 30  # it lands nearly everywhere
    reloaded = lab.Pilot("starter")
    assert reloaded.model_path(1).exists()
    with pytest.raises(RuntimeError):
        lab.load_starter("starter")


def test_delete_generations_moves_them_to_the_trash(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "RUNS", tmp_path)
    monkeypatch.setattr(lab, "TRASH", tmp_path / ".trash")
    pilot = lab.load_starter("starter")
    pilot.delete_generations([1])
    assert not lab.Pilot("starter").generations
    assert any((tmp_path / ".trash").rglob("best_model.zip"))


def test_browser_game_uses_the_same_worlds():
    """ui/data/worlds.json (from export_web.py) must match lab.PLANETS, or the game flies other physics."""
    worlds = json.loads((UI_DATA / "worlds.json").read_text())
    assert set(worlds) == set(lab.PLANETS)
    for name, world in worlds.items():
        for key in lab.PHYSICS_KEYS:
            assert world[key] == lab.PLANETS[name][key], (name, key)


def test_shipped_game_pilot_exists():
    index = json.loads((UI_DATA / "pilots.json").read_text())
    for entry in index:
        weights = json.loads((UI_DATA / entry["file"]).read_text())
        assert weights["control"] == entry["control"]
        assert len(weights["actor"][0]["w"][0]) == 8  # 8 sensor readings in
