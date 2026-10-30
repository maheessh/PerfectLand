"""Mission control's HTTP API, on a throwaway runs/ folder."""

import json
import mimetypes
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import app
import lab


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    patch = pytest.MonkeyPatch()
    patch.setattr(lab, "RUNS", runs)
    patch.setattr(lab, "TRASH", runs / ".trash")
    mimetypes.add_type("text/javascript", ".js")
    app.Handler.hub = app.Hub()
    app.Handler.mission = app.MissionControl(app.Handler.hub)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    patch.undo()


def get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def post(url, body):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def test_state_starts_with_no_pilots(server):
    status, kind, body = get(f"{server}/api/state")
    state = json.loads(body)
    assert status == 200 and "json" in kind
    assert state["pilots"] == [] and state["training"] is None
    assert set(state["planets"]) == set(lab.PLANETS)


def test_pages_are_served(server):
    for path, kind in [("/", "text/html"), ("/play", "text/html"), ("/sim.js", "javascript"), ("/data/worlds.json", "json")]:
        status, content_type, _ = get(server + path)
        assert status == 200 and kind in content_type, path


def test_unknown_paths_404(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        get(f"{server}/../lab.py")
    assert error.value.code == 404


def test_random_flight(server):
    flight = post(f"{server}/api/fly", {"random": True, "seed": 5, "conditions": {"planet": "titan"}})
    assert flight["outcome"] in lab.OUTCOMES and flight["conditions"]["planet"] == "titan"


def test_load_the_starter_pilot_then_fly_it(server):
    name = post(f"{server}/api/load-starter", {})["pilot"]
    state = json.loads(get(f"{server}/api/state")[2])
    assert [p["name"] for p in state["pilots"]] == [name]
    flight = post(f"{server}/api/fly", {"pilot": name, "gen": 1, "seed": 10_000, "conditions": {"planet": "mars"}})
    assert flight["outcome"] == "landed on pad"
    weights = json.loads(get(f"{server}/api/weights?pilot={name}&gen=1")[2])
    assert weights["control"] == "buttons"
