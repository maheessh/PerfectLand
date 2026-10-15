"""PerfectLand mission control: watch the rocket train, fly it, and look inside its brain.

    python app.py              # then open http://localhost:8000
    python app.py --port 9000

The browser talks to this server: it starts and stops training, asks for test
flights, and listens to a live stream of what the training process is doing.
"""

import argparse
import json
import mimetypes
import multiprocessing as mp
import queue
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import lab

UI_DIR = Path(__file__).resolve().with_name("ui")


class Hub:
    """Fans live events out to every open browser tab."""

    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.clients.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)

    def publish(self, event):
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(event)
                except queue.Full:  # a tab that stopped reading: drop it so it reconnects fresh
                    q.dropped = True
                    self.clients.discard(q)


class MissionControl:
    """Owns the pilots on disk and the one training run that may be in progress."""

    def __init__(self, hub):
        self.hub = hub
        self.lock = threading.RLock()
        self.ctx = mp.get_context("spawn")
        self.events = self.ctx.Queue()
        self.pilots = {name: lab.Pilot(name) for name in lab.discover_pilots()}
        self.job = None
        self.process = None
        self.stop_event = None
        threading.Thread(target=self._pump, daemon=True).start()

    # Pilots ---------------------------------------------------------------

    def pilot(self, name):
        with self.lock:
            if name not in self.pilots:
                if not (lab.RUNS / name / "lab.json").exists():
                    raise KeyError(f"No pilot named {name!r}")
                self.pilots[name] = lab.Pilot(name)
            return self.pilots[name]

    def new_pilot_name(self):
        n = 1
        while f"pilot-{n}" in self.pilots or (lab.RUNS / f"pilot-{n}").exists():
            n += 1
        return f"pilot-{n}"

    def state(self):
        with self.lock:
            pilots = sorted(self.pilots.values(), key=lambda p: p.created)
            return {
                "pilots": [p.summary() for p in pilots],
                "training": self.public_job(),
                "defaults": {
                    "reward": lab.default_reward(),
                    "conditions": lab.DEFAULT_CONDITIONS,
                    "ppo": lab.DEFAULT_PPO,
                },
                "planets": lab.PLANETS,
                "new_pilot_name": self.new_pilot_name(),
                "frame_fields": lab.FRAME_FIELDS,
                "outcomes": lab.OUTCOMES,
            }

    # Training -----------------------------------------------------------

    def public_job(self):
        if not self.job:
            return None
        keys = ("pilot", "gen", "mode", "origin", "parent", "frames", "auto", "conditions", "reward", "ppo", "started", "progress")
        return {k: self.job.get(k) for k in keys}

    def start_training(self, request):
        with self.lock:
            if self.job:
                raise RuntimeError("A training run is already in progress.")
            conditions = lab.clean_conditions(request.get("conditions"))
            if request.get("mode") != "continue":  # a brand-new pilot
                name = (request.get("pilot") or self.new_pilot_name()).strip()
                if not name.replace("-", "").replace("_", "").isalnum():
                    raise ValueError("Pilot names may use letters, numbers, - and _.")
                if name in self.pilots and self.pilots[name].generations:
                    raise ValueError(f"Pilot {name!r} already exists. Retrain it, or pick a new name.")
                pilot = self.pilots.get(name) or lab.Pilot(name)
                self.pilots[name] = pilot
                mode, origin, parent, from_path = "new", "new pilot", None, None
                ppo = lab.clean_ppo(request.get("ppo"))
            else:
                pilot = self.pilot(request["pilot"])
                shape = pilot.latest()
                if shape is None:
                    raise ValueError("This pilot has no finished generation yet. Train a new pilot instead.")
                # The network's shape and controls are fixed once it exists; only the learning knobs change.
                fixed = {k: shape["config"]["ppo"][k] for k in ("net_arch", "activation", "control")}
                ppo = lab.clean_ppo({**shape["config"]["ppo"], **(request.get("ppo") or {}), **fixed})
                if request.get("fresh"):  # a blank network of the same shape, e.g. to start a world from scratch
                    mode, origin, parent, from_path = "new", "fresh", None, None
                else:
                    # Each world has its own line of generations: by default, continue the latest
                    # one trained on this world, or else the latest of any world.
                    from_gen = request.get("from_gen")
                    parent = (pilot.generation(int(from_gen)) if from_gen
                              else pilot.latest(conditions["planet"]) or shape)
                    from_path = pilot.model_path(parent["gen"], request.get("from_model", "final"))
                    if from_path is None:
                        raise ValueError(f"Generation {parent['gen']} has no saved network to continue from.")
                    mode, origin = "continue", "continue"
            gen = max((g["gen"] for g in pilot.generations), default=0) + 1
            job = {
                "pilot": pilot.name, "gen": gen, "mode": mode, "origin": origin,
                "parent": parent["gen"] if parent else None,
                "from_path": str(from_path) if from_path else None,
                "from_model": request.get("from_model", "final"),
                "frames": min(max(int(request.get("frames", 1_000_000)), 20_000), 20_000_000),
                "reward": lab.clean_reward(request.get("reward")),
                "conditions": conditions,
                "ppo": ppo,
                "auto": bool(request.get("auto")),
                "attempt_offset": len(pilot.episodes),
                "started": time.time(),
                "progress": {"done": 0, "fps": 0, "elapsed": 0},
            }
            pilot.generations.append({
                "gen": gen, "mode": "fresh" if origin == "fresh" else mode, "parent": job["parent"], "status": "training",
                "world": conditions["planet"],
                "started": lab.now_iso(), "duration": None, "frames": job["frames"],
                "frames_start": None, "frames_end": None,
                "attempts_start": job["attempt_offset"], "attempts_end": None,
                "config": {k: job[k] for k in ("reward", "conditions", "ppo", "frames", "from_model")},
                "models": {}, "summary": {},
            })
            pilot.save()
            self.job = job
            self.stop_event = self.ctx.Event()
            self.process = self.ctx.Process(
                target=lab.train_worker, args=(job, self.events, self.stop_event), daemon=True
            )
            self.process.start()
            threading.Thread(target=self._watch, args=(self.process, pilot.name, gen), daemon=True).start()
            self.hub.publish({"type": "job", "training": self.public_job(), "pilots": self.state()["pilots"]})
            return self.public_job()

    def stop_training(self):
        with self.lock:
            if not self.job:
                return False
            self.job["auto"] = False
            self.stop_event.set()
            self.hub.publish({"type": "stopping", "pilot": self.job["pilot"], "gen": self.job["gen"]})
            return True

    def _watch(self, process, pilot, gen):
        process.join()
        self.events.put({"type": "exit", "pilot": pilot, "gen": gen, "code": process.exitcode})

    def _pump(self):
        while True:
            event = self.events.get()
            try:
                self._handle(event)
            except Exception:
                traceback.print_exc()

    def _handle(self, event):
        kind = event["type"]
        with self.lock:
            job = self.job
            current = job and job["pilot"] == event.get("pilot") and job["gen"] == event.get("gen")
            pilot = self.pilots.get(event.get("pilot"))
            if kind == "exit":
                if not current:
                    return
                event = {"type": "error", "pilot": job["pilot"], "gen": job["gen"],
                         "message": f"The training process exited unexpectedly (code {event['code']})."}
                kind = "error"
            if pilot is None:
                return
            if kind == "started" and current:
                pilot.generation(job["gen"])["frames_start"] = event["frames"]
                pilot.save()
            elif kind == "round":
                pilot.add_episodes(event["episodes"], event["trails"])
                if current:
                    job["progress"] = {k: event[k] for k in ("done", "fps", "elapsed", "frames")}
            elif kind == "test":
                pilot.add_test(event["row"])
                summary = pilot.generation(event["gen"])["summary"]
                score = event["row"][2]
                summary["last_test"], summary["last_landed"] = score, event["row"][3]
                summary["last_miss"], summary["last_touchdown"] = event["result"]["miss"], event["result"]["touchdown"]
                best = summary.get("best_test")
                summary["best_test"] = score if best is None else max(best, score)
            elif kind == "done":
                g = pilot.generation(event["gen"])
                gen_dir = f"gen-{event['gen']}"
                g.update(status="stopped" if event["stopped"] else "done", duration=event["duration"],
                         frames_end=event["frames_end"], attempts_end=event["attempts_end"],
                         models={"final": f"{gen_dir}/final_model.zip", "best": f"{gen_dir}/best_model.zip"})
                pilot.save()
            elif kind == "error":
                g = pilot.generation(event["gen"])
                g.update(status="failed", error=event["message"][-2000:])
                pilot.save()
                print(event["message"])
            if kind in ("done", "error") and current:
                self.job = None
                event = {**event, "pilots": [p.summary() for p in self.pilots.values()]}
        self.hub.publish(event)
        if kind == "done" and current and job["auto"] and not event["stopped"]:
            try:  # keep improving: start the next generation from this one
                self.start_training({"mode": "continue", "pilot": job["pilot"], "frames": job["frames"],
                                     "reward": job["reward"], "conditions": job["conditions"],
                                     "ppo": job["ppo"], "auto": True, "from_model": job["from_model"]})
            except Exception as error:
                self.hub.publish({"type": "notice", "message": f"Auto-retrain stopped: {error}"})


class Handler(BaseHTTPRequestHandler):
    server_version = "PerfectLand/1.0"
    mission = None
    hub = None

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    # Responses ----------------------------------------------------------

    def send_json(self, payload, status=200):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status)

    def read_json(self):
        # Browsers can't send JSON to another site without asking first, so this
        # keeps other web pages from starting training on your machine.
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise ValueError("Send requests as application/json.")
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    # Routes -------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/events":
                return self.stream_events()
            if url.path == "/api/state":
                return self.send_json(self.mission.state())
            if url.path == "/api/pilot":
                with self.mission.lock:
                    payload = self.mission.pilot(query["name"]).payload()
                return self.send_json(payload)
            if url.path == "/api/weights":
                pilot = self.mission.pilot(query["pilot"])
                path = pilot.model_path(int(query["gen"]) if query.get("gen") else None, query.get("model", "best"))
                if path is None:
                    return self.send_error_json(404, "No saved network for that generation yet.")
                return self.send_json(lab.network_weights(lab.load_model(path)))
            if url.path == "/api/episodes.csv":
                pilot = self.mission.pilot(query["pilot"])
                return self.send_file(pilot.dir / "episodes.csv", "text/csv", download=f"{pilot.name}-episodes.csv")
            return self.send_static(url.path)
        except KeyError as error:
            self.send_error_json(404, str(error))
        except Exception as error:
            traceback.print_exc()
            self.send_error_json(500, str(error))

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self.read_json()
            if url.path == "/api/train":
                return self.send_json(self.mission.start_training(body))
            if url.path == "/api/stop":
                return self.send_json({"stopping": self.mission.stop_training()})
            if url.path in ("/api/fly", "/api/test"):
                model = None
                if not body.get("random"):
                    pilot = self.mission.pilot(body["pilot"])
                    gen = int(body["gen"]) if body.get("gen") else None
                    path = pilot.model_path(gen, body.get("model", "best"))
                    if path is None:
                        return self.send_error_json(404, "This pilot has no saved network yet.")
                    model = lab.load_model(path)
                control = body.get("control", "buttons")  # for the random pilot
                if url.path == "/api/test":
                    return self.send_json(lab.test_flights(model, body.get("conditions"),
                                                           seed=int(body.get("seed", 1000)), control=control))
                return self.send_json(lab.record_flight(model, body.get("seed"), body.get("conditions"), control))
            self.send_error_json(404, "Unknown endpoint")
        except (ValueError, KeyError) as error:
            self.send_error_json(400, str(error))
        except RuntimeError as error:
            self.send_error_json(409, str(error))
        except Exception as error:
            traceback.print_exc()
            self.send_error_json(500, str(error))

    def stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = self.hub.subscribe()
        try:
            self.wfile.write(b"retry: 1500\n\n")
            self.wfile.flush()
            while not getattr(q, "dropped", False):
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                else:
                    self.wfile.write(b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.hub.unsubscribe(q)

    def send_static(self, path):
        target = (UI_DIR / (path.lstrip("/") or "index.html")).resolve()
        if target.is_dir():
            target = target / "index.html"
        if not target.is_relative_to(UI_DIR) or not target.is_file():
            return self.send_error_json(404, "Not found")
        self.send_file(target, mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    def send_file(self, path, content_type, download=None):
        if not path.is_file():
            return self.send_error_json(404, "Not found")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    mimetypes.add_type("text/javascript", ".js")
    Handler.hub = Hub()
    Handler.mission = MissionControl(Handler.hub)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://localhost:{args.port}"
    print(f"PerfectLand mission control: {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        Handler.mission.stop_training()


if __name__ == "__main__":
    main()
