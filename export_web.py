"""Put a trained pilot into the browser game (ui/play.html), which flies without a server.

    python export_web.py --pilot main --gen 16             # runs/main/gen-16/best_model.zip
    python export_web.py --model path/to/best_model.zip --name universal

Writes the network's weights to ui/data/pilots/<name>.json, lists it in
ui/data/pilots.json, and refreshes ui/data/worlds.json from the worlds in lab.py.
The browser runs the same math as PyTorch (ui/nn.js), so the exported pilot flies
the browser's copy of the simulator (ui/sim.js) the way it flies the real one.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import lab

DATA = Path(__file__).resolve().with_name("ui") / "data"


def weights(model, digits=6):
    """lab.network_weights, at full enough precision to fly exactly like the original."""
    policy = model.policy

    def linear_layers(modules):
        return [
            {"w": np.round(m.weight.detach().cpu().numpy().astype(float), digits).tolist(),
             "b": np.round(m.bias.detach().cpu().numpy().astype(float), digits).tolist()}
            for m in modules if isinstance(m, torch.nn.Linear)
        ]

    summary = lab.network_weights(model)
    summary["actor"] = linear_layers(policy.mlp_extractor.policy_net) + linear_layers([policy.action_net])
    summary["critic"] = linear_layers(policy.mlp_extractor.value_net) + linear_layers([policy.value_net])
    return summary


def write_json(path, payload, compact=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":")) if compact else json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text + "\n")


def export_worlds():
    worlds = {k: {key: v[key] for key in ("name", *lab.PHYSICS_KEYS, "fact")} for k, v in lab.PLANETS.items()}
    write_json(DATA / "worlds.json", worlds, compact=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", help="a pilot in runs/")
    parser.add_argument("--gen", type=int, help="its generation (default: the latest)")
    parser.add_argument("--model", type=Path, help="or a saved network .zip, from anywhere")
    parser.add_argument("--name", default="universal", help="file name in ui/data/pilots/")
    parser.add_argument("--title", default="Universal pilot", help="what the game calls it")
    parser.add_argument("--note", default="", help="one line about how it was trained")
    args = parser.parse_args()

    if args.model:
        path = args.model
    elif args.pilot:
        path = lab.Pilot(args.pilot).model_path(args.gen, "best")
        if path is None:
            raise SystemExit(f"runs/{args.pilot} has no saved network for that generation.")
    else:
        raise SystemExit("Pass --pilot (and --gen) or --model.")

    export_worlds()
    model = lab.load_model(Path(path))
    write_json(DATA / "pilots" / f"{args.name}.json", weights(model))

    index_path = DATA / "pilots.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [p for p in index if p["id"] != args.name]
    index.insert(0, {"id": args.name, "title": args.title, "note": args.note,
                     "control": "throttle" if lab.is_throttle(model) else "buttons",
                     "file": f"pilots/{args.name}.json"})
    write_json(index_path, index, compact=False)
    print(f"Exported {path} -> ui/data/pilots/{args.name}.json")


if __name__ == "__main__":
    main()
