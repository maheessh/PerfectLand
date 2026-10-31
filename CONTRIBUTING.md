# Contributing to PerfectLand

Thanks for wanting to help. PerfectLand is a teaching tool first, so changes that make reinforcement learning easier to see and understand are the most welcome.

## Set up

```bash
git clone https://github.com/maheessh/PerfectLand
cd PerfectLand
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python app.py            # mission control at http://localhost:8000, the game at /play
```

Or open the repository in GitHub Codespaces: it installs everything and starts mission control for you.

## Run the tests

```bash
pytest -q                                         # simulator, reward, lab and server (about 5 s)
node --test tests/js/*.test.mjs                   # the browser game flies like the real simulator
```

Both run on every push in GitHub Actions.

## How the code fits together

| File | What it does |
| --- | --- |
| `reward.py` | The reward function: what earns and costs points. Edit it and train to see what changes. |
| `lander.py` | The simulator (Gymnasium Lunar Lander) plus wrappers for gravity, air, engines and start altitude. |
| `lab.py` | Pilots, generations, test flights, distillation and the training loop. |
| `app.py` | Mission control's server: a standard-library HTTP server with a live event stream. |
| `ui/` | Mission control (`index.html`, `app.js`) and the game (`play.html`, `play.js`). |
| `ui/sim.js` | The simulator ported to the browser on planck.js, so the game needs no server. |
| `ui/lander.js` | The 3D lander. Its legs are rebuilt every frame from the physics legs, so keep it in step with `lander.py`'s shapes. |
| `export_web.py` | Puts a trained pilot into the game (`ui/data/pilots/`). |

## Good first contributions

- **Add a world.** Add it to `PLANETS` in `lab.py`, give it a sky in `LOOKS` and a color in `WORLD_DOT` (`ui/worlds.js`), then run `python export_web.py --model pretrained/universal.zip` to refresh the game's world list.
- **A new chart** in the Telemetry section (`ui/charts.js`), such as fuel used per attempt.
- **Accessibility:** keyboard focus order and screen reader labels in the game and the drawer.
- **Translations** of the game page.

If you change the simulator in `lander.py`, make the same change in `ui/sim.js` so the game stays faithful. `tests/js/sim.test.mjs` checks that the trained pilot still lands there.

## Style

- Python: standard library first, small functions, a docstring that says *why*.
- JavaScript: plain ES modules, no build step, no framework.
- Write user-facing text in plain words: "landed 0.3 m from the center", not "Δx = 0.3".
