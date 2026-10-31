<div align="center">

# PerfectLand

**Teach a neural network to land a rocket on 13 worlds. Then try to beat it.**

[![Tests](https://github.com/maheessh/PerfectLand/actions/workflows/ci.yml/badge.svg)](https://github.com/maheessh/PerfectLand/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-111111.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-111111.svg)](#quick-start)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Codespaces-open-111111.svg?logo=github)](https://codespaces.new/maheessh/PerfectLand)

### [▶ Play it in your browser](https://maheessh.github.io/PerfectLand/)

<sub>No install, no account. [Watch the 23-second tour](media/perfectland-tour.mp4?raw=true) · [Train your own](#quick-start) · [How it works](#how-it-fits-together)</sub>

</div>

<br>

<a href="https://maheessh.github.io/PerfectLand/"><img src="media/highlights/race.gif" alt="The game on Mars: a black rocket labelled YOU and a see-through blue rocket labelled AI descend to the same pad. The AI stops 0.10 m from the center and wins 245 to 238." width="100%"></a>

<p align="center"><sub>The game: race the trained network on the same Mars terrain, with the same gusts and the same starting push. The see-through blue rocket is the AI. (In this recording, the classic Lunar Lander heuristic flies "you".)</sub></p>

A neural network learns to land a rocket in the Lunar Lander physics simulator, using
reinforcement learning (PPO) and **reward shaping**. Nobody tells it how to fly: every
frame, the reward function in [`reward.py`](reward.py) turns what just happened into
points, and the network practises until it finds the flying style that earns the most.
A first pilot learns in about 80 seconds on a laptop.

PerfectLand then takes it further than the textbook exercise:

- **13 worlds with real gravity**, from Pluto's 0.7 m/s² to Jupiter's 23.1, with air,
  wind, engines sized like a real lander's, and start altitudes from 5 to 40 m.
  Gymnasium's simulator stops at 12 m/s²; [`lander.py`](lander.py) wraps it to go further.
- **One network for every world.** The Universal pilot first studies each world's
  best specialist (policy distillation, DAgger-style), then practises on all 13 worlds
  at once. Landings across the worlds rose from 62% to 87%.
- **Mission control**, a local web app where you watch a pilot learn live: every
  attempt charted, a 3D flight deck, and the network's neurons firing as it decides.
- **Learn it by playing.** Two hand-drawn machines you can turn round, open up and play with: a
  lander you fly and refuel, and the AI pilot's flight computer, running the real network, whose
  neurons light up as it lands. Made for kids and newcomers as much as engineers.
- **Train your own AI in the browser.** PPO is reimplemented in plain JavaScript
  ([`ui/ppo.js`](ui/ppo.js)) and runs in a background thread: name your pilot, pick a world, and
  watch it go from random crashes to landings in about a minute and a half, then race it. Your AIs
  are saved in your own browser (IndexedDB), never uploaded; export one to move it to another device.
- **A game anyone can play.** The simulator is ported to the browser (planck.js, a
  JavaScript Box2D) and checked against the original in CI, so the trained network
  flies there too and you can race it. It is hosted for free on GitHub Pages.

### Learn it by playing

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="media/highlights/figure-lander.gif" alt="A hand-drawn lunar lander on its pad: the autopilot lands it while See-through shows the fuel and oxidizer tanks, helium, main valve and flight computer, each label lighting up as it works" width="100%">
      <br><b>Fig. 01: the lander.</b>
      <br><sub>Fly it with the keypad or the autopilot, refuel it from the cart, and see the fuel flow inside. Probes touch first ("contact light!"), then the shock absorbers catch the bump.</sub>
    </td>
    <td width="50%" valign="top">
      <img src="media/highlights/figure-brain.gif" alt="A hand-drawn flight computer: eight sensor dials, a screen with the lander, and two grids of 64 neuron lights under glass, flickering orange and blue as the real network lands the rocket" width="100%">
      <br><b>Fig. 02: the AI pilot's flight computer.</b>
      <br><sub>Eight sensor dials in, two layers of 64 neurons, four engine lamps out. It runs the real trained network: turn a dial and watch it change its mind.</sub>
    </td>
  </tr>
</table>

### Results

| Pilot | Test | Landed on the pad | Average score |
|---|---|---|---|
| Random button presses | 100 new Classic flights | 0% | −201 |
| One pilot, 1M frames on Classic (about 80 s) | 100 new Classic flights | **100%** | **256** (200 = solved) |
| Each world's own specialist, else the newest one | 20 flights × 13 worlds | 62% (161/260) | |
| **Universal pilot**: studied 6 specialists, then 1M frames on all worlds | 20 flights × 13 worlds | **87% (227/260)** | |
| The same network in the browser port ([`ui/sim.js`](ui/sim.js)) | 20 flights × 13 worlds | 91% (237/260) | 258 |

## Highlights

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="media/highlights/hook.gif" alt="An untrained network pressing random buttons tumbles out of an Earth sky and crashes beside the pad" width="100%">
      <br><b>Attempt 1: random buttons.</b>
      <br><sub>An untrained network on Earth. Nobody tells it how to fly, so it crashes.</sub>
    </td>
    <td width="50%" valign="top">
      <img src="media/highlights/telemetry.gif" alt="Telemetry charts fill up while a fresh pilot trains: scores climb and the outcome bars turn from crashes to landings" width="100%">
      <br><b>Every attempt, charted.</b>
      <br><sub>A fresh pilot went from 0% to 95% landed in 3,960 attempts, about 80 s of training.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="media/highlights/worlds.gif" alt="The dotted 3D globe steps from Jupiter to Europa to Saturn, each with its real gravity" width="100%">
      <br><b>13 worlds, real gravity.</b>
      <br><sub>From Pluto's 0.7 m/s² to Jupiter's 23.1, each with its own sky, air and wind.</sub>
    </td>
    <td width="50%" valign="top">
      <img src="media/highlights/altitude.gif" alt="The start altitude slider goes to 40 m; the camera backs away to show the rocket high above the pad, then follows it down to a landing" width="100%">
      <br><b>Start from up to 40 m.</b>
      <br><sub>Higher starts get more time on the clock, and pilots trained there touch down about 3× softer.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="media/highlights/neurons.gif" alt="The pilot's network in 3D: 8 sensor inputs, two layers of 64 neurons and two throttle outputs, with connections glowing as signals flow" width="100%">
      <br><b>Inside the pilot's brain.</b>
      <br><sub>Every connection glows with the signal it carries on this frame. Click any neuron.</sub>
    </td>
    <td width="50%" valign="top">
      <img src="media/highlights/training.gif" alt="The training panel: pick a world, drag the start altitude to 30 m, pick a duration" width="100%">
      <br><b>Train each world its own way.</b>
      <br><sub>Pick the world, the start altitude and how long; each world keeps its own line of generations.</sub>
    </td>
  </tr>
</table>

## Quick start

**Just play.** Open [the game](https://maheessh.github.io/PerfectLand/) in any modern
browser, phones included.

**Run everything in the cloud.** [Open the repository in GitHub Codespaces](https://codespaces.new/maheessh/PerfectLand):
it installs everything and starts mission control in a forwarded tab. Personal accounts
get free Codespaces hours every month; this one asks for a 4-core machine, so training
stays quick.

**Run it on your computer.** You need Python 3.10 to 3.13. (3.14 also works, but then
Box2D compiles from source and needs a C++ compiler.) Check with `python3 --version`
first: the `python3` that ships with macOS can be 3.9, which is too old. In that case
install a newer one (for example `brew install python@3.12`) and use `python3.12` below.

```bash
git clone https://github.com/maheessh/PerfectLand
cd PerfectLand
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # mission control at http://localhost:8000, the game at /play
```

On a fresh install, mission control offers **Load the pre-trained pilot**: the Universal
network from [`pretrained/`](pretrained/), ready to fly, test and inspect on every world.
Or press **Train a pilot**, and the first one lands in about 80 seconds.

Everything you train is yours and stays on your computer: pilots live in `runs/`, which Git
ignores, so nothing you train is uploaded or shared. Everyone who downloads PerfectLand starts
with an empty `runs/`. The only network that ships with the project is the demo Universal pilot
the game races against.

Prefer the command line? The same simulator and reward, without the web app:

```bash
python train.py                    # practice for 1 million frames
python watch.py                    # watch 3 landings in a window
python evaluate.py                 # score the pilot on 100 new flights
python evaluate.py --random        # compare with random button presses
python watch.py --gif flight.gif   # save landings as a GIF instead
```

What `watch.py` shows, before and after training:

| Before training: random buttons | After 1 million frames of practice |
|---|---|
| ![A random pilot crashing twice](media/random_pilot.gif) | ![The trained pilot landing three times](media/landing.gif) |

## How it fits together

```mermaid
flowchart LR
  R["reward.py<br/>what earns points"] --> T
  W["lander.py<br/>13 worlds, start altitude"] --> T["PPO training<br/>Stable-Baselines3, 16–26 simulators"]
  T -- "every 100k frames" --> E["20 test flights"]
  T --> G[("runs/&lt;pilot&gt;/gen-N<br/>a line of generations per world")]
  G --> D["Universal pilot<br/>distill each world's best, then train on all"]
  D --> G
  G --> M["mission control<br/>app.py + three.js"]
  G -- "export_web.py" --> J["weights as JSON"]
  J --> P["the game<br/>ui/sim.js on planck.js, no server"]
```

- **Training** ([`lab.py`](lab.py)) runs in its own process and streams every attempt,
  test and checkpoint to mission control ([`app.py`](app.py), a standard-library HTTP
  server with server-sent events), which charts them as they arrive.
- **Each world keeps its own line of generations**, so training on Mars never overwrites
  what the pilot knows about Jupiter. The Universal line learns from all of them.
- **The browser runs the network itself** ([`ui/nn.js`](ui/nn.js), the same math as
  PyTorch), which is what makes the neuron view and the game instant.

## The game: race the AI

[`ui/play.html`](ui/play.html) is a game made from the same pieces. It runs entirely in
the browser, so it is [hosted for free on GitHub Pages](https://maheessh.github.io/PerfectLand/),
and app.py serves it at `/play` too.

- **Race the AI:** you and the network fly the same terrain at the same time, with the
  same gusts and starting push; its see-through blue rocket flies beside yours. The
  official Lunar Lander score decides: where you come to rest and how much fuel you burn.
- **Fly solo** or **watch the AI**, with its decision on every frame (how likely it is to
  press each engine) and how many more points it expects.
- **Keys:** ↑ or W fires the main engine, ← → or A D the side engines. On a phone, hold
  the three buttons on the view. Slow motion halves the speed while you learn.
- **Records** per world stay in your browser. **Copy my result** gives a link to the same
  terrain, so a friend can try to beat your score on it.
- **Train your AI:** right under the game. It practises in a background thread (about 7,500 frames
  a second on a laptop; a first landing usually comes around 500k frames), with every practice flight
  charted live and a 10-flight test every 20k frames. **My AIs** keeps them in this browser, with race,
  train more, export and import.
- **Race your own pilots:** served by app.py, the opponent list includes every generation
  you trained in mission control.
- **Learn the parts:** under the game, the hand-drawn lander and the flight computer explain
  how a lander lands and how the network decides, in words a ten-year-old can follow.

The physics is a line-by-line port of Gymnasium's Lunar Lander and PerfectLand's world
wrappers onto [planck.js](https://piqnt.com/planck.js/), a JavaScript rewrite of the
same Box2D engine. `tests/js/sim.test.mjs` flies the trained Universal network in it on
every world and fails if fewer than 80% land (Python measured 87%; the port gets about
90%). To put another pilot in the game:

```bash
python export_web.py --pilot main --gen 16 --name universal --title "Universal pilot"
```

## Mission control: the web UI

```bash
python app.py                      # opens http://localhost:8000
```

A local web app for training, flying and inspecting pilots on 13 worlds. It needs no
extra Python packages. The page loads three.js (3D graphics) and its fonts from a CDN,
so keep an internet connection.

<img src="media/highlights/landing.gif" alt="A throttle pilot lands on the Moon 0.11 m from the pad center, then on Mars 0.02 m from it, while the flight log fills in" width="100%">

The page has five numbered sections: **01 Flight deck**, **02 How it learns**,
**03 Telemetry**, **04 Neurons** and **05 Generations**, described [below](#the-page).
With no pilots yet, it offers the pre-trained Universal pilot, so there is something to
fly and inspect before you train anything. **Fly it yourself vs this pilot** in the
flight deck opens [the game](#the-game-race-the-ai) against whichever generation you
are looking at.

### Worlds

| World | Gravity (m/s²) | Air and wind | Engine |
|---|---|---|---|
| Classic | 10 | none (the original Lunar Lander) | 1× |
| Mercury | 3.7 | none | 0.37× |
| Venus | 8.9 | very thick air, light wind | 1× |
| Earth | 9.8 | thin air, breeze | 1× |
| Moon | 1.6 | none | 0.34× |
| Mars | 3.7 | very thin air, dusty gusts | 0.37× |
| Jupiter | 23.1 | pad floats in the cloud tops; strong wind | 2.3× |
| Europa | 1.31 | none | 0.34× |
| Saturn | 9.0 | pad floats above the clouds; strong wind | 1× |
| Titan | 1.35 | thick air, light wind | 0.34× |
| Uranus | 8.7 | strong wind | 1× |
| Neptune | 11.0 | the strongest wind | 1.1× |
| Pluto | 0.7 | none | 0.34× |

Gravity is real (NASA's Planetary Fact Sheet; Titan's and Europa's come from their mass
and radius). Air drag, wind and turbulence are simplified to fit the simulator. Two
things had to change to make other worlds landable:

- **Engines sized like a real lander's.** The standard engine pushes 17.8 m/s². That
  can't even hover on Jupiter, and on the Moon it is 11 times gravity, so every burn is
  a violent kick. Each world's engine is about 1.8 times its gravity, and at least
  6 m/s² so it can still cancel the random push every flight starts with.
- **Landing legs rated for the world.** The simulator's leg springs are sized for
  10 m/s². On Jupiter they folded under the rocket's own weight, so even a perfect
  touchdown scraped the hull and counted as a crash. Above 10 m/s² the springs now get
  stronger with gravity.

Picking a world restyles everything: the dotted globe in the hero, and the flight
deck's sky, ground, weather and neighbors (Earth hangs over the Moon, Jupiter over
Europa, Saturn behind Titan's haze).

Here is a pilot trained only on Classic for 1 million frames, flown on every world with
no extra training (20 test flights each, new terrain):

| Lands easily (18–20 of 20) | Struggles |
|---|---|
| Classic 20, Earth 20, Mercury 19, Venus 19, Mars 19, Saturn 19, Uranus 19, Jupiter 18 | Neptune 12 (wind), Titan 9, Moon 8, Europa 8, Pluto 3 |

On the low-gravity worlds it floats, drifts and often leaves the screen. That is what
per-world training is for.

### The page

- **Worlds:** the selected world as a rotating dotted globe, with its facts. Arrows or
  the chips in the flight deck switch worlds.
- **Flight deck:** the 3D replay on the selected world. The HUD shows altitude, speed,
  tilt, main and side engine power, and how your reward function scored the frame. The
  flight trace below charts altitude, speed and engine power over the whole flight;
  click it to jump to a moment. The console sets the conditions, including the
  **start altitude**, and has **Launch flight**, **Run 20 test flights**, **Tour every
  world** (20 test flights on each) and **Train on this world**. While a pilot trains, its newest test flight plays here, and
  recent practice attempts appear as dotted trails.
- **Practice · Score · Learn:** three live cards: recent practice paths, the reward
  "potential" field with the rocket on it, and the hidden neurons firing.
- **Telemetry:** every practice attempt as a dot, how attempts end, where each one
  touches down, and test-flight scores, filterable by world. All update live.
- **Neurons:** the network in 3D. The glow on each connection is the signal flowing
  through it on the current frame. Click a neuron to see what it listens to and how it
  fires over the flight, or switch on **What if** to set the 8 sensor readings yourself
  and watch the decision change. **Brain health** counts neurons that are active, quiet
  (barely move) or saturated (stuck at their limit).
- **Generations:** every training run, filterable by world, with where it started from.
  Delete a generation from its row, a whole world's line, or every generation at once
  (see [Deleting generations](#deleting-generations)).

### Training: pick the world and how long

Every training run starts by asking four things:

- **World:** where to practice. The chips show how many generations the pilot already
  has on each world. **Universal** practises on all 13 at once (see below).
- **Start altitude:** how high above the pad each practice flight starts (see below).
- **Duration:** Quick (250k frames) to Extra long (5M), with time estimates based on
  your computer's measured training speed.
- **Start from:** each world has its own line of generations. Training on Mars continues
  the pilot's latest Mars generation (Mars #1, Mars #2, …). The first time a pilot
  trains on a world, it can bring another world's skills over (start from that world's
  latest generation) or begin with a blank network.

The world's physics, the reward weights and the learning settings sit in fold-out
sections below, each knob with a one-line explanation. **Keep retraining automatically**
chains generations on the same world until you press Stop. In the flight deck, each
world is flown by its own specialist (its latest generation), or by the newest
generation if the pilot hasn't trained there yet.

### Universal: one network for every world

Each world keeps its own line of generations. The **Universal** line adds one network
meant to land anywhere, trained in two steps:

1. **Study.** It first learns from the pilot's best generation on each world (most test
   landings): that teacher flies its world while every decision is recorded, then the new
   network flies and each frame it sees is labelled with what that world's teacher would
   have done, so it also learns to recover from its own mistakes. Three rounds of this
   take a few seconds per world. Every world counts equally.
2. **Practise.** Then it trains on every world at once: 26 simulators, two per world,
   each with that world's gravity, air, wind and engine. Its tests are 3 flights on each
   world (39 in all); hover over Landed in the Generations table for the per-world count.

The lines link both ways. The flight deck flies each world's own latest generation where
there is one and the Universal one everywhere else, and a world's first generation can
start from the Universal one ("bring every world's skills to Mars").

On a copy of the `main` pilot, which had lines on Classic, Jupiter, Saturn, the Moon,
Europa and a custom world, one Universal generation (studying those six, then 1M frames
of practice, 84 s in all) gave these results over 20 new flights per world:

| | Landed on the pad, all 13 worlds |
|---|---|
| Before: each world's own line, else the newest generation | 161 of 260 (62%) |
| Right after studying, before any practice (39 test flights) | 32 of 39 (82%) |
| **Universal, after practice** | **227 of 260 (87%)** |

The biggest gains were on worlds with no line of their own (Earth 2 → 19 of 20, Mars
2 → 18, Neptune 0 → 13). It also beat the Moon's own line (20 vs 15) and matched Classic,
Jupiter and Europa. It trailed the Saturn line (14 vs 20), and the Europa generation on
the other low-gravity worlds, Titan (13 vs 16) and Pluto (17 vs 20). The deck still flies
a world's own line when it has one; train that world from the Universal generation, or
give the Universal line a longer run, to close such gaps.

### Buttons or throttle

A new pilot can use **buttons** (each engine fully on or off) or a **throttle**: the
network outputs two numbers every frame, and the simulator turns them into main-engine
power (50–100% when it fires) and side-engine power. The pilot learns how much power to
use where. Three throttle pilots trained on Classic for 1.3 million frames each landed
20 of 20 test flights with scores of 282–286, against about 260–275 for button pilots,
because they spend less fuel. A typical throttle flight keeps the main engine on for
most of the descent at only 50–70% power: gentle, continuous burns instead of full-power
pulses. The HUD, flight trace, decision panel and 3D brain all show the throttle values.

### Start altitude: more room to land

The simulator always drops the rocket 10 m above the pad. **Start altitude**, in the
flight deck console and in every training run, moves the start anywhere from 5 to 40 m,
on any world. It belongs to the flight, not the world: Mars from 30 m is still in Mars's
line of generations, and the Generations table notes it under Settings.

- **How it works.** `StartHeight` in [`lander.py`](lander.py) lets the simulator build each
  flight as usual (terrain, random push), then lifts the rocket and its legs and runs one
  frame with the engines off, so the pilot's readings and the score start from there.
  Flights from 10 m are exactly as before.
- **More time on the clock.** A flight times out after 20 s from 10 m. Falling 4 times as
  far takes twice as long, so the limit grows with the square root of the height: 28 s
  from 20 m, 35 s from 30 m, 40 s from 40 m.
- **Scores grow with height.** The official score counts progress toward the pad, so a
  landing from 30 m scores about 560 instead of about 280, and "200 = solved" only means
  something from 10 m. Compare generations by landings and precision instead.
- **Precision.** Test flights now measure how far from the pad center each landing stopped
  and how fast the legs touched down. You'll find both in the console after **Run 20 test
  flights**, in the flight log, and in the Generations table's Precision column.

Does starting higher make better pilots? I trained button pilots on Classic for 2 million
frames from 10 m (2 training seeds) and from 30 m (3 seeds), then flew each 100 times from
every height:

| Trained from | Landed from 10 m | 20 m | 30 m | 40 m | Touchdown speed | Distance from center |
|---|---|---|---|---|---|---|
| 10 m | 98–99% | 97–100% | 87–90% | 78–79% | 2.2–2.5 m/s | 0.2–0.4 m |
| 30 m | 97–100% | 100% | 100% | 100% | 0.6–1.2 m/s | 0.3–0.4 m |

- **Softer, not closer.** Pilots trained from 30 m touch down about three times as softly
  and land from every height, even 40 m, which they never practiced. They stop no closer
  to the center. Pilots trained from 10 m learn to drop the last stretch: with more
  practice their touchdowns got harder (about 1 m/s after 1M frames, 2.4 m/s after 2M),
  since the landing still counts and coming in fast saves fuel and time.
- **A blank network needs longer from high up.** In 3 of the 4 runs from 30 m, test
  flights that had already been landing went through a stretch (somewhere between 0.8M
  and 1.2M frames) where every one hovered a few centimeters above the pad until time ran
  out, while practice flights still landed. Near the ground the network was nearly
  undecided between its four buttons (about 20–33% each). Test flights always take its
  top choice, which kept an engine firing; practice flights pick by those odds, so sooner
  or later they let it settle. The likely reason it takes longer: from 30 m, the last
  meter is a smaller share of every practice flight, so the network gets fewer chances
  to sharpen that decision. Every run got past it by about 1.3M frames, so train blank
  networks from high up for 2M frames (Long). Flights and tests in the deck use each
  generation's best-scoring network, so a run that ends mid-hover still flies well.
- **Low gravity cuts the other way.** On the Moon, extra height mostly means more time
  for the random push to carry the rocket sideways. Two Moon pilots, each continued from
  the same Classic pilot for 1M frames: the one trained from 10 m landed 63 of 100 flights
  from 10 m; the one trained from 30 m landed 49 of 100 from 30 m, and 38 drifted off
  screen.

### Deleting generations

The Generations table can remove runs you don't want. Each takes a second click to
confirm, and none works while that pilot is training:

- **Delete** on a row removes that generation: its networks, practice attempts, test
  flights and trails. Generations trained after it keep their own networks and still
  fly; their "Started from" says the parent was deleted.
- **Delete the Mars line** (it appears when you filter the table to a world) removes
  every generation trained on that world. The pilot's other worlds stay.
- **Start over** removes every generation but keeps the pilot's name. Its next training
  starts again at gen 1 from a blank network, and you can pick buttons or throttle and
  the network size again.
- **Delete the pilot** removes it and everything it learned from the pilot list.

Nothing is destroyed: deleted generations move to `runs/.trash/<pilot>/` (a deleted
pilot to `runs/.trash/`), with the networks and the removed rows of `episodes.csv` and
`tests.csv`, so you can put them back by hand. Empty `runs/.trash/` when you're sure.

### Where it saves things

Each pilot lives in `runs/<pilot>/`: `lab.json` (its generations and their settings),
`episodes.csv` (every practice attempt), `tests.csv`, `trails.json`, and a `gen-N/`
folder of saved networks per generation. Runs made with `train.py` are imported the
first time `app.py` starts (it flies 20 test flights to fill in the scoreboard). If you
train with `train.py` while `app.py` is running, restart `app.py`. `train.py` won't
overwrite a pilot that mission control has trained; give it another `--run` name.

## The four pieces

**1. The simulator.** [Gymnasium](https://gymnasium.farama.org)'s `LunarLander-v3`, a 2D
physics world (Box2D) with gravity, a main engine and two side engines. Lunar Lander
first shipped with OpenAI Gym; the Farama Foundation maintains it today. Each frame (50
per second) the rocket reports 8 numbers, and the pilot presses one of 4 buttons:

| The pilot sees | The pilot can press |
|---|---|
| position x, y (relative to the pad) | nothing |
| velocity vx, vy | left engine |
| tilt and spin | main engine |
| whether each leg touches the ground | right engine |

A flight ends when the rocket comes to rest (landed), its hull touches the ground
(crashed), it leaves the screen, or 1,000 frames pass (timed out).

**2. The brain.** A neural network with 8 inputs, two layers of 64 neurons, and 4
outputs: a probability for each button. A second network of the same shape, the
*critic*, learns to predict how many points are coming. That's 9,797 adjustable numbers,
created in one line by [Stable-Baselines3](https://stable-baselines3.readthedocs.io).

**3. The reward function.** [`reward.py`](reward.py), the only place you say what you
want. See the next section.

**4. The training.** PPO (Proximal Policy Optimization). Sixteen simulators fly at once.
Every 16,384 frames, PPO goes through every button press and asks: did more points
follow than the critic predicted? It makes the better-than-expected presses more likely
and the rest less likely, in small capped steps (the "proximal" part), then flies
again. One million frames is about 60 of these rounds, or about 4,000 landing attempts.

## Reward shaping, the key idea

Suppose you only said "+100 for landing, −100 for crashing." A random pilot almost never
lands (0 times in 100 flights), so it never sees the +100 and has no way to learn that
landing is even possible. The reward is too *sparse*.

Reward shaping adds small, frequent rewards that point toward the goal. Here's the
heart of `reward.py`:

```python
def potential(obs):
    """How good is this moment? Closer, slower, upright, legs down = higher."""
    x, y, vx, vy, angle, _spin, left_leg, right_leg = obs
    return (
        -DISTANCE_WEIGHT * np.hypot(x, y)
        - SPEED_WEIGHT * np.hypot(vx, vy)
        - TILT_WEIGHT * abs(angle)
        + LEG_CONTACT_BONUS * (left_leg + right_leg)
    )

reward = potential(obs) - potential(prev_obs)   # every frame
```

The rocket earns points for *progress*: getting closer to the pad, slowing down,
straightening up, putting its legs down. If it moves back the other way, it loses
exactly the points it gained. Over a whole flight, the progress points add up to "where it
ended" minus "where it started," so hovering near the pad can't farm them. This is
called potential-based reward shaping (Ng, Harada & Russell, 1999). It's a standard
technique because it guides learning while rewarding nothing except real progress.

The full scorecard:

| What happens | When | Points |
|---|---|---|
| Gets closer to the pad, slows down, or straightens up | every frame | 100 × the improvement (minus the same for getting worse) |
| A leg touches the ground | every frame | +10 per leg (−10 if it lifts off again) |
| Main engine fires | every frame | −0.3 × its power, for fuel |
| Side engine fires | every frame | −0.03 × its power, for fuel |
| Time passes | every frame | −0.05 (added by this project) |
| An engine fires while parked (both legs down, nearly stopped) | every frame | −0.3 (added by this project) |
| Comes to rest on its legs | once, at the end | +100 |
| Hull hits the ground, or flies off screen | once, at the end | −100 |

Every row not marked "added" is Lunar Lander's built-in reward. With the two additions
set to zero, `reward.py` reproduces the built-in reward to within 0.0001 points per
frame. Training logs and `evaluate.py` always report the official built-in score, so
your results stay comparable with everyone else's (200+ = solved) however you change
the reward.

## A real reward bug, found while building this

The first version of `reward.py` had no ground-engine cost. Its pilot aced the test
flights (99 of 100 on the pad), but during practice 83% of attempts timed out.

The cause: the rocket touched down on the pad and then kept tapping its side engines,
about one frame in three. A side engine costs only 0.03 points, so nothing
discouraged it. But the simulator only counts a landing once the rocket has been
almost perfectly still for half a second, so the +100 never arrived. A pilot that has
never collected that bonus has no reason to keep still.

One new line fixed it: −0.3 for firing any engine while parked (both legs down and
almost stopped). With the same random seed, practice landings at 1 million frames went
from 4% to 95%. Touching down stays free, so the pilot can still brake all the way to
the ground.

That's reward shaping in practice: watch what the agent actually does, find the
incentive that explains it, change the reward, retrain.

### A second one: the pilot that crashed on purpose

Throttle pilots (see mission control) brought a new surprise. They learned to land
cleanly in test flights, then got worse with more practice, until every test flight
ended with the hull slammed into the pad: a deliberate crash.

A throttle pilot explores by adding random noise to every engine setting, and that
noise doesn't switch off on the ground. Parked on the pad, it kept nudging the engines,
paid the parking cost again and again, and never came to rest, so the +100 never came.
For the rest of the episode that added up to more than the −100 crash penalty. The
cheapest way out the pilot could find was to end the flight by crashing.

Raising the penalty would only have moved the problem around. The fix is a rule real
landers use: the Apollo lunar module cut its engine on "contact light". For throttle
pilots, the engines now shut off automatically once the rocket is parked
(`TouchdownCutoff` in `lander.py`). It doesn't help them land; it only stops the noise
once they have. Three throttle pilots then landed 20 of 20 test flights each, and
practice landings rose from about 0% to over 90%.

## Reading the training output

```
                        practice: last 100 attempts     test: 20 flights
    frames  attempts    score  on pad  crashed  timeout     score  on pad
   114,688     1,198    -63.0      0%     100%       0%     -98.4    0/20  best
   311,296     2,169      9.1      0%      67%      33%    -130.3    0/20
   507,904     2,455    128.8     27%      28%      38%     227.9   20/20  best
   704,512     3,102    245.7     87%       6%       1%     213.3   19/20
 1,015,808     4,158    265.9     97%       3%       0%     257.5   20/20  best
```

During **practice** the agent picks buttons at random according to its probabilities
(say 90% nothing, 10% side engine), so it keeps trying new things. **Test** flights
always press the most likely button, which shows what it has actually learned. Every
time the test score beats the previous best, that pilot is saved as
`runs/main/best_model.zip`.

Training ends with a learning curve (`runs/main/learning_curve.png`). Here's the one
from this run. First it learns to crash more gently. Then it learns to touch down but
not to settle, so those flights time out. Finally it learns to switch off and land:

![Learning curve: score rises from -190 to 266 while crashes give way to landings](media/learning_curve.png)

## Experiments: break the reward on purpose

Each row below is one change to `reward.py`, trained for 1 million frames with the same
seed:

| Change in `reward.py` | Practice landings on pad | Test flights landed | What the rocket learned |
|---|---|---|---|
| none (the default) | 95% | 20/20 | to land |
| `GROUND_ENGINE_COST = 0.0` | 4% (83% time out) | 19/20 | to land, then fidget on the pad |
| `DISTANCE_WEIGHT`, `SPEED_WEIGHT`, `TILT_WEIGHT`, `LEG_CONTACT_BONUS` all `0.0` (no shaping) | 0% | 0/20 | engines off, free-fall into the ground |
| `reward = potential(obs)` (the state, not the change) | 0% | 0/20 | to crash as fast as possible |

- **No shaping:** the pilot never stumbles onto a landing, so the best outcome it knows
  is the cheapest crash: no fuel spent and no time wasted.
- **The state instead of the change:** the potential is negative while flying (far
  away, moving fast), so every frame in the air costs points. The quickest way to stop
  losing points is to hit the ground. Reward progress, not position.

To run one yourself, edit `reward.py`, then train under a new name so the main pilot
stays intact:

```bash
python train.py --run no-shaping
python watch.py --run no-shaping
```

Each run keeps a copy of the `reward.py` that trained it in `runs/<name>/reward.py`.
Copy `runs/main/reward.py` back over `reward.py` to undo your edits.

More ideas to try (these haven't been measured):

- **Wind:** in `lander.py`, change the `gym.make(...)` call to
  `gym.make(ENV_ID, render_mode=render_mode, enable_wind=True, wind_power=15.0)`.
- **Expensive fuel:** multiply `MAIN_ENGINE_COST` by 10 and watch the landing style change.
- **Precision:** pay the landing bonus only between the flags (the outcome is
  `"landed on pad"`).
- **More practice:** `python train.py --steps 3000000`.

## Files

| File | What it does |
|---|---|
| [`reward.py`](reward.py) | The reward function. The file you're meant to edit. |
| [`lander.py`](lander.py) | Plugs the reward into the simulator, adds other worlds' physics and start altitudes, labels how each flight ended (and how precisely it landed), and flies a pilot. |
| [`train.py`](train.py) | Trains a pilot with PPO and saves it in `runs/<name>/`. |
| [`watch.py`](watch.py) | Watches a pilot in a window or saves a GIF (`--random` for an untrained one). |
| [`evaluate.py`](evaluate.py) | Scores a pilot on 100 flights. |
| [`plot.py`](plot.py) | Redraws a run's learning curve, even while it's still training. |
| [`app.py`](app.py) | The mission control web server: serves the UI and streams live training events. |
| [`lab.py`](lab.py) | The worlds, pilots and their generations, recorded flights, network weights, and the training process behind the UI. |
| [`ui/`](ui/) | Mission control: [`stage.js`](ui/stage.js) (3D flights), [`worlds.js`](ui/worlds.js) (planet globes and looks), [`brain.js`](ui/brain.js) (3D network), [`charts.js`](ui/charts.js), [`nn.js`](ui/nn.js) (the network and reward, in JavaScript), [`app.js`](ui/app.js) (wiring). |
| [`ui/play.html`](ui/play.html), [`ui/play.js`](ui/play.js) | The game: race the AI in the browser. |
| [`ui/sim.js`](ui/sim.js) | The simulator and world wrappers ported to the browser on planck.js. |
| [`ui/lander.js`](ui/lander.js) | The 3D lander: hull, legs that follow the physics, the engine plume and thrusters. |
| [`ui/sound.js`](ui/sound.js), [`ui/annunciator.js`](ui/annunciator.js) | Engine sound (Web Audio, generated) and the cockpit lamps and callouts over the 3D views. |
| [`ui/figures/`](ui/figures/) | Hand-drawn interactive figures made with [Colorfigures](https://april-zhu.com/colorfigures): the lander you can fly and see inside. Sources in `ui/figures/src/`. |
| [`ui/specs.js`](ui/specs.js) | The per-world spec sheet (thrust to weight, hover throttle) from the simulator's numbers. |
| [`ui/ppo.js`](ui/ppo.js), [`ui/train-worker.js`](ui/train-worker.js) | PPO in plain JavaScript, and the background thread that trains with it. |
| [`ui/trainer-ui.js`](ui/trainer-ui.js), [`ui/my-ais.js`](ui/my-ais.js) | The "Train your AI" section, and pilots saved in the browser (IndexedDB). |
| [`ui/vendor/planck.mjs`](ui/vendor/planck.mjs) | planck.js 1.5.0 (MIT), the browser's physics engine, vendored so workers and tests can load it. |
| [`ui/data/`](ui/data/) | The game's worlds and pilots (network weights as JSON), written by `export_web.py`. |
| [`export_web.py`](export_web.py) | Puts a trained pilot into the game. |
| [`pretrained/`](pretrained/) | The Universal pilot (150 KB), for a first visit to mission control. |
| [`tests/`](tests/) | pytest for the reward, simulator, lab and server; `tests/js/` checks the browser simulator. |
| [`media/`](media/) | The tour video and README GIFs, recorded from the running app, plus the `watch.py` GIFs. |

Each `runs/<name>/` folder made by `train.py` holds `best_model.zip`, `final_model.zip`,
the training logs, the learning curve, and the `reward.py` snapshot.

## Tests

```bash
pip install pytest && pytest -q                   # 35 tests, about 5 s
node --test tests/js/*.test.mjs                   # the browser port still lands like the original
```

GitHub Actions runs both on every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)),
and [`pages.yml`](.github/workflows/pages.yml) publishes the game to GitHub Pages.

## Ideas and roadmap

- **Describe a landing style in plain English and let an LLM write `reward.py`**, then
  train it and compare, in the spirit of NVIDIA's [Eureka](https://arxiv.org/abs/2310.12931),
  with a gallery of rewards that the pilot learned to game.
- **A held-out world:** train the Universal pilot on 12 worlds and test it on the 13th.
- **Sensor faults:** a late or noisy altimeter, like the one behind ispace's 2025 lunar
  crash, to see how robust each pilot is.
- **More algorithms:** DQN and SAC beside PPO, on the same worlds and tests.
- Publish pilots to the Hugging Face Hub with a model card.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Built on

- [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/)'s Lunar Lander, [Stable-Baselines3](https://stable-baselines3.readthedocs.io)' PPO and [PyTorch](https://pytorch.org)
- [three.js](https://threejs.org) for 3D, [planck.js](https://piqnt.com/planck.js/) for physics in the browser
- The hand-drawn figures are made with [Colorfigures](https://april-zhu.com/colorfigures) by April Zhu (MIT; its engine and license are in [`ui/figures/`](ui/figures/))
- Ideas: PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)), potential-based reward shaping ([Ng, Harada & Russell, 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)), policy distillation ([Rusu et al., 2015](https://arxiv.org/abs/1511.06295)) and DAgger ([Ross, Gordon & Bagnell, 2011](https://arxiv.org/abs/1011.0686))
- Planet gravity from NASA's [Planetary Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/)

PerfectLand is MIT-licensed ([LICENSE](LICENSE)). If you use it in a class or a paper,
[CITATION.cff](CITATION.cff) has the details.

## Troubleshooting

- **Old tutorials use `import gym` and `LunarLander-v2`.** This project uses their
  successors, `import gymnasium` and `LunarLander-v3`.
- **Box2D won't install:** use Python 3.10 to 3.13, which get a prebuilt Box2D. On
  3.14 it compiles from source and needs a C++ compiler (`xcode-select --install` on macOS).
- **No window opens** (for example on a server): use `python watch.py --gif flight.gif`.
- **Port 8000 is taken:** `python app.py --port 9000`.
- **The game's keys scroll the page:** click the game view once, or press Launch first;
  while a round is flying, the arrow keys and space only fly the rocket.
- **The mission control page is blank:** it loads three.js from cdn.jsdelivr.net, so
  check your internet connection.
- **Training is slow:** everything runs on the CPU, because a network this small trains
  fastest there. Time scales with your processor; `--envs` sets how many simulators fly
  at once.
- **The server stopped in the middle of training** (closed, crashed, or the computer
  slept): the next start marks that generation stopped and keeps its best-tested
  network, so you can still fly it and continue from it. For long runs, start
  `app.py` from a terminal you keep open.
- **Practice lands but every test flight times out:** the pilot is hovering just above
  the pad (see [Start altitude](#start-altitude-more-room-to-land)). Keep training, or
  continue from the generation's best test (**Which version** in the training panel).
