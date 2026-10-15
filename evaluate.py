"""Score a pilot over many flights, without graphics (takes seconds).

    python evaluate.py              # 100 flights with the trained pilot
    python evaluate.py --random     # the same 100 flights, pressing random buttons
"""

import argparse
from collections import Counter

import numpy as np

from lander import FPS, OUTCOMES, fly, load_pilot, make_flight_env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="main", help="folder under runs/ to load the pilot from (default: main)")
    parser.add_argument("--random", action="store_true", help="press random buttons instead of using a trained pilot")
    parser.add_argument("--episodes", type=int, default=100, help="number of flights (default: 100)")
    parser.add_argument("--seed", type=int, default=1000, help="terrain seed for the first flight (default: 1000)")
    args = parser.parse_args()

    pilot = None if args.random else load_pilot(args.run)
    env = make_flight_env()
    results = [fly(env, pilot, seed=args.seed + i) for i in range(args.episodes)]
    env.close()

    outcomes = Counter(outcome for outcome, _, _ in results)
    scores = np.array([score for _, score, _ in results])
    seconds = np.array([steps for _, _, steps in results]) / FPS
    print(f"\n{args.episodes} flights by {'a random pilot' if args.random else 'the trained pilot'}:\n")
    for outcome in OUTCOMES:
        share = outcomes[outcome] / args.episodes
        print(f"  {outcome:<16} {share:>5.0%}  {'█' * round(share * 40)}")
    print(f"\n  official score   {scores.mean():.1f} average (200+ = solved), "
          f"worst {scores.min():.1f}, best {scores.max():.1f}")
    print(f"  flight time      {seconds.mean():.1f} s average")


if __name__ == "__main__":
    main()
