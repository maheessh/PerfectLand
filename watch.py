"""Watch a pilot fly.

    python watch.py                     # 3 flights in a window
    python watch.py --gif landing.gif   # save them as an animated GIF instead
    python watch.py --random            # a pilot that presses random buttons
    python watch.py --run my-reward     # the pilot trained in runs/my-reward/
"""

import argparse

from PIL import Image

from lander import FPS, fly, load_pilot, make_flight_env


def save_gif(flights, path):
    """Write flights to a GIF at 25 fps, holding the last frame of each for a second."""
    images, durations = [], []
    for frames in flights:
        for frame in frames[::2]:
            images.append(Image.fromarray(frame).resize((450, 300), Image.Resampling.LANCZOS))
            durations.append(2 * 1000 // FPS)
        durations[-1] = 1000
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="main", help="folder under runs/ to load the pilot from (default: main)")
    parser.add_argument("--random", action="store_true", help="press random buttons instead of using a trained pilot")
    parser.add_argument("--episodes", type=int, default=3, help="number of flights (default: 3)")
    parser.add_argument("--gif", metavar="PATH", help="save the flights as an animated GIF instead of opening a window")
    parser.add_argument("--seed", type=int, help="terrain seed for the first flight (default: random)")
    args = parser.parse_args()

    pilot = None if args.random else load_pilot(args.run)
    env = make_flight_env(render_mode="rgb_array" if args.gif else "human")
    flights = []
    try:
        for i in range(args.episodes):
            frames = [] if args.gif else None
            seed = None if args.seed is None else args.seed + i
            outcome, score, steps = fly(env, pilot, seed=seed, frames=frames)
            print(f"Flight {i + 1}: {outcome:<16} score {score:6.1f}   {steps / FPS:4.1f} s")
            if frames:
                flights.append(frames)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    if flights:
        save_gif(flights, args.gif)
        print(f"Saved {args.gif}")


if __name__ == "__main__":
    main()
