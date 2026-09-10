import argparse
import time

from reachy_mini import ReachyMini

from emotions import EMOTIONS, play_emotion

parser = argparse.ArgumentParser(description="Play robot emotions for testing.")
parser.add_argument(
    "emotion",
    nargs="?",
    choices=sorted(EMOTIONS),
    help="Name of a single emotion to play. Omit to play all of them.",
)
args = parser.parse_args()

names = [args.emotion] if args.emotion else list(EMOTIONS)

with ReachyMini(media_backend="no_media") as mini:
    for name in names:
        print(f"Playing: {name}")
        play_emotion(mini, name)
        time.sleep(2.0)
