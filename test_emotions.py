import time

from reachy_mini import ReachyMini

from emotions import EMOTIONS, play_emotion

with ReachyMini(media_backend="no_media") as mini:
    for name in EMOTIONS:
        print(f"Playing: {name}")
        play_emotion(mini, name)
        time.sleep(2.0)
