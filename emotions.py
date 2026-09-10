import random

import numpy as np

from reachy_mini.utils import create_head_pose

# Each emotion is a list of pose variants (head pose kwargs, antenna angles in
# degrees, movement duration). play_emotion() picks one at random so the same
# emotion doesn't look identical every time it plays. Add more variants or
# emotions here - this file is the single place other scripts pull poses from.
EMOTIONS = {
    "neutral": [
        {"head": {"yaw": 0, "pitch": 0, "z": 0, "mm": True}, "antennas": [-30, 30], "duration": 1.0},
    ],
    "freude": [
        {"head": {"yaw": 15, "pitch": -10, "z": 10, "mm": True}, "antennas": [-90, 90], "duration": 0.6},
        {"head": {"yaw": -15, "pitch": -10, "z": 10, "mm": True}, "antennas": [-90, 90], "duration": 0.6},
        {"head": {"yaw": 0, "pitch": -15, "z": 15, "mm": True}, "antennas": [-60, 60], "duration": 0.6},
    ],
    "angst": [
        {"head": {"yaw": 0, "pitch": 10, "z": -30, "mm": True}, "antennas": [-180, 180], "duration": 1.0},
    ],
    "neugierig": [
        {"head": {"yaw": 20, "pitch": -5, "z": 0, "mm": True}, "antennas": [-120, 110], "duration": 1.2},
        {"head": {"yaw": -20, "pitch": -5, "z": 0, "mm": True}, "antennas": [-120, 110], "duration": 1.2},
    ],
    "traurig": [
        {"head": {"yaw": 0, "pitch": 20, "z": -15, "mm": True}, "antennas": [-10, 10], "duration": 1.5},
    ],
    "vorsichtig": [
        {"head": {"yaw": 0, "pitch": 0, "z": 0, "mm": True}, "antennas": [-120, 110], "duration": 1.0},
    ],
    "verwirrt": [
        {"head": {"yaw": 20, "pitch": -10, "z": 5, "mm": True}, "antennas": [-90, 30], "duration": 0.5},
        {"head": {"yaw": -20, "pitch": -10, "z": 5, "mm": True}, "antennas": [30, -90], "duration": 0.5},
    ],
    # Wave: keeps following the tracked face (head yaw/pitch come from the
    # detection) while both antennas flap 0 -> -90 -> 0 twice. Uses
    # "antenna_sequence" instead of "antennas" so play_emotion runs the
    # keyframes in order rather than as a single move.
    "winken": [
        {
            "head": {"yaw": 0, "pitch": -5, "z": 5, "mm": True},
            "antenna_sequence": [[0, 0], [-90, 90], [0, 0], [-90, 90], [0, 0]],
            "duration": 0.25,
        },
    ],
}


# Emotions that keep their own fixed head yaw/pitch instead of following the
# center of the detected face box.
IGNORE_FACE_POSITION = {"traurig"}


def play_emotion(mini, name: str, yaw: float | None = None, pitch: float | None = None) -> None:
    variants = EMOTIONS.get(name)
    if not variants:
        raise KeyError(f"Unknown emotion: {name!r}. Available: {sorted(EMOTIONS)}")

    variant = random.choice(variants)
    head = dict(variant["head"])
    if name not in IGNORE_FACE_POSITION:
        if yaw is not None:
            head["yaw"] = yaw
        if pitch is not None:
            head["pitch"] = pitch

    pose = create_head_pose(degrees=True, **head)

    # Multi-step emotions (e.g. winken) flap the antennas through a list of
    # keyframes. goto_target blocks until each move finishes, so a plain loop
    # plays the sequence. The head pose stays fixed (still following the face).
    if "antenna_sequence" in variant:
        for antennas in variant["antenna_sequence"]:
            mini.goto_target(
                head=pose,
                antennas=np.deg2rad(antennas),
                duration=variant["duration"],
                method="minjerk",
            )
        return

    mini.goto_target(
        head=pose,
        antennas=np.deg2rad(variant["antennas"]),
        duration=variant["duration"],
        method="minjerk",
    )
