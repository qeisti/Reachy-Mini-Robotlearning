"""Eingabequellen: Webcam, Stimulus-Video (Echtzeit-Replay), Reachy-Kamera,
oder ein Ereignis-Skript (ohne Kamera/Wahrnehmung, fuer reine Entscheidungstests).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Frame:
    image: object
    t_frame: float
    video_time: float | None


def webcam_frames(index: int = 0) -> Iterator[Frame]:
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam {index} laesst sich nicht oeffnen (Index in --source aendern).")
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield Frame(img, time.perf_counter(), None)
    finally:
        cap.release()


def video_frames(path: str | Path) -> Iterator[Frame]:
    """Spielt ein Video in ECHTZEIT ab: ist die Verarbeitung langsamer als die
    Bildrate, werden Frames uebersprungen – wie bei einer echten Kamera. So
    bleibt der Stimulus fuer alle Konfigurationen zeitlich identisch."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video {path} laesst sich nicht oeffnen.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    start = time.perf_counter()
    idx = -1
    try:
        while True:
            target = int((time.perf_counter() - start) * fps)
            if target >= n_total:
                break
            if target <= idx:  # zu schnell -> auf naechsten Frame warten
                time.sleep((idx + 1) / fps - (time.perf_counter() - start))
                continue
            while idx < target - 1:  # zu langsam -> Frames verwerfen
                if not cap.grab():
                    return
                idx += 1
            ok, img = cap.read()
            if not ok:
                break
            idx += 1
            yield Frame(img, time.perf_counter(), idx / fps)
    finally:
        cap.release()


def reachy_frames(robot) -> Iterator[Frame]:
    while True:
        img = robot.get_frame()
        if img is None:
            time.sleep(0.005)
            continue
        yield Frame(img, time.perf_counter(), None)


def load_scenario(path: str | Path) -> list[dict]:
    """Ereignis-Skript: {"events": [{"t": 1.0, "type": "person_appeared",
    "data": {...}, "state": {...}}, ...]}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    events = data["events"] if isinstance(data, dict) else data
    return sorted(events, key=lambda e: e["t"])
