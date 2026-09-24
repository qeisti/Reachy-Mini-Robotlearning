"""Motion-Layer: Kopf folgt dem Gesicht und spielt Ausdruecke nicht-blockierend ab.

Laeuft in einem eigenen Thread mit fester Rate (``config.MOTION_RATE_HZ``) und
setzt bei jedem Tick per ``set_target`` die Zielpose (wie in
``legacy/facedetection3.py``). Ausdruecke werden als Keyframe-Folge mit
Minimum-Jerk-Interpolation ueberlagert. Dadurch

* laeuft das Tracking immer weiter, egal wie lange die Policy nachdenkt, und
* wartet keine Policy auf eine Bewegung (``submit`` kehrt sofort zurueck).

Fuer die Messung meldet der Motion-Layer ueber ``on_started`` den Zeitpunkt,
an dem der erste Motorbefehl einer neuen Aktion abgeschickt wurde (t3).
"""

from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import config
from .actions import NO_ACTION, Action, pick_variant
from .robot import RobotBackend

POSE_KEYS = ("yaw", "pitch", "roll", "z")
NEUTRAL_ANTENNAS = (-30.0, 30.0)


def minjerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass
class Pose:
    """Ausdrucks-Anteil der Pose (wird zum Tracking addiert)."""

    antennas: tuple[float, float] = NEUTRAL_ANTENNAS
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    z: float = 0.0
    tracking: float = 1.0

    @staticmethod
    def from_keyframe(kf: dict, intensity: float, tracking: float, base: "Pose") -> "Pose":
        ant = kf.get("antennas")
        if ant is None:
            ant = base.antennas
        else:
            ant = tuple(n + intensity * (a - n) for a, n in zip(ant, NEUTRAL_ANTENNAS))
        head = kf.get("head", {})
        return Pose(ant, *(intensity * head.get(k, 0.0) for k in POSE_KEYS), tracking)

    def lerp(self, other: "Pose", s: float) -> "Pose":
        a = tuple(x + (y - x) * s for x, y in zip(self.antennas, other.antennas))
        vals = [getattr(self, k) + (getattr(other, k) - getattr(self, k)) * s for k in POSE_KEYS]
        tr = self.tracking + (other.tracking - self.tracking) * s
        return Pose(a, *vals, tr)


@dataclass
class _Playback:
    action: Action
    keyframes: list[Pose]
    step_s: float
    start_pose: Pose
    t_start: float
    persistent: bool
    on_started: Callable[[float], None] | None
    announced: bool = False


class MotionController:
    def __init__(self, robot: RobotBackend, rng: random.Random | None = None) -> None:
        self.robot = robot
        self.rng = rng or random.Random()
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._target = (0.0, 0.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="motion", daemon=True)
        self.posture = Pose()          # persistente Grundhaltung (letzte Emotion)
        self.current = Pose()          # aktuell kommandierter Ausdrucks-Anteil
        self.current_action = "neutral"
        self.n_ticks = 0

    # --- API ------------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def set_tracking_target(self, yaw: float, pitch: float) -> None:
        with self._lock:
            self._target = (yaw, pitch)

    def submit(self, action: Action, on_started: Callable[[float], None] | None = None) -> None:
        """Nicht-blockierend. Eine neue Aktion unterbricht eine laufende."""
        self._queue.put((action, on_started))

    # --- Thread ---------------------------------------------------------------
    def _start_playback(self, action: Action, on_started, now: float) -> _Playback | None:
        if action.name == NO_ACTION:
            if on_started:
                on_started(now)
            return None
        variant = pick_variant(action.name, self.rng)
        tracking = variant.get("tracking", 1.0)
        persistent = variant.get("persistent", True)
        base = self.posture
        kfs = [Pose.from_keyframe(kf, action.intensity, tracking if persistent else base.tracking, base)
               for kf in variant["keyframes"]]
        if not persistent:
            kfs.append(base)  # zurueck zur Grundhaltung
        self.current_action = action.name
        return _Playback(action, kfs, variant["step_s"], self.current, now, persistent, on_started)

    def _evaluate(self, pb: _Playback, now: float) -> tuple[Pose, bool]:
        tau = now - pb.t_start
        k = int(tau // pb.step_s)
        if k >= len(pb.keyframes):
            return pb.keyframes[-1], True
        prev = pb.start_pose if k == 0 else pb.keyframes[k - 1]
        s = minjerk((tau - k * pb.step_s) / pb.step_s)
        return prev.lerp(pb.keyframes[k], s), False

    def _loop(self) -> None:
        period = 1.0 / config.MOTION_RATE_HZ
        sm_yaw = sm_pitch = body_yaw = 0.0
        playback: _Playback | None = None
        while not self._stop.is_set():
            tick = time.perf_counter()
            try:
                while True:  # nur die neueste Aktion zaehlt
                    action, cb = self._queue.get_nowait()
                    pb = self._start_playback(action, cb, tick)
                    if action.name != NO_ACTION:
                        playback = pb
            except queue.Empty:
                pass

            with self._lock:
                yaw_t, pitch_t = self._target
            sm_yaw += (yaw_t - sm_yaw) * config.SMOOTHING_ALPHA
            sm_pitch += (pitch_t - sm_pitch) * config.SMOOTHING_ALPHA

            if playback is not None:
                self.current, done = self._evaluate(playback, tick)
                if done:
                    if playback.persistent:
                        self.posture = self.current
                    playback = None
            else:
                self.current = self.posture

            p = self.current
            yaw = p.tracking * sm_yaw + p.yaw
            pitch = p.tracking * sm_pitch + p.pitch
            body_yaw += (yaw - body_yaw) * config.BODY_FOLLOW_ALPHA
            self.robot.set_target(yaw, pitch, p.roll, p.z, p.antennas, body_yaw)
            self.n_ticks += 1

            if playback is not None and not playback.announced:
                playback.announced = True
                if playback.on_started:
                    playback.on_started(time.perf_counter())

            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
