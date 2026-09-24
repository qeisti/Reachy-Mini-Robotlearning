"""Roboter-Backends: echter Reachy Mini (Hardware oder MuJoCo-Sim) oder Mock.

Der Mock erlaubt, Policies und Metriken ohne Daemon zu testen (z. B. auf einem
Laptop ohne Simulation oder in CI).
"""

from __future__ import annotations

import time

import numpy as np


class RobotBackend:
    """Minimale Schnittstelle, die der Motion-Layer braucht."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def set_target(self, yaw: float, pitch: float, roll: float, z_mm: float,
                   antennas_deg: tuple[float, float], body_yaw_deg: float) -> None:
        raise NotImplementedError

    def get_frame(self):
        raise NotImplementedError("Dieses Backend liefert keine Kamerabilder.")


class ReachyBackend(RobotBackend):
    """Verbindet sich mit dem laufenden ``reachy-mini-daemon`` (mit oder ohne --sim)."""

    def __init__(self, use_camera: bool = False) -> None:
        self.use_camera = use_camera
        self._ctx = None
        self.mini = None

    def open(self) -> None:
        from reachy_mini import ReachyMini

        from . import config

        media = config.REACHY_MEDIA_BACKEND if self.use_camera else "no_media"
        self._ctx = ReachyMini(media_backend=media, automatic_body_yaw=False)
        self.mini = self._ctx.__enter__()

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = self.mini = None

    def set_target(self, yaw, pitch, roll, z_mm, antennas_deg, body_yaw_deg) -> None:
        from reachy_mini.utils import create_head_pose

        pose = create_head_pose(z=z_mm, mm=True, yaw=yaw, pitch=pitch, roll=roll, degrees=True)
        self.mini.set_target(
            head=pose,
            antennas=np.deg2rad(antennas_deg),
            body_yaw=np.deg2rad(body_yaw_deg),
        )

    def get_frame(self):
        return self.mini.media.get_frame()


class MockBackend(RobotBackend):
    """Tut nichts ausser mitzuzaehlen – fuer Tests und Trockenlaeufe."""

    def __init__(self) -> None:
        self.n_commands = 0
        self.last: dict | None = None
        self.t_last = 0.0

    def set_target(self, yaw, pitch, roll, z_mm, antennas_deg, body_yaw_deg) -> None:
        self.n_commands += 1
        self.t_last = time.perf_counter()
        self.last = dict(yaw=yaw, pitch=pitch, roll=roll, z_mm=z_mm,
                         antennas=tuple(antennas_deg), body_yaw=body_yaw_deg)


def make_backend(name: str, use_camera: bool = False) -> RobotBackend:
    if name == "reachy":
        return ReachyBackend(use_camera=use_camera)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"Unbekanntes Roboter-Backend: {name}")
