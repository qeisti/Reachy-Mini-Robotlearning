"""Verbindet Quelle -> Wahrnehmung -> Ereignisse -> Policy -> Motion und misst.

Die Policy laeuft in einem eigenen Worker-Thread. Wahrnehmung und Tracking
laufen dadurch immer in Kamerarate weiter, auch wenn der Agent nachdenkt.
Ereignisse, die waehrend einer laufenden Entscheidung eintreffen, warten in
einer FIFO-Queue (die Wartezeit wird als t_dispatch - t_event gemessen).
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .actions import Action
from .metrics import EventRecord, MetricsLogger, ResourceSampler
from .motion import MotionController, Pose
from .policies import Policy
from .robot import RobotBackend
from .state import Event, EventDetector


@dataclass
class RunConfig:
    source: str                      # "webcam:0" | "reachy" | pfad.mp4 | pfad.json
    run_idx: int = 0
    duration: float | None = None    # nur Webcam/Reachy: Laufzeit in s
    gestures: bool = True
    speech: bool = False
    display: bool = True
    drain_timeout: float = 60.0
    seed: int | None = None


class PolicyWorker(threading.Thread):
    def __init__(self, policy: Policy, motion: MotionController, logger: MetricsLogger,
                 run_id: str, run_idx: int) -> None:
        super().__init__(daemon=True, name="policy")
        self.policy, self.motion, self.logger = policy, motion, logger
        self.run_id, self.run_idx = run_id, run_idx
        self.events: queue.Queue[Event | None] = queue.Queue()
        self.busy = threading.Event()
        self.model = policy.describe().get("model", "")

    def submit(self, ev: Event) -> None:
        self.events.put(ev)

    def run(self) -> None:
        while True:
            ev = self.events.get()
            if ev is None:
                return
            self.busy.set()
            try:
                self._handle(ev)
            finally:
                self.busy.clear()
                self.events.task_done()

    def _handle(self, ev: Event) -> None:
        rec = EventRecord(self.run_id, self.policy.name, self.model, self.run_idx, ev.id, ev.type,
                          json.dumps(ev.data, ensure_ascii=False), ev.video_time, ev.t_frame, ev.t_event,
                          queue_len=self.events.qsize())
        actions: list[Action] = []

        def on_started(t: float, rec=rec) -> None:
            if rec.t_motor is None:
                rec.t_motor = t

        def emit(action: Action) -> None:
            if rec.t_decision is None:
                rec.t_decision = time.perf_counter()
            actions.append(action)
            # nur die erste Aktion misst t3; weitere werden trotzdem ausgefuehrt
            self.motion.submit(action, on_started if len(actions) == 1 else None)

        rec.t_dispatch = time.perf_counter()
        info = self.policy.decide(ev, emit)
        rec.t_done = time.perf_counter()
        rec.actions = "|".join(a.name for a in actions)
        rec.n_llm_calls, rec.n_tool_calls, rec.n_invalid = info.n_llm_calls, info.n_tool_calls, info.n_invalid
        rec.tokens_in, rec.tokens_out = info.tokens_in, info.tokens_out
        rec.error, rec.route = info.error or "", info.route
        self.logger.add(rec)
        print(f"  [{self.policy.name}] {ev.type:<17} -> {rec.actions or '-':<12} "
              f"{(rec.t_done - rec.t_dispatch) * 1000:7.1f} ms"
              + (f"  FEHLER {info.error}" if info.error else ""))


def run_once(policy: Policy, robot: RobotBackend, cfg: RunConfig, out_dir: Path,
             perception=None) -> Path:
    """Ein Durchlauf einer Policy ueber eine Quelle. Schreibt events.csv,
    resources.csv und meta.json nach ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run_id = f"{policy.name}_run{cfg.run_idx:02d}"
    logger = MetricsLogger(out_dir, t0)
    sampler = ResourceSampler(out_dir / "resources.csv", t0)
    motion = MotionController(robot, random.Random(cfg.seed))
    motion.posture = Pose()
    policy.history.clear()
    worker = PolicyWorker(policy, motion, logger, run_id, cfg.run_idx)

    sampler.start()
    motion.start()
    worker.start()
    n_frames = 0
    try:
        if cfg.source.lower().endswith(".json"):
            _run_scenario(cfg, worker)
        else:
            n_frames = _run_camera(cfg, worker, motion, robot, perception)
    except KeyboardInterrupt:
        print("abgebrochen")
    finally:
        _drain(worker, cfg.drain_timeout)
        time.sleep(0.3)  # letzter Aktion Zeit fuer den ersten Motorbefehl geben
        worker.events.put(None)
        motion.stop()
        sampler.stop()
    meta = {**policy.describe(), "run_id": run_id, "run_idx": cfg.run_idx, "source": cfg.source,
            "frames": n_frames, "duration_s": round(time.perf_counter() - t0, 2),
            "n_events": len(logger.records), "motion_ticks": motion.n_ticks,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    logger.write(meta)
    return out_dir


def _drain(worker: PolicyWorker, timeout: float) -> None:
    end = time.perf_counter() + timeout
    while (not worker.events.empty() or worker.busy.is_set()) and time.perf_counter() < end:
        time.sleep(0.05)


def _run_scenario(cfg: RunConfig, worker: PolicyWorker) -> None:
    from .sources import load_scenario

    start = time.perf_counter()
    for i, e in enumerate(load_scenario(cfg.source), start=1):
        delay = e["t"] - (time.perf_counter() - start)
        if delay > 0:
            time.sleep(delay)
        now = time.perf_counter()
        data = e.get("data", {})
        state = {"face_present": e["type"] != "person_left", "distance": data.get("distance"),
                 "emotion": data.get("emotion", "neutral"), **e.get("state", {})}
        worker.submit(Event(i, e["type"], now, now, e["t"], data, state))


def _run_camera(cfg: RunConfig, worker: PolicyWorker, motion: MotionController,
                robot: RobotBackend, perception) -> int:
    from . import sources

    if perception is None:
        from .perception import Perception
        perception = Perception(gestures=cfg.gestures, speech=cfg.speech)

    if cfg.source.startswith("webcam"):
        idx = int(cfg.source.split(":")[1]) if ":" in cfg.source else 0
        frames = sources.webcam_frames(idx)
    elif cfg.source == "reachy":
        frames = sources.reachy_frames(robot)
    else:
        frames = sources.video_frames(cfg.source)

    detector = EventDetector()
    start = time.perf_counter()
    n = 0
    cv2 = None
    if cfg.display:
        import cv2
    for fr in frames:
        n += 1
        st = perception.process(fr.image, fr.t_frame, fr.video_time)
        if st.face_present:
            motion.set_tracking_target(st.target_yaw, st.target_pitch)
        for ev in detector.update(st):
            if ev.type == "person_left":
                motion.set_tracking_target(0.0, 0.0)
            worker.submit(ev)
        if cv2 is not None:
            _draw(cv2, fr.image, st, perception, motion, worker)
            cv2.imshow("Reachy HRI", fr.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if cfg.duration and time.perf_counter() - start > cfg.duration:
            break
    if cv2 is not None:
        cv2.destroyAllWindows()
    return n


def _draw(cv2, img, st, perception, motion, worker) -> None:
    h, w = img.shape[:2]
    rect = getattr(perception.face, "last_rect", None)
    if rect:
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)
    lines = [
        f"Policy: {worker.policy.name}   Aktion: {motion.current_action}",
        f"Gesicht: {st.emotion} ({st.emotion_score:.2f})  Distanz: {st.distance or '-'}",
        f"Geste: {st.gesture or '-'}   Queue: {worker.events.qsize()}"
        + ("   [denkt...]" if worker.busy.is_set() else ""),
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
