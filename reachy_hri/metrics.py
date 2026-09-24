"""Messung: Zeitstempel pro Ereignis und Ressourcenverbrauch.

Zeitpunkte (alle ``time.perf_counter()``, relativ zum Start des Laufs):

    t_frame     Frame eingelesen, in dem das Ereignis erkannt wurde   (t0)
    t_event     Ereignis vom EventDetector gemeldet
    t_dispatch  Policy beginnt mit dem Ereignis (danach Warteschlange)  (t1)
    t_decision  erste Aktion ausgegeben                                 (t2)
    t_motor     erster Motorbefehl dieser Aktion gesendet               (t3)
    t_done      Policy fertig (inkl. evtl. weiterer LLM-Schritte)

Abgeleitet (in ``compare.py``): Entscheidungslatenz t2-t1, Wartezeit t1-t_event,
Ende-zu-Ende t3-t0.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

EVENT_FIELDS = [
    "run_id", "policy", "model", "run_idx", "event_id", "event_type", "event_data", "video_time",
    "t_frame", "t_event", "t_dispatch", "t_decision", "t_motor", "t_done",
    "actions", "n_llm_calls", "n_tool_calls", "n_invalid", "tokens_in", "tokens_out",
    "error", "route", "queue_len",
]


@dataclass
class EventRecord:
    run_id: str
    policy: str
    model: str
    run_idx: int
    event_id: int
    event_type: str
    event_data: str
    video_time: float | None
    t_frame: float
    t_event: float
    t_dispatch: float | None = None
    t_decision: float | None = None
    t_motor: float | None = None
    t_done: float | None = None
    actions: str = ""
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    n_invalid: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    route: str = ""
    queue_len: int = 0


class MetricsLogger:
    def __init__(self, out_dir: Path, t0: float) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = t0
        self.records: list[EventRecord] = []
        self.lock = threading.Lock()

    def rel(self, t: float | None) -> float | None:
        return None if t is None else round(t - self.t0, 6)

    def add(self, rec: EventRecord) -> None:
        with self.lock:
            self.records.append(rec)

    def write(self, meta: dict) -> Path:
        path = self.out_dir / "events.csv"
        with self.lock, open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
            w.writeheader()
            for r in self.records:
                row = asdict(r)
                for k in ("t_frame", "t_event", "t_dispatch", "t_decision", "t_motor", "t_done"):
                    row[k] = self.rel(row[k])
                w.writerow(row)
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        return path


class ResourceSampler(threading.Thread):
    """Misst periodisch CPU/RAM dieses Prozesses, des LLM-Servers (Ollama) und
    – falls vorhanden – die NVIDIA-GPU."""

    FIELDS = ["t", "proc_cpu", "proc_rss_mb", "llm_cpu", "llm_rss_mb", "sys_cpu", "gpu_util", "gpu_mem_mb"]

    def __init__(self, out_path: Path, t0: float, interval: float = 0.5) -> None:
        super().__init__(daemon=True, name="resources")
        self.out_path, self.t0, self.interval = Path(out_path), t0, interval
        self._halt = threading.Event()
        self.rows: list[dict] = []
        self._nvsmi = shutil.which("nvidia-smi")

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)
        with open(self.out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    def _llm_procs(self, psutil):
        procs = []
        for p in psutil.process_iter(["name"]):
            name = (p.info.get("name") or "").lower()
            if "ollama" in name or "llama" in name:
                procs.append(p)
        return procs

    def _gpu(self) -> tuple[float | None, float | None]:
        if not self._nvsmi:
            return None, None
        try:
            out = subprocess.run(
                [self._nvsmi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2).stdout.strip().splitlines()[0]
            util, mem = (float(x) for x in out.split(","))
            return util, mem
        except Exception:
            return None, None

    def run(self) -> None:
        try:
            import psutil
        except ImportError:
            print("[metrics] psutil fehlt – keine Ressourcenmessung (pip install psutil)")
            return
        me = psutil.Process(os.getpid())
        me.cpu_percent(None)
        llm = self._llm_procs(psutil)
        for p in llm:
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        psutil.cpu_percent(None)
        n = 0
        while not self._halt.wait(self.interval):
            n += 1
            if n % 20 == 0:  # neu gestartete Runner-Prozesse von Ollama mitnehmen
                llm = self._llm_procs(psutil)
            llm_cpu = llm_rss = 0.0
            for p in llm:
                try:
                    llm_cpu += p.cpu_percent(None)
                    llm_rss += p.memory_info().rss / 2**20
                except psutil.Error:
                    pass
            gpu_util, gpu_mem = self._gpu()
            self.rows.append({
                "t": round(time.perf_counter() - self.t0, 3),
                "proc_cpu": me.cpu_percent(None),
                "proc_rss_mb": round(me.memory_info().rss / 2**20, 1),
                "llm_cpu": round(llm_cpu, 1),
                "llm_rss_mb": round(llm_rss, 1),
                "sys_cpu": psutil.cpu_percent(None),
                "gpu_util": gpu_util,
                "gpu_mem_mb": gpu_mem,
            })
