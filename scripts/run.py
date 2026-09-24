"""Startet eine oder mehrere Policies ueber eine Quelle und speichert Messdaten.

Beispiele (vorher Daemon starten: ``reachy-mini-daemon --sim``):

    # Live mit Webcam, regelbasiert
    python scripts/run.py --policy rule

    # Live mit LangChain-Agent (vorher: ollama pull qwen2.5:3b)
    python scripts/run.py --policy agent_tc --model qwen2.5:3b

    # Experiment: alle Policies je 10x auf dem Stimulus-Video, ohne Fenster
    python scripts/run.py --policy rule agent_tc agent_so hybrid --runs 10 \
        --source experiments/stimuli/stimulus.mp4 --no-display --tag pilot

    # Nur Entscheidungslogik, ohne Kamera und ohne Roboter
    python scripts/run.py --policy rule agent_so --source experiments/scenarios/demo.json --robot mock
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reachy_hri import config  # noqa: E402
from reachy_hri.policies import POLICY_NAMES, make_policy  # noqa: E402
from reachy_hri.robot import make_backend  # noqa: E402
from reachy_hri.runner import RunConfig, run_once  # noqa: E402


def write_system_info(session: Path, args, policies) -> None:
    """Haelt fest, womit gemessen wurde (fuer den Methodenteil des Papers)."""
    import json
    import platform
    import subprocess
    from importlib import metadata

    info = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), "policies": policies,
            "platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version()}
    try:
        import psutil
        info["cpu_cores_logical"] = psutil.cpu_count()
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    versions = {}
    for pkg in ["reachy-mini", "mediapipe", "opencv-python", "langchain", "langchain-ollama", "numpy"]:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    info["packages"] = versions
    root = Path(__file__).resolve().parent.parent
    for key, cmd in [("git_commit", ["git", "-C", str(root), "rev-parse", "--short", "HEAD"]),
                     ("git_dirty", ["git", "-C", str(root), "status", "--porcelain"]),
                     ("ollama_version", ["ollama", "--version"]),
                     ("gpu", ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])]:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
            info[key] = bool(out) if key == "git_dirty" else out
        except (OSError, subprocess.SubprocessError):
            pass
    session.mkdir(parents=True, exist_ok=True)
    (session / "system.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", nargs="+", default=["rule"], choices=POLICY_NAMES + ["all"])
    ap.add_argument("--source", default="webcam:0",
                    help="webcam:<idx> | reachy | <video.mp4> | <szenario.json>")
    ap.add_argument("--robot", default="reachy", choices=["reachy", "mock"])
    ap.add_argument("--model", default=config.DEFAULT_MODEL, help="Ollama-Modell, z. B. qwen2.5:3b")
    ap.add_argument("--ollama-url", default=config.OLLAMA_BASE_URL)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--duration", type=float, default=None, help="Laufzeit bei Webcam/Reachy [s]")
    ap.add_argument("--no-gestures", action="store_true")
    ap.add_argument("--speech", action="store_true", help="Vosk-Spracherkennung aktivieren")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--tag", default="", help="Name des Experiments (Ordnername)")
    ap.add_argument("--out", default=None, help="Zielordner (Standard: experiments/results/<zeit>_<tag>)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--order", choices=["interleaved", "blocked"], default="interleaved",
                    help="interleaved: R, A-TC, A-SO, H, R, ... (gleicht Drift/Erwaermung aus); "
                         "blocked: erst alle Laeufe von R, dann A-TC, ...")
    ap.add_argument("--pause", type=float, default=3.0, help="Pause zwischen Laeufen [s]")
    args = ap.parse_args()

    policies = POLICY_NAMES if "all" in args.policy else args.policy
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = Path(args.out) if args.out else config.RESULTS_DIR / (stamp + (f"_{args.tag}" if args.tag else ""))
    print(f"Ergebnisse -> {session}")

    robot = make_backend(args.robot, use_camera=args.source == "reachy")
    robot.open()
    perception = None
    try:
        if not args.source.lower().endswith(".json"):
            from reachy_hri.perception import Perception
            perception = Perception(gestures=not args.no_gestures, speech=args.speech)

        instances = {}
        for name in policies:
            instances[name] = make_policy(name, args.model, args.ollama_url)
            if not args.no_warmup:
                t = time.perf_counter()
                instances[name].warmup()
                print(f"[{name}] Warm-up {time.perf_counter() - t:.1f} s")
        write_system_info(session, args, policies)

        if args.order == "interleaved":
            schedule = [(k, n) for k in range(args.runs) for n in policies]
        else:
            schedule = [(k, n) for n in policies for k in range(args.runs)]
        for i, (k, name) in enumerate(schedule, start=1):
            print(f"\n=== [{i}/{len(schedule)}] {name}  Lauf {k + 1}/{args.runs} ===")
            cfg = RunConfig(source=args.source, run_idx=k, duration=args.duration,
                            gestures=not args.no_gestures, speech=args.speech,
                            display=not args.no_display,
                            seed=None if args.seed is None else args.seed + k)
            run_once(instances[name], robot, cfg, session / f"{name}_run{k:02d}", perception)
            if i < len(schedule) and args.pause > 0:
                time.sleep(args.pause)
    finally:
        if perception is not None:
            perception.close()
        robot.close()
    print(f"\nFertig. Auswertung: python scripts/compare.py {session}")


if __name__ == "__main__":
    main()
