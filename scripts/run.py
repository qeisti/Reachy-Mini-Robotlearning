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

        for name in policies:
            policy = make_policy(name, args.model, args.ollama_url)
            if not args.no_warmup:
                t = time.perf_counter()
                policy.warmup()
                print(f"[{name}] Warm-up {time.perf_counter() - t:.1f} s")
            for k in range(args.runs):
                print(f"\n=== {name}  Lauf {k + 1}/{args.runs} ===")
                cfg = RunConfig(source=args.source, run_idx=k, duration=args.duration,
                                gestures=not args.no_gestures, speech=args.speech,
                                display=not args.no_display,
                                seed=None if args.seed is None else args.seed + k)
                run_once(policy, robot, cfg, session / f"{name}_run{k:02d}", perception)
    finally:
        if perception is not None:
            perception.close()
        robot.close()
    print(f"\nFertig. Auswertung: python scripts/compare.py {session}")


if __name__ == "__main__":
    main()
