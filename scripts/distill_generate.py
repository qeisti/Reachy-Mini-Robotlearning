"""Erzeugt Trainingsdaten fuer die Destillation (A-FT).

Simuliert zufaellige, plausible Interaktions-Episoden (Person erscheint,
Emotion/Distanz wechseln, Gesten, Sprache, Person geht). Ein grosses Lehrer-
Modell entscheidet Schritt fuer Schritt mit seinem eigenen Verlauf – genau wie
``agent_so`` im Betrieb. Jede Entscheidung wird als Trainingsbeispiel im
Tool-Calling-Format von ``agent_tc`` gespeichert (gleicher System-Prompt, gleiche
Tools, gleiche Beobachtung), damit der Schueler spaeter exakt das sieht, was er
im Betrieb bekommt.

Beispiele:

    # Probelauf: 20 Episoden
    python scripts/distill_generate.py --episodes 20 --out experiments/distill/probe.jsonl   # nicht im Git

    # Voller Datensatz
    python scripts/distill_generate.py --episodes 400 --out experiments/distill/dataset_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from reachy_hri.policies import make_policy  # noqa: E402
from reachy_hri.policies.agent import TOOL_SUFFIX, build_observation, build_system_prompt  # noqa: E402
from reachy_hri.state import Event  # noqa: E402

# Werte genau so, wie Wahrnehmung und EventDetector sie liefern.
FACE_EMOTIONS = ["neutral", "freude", "traurig", "ueberrascht", "wuetend"]
DISTANCES = ["weit", "mittel", "nah"]
GESTURES = ["winken", "Thumb_Up", "Thumb_Down", "Victory", "ILoveYou", "mittelfinger"]
UTTERANCES = [
    "hallo", "hallo reachy", "hi", "servus", "guten morgen", "na du",
    "wie geht es dir", "was machst du", "wer bist du", "kannst du mich sehen",
    "du bist so suess", "gut gemacht", "das finde ich toll", "danke",
    "ich bin heute traurig", "mir geht es nicht gut", "ich hatte einen schlechten tag",
    "geh weg", "du nervst", "sei still",
    "schau mal her", "komm her", "ja", "nein", "okay",
    "tschuess", "bis spaeter", "ich muss jetzt gehen",
]
# Gewichte der Folge-Ereignisse innerhalb einer Episode.
NEXT_EVENT = {"emotion_changed": 0.40, "distance_changed": 0.25, "gesture": 0.20, "speech": 0.15}


def simulate_episode(rng: random.Random) -> list[Event]:
    """Eine Episode vom Erscheinen bis zum Verlassen der Person."""
    emotion = rng.choices(FACE_EMOTIONS, weights=[5, 3, 1, 1, 1])[0]
    distance = rng.choices(DISTANCES, weights=[4, 3, 1])[0]
    events: list[Event] = []

    def add(etype: str, **data) -> None:
        state = {"face_present": etype != "person_left", "distance": distance if etype != "person_left" else None,
                 "emotion": emotion if etype != "person_left" else "neutral"}
        events.append(Event(len(events) + 1, etype, 0.0, 0.0, None, data, state))

    add("person_appeared", distance=distance, emotion=emotion)
    for _ in range(rng.randint(2, 9)):
        etype = rng.choices(list(NEXT_EVENT), weights=list(NEXT_EVENT.values()))[0]
        if etype == "emotion_changed":
            previous, emotion = emotion, rng.choice([e for e in FACE_EMOTIONS if e != emotion])
            add(etype, emotion=emotion, previous=previous, score=round(rng.uniform(0.2, 0.9), 2),
                distance=distance)
        elif etype == "distance_changed":
            # Distanz aendert sich schrittweise (weit <-> mittel <-> nah)
            i = DISTANCES.index(distance)
            previous, distance = distance, DISTANCES[rng.choice([j for j in (i - 1, i + 1) if 0 <= j < 3])]
            add(etype, distance=distance, previous=previous, emotion=emotion)
        elif etype == "gesture":
            add(etype, gesture=rng.choice(GESTURES), emotion=emotion, distance=distance)
        else:
            add(etype, text=rng.choice(UTTERANCES), emotion=emotion, distance=distance)
    add("person_left")
    return events


def tool_call(action: str) -> dict:
    """Aktion -> Tool-Aufruf. Jedes Tool heisst wie seine Aktion und hat keine Argumente."""
    return {"type": "function", "function": {"name": action, "arguments": {}}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--teacher", default="qwen2.5:14b", help="Ollama-Modell des Lehrers")
    ap.add_argument("--out", type=Path, default=Path("experiments/distill/dataset_raw.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    teacher = make_policy("agent_so", model=args.teacher)
    # Tools und Prompt genau so, wie agent_tc sie an das Modell schickt.
    student_view = make_policy("agent_tc", model="functiongemma")
    tools = [convert_to_openai_tool(t) for t in student_view._make_tools()]
    system = build_system_prompt(TOOL_SUFFIX)

    print(f"Lehrer {args.teacher} laden ...")
    teacher.warmup()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, Counter] = defaultdict(Counter)
    n_ok = n_skip = 0
    t_start = time.perf_counter()

    with args.out.open("w", encoding="utf-8") as f:
        for ep in range(args.episodes):
            teacher.history.clear()
            for event in simulate_episode(rng):
                observation = build_observation(event, teacher.history)
                actions = []
                info = teacher.decide(event, actions.append)
                if not actions or info.error:
                    n_skip += 1
                    continue
                action = actions[0]
                by_type[event.type][action.name] += 1
                n_ok += 1
                f.write(json.dumps({
                    "episode": ep,
                    "event_type": event.type,
                    "event_data": event.data,
                    "event_state": event.state,
                    "teacher_action": action.name,
                    "tools": tools,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": observation},
                        {"role": "assistant", "tool_calls": [tool_call(action.name)]},
                    ],
                }, ensure_ascii=False) + "\n")
            print(f"  Episode {ep + 1}/{args.episodes}: {n_ok} Beispiele", end="\r")

    dt = time.perf_counter() - t_start
    print(f"\n{n_ok} Beispiele in {dt:.0f} s ({dt / max(n_ok, 1):.2f} s/Beispiel), "
          f"{n_skip} uebersprungen -> {args.out}\n")
    print("Verteilung der Lehrer-Aktionen je Ereignistyp:")
    for etype, counts in sorted(by_type.items()):
        total = sum(counts.values())
        dist = ", ".join(f"{a} {100 * c / total:.0f}%" for a, c in counts.most_common())
        print(f"  {etype:<17} n={total:<4} {dist}")

    # Fuer Reproduzierbarkeit: Einstellungen, Statistik, Tools und Prompt ablegen.
    stem = args.out.with_suffix("")
    Path(f"{stem}.stats.json").write_text(json.dumps({
        "teacher": args.teacher, "episodes": args.episodes, "seed": args.seed,
        "examples": n_ok, "skipped": n_skip, "seconds": round(dt, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "actions_per_event_type": {e: dict(c.most_common()) for e, c in sorted(by_type.items())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(f"{stem}.tools.json").write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(f"{stem}.system_prompt.txt").write_text(system, encoding="utf-8")
    print(f"\nStatistik, Tools und System-Prompt: {stem}.stats.json / .tools.json / .system_prompt.txt")


if __name__ == "__main__":
    main()
