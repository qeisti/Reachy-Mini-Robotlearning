"""Setzt die Regeln aus rule_based.py in den Trainingsdaten durch ("Regeln Pflicht, KI Rest").

Der Lehrer (qwen2.5:14b) haelt die Regeln aus dem Prompt in Einzeltests ein, in langen
Episoden mit Verlauf aber nicht immer: Aendert sich das Gesicht, waehrend die Person
'mittel' ist, spiegelt er oft das Gesicht statt 'vorsichtig' (Vorrangregel). Damit der
Schueler die Regeln 1:1 lernt, wird das Trainingsziel so gesetzt:

* Event mit fester Regel               -> Aktion der Regel-Policy
* Sprache ohne Begruessungswort        -> Aktion des Lehrers (hier ist die KI frei)

Die Lehrer-Antwort bleibt als ``llm_action`` erhalten, ``label_source`` sagt, woher das
Ziel kommt. ``teacher_action`` (das Feld, das Training und Auswertung benutzen) ist danach
das finale Ziel.

Aeltere Dateien ohne ``event_data`` werden ueber Seed/Episoden aus ``<stem>.stats.json``
rekonstruiert; die Uebereinstimmung mit den Zeilen wird geprueft.

    python scripts/distill_apply_rules.py experiments/distill/data_v2.jsonl \
        --out experiments/distill/data_v2_rules.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from distill_generate import simulate_episode, tool_call  # noqa: E402

from reachy_hri import config  # noqa: E402
from reachy_hri.policies.agent import build_observation  # noqa: E402

from reachy_hri.policies.rule_based import GREETING_WORDS, RuleBasedPolicy  # noqa: E402
from reachy_hri.state import Event  # noqa: E402


def is_free(event: Event) -> bool:
    """Faelle ohne feste Regel: Sprache ohne Begruessung."""
    if event.type != "speech":
        return False
    text = (event.data.get("text") or "").lower()
    return not any(w in text for w in GREETING_WORDS)


def events_for(rows: list[dict], path: Path) -> list[Event]:
    if all("event_data" in r for r in rows):
        return [Event(0, r["event_type"], 0.0, 0.0, None, r["event_data"], r.get("event_state", {}))
                for r in rows]
    stats = json.loads(path.with_suffix(".stats.json").read_text(encoding="utf-8"))   # data.jsonl -> data.stats.json
    rng = random.Random(stats["seed"])
    events = [e for _ in range(stats["episodes"]) for e in simulate_episode(rng)]
    if stats.get("skipped"):
        raise SystemExit("Datensatz hat uebersprungene Beispiele – Rekonstruktion nicht eindeutig.")
    if len(events) != len(rows) or any(e.type != r["event_type"] for e, r in zip(events, rows)):
        raise SystemExit("Rekonstruierte Events passen nicht zu den Zeilen (Seed/Episoden pruefen).")
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    events = events_for(rows, args.data)
    rules = RuleBasedPolicy()
    changed: Counter = Counter()
    sources: Counter = Counter()
    history: deque[str] = deque(maxlen=config.AGENT_HISTORY)
    episode, same_obs = None, 0

    with args.out.open("w", encoding="utf-8") as f:
        for event, row in zip(events, rows):
            if row["episode"] != episode:
                episode = row["episode"]
                history.clear()
            llm = row.get("llm_action", row["teacher_action"])
            target = llm if is_free(event) else rules._choose(event)
            source = "teacher" if is_free(event) else "rule"
            sources[source] += 1
            if target != llm:
                changed[(event.type, llm, target)] += 1
            # Verlauf mit den korrigierten Aktionen neu aufbauen – im Betrieb sieht das Modell
            # seinen eigenen (regeltreuen) Verlauf, nicht die Abweichungen des Lehrers.
            observation = build_observation(event, history)
            same_obs += observation == row["messages"][1]["content"]
            history.append(f"{event.describe()} -> Roboter: {target}")
            row.update({"event_data": event.data, "event_state": event.state, "llm_action": llm,
                        "teacher_action": target, "label_source": source})
            row["messages"][1] = {"role": "user", "content": observation}
            row["messages"][2] = {"role": "assistant", "tool_calls": [tool_call(target)]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"{n} Beispiele -> {args.out}")
    print(f"  Ziel aus Regel: {sources['rule']}, aus Lehrer (frei): {sources['teacher']}")
    print(f"  geaendert: {sum(changed.values())} ({100 * sum(changed.values()) / n:.1f} %)")
    print(f"  Beobachtung unveraendert: {same_obs}, mit korrigiertem Verlauf neu: {n - same_obs}")
    for (etype, llm, target), k in changed.most_common(10):
        print(f"    {etype:<17} Lehrer {llm:<10} -> Regel {target:<10} {k}x")
    stats = {"source": str(args.data), "examples": n, "label_sources": dict(sources),
             "changed": sum(changed.values()),
             "changes": [{"event_type": e, "llm": l, "rule": t, "count": k} for (e, l, t), k in changed.most_common()]}
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
