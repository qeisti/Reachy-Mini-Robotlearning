import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from reachy_hri.actions import Action
from reachy_hri.motion import MotionController
from reachy_hri.policies import make_policy
from reachy_hri.policies.rule_based import RuleBasedPolicy
from reachy_hri.robot import MockBackend
from reachy_hri.runner import RunConfig, run_once
from reachy_hri.state import Event, EventDetector, PerceptionState

from fakes import FakeStructuredModel, tool_model

ROOT = Path(__file__).resolve().parent.parent
SCENARIO = ROOT / "experiments" / "scenarios" / "demo.json"


def ev(etype, **data):
    return Event(1, etype, 0.0, 0.0, None, data, {"face_present": True, **data})


def collect(policy, event):
    out = []
    info = policy.decide(event, out.append)
    return out, info


# --- Regeln ------------------------------------------------------------------
@pytest.mark.parametrize("event,expected", [
    (ev("gesture", gesture="winken"), "winken"),
    (ev("emotion_changed", emotion="freude", distance="weit"), "freude"),
    (ev("emotion_changed", emotion="wuetend", distance="weit"), "angst"),
    (ev("emotion_changed", emotion="freude", distance="nah"), "angst"),
    (ev("distance_changed", distance="mittel", emotion="freude"), "vorsichtig"),
    (ev("person_left"), "neutral"),
    (ev("speech", text="Hallo du"), "winken"),
])
def test_rules(event, expected):
    acts, info = collect(RuleBasedPolicy(), event)
    assert [a.name for a in acts] == [expected]
    assert info.n_invalid == 0


# --- Agent mit Tool-Calling ----------------------------------------------------
def test_agent_tool_calling_emits_action():
    p = make_policy("agent_tc", llm=tool_model([("express_emotion", {"emotion": "freude", "intensity": 0.7})]))
    acts, info = collect(p, ev("emotion_changed", emotion="freude", distance="weit"))
    assert [a.name for a in acts] == ["freude"]
    assert acts[0].intensity == pytest.approx(0.7)
    assert info.n_llm_calls == 1          # return_direct: kein zweiter LLM-Aufruf
    assert info.n_tool_calls == 1 and info.n_invalid == 0
    assert info.tokens_in == 100
    assert "Bisheriger Verlauf" not in p.history[-1]


def test_agent_tool_calling_invalid_argument():
    p = make_policy("agent_tc", llm=tool_model([("express_emotion", {"emotion": "wuetend"})]))
    acts, info = collect(p, ev("emotion_changed", emotion="wuetend"))
    assert acts == []
    assert info.n_invalid == 1


def test_agent_tool_calling_gesture_tools():
    p = make_policy("agent_tc", llm=tool_model([("wave", {}), ("nod", {}), ("do_nothing", {})]))
    names = [collect(p, ev("gesture", gesture="winken"))[0][0].name for _ in range(3)]
    assert names == ["winken", "nicken", "nichts"]


# --- Agent mit Structured Output ----------------------------------------------
def test_agent_structured():
    llm = FakeStructuredModel([{"action": "neugierig", "intensity": 0.5}, None])
    p = make_policy("agent_so", llm=llm)
    acts, info = collect(p, ev("person_appeared", distance="weit"))
    assert [a.name for a in acts] == ["neugierig"] and info.n_invalid == 0
    acts, info = collect(p, ev("person_left"))
    assert acts == [] and info.n_invalid == 1 and info.error


def test_hybrid_routes():
    llm = FakeStructuredModel([{"action": "traurig"}])
    p = make_policy("hybrid", llm=llm)
    acts, info = collect(p, ev("gesture", gesture="winken"))
    assert acts[0].name == "winken" and info.route == "rules"
    acts, info = collect(p, ev("emotion_changed", emotion="traurig"))
    assert acts[0].name == "traurig" and info.route == "agent"
    assert len(p.agent.history) == 2


# --- Ereignisse ------------------------------------------------------------------
def test_event_detector_debounce():
    d = EventDetector()
    st = lambda **k: PerceptionState(t_frame=0, face_present=True, distance="weit", **k)  # noqa: E731
    assert [e.type for e in d.update(st(emotion="neutral"), now=0.0)] == ["person_appeared"]
    assert d.update(st(emotion="freude"), now=0.1) == []          # noch nicht stabil
    assert d.update(st(emotion="freude"), now=0.3) == []
    evs = d.update(st(emotion="freude"), now=0.7)
    assert [e.type for e in evs] == ["emotion_changed"] and evs[0].data["previous"] == "neutral"
    assert d.update(PerceptionState(t_frame=0), now=1.0) == []    # Timeout noch nicht erreicht
    assert [e.type for e in d.update(PerceptionState(t_frame=0), now=1.9)] == ["person_left"]


def test_gesture_cooldown():
    d = EventDetector()
    st = PerceptionState(t_frame=0, gesture="winken")
    assert len(d.update(st, now=0.0)) == 1
    assert d.update(st, now=0.5) == []
    assert len(d.update(st, now=3.0)) == 1


# --- Motion ------------------------------------------------------------------------
def test_motion_nonblocking_and_t3():
    robot = MockBackend()
    m = MotionController(robot)
    m.start()
    started = []
    t = time.perf_counter()
    m.submit(Action("freude"), started.append)
    assert time.perf_counter() - t < 0.01            # submit blockiert nicht
    time.sleep(1.0)
    m.submit(Action("winken"), started.append)
    time.sleep(0.2)
    assert m.current_action == "winken"
    time.sleep(1.5)
    m.stop()
    assert len(started) == 2 and started[0] - t < 0.1
    assert robot.last["antennas"][1] > 50            # nach dem Winken wieder Freude-Haltung
    assert robot.n_commands > 50


# --- Ende-zu-Ende: Szenario -> CSV -> Report ---------------------------------------
def test_end_to_end(tmp_path):
    scenario = json.loads(SCENARIO.read_text(encoding="utf-8"))
    for e in scenario["events"]:
        e["t"] = e["t"] / 10                          # 10x schneller fuer den Test
    sc = tmp_path / "sc.json"
    sc.write_text(json.dumps(scenario), encoding="utf-8")

    policies = {
        "rule": make_policy("rule"),
        "agent_so": make_policy("agent_so", llm=FakeStructuredModel(
            [{"action": "freude"}, {"action": "neugierig"}, None], delay=0.05)),
        "agent_tc": make_policy("agent_tc", llm=tool_model(
            [("express_emotion", {"emotion": "freude"}), ("wave", {})], delay=0.08)),
    }
    for name, pol in policies.items():
        for k in range(2):
            run_once(pol, MockBackend(), RunConfig(source=str(sc), run_idx=k, drain_timeout=10),
                     tmp_path / "session" / f"{name}_run{k:02d}")
    df = pd.read_csv(tmp_path / "session" / "agent_tc_run00" / "events.csv")
    assert len(df) == 12
    assert (df["t_decision"] >= df["t_dispatch"]).all()
    assert df["t_motor"].notna().mean() > 0.9

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare.py"), str(tmp_path / "session"),
                        "--annotations", str(ROOT / "experiments" / "annotations_template.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = tmp_path / "session" / "report"
    for f in ["summary.csv", "summary_table.tex", "report.html", "latency.pdf", "quality.png", "actions.png",
              "stats.csv", "checks.txt"]:
        assert (rep / f).exists(), f
    s = pd.read_csv(rep / "summary.csv").set_index("policy")
    assert s.loc["rule", "decision_median_ms"] < s.loc["agent_tc", "decision_median_ms"]
    assert s.loc["rule", "consistency"] == 1.0


def test_run_cli_interleaved(tmp_path):
    scenario = json.loads(SCENARIO.read_text(encoding="utf-8"))
    scenario["events"] = scenario["events"][:3]
    for e in scenario["events"]:
        e["t"] = e["t"] / 10
    sc = tmp_path / "sc.json"
    sc.write_text(json.dumps(scenario), encoding="utf-8")
    out = tmp_path / "sess"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), "--policy", "rule", "--runs", "2",
                        "--source", str(sc), "--robot", "mock", "--pause", "0", "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert (out / "system.json").exists()
    assert (out / "rule_run00" / "events.csv").exists() and (out / "rule_run01" / "events.csv").exists()
