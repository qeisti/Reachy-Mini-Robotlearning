"""A-TC / A-SO – Verhaltensauswahl durch ein lokales LLM ueber LangChain.

* ``tool_calling`` (A-TC): LangChain-Agent (``create_agent``) mit einem Tool pro
  Aktionsart. Das Tool legt die Aktion sofort in die Motion-Queue.
* ``structured`` (A-SO): ein einziger LLM-Aufruf, der ein validiertes
  Aktions-Objekt zurueckgibt (``with_structured_output``). Kein Agent-Loop.

Das Modell laeuft lokal ueber Ollama (``ollama serve`` + ``ollama pull <modell>``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .. import config
from ..actions import ACTION_DESCRIPTIONS, ACTION_NAMES, Action
from ..state import Event
from .base import DecisionInfo, Emit, Policy
from .rule_based import rules_prompt

SYSTEM_PROMPT = """Du bist Reachy Mini, ein kleiner, freundlicher Tischroboter ohne Gesicht.
Du drueckst dich nur ueber Kopfbewegungen und zwei Antennen aus.
Du bekommst Beobachtungen ueber eine Person vor dir und reagierst so, dass sie sich
verstanden und wohl fuehlt. Reagiere zeitnah und passend; wiederhole dich nicht
unnoetig.

Moegliche Reaktionen:
{actions}

{rules}

Reagiere auf jede Beobachtung mit GENAU EINER Aktion."""

TOOL_SUFFIX = "\nRufe dafuer genau ein Tool auf. Schreibe keinen weiteren Text."
STRUCT_SUFFIX = "\nAntworte nur mit dem geforderten JSON-Objekt."


def _action_list() -> str:
    return "\n".join(f"- {n}: {d}" for n, d in ACTION_DESCRIPTIONS.items())


def build_system_prompt(suffix: str = "") -> str:
    """System-Prompt fuer alle LLM-Varianten (agent_tc, agent_so, Hybrid, Destillations-
    Lehrer). Die Regeln kommen aus rule_based.py, damit LLM und Regel-Policy dasselbe
    Verhalten beschrieben bekommen."""
    return SYSTEM_PROMPT.format(actions=_action_list(), rules=rules_prompt()) + suffix


def build_observation(event: Event, history) -> str:
    st = event.state or {}
    lines = []
    if history:
        lines.append("Bisheriger Verlauf:")
        lines += [f"- {h}" for h in history]
    lines.append(
        "Aktueller Zustand: "
        f"Person sichtbar: {'ja' if st.get('face_present') else 'nein'}, "
        f"Distanz: {st.get('distance') or '-'}, Gesichtsausdruck: {st.get('emotion') or '-'}."
    )
    lines.append(f"Neue Beobachtung: {event.describe()}")
    lines.append("Wie reagierst du?")
    return "\n".join(lines)


ActionName = Literal[tuple(ACTION_NAMES)]  # type: ignore[valid-type]


class ActionChoice(BaseModel):
    """Die gewaehlte Reaktion des Roboters."""

    action: ActionName = Field(description="Name der Aktion")  # type: ignore[valid-type]
    intensity: float = Field(1.0, ge=0.0, le=1.0, description="Staerke 0..1")


def make_llm(model: str = config.DEFAULT_MODEL, base_url: str = config.OLLAMA_BASE_URL):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=config.LLM_TEMPERATURE,
        num_predict=config.LLM_NUM_PREDICT,
        keep_alive="30m",
    )


class AgentPolicy(Policy):
    def __init__(self, mode: str = "tool_calling", model: str = config.DEFAULT_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL, llm=None) -> None:
        super().__init__()
        if mode not in ("tool_calling", "structured"):
            raise ValueError(mode)
        self.mode = mode
        self.model_name = model
        self.name = "agent_tc" if mode == "tool_calling" else "agent_so"
        self.llm = llm if llm is not None else make_llm(model, base_url)
        self._emit: Emit | None = None
        self._info: DecisionInfo | None = None
        self._actions: list[Action] = []

        if mode == "tool_calling":
            from langchain.agents import create_agent

            self.agent = create_agent(
                model=self.llm,
                tools=self._make_tools(),
                system_prompt=build_system_prompt(TOOL_SUFFIX),
            )
        else:
            self.structured = self.llm.with_structured_output(ActionChoice, include_raw=True)
            self.system = build_system_prompt(STRUCT_SUFFIX)

    # --- Tools (A-TC) ---------------------------------------------------------
    def _push(self, name: str, intensity: float = 1.0) -> str:
        info = self._info
        info.n_tool_calls += 1
        try:
            action = Action(name, intensity, source=self.name)
        except (ValueError, TypeError) as exc:
            info.n_invalid += 1
            return f"Fehler: {exc}"
        self._actions.append(action)
        if self._emit is not None:
            self._emit(action)
        return "ok"

    def _make_tools(self):
        """Ein Tool pro Aktion, benannt wie die Aktion (freude, traurig, winken ...).

        Prompt, Regeln und Tools benutzen dieselben Namen: "Mittelfinger -> traurig"
        heisst einfach ``traurig()`` aufrufen. Mit dem frueheren Design (4 Tools,
        Emotionen als Argument von ``express_emotion``) musste das Modell den Namen
        erst uebersetzen; qwen2.5:3b schaffte damit nur 1/12 Regelfaellen und nickte
        fast nur noch, mit diesem Design 10/12.
        """
        from langchain_core.tools import StructuredTool

        def action_fn(name: str):
            def run() -> str:   # keine Parameter -> leeres Schema fuer das Modell
                return self._push(name)
            return run

        # return_direct: Der Agent endet direkt nach dem Tool, ohne zweiten
        # LLM-Aufruf fuer eine Abschlussantwort -> spart eine volle Inferenz.
        return [StructuredTool.from_function(func=action_fn(name), name=name,
                                             description=ACTION_DESCRIPTIONS[name], return_direct=True)
                for name in ACTION_NAMES]

    # --- Policy-API -------------------------------------------------------------
    def warmup(self) -> None:
        """Laedt das Modell in den Speicher; Ergebnis wird verworfen."""
        from ..state import Event as _E

        dummy = _E(0, "person_appeared", 0.0, 0.0, None,
                   {"distance": "weit", "emotion": "neutral"}, {"face_present": True})
        saved = list(self.history)
        self.decide(dummy, emit=lambda a: None)
        self.history.clear()
        self.history.extend(saved)

    def decide(self, event: Event, emit: Emit) -> DecisionInfo:
        info = DecisionInfo()
        self._emit, self._info, self._actions = emit, info, []
        observation = build_observation(event, self.history)
        try:
            if self.mode == "tool_calling":
                self._decide_tools(observation, info)
            else:
                self._decide_structured(observation, info, emit)
        except Exception as exc:  # LLM-Server weg, Timeout, Parser ...
            info.error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self._emit = None
        if not self._actions:
            # Modell hat geantwortet, aber keine gueltige Aktion gewaehlt.
            info.extra["no_action"] = True
            if info.error is None and info.n_invalid == 0:
                info.n_invalid += 1
        self.remember(event, self._actions)
        return info

    def _decide_tools(self, observation: str, info: DecisionInfo) -> None:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": observation}]},
            config={"recursion_limit": config.AGENT_RECURSION_LIMIT},
        )
        for msg in result.get("messages", []):
            if getattr(msg, "type", "") == "ai":
                info.n_llm_calls += 1
                usage = getattr(msg, "usage_metadata", None) or {}
                info.tokens_in += usage.get("input_tokens", 0)
                info.tokens_out += usage.get("output_tokens", 0)
                calls = getattr(msg, "tool_calls", None) or []
                info.raw += "; ".join(f"{c['name']}({c.get('args')})" for c in calls) or str(msg.content)[:200]
                # Unbekannter Tool-Name (z. B. "wuetend"): LangChain meldet dem Modell einen
                # Fehler und fragt erneut – das zaehlt als ungueltiger Aufruf.
                for c in calls:
                    if c["name"] not in ACTION_NAMES:
                        info.n_tool_calls += 1
                        info.n_invalid += 1

    def _decide_structured(self, observation: str, info: DecisionInfo, emit: Emit) -> None:
        out = self.structured.invoke([("system", self.system), ("human", observation)])
        info.n_llm_calls += 1
        raw = out.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        info.tokens_in += usage.get("input_tokens", 0)
        info.tokens_out += usage.get("output_tokens", 0)
        info.raw = str(getattr(raw, "content", ""))[:200]
        parsed: ActionChoice | None = out.get("parsed")
        if parsed is None or out.get("parsing_error") is not None:
            info.n_invalid += 1
            info.error = f"parse: {out.get('parsing_error')}"[:300]
            return
        action = Action(parsed.action, parsed.intensity, source=self.name)
        self._actions.append(action)
        emit(action)

    def describe(self) -> dict:
        return {"policy": self.name, "model": self.model_name, "mode": self.mode}

