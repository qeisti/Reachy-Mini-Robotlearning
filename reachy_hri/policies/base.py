"""Gemeinsame Schnittstelle aller Verhaltens-Policies.

Jede Policy bekommt dasselbe Ereignis und gibt Aktionen ueber ``emit`` aus.
``emit`` legt die Aktion sofort in die Queue des Motion-Layers – der
Zeitpunkt des ersten ``emit`` ist die Entscheidungszeit t2.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .. import config
from ..actions import Action
from ..state import Event

Emit = Callable[[Action], None]


@dataclass
class DecisionInfo:
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    n_invalid: int = 0          # ungueltige Aktionen/Argumente oder Parse-Fehler
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None
    raw: str = ""
    route: str = ""             # nur Hybrid: "rules" oder "agent"
    extra: dict = field(default_factory=dict)


class Policy:
    name = "base"

    def __init__(self) -> None:
        self.history: deque[str] = deque(maxlen=config.AGENT_HISTORY)

    def warmup(self) -> None:
        """Einmal vor der Messung aufrufen (z. B. LLM in den Speicher laden)."""

    def decide(self, event: Event, emit: Emit) -> DecisionInfo:
        raise NotImplementedError

    def remember(self, event: Event, actions: list[Action]) -> None:
        names = ", ".join(a.name for a in actions) or "keine"
        self.history.append(f"{event.describe()} -> Roboter: {names}")

    def describe(self) -> dict:
        return {"policy": self.name}
