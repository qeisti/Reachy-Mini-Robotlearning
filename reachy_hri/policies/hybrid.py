"""H – hybrid: zeitkritische Ereignisse per Regel, der Rest per Agent.

Idee: Reflexe (Begruessung, Winken erwidern, zu nah) muessen sofort kommen,
fuer affektive Aenderungen und Sprache darf der Agent laenger nachdenken.
"""

from __future__ import annotations

from ..state import Event
from .agent import AgentPolicy
from .base import DecisionInfo, Emit, Policy
from .rule_based import RuleBasedPolicy

FAST_EVENTS = {"person_appeared", "person_left", "gesture", "distance_changed"}


class HybridPolicy(Policy):
    name = "hybrid"

    def __init__(self, agent: AgentPolicy, rules: RuleBasedPolicy | None = None) -> None:
        super().__init__()
        self.agent = agent
        self.rules = rules or RuleBasedPolicy()

    def warmup(self) -> None:
        self.agent.warmup()

    def decide(self, event: Event, emit: Emit) -> DecisionInfo:
        if event.type in FAST_EVENTS:
            info = self.rules.decide(event, emit)
            info.route = "rules"
        else:
            info = self.agent.decide(event, emit)
            info.route = "agent"
        # Beide Teile sollen denselben Verlauf sehen.
        latest = (self.rules if info.route == "rules" else self.agent).history[-1]
        for p in (self.rules, self.agent):
            if not p.history or p.history[-1] != latest:
                p.history.append(latest)
        return info

    def describe(self) -> dict:
        d = self.agent.describe()
        d["policy"] = self.name
        return d
