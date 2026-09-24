"""Austauschbare Verhaltensauswahl – die unabhaengige Variable des Experiments."""

from __future__ import annotations

from .. import config
from .base import DecisionInfo, Policy

POLICY_NAMES = ["rule", "agent_tc", "agent_so", "hybrid"]


def make_policy(name: str, model: str = config.DEFAULT_MODEL,
                base_url: str = config.OLLAMA_BASE_URL, llm=None) -> Policy:
    if name == "rule":
        from .rule_based import RuleBasedPolicy
        return RuleBasedPolicy()
    from .agent import AgentPolicy
    if name == "agent_tc":
        return AgentPolicy("tool_calling", model, base_url, llm)
    if name == "agent_so":
        return AgentPolicy("structured", model, base_url, llm)
    if name == "hybrid":
        from .hybrid import HybridPolicy
        return HybridPolicy(AgentPolicy("structured", model, base_url, llm))
    raise ValueError(f"Unbekannte Policy {name!r}. Erlaubt: {POLICY_NAMES}")


__all__ = ["DecisionInfo", "Policy", "POLICY_NAMES", "make_policy"]
