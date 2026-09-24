"""Fake-LLMs, damit Agent-Policies ohne Ollama getestet werden koennen."""

import itertools
import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class FakeToolModel(GenericFakeChatModel):
    """Antwortet reihum mit vorbereiteten Tool-Calls."""

    delay: float = 0.0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        return super()._generate(*args, **kwargs)


def tool_model(calls, delay=0.0):
    msgs = [AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"}],
                      usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
            for i, (n, a) in enumerate(calls)]
    return FakeToolModel(messages=itertools.cycle(msgs), delay=delay)


class FakeStructuredModel:
    """Minimaler Ersatz: with_structured_output liefert vorbereitete Objekte."""

    def __init__(self, choices, delay=0.0):
        self.choices = itertools.cycle(choices)
        self.delay = delay

    def with_structured_output(self, schema, include_raw=False):
        def run(_):
            if self.delay:
                time.sleep(self.delay)
            c = next(self.choices)
            raw = AIMessage(content=str(c), usage_metadata={"input_tokens": 80, "output_tokens": 8,
                                                            "total_tokens": 88})
            if c is None:
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("kaputt")}
            return {"raw": raw, "parsed": schema(**c), "parsing_error": None}
        return RunnableLambda(run)
