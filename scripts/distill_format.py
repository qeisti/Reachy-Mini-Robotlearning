"""Prompt-Format fuer FunctionGemma – exakt so, wie Ollama es rendert.

Ollama baut den Prompt fuer FunctionGemma nicht aus dem Hugging-Face-Chat-
Template, sondern mit einem eigenen Renderer (``RENDERER functiongemma``). Die
beiden weichen ab: Ollama laesst z. B. ``enum``/``default`` im Tool-Schema weg
und rendert Tools ohne Parameter anders. Damit das Modell im Training genau das
sieht, was es spaeter im Betrieb bekommt, portiert dieses Modul den Renderer.

Quelle: ollama/ollama v0.34.4, ``model/renderers/functiongemma.go``.
Bei einem Ollama-Update pruefen: ``python scripts/distill_format.py`` vergleicht
die Token-Zahlen mit dem laufenden Ollama.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reachy_hri.actions import ACTION_NAMES  # noqa: E402  (eine Quelle fuer die Aktionsnamen)

DEFAULT_SYSTEM = "You can do function calling with the following functions:"
STOP = "<start_function_response>"   # FunctionGemma beendet einen Tool-Aufruf damit (eos_token_id 50)
STOP_TOKENS = ("<eos>", "<end_of_turn>", STOP)   # = generation_config.eos_token_id

# Rueckrichtung: Modell-Ausgabe -> Aktion (gemeinsam fuer Training und Auswertung).
# Seit "Variante A" heisst jedes Tool wie seine Aktion (freude, traurig, winken ...).
CALL_RE = re.compile(r"call:(\w+)\{(.*?)\}<end_function_call>", re.S)


def to_action(name: str | None) -> str | None:
    """Tool-Name -> Aktionsname; None, wenn es das Tool nicht gibt (= ungueltig)."""
    return name if name in ACTION_NAMES else None


def parse_output(text: str) -> tuple[str | None, str]:
    """Rohe Modell-Ausgabe (mit Spezial-Tokens) -> (Aktion oder None, gekuerzter Rohtext)."""
    m = CALL_RE.search(text)
    if not m:
        return None, text[:80]
    return to_action(m.group(1)), m.group(0)[:80]


def _value(v) -> str:
    if isinstance(v, str):
        return f"<escape>{v}<escape>"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_value(v[k])}" for k in sorted(v)) + "}"
    if isinstance(v, list):
        return "[" + ",".join(_value(x) for x in v) + "]"
    return str(v)


def _declaration(tool: dict) -> str:
    fn = tool["function"]
    out = f"<start_function_declaration>declaration:{fn['name']}{{description:<escape>{fn.get('description', '')}<escape>"
    params = fn.get("parameters") or {}
    props = params.get("properties")
    if props is not None or params.get("type"):
        out += ",parameters:{"
        parts = []
        if props:
            rendered = []
            for name in sorted(props):
                p = props[name]
                s = f"{name}:{{description:<escape>{p.get('description', '')}<escape>"
                ptype = p.get("type")
                if ptype:
                    s += f",type:<escape>{(ptype[0] if isinstance(ptype, list) else ptype).upper()}<escape>"
                rendered.append(s + "}")
            parts.append("properties:{" + ",".join(rendered) + "}")
        if params.get("required"):
            parts.append("required:[" + ",".join(f"<escape>{r}<escape>" for r in params["required"]) + "]")
        if params.get("type"):
            parts.append(f"type:<escape>{params['type'].upper()}<escape>")
        out += ",".join(parts) + "}"
    return out + "}<end_function_declaration>"


def render_prompt(messages: list[dict], tools: list[dict]) -> str:
    """System- und User-Nachricht + Tools -> Prompt bis einschliesslich ``<start_of_turn>model``."""
    out = "<bos>"
    system, rest = "", messages
    if messages and messages[0]["role"] in ("system", "developer"):
        system, rest = messages[0]["content"], messages[1:]
    if system or tools:
        out += "<start_of_turn>developer\n" + system.strip()
        if tools:
            if system:
                out += "\n"
            if system.strip() != DEFAULT_SYSTEM:
                out += DEFAULT_SYSTEM
        out += "".join(_declaration(t) for t in tools) + "<end_of_turn>\n"
    for m in rest:
        if m["role"] != "user":
            raise ValueError(f"nur user-Nachrichten im Prompt erwartet, nicht {m['role']!r}")
        out += "<start_of_turn>user\n" + m["content"].strip() + "<end_of_turn>\n"
    return out + "<start_of_turn>model\n"


def render_target(tool_calls: list[dict]) -> str:
    """Gewuenschte Modell-Ausgabe: Tool-Aufruf(e) + Stop-Token."""
    out = ""
    for tc in tool_calls:
        fn = tc["function"]
        args = fn.get("arguments") or {}
        out += f"<start_function_call>call:{fn['name']}{{" + ",".join(
            f"{k}:{_value(args[k])}" for k in sorted(args)) + "}<end_function_call>"
    return out + STOP


if __name__ == "__main__":
    # Selbsttest: Token-Zahl unseres Prompts == prompt_eval_count von Ollama?
    import json
    import sys
    import urllib.request

    from transformers import AutoTokenizer

    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/distill/probe.jsonl"
    tok = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    ok = True
    for i, line in enumerate(open(path, encoding="utf-8")):
        if i >= 10:
            break
        d = json.loads(line)
        prompt = render_prompt(d["messages"][:2], d["tools"])
        n_ours = len(tok(prompt, add_special_tokens=False)["input_ids"])
        body = {"model": "functiongemma", "stream": False, "messages": d["messages"][:2], "tools": d["tools"],
                "options": {"num_predict": 1}}
        req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode())
        n_ollama = json.loads(urllib.request.urlopen(req).read())["prompt_eval_count"]
        ok &= n_ours == n_ollama
        print(f"Beispiel {i}: unser Renderer {n_ours}, Ollama {n_ollama}  {'OK' if n_ours == n_ollama else 'ABWEICHUNG'}")
    print("\nZiel-Beispiel:", render_target(json.loads(open(path).readline())["messages"][2]["tool_calls"]))
    sys.exit(0 if ok else 1)
