"""Wertet ein (destilliertes) Modell auf dem Test-Split aus: Stimmt es mit dem Lehrer ueberein?

Zwei Wege, dasselbe Modell zu befragen:

* ``--backend hf``     transformers-Checkpoint (z. B. experiments/distill/model/final)
* ``--backend ollama`` Ollama-Modell (z. B. functiongemma oder reachy-fg) – so wie im Betrieb

Liegen beide fuer dasselbe Modell nah beieinander, spielt der kleine Unterschied
in der Tokenisierung zwischen Hugging Face und Ollama keine Rolle.

    python scripts/distill_eval.py --backend ollama --model functiongemma
    python scripts/distill_eval.py --backend hf --model experiments/distill/model/final
    python scripts/distill_eval.py --backend ollama --model reachy-fg
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_format import STOP_TOKENS, parse_output, render_prompt, to_action  # noqa: E402
from distill_train import load_split  # noqa: E402


class OllamaBackend:
    def __init__(self, model: str, url: str = "http://localhost:11434") -> None:
        self.model, self.url = model, url

    def __call__(self, d: dict) -> tuple[str | None, str]:
        body = {"model": self.model, "stream": False, "messages": d["messages"][:2], "tools": d["tools"],
                "options": {"temperature": 0, "num_predict": 64}}
        req = urllib.request.Request(f"{self.url}/api/chat", json.dumps(body).encode())
        msg = json.loads(urllib.request.urlopen(req).read())["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return None, (msg.get("content") or "")[:80]
        fn = calls[0]["function"]
        args = fn.get("arguments") or {}
        return to_action(fn["name"]), f"{fn['name']}({args})"


class HFBackend:
    def __init__(self, path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        self.eos = [self.tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]

    def __call__(self, d: dict) -> tuple[str | None, str]:
        prompt = render_prompt(d["messages"][:2], d["tools"])
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=64, do_sample=False, eos_token_id=self.eos)
        return parse_output(self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "ollama"], required=True)
    ap.add_argument("--model", required=True, help="Ollama-Name oder Pfad zum HF-Checkpoint")
    ap.add_argument("--data", type=Path, default=Path("experiments/distill/dataset.jsonl"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, help="optional: fester Pfad fuer das Log (sonst automatisch in --log-dir)")
    ap.add_argument("--log-dir", type=Path, default=Path("experiments/distill/logs"))
    args = ap.parse_args()

    ask = OllamaBackend(args.model) if args.backend == "ollama" else HFBackend(args.model)
    rows = load_split(args.data, args.split)
    per_type: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    results = []
    for i, d in enumerate(rows, 1):
        action, raw = ask(d)
        teacher = d["teacher_action"]
        status = "invalid" if action is None else ("match" if action == teacher else "other")
        per_type[d["event_type"]][status] += 1
        if status == "other":
            confusion[(teacher, action)] += 1
        results.append({"event_type": d["event_type"], "teacher": teacher, "student": action, "raw": raw})
        print(f"  {i}/{len(rows)}", end="\r")

    total = Counter()
    for c in per_type.values():
        total.update(c)
    n = sum(total.values())
    print(f"\n{args.backend}:{args.model} auf {args.split} (n={n})")
    print(f"  Uebereinstimmung mit Lehrer: {100 * total['match'] / n:5.1f} %")
    print(f"  andere gueltige Aktion:      {100 * total['other'] / n:5.1f} %")
    print(f"  ungueltig / kein Tool:       {100 * total['invalid'] / n:5.1f} %\n")
    for etype, c in sorted(per_type.items()):
        k = sum(c.values())
        print(f"  {etype:<17} n={k:<4} Treffer {100 * c['match'] / k:5.1f} %   ungueltig {100 * c['invalid'] / k:5.1f} %")
    if confusion:
        print("\n  Haeufigste Abweichungen (Lehrer -> Schueler):")
        for (t, s), k in confusion.most_common(8):
            print(f"    {t:<11} -> {s:<11} {k}x")
    # Immer mitschreiben: Kennzahlen + jede einzelne Entscheidung.
    import time
    args.log_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]+", "_", Path(args.model).name if args.backend == "hf" else args.model)
    path = args.out or args.log_dir / f"eval_{args.backend}_{name}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend, "model": args.model, "data": str(args.data), "split": args.split, "n": n,
        "match": total["match"] / n, "other": total["other"] / n, "invalid": total["invalid"] / n,
        "per_event_type": {e: dict(c) for e, c in sorted(per_type.items())},
        "confusion": [{"teacher": t, "student": s, "count": k} for (t, s), k in confusion.most_common()],
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nLog: {path}")


if __name__ == "__main__":
    main()
