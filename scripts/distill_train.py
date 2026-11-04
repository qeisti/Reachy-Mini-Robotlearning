"""Trainiert FunctionGemma auf den Lehrer-Daten (Destillation, A-FT).

Die Daten stammen aus ``distill_generate.py``. Aufteilung nach Episoden (nicht
nach Beispielen), damit fast gleiche Verlaeufe nicht in Training und Test landen:
Episode % 10 == 8 -> Validierung, == 9 -> Test, sonst Training.

Der Prompt wird mit ``distill_format.render_prompt`` gebaut – also genau so, wie
Ollama ihn im Betrieb rendert – und selbst tokenisiert. Der Loss wird nur auf den
Tool-Aufruf berechnet, nicht auf den Prompt.

Laeuft in der Conda-Umgebung ``distill`` (PyTorch mit CUDA):

    python scripts/distill_train.py --data experiments/distill/dataset.jsonl

Live mitschauen (zweites Terminal):

    tensorboard --logdir experiments/distill/logs/tensorboard
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_format import STOP_TOKENS, parse_output, render_prompt, render_target  # noqa: E402

BASE_MODEL = "google/functiongemma-270m-it"


def split_of(episode: int) -> str:
    return {8: "val", 9: "test"}.get(episode % 10, "train")


def load_split(path: Path, split: str) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        if split_of(d["episode"]) == split:
            rows.append(d)
    return rows


def encode(rows: list[dict], tok) -> Dataset:
    data = {"input_ids": [], "attention_mask": [], "labels": []}
    for d in rows:
        prompt = tok(render_prompt(d["messages"][:2], d["tools"]), add_special_tokens=False)["input_ids"]
        target = tok(render_target(d["messages"][2]["tool_calls"]), add_special_tokens=False)["input_ids"]
        data["input_ids"].append(prompt + target)
        data["attention_mask"].append([1] * (len(prompt) + len(target)))
        data["labels"].append([-100] * len(prompt) + target)   # nur auf die Antwort trainieren
    return Dataset.from_dict(data)


class TargetOnlyTrainer(Trainer):
    """Berechnet Logits nur an den Positionen des Tool-Aufrufs.

    Der Standard-Loss wendet ``lm_head`` auf jede Position an (Batch x ~700 Tokens
    x 262k Vokabel) und braucht dafuer mehrere GB. Wir trainieren aber nur auf die
    ~20 Tokens der Antwort. Gleicher Loss, ein Bruchteil des Speichers.
    Gilt fuer Gemma 3 ohne final_logit_softcapping (bei FunctionGemma: null).
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        hidden = model.model(input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"]).last_hidden_state
        # Position t sagt Token t+1 vorher
        hidden, targets = hidden[:, :-1], labels[:, 1:]
        mask = targets != -100
        logits = model.lm_head(hidden[mask])
        loss = torch.nn.functional.cross_entropy(logits.float(), targets[mask])
        return (loss, {"logits": logits}) if return_outputs else loss

    # --- Uebereinstimmung mit dem Lehrer, live bei jeder Auswertung -------------
    agree_rows: list[dict] = []
    gen_tok = None

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self.agree_rows:
            extra = self.agreement()
            metrics.update(extra)
            self.log(extra)        # -> TensorBoard: eval/agreement, eval/invalid
        return metrics

    @torch.no_grad()
    def agreement(self, batch_size: int = 32) -> dict:
        """Laesst das Modell die Validierungsbeispiele beantworten (greedy, wie im
        Betrieb mit temperature 0) und vergleicht die Aktion mit der des Lehrers."""
        model, tok = self.model, self.gen_tok
        was_training = model.training
        model.eval()
        eos = [tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]
        match = invalid = 0
        for i in range(0, len(self.agree_rows), batch_size):
            rows = self.agree_rows[i:i + batch_size]
            prompts = [render_prompt(d["messages"][:2], d["tools"]) for d in rows]
            enc = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(**enc, max_new_tokens=32, do_sample=False, eos_token_id=eos,
                                     pad_token_id=tok.pad_token_id)
            for d, seq in zip(rows, out[:, enc.input_ids.shape[1]:]):
                action, _ = parse_output(tok.decode(seq, skip_special_tokens=False))
                match += action == d["teacher_action"]
                invalid += action is None
        if was_training:
            model.train()
        n = len(self.agree_rows)
        return {"eval_agreement": match / n, "eval_invalid": invalid / n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("experiments/distill/dataset.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("experiments/distill/model"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1, help="effektive Batch = batch * grad-accum")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-steps", type=int, default=50, help="alle N Schritte Loss + Uebereinstimmung messen")
    ap.add_argument("--log-dir", type=Path, default=Path("experiments/distill/logs"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_rows, val_rows = load_split(args.data, "train"), load_split(args.data, "val")
    print(f"Train {len(train_rows)}, Validierung {len(val_rows)}, "
          f"Test {len(load_split(args.data, 'test'))} Beispiele")
    train_ds, val_ds = encode(train_rows, tok), encode(val_rows, tok)
    lengths = [len(x) for x in train_ds["input_ids"]]
    print(f"Sequenzlaenge: {min(lengths)}-{max(lengths)} Tokens")

    # float32-Gewichte + bf16-Autocast: stabiler als reines bf16 (Optimizer-Zustaende).
    # sdpa statt eager spart die volle Attention-Matrix pro Schicht; eager braucht
    # Gemma 3 nur bei attn_logit_softcapping, das bei FunctionGemma null ist.
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32,
                                                 attn_implementation="sdpa")
    # transformers >= 5: TensorBoard-Ordner nur noch ueber diese Umgebungsvariable.
    tb_dir = args.log_dir / "tensorboard" / time.strftime("%Y%m%d_%H%M%S")
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)
    print(f"TensorBoard: tensorboard --logdir {args.log_dir / 'tensorboard'}")
    trainer = TargetOnlyTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,          # Anteil der Schritte (transformers >= 5)
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            per_device_eval_batch_size=args.batch,
            bf16=True,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_agreement",   # bestes Modell = hoechste Uebereinstimmung
            greater_is_better=True,
            logging_steps=10,
            report_to=["tensorboard"],
            prediction_loss_only=True,
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.agree_rows = val_rows
    trainer.gen_tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")  # Links-Padding fuer generate
    t_start = time.perf_counter()
    result = trainer.train()
    final = args.out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    # Der Tokenizer wird nicht trainiert -> alle Tokenizer-Dateien aus dem Original kopieren.
    # transformers 5 speichert kein tokenizer.model und kein added_tokens_decoder in
    # tokenizer_config.json. Ohne beides behandelt llama.cpp bei der GGUF-Konvertierung
    # <start_of_turn>, <escape> usw. als normalen Text (7 statt 1 Token) und das Modell
    # sieht in Ollama eine andere Token-Folge als im Training.
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(BASE_MODEL))
    for name in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
                 "special_tokens_map.json", "chat_template.jinja"):
        shutil.copyfile(src / name, final / name)   # copyfile: HF-Cache-Dateien sind schreibgeschuetzt
    print(f"Modell gespeichert: {final}")

    # Log fuer das Paper: Einstellungen, Datensatz, Versionen und Loss-Verlauf.
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": BASE_MODEL,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data": {"train": len(train_rows), "val": len(val_rows), "test": len(load_split(args.data, "test")),
                 "seq_len_min": min(lengths), "seq_len_max": max(lengths)},
        "training": {"seconds": round(time.perf_counter() - t_start, 1), "steps": result.global_step,
                     "train_loss": result.training_loss,
                     "best_eval_agreement": trainer.state.best_metric,
                     "best_checkpoint": trainer.state.best_model_checkpoint,
                     "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20)},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0)},
        "log_history": trainer.state.log_history,
    }
    path = args.log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Trainings-Log: {path}")


if __name__ == "__main__":
    main()
