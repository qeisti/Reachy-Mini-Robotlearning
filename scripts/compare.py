"""Wertet Messlaeufe aus und stellt die Policies gegenueber.

    python scripts/compare.py experiments/results/<session> [weitere Sessions ...]
        [--annotations experiments/annotations.json] [--threshold-ms 1000]

Erzeugt in ``<erste session>/report/``:
    summary.csv / summary.md / summary_table.tex   Kennzahlen je Policy
    latency.pdf/.png        Entscheidungs- und Ende-zu-Ende-Latenz (Boxplot)
    latency_timeline.pdf/.png  Latenz je Ereignis ueber die Zeit
    resources.pdf/.png      CPU/RAM (Prozess + LLM-Server)
    quality.pdf/.png        Gueltigkeit, Konsistenz, Angemessenheit
    actions.pdf/.png        Welche Aktion bei welchem Ereignistyp
    report.html             alles zusammen, im Browser oeffnen
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ORDER = ["rule", "agent_tc", "agent_so", "hybrid"]
LABELS = {"rule": "R (Regeln)", "agent_tc": "A-TC (Tool-Calling)",
          "agent_so": "A-SO (Structured)", "hybrid": "H (Hybrid)"}
# Feste Farbe je Policy (validierte Kategorial-Palette, Slots 1-4).
COLORS = {"rule": "#2a78d6", "agent_tc": "#eb6834", "agent_so": "#1baf7a", "hybrid": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --- Laden ---------------------------------------------------------------------
def load(sessions: list[Path], by_session: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev, res = [], []
    for s in sessions:
        for f in sorted(Path(s).rglob("events.csv")):
            df = pd.read_csv(f)
            meta_f = f.parent / "meta.json"
            meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
            df["run_dir"] = str(f.parent)
            df["session"] = Path(s).name
            df["model"] = df["model"].fillna(meta.get("model", ""))
            ev.append(df)
            rf = f.parent / "resources.csv"
            if rf.exists() and rf.stat().st_size > 0:
                r = pd.read_csv(rf)
                r["policy"] = df["policy"].iloc[0] if len(df) else meta.get("policy")
                r["run_dir"] = str(f.parent)
                res.append(r)
    if not ev:
        raise SystemExit("Keine events.csv gefunden.")
    events = pd.concat(ev, ignore_index=True)
    resources = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    events["label"] = events["policy"].astype(str) + np.where(
        events["model"].fillna("").astype(str) != "", " · " + events["model"].fillna("").astype(str), "")
    if by_session:  # z. B. Simulation vs. Roboter: gleiche Policy getrennt nach Session
        events["label"] = events["label"] + " · " + events["session"].astype(str)
    events["lat_decision_ms"] = (events["t_decision"] - events["t_dispatch"]) * 1000
    events["lat_wait_ms"] = (events["t_dispatch"] - events["t_event"]) * 1000
    events["lat_e2e_ms"] = (events["t_motor"] - events["t_frame"]) * 1000
    events["lat_policy_total_ms"] = (events["t_done"] - events["t_dispatch"]) * 1000
    events["first_action"] = events["actions"].fillna("").astype(str).str.split("|").str[0].replace("", "(keine)")
    events["failed"] = (events["n_invalid"].fillna(0) > 0) | events["error"].fillna("").astype(str).ne("")
    # Ereignisse ueber Laeufe hinweg zuordnen: n-tes Ereignis dieses Typs im Lauf.
    # (Bei identischem Stimulus entspricht das n-te "emotion_changed" in jedem
    # Lauf demselben Moment im Video.)
    events = events.sort_values(["run_dir", "t_event"]).reset_index(drop=True)
    events["event_key"] = events["event_type"] + "#" + (
        events.groupby(["run_dir", "event_type"]).cumcount() + 1).astype(str)
    return events, resources


# --- Kennzahlen ----------------------------------------------------------------
def q(s: pd.Series, p: float) -> float:
    s = s.dropna()
    return float(np.percentile(s, p)) if len(s) else float("nan")


def consistency(df: pd.DataFrame) -> float:
    """Mittlerer Anteil der Laeufe, die bei einem Ereignis die haeufigste Aktion waehlen."""
    vals = []
    for _, g in df.groupby("event_key"):
        if g["run_idx"].nunique() >= 2:
            vals.append(g["first_action"].value_counts(normalize=True).iloc[0])
    return float(np.mean(vals)) if vals else float("nan")


def rule_agreement(df: pd.DataFrame, rule: pd.DataFrame) -> float:
    if rule.empty:
        return float("nan")
    modal = rule.groupby("event_key")["first_action"].agg(lambda s: s.value_counts().index[0])
    m = df["event_key"].map(modal)
    ok = m.notna()
    return float((df.loc[ok, "first_action"] == m[ok]).mean()) if ok.any() else float("nan")


def appropriateness(df: pd.DataFrame, ann: list[dict] | None) -> float:
    if not ann:
        return float("nan")
    hits = []
    for _, r in df.iterrows():
        if pd.isna(r["video_time"]):
            continue
        for a in ann:
            if a["t_start"] <= r["video_time"] <= a["t_end"] and a.get("type", r["event_type"]) == r["event_type"]:
                hits.append(r["first_action"] in a["acceptable"])
                break
    return float(np.mean(hits)) if hits else float("nan")


def summarize(events: pd.DataFrame, resources: pd.DataFrame, ann, threshold_ms: float) -> pd.DataFrame:
    rule = events[events["policy"] == "rule"]
    rows = []
    for label, g in events.groupby("label", sort=False):
        pol = g["policy"].iloc[0]
        r = resources[resources["run_dir"].isin(g["run_dir"].unique())] if not resources.empty else pd.DataFrame()
        row = {
            "label": label, "policy": pol, "model": g["model"].iloc[0] if pd.notna(g["model"].iloc[0]) else "",
            "runs": g["run_dir"].nunique(), "events": len(g),
            "decision_median_ms": g["lat_decision_ms"].median(),
            "decision_iqr_ms": q(g["lat_decision_ms"], 75) - q(g["lat_decision_ms"], 25),
            "decision_p95_ms": q(g["lat_decision_ms"], 95),
            "wait_median_ms": g["lat_wait_ms"].median(),
            "e2e_median_ms": g["lat_e2e_ms"].median(),
            "e2e_p95_ms": q(g["lat_e2e_ms"], 95),
            f"share_e2e_over_{int(threshold_ms)}ms": float((g["lat_e2e_ms"] > threshold_ms).mean()),
            "invalid_rate": float(g["failed"].mean()),
            "llm_calls_per_event": g["n_llm_calls"].mean(),
            "tokens_per_event": (g["tokens_in"] + g["tokens_out"]).mean(),
            "consistency": consistency(g),
            "distinct_actions_per_type": g.groupby("event_type")["first_action"].nunique().mean(),
            "agreement_with_rules": rule_agreement(g, rule) if pol != "rule" else 1.0,
            "appropriateness": appropriateness(g, ann),
        }
        if not r.empty:
            row.update({
                "proc_cpu_mean": r["proc_cpu"].mean(), "llm_cpu_mean": r["llm_cpu"].mean(),
                "llm_cpu_peak": r["llm_cpu"].max(), "proc_rss_peak_mb": r["proc_rss_mb"].max(),
                "llm_rss_peak_mb": r["llm_rss_mb"].max(),
                "gpu_util_mean": pd.to_numeric(r["gpu_util"], errors="coerce").mean(),
                "gpu_mem_peak_mb": pd.to_numeric(r["gpu_mem_mb"], errors="coerce").max(),
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_o"] = df["policy"].map({p: i for i, p in enumerate(ORDER)}).fillna(99)
    return df.sort_values(["_o", "label"]).drop(columns="_o").reset_index(drop=True)


# --- Statistik -------------------------------------------------------------------
def bootstrap_median_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def stats_tests(events: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney-U je Policy gegen R (und A-TC gegen A-SO), mit Effektstaerke
    (rang-biseriale Korrelation r) und Bootstrap-95%-KI des Medians."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("[compare] scipy fehlt – keine Signifikanztests (pip install scipy)")
        return pd.DataFrame()
    pols = [p for p in ORDER if p in set(events["policy"])]
    pairs = [("rule", p) for p in pols if p != "rule"]
    if {"agent_tc", "agent_so"} <= set(pols):
        pairs.append(("agent_so", "agent_tc"))
    rows = []
    for metric in ("lat_decision_ms", "lat_e2e_ms"):
        for a, b in pairs:
            xa = events.loc[events["policy"] == a, metric].dropna().values
            xb = events.loc[events["policy"] == b, metric].dropna().values
            if len(xa) < 3 or len(xb) < 3:
                continue
            u, pval = mannwhitneyu(xa, xb, alternative="two-sided")
            lo_a, hi_a = bootstrap_median_ci(xa)
            lo_b, hi_b = bootstrap_median_ci(xb)
            rows.append({"metric": metric, "a": a, "b": b, "n_a": len(xa), "n_b": len(xb),
                         "median_a": np.median(xa), "ci95_a": f"[{lo_a:.3g}, {hi_a:.3g}]",
                         "median_b": np.median(xb), "ci95_b": f"[{lo_b:.3g}, {hi_b:.3g}]",
                         "U": u, "p": pval, "r_rank_biserial": 1 - 2 * u / (len(xa) * len(xb))})
    return pd.DataFrame(rows)


def data_checks(events: pd.DataFrame) -> list[str]:
    """Plausibilitaetspruefungen: sind die Laeufe vergleichbar?"""
    msgs = []
    per_run = events.groupby(["policy", "run_dir"]).size()
    for pol, g in per_run.groupby(level=0):
        if g.max() - g.min() > max(1, 0.2 * g.median()):
            msgs.append(f"{pol}: Ereignisse pro Lauf schwanken stark ({g.min()}–{g.max()}) – "
                        "Wahrnehmung instabil oder Frames verworfen? (Stimulus/Schwellen pruefen)")
    med = events.groupby("policy").size() / events.groupby("policy")["run_dir"].nunique()
    if len(med) > 1 and med.max() > 1.3 * med.min():
        msgs.append("Unterschiedlich viele Ereignisse je Policy – Perception wurde durch die Last "
                    "des Agenten beeinflusst (Frames verworfen). Im Paper erwaehnen.")
    miss = events["t_motor"].isna().mean()
    if miss > 0.05:
        msgs.append(f"{miss:.0%} der Ereignisse ohne Motorbefehl (Aktion 'nichts', ungueltig oder verdraengt).")
    return msgs


# --- Plots -----------------------------------------------------------------------
def _style(ax, ylabel: str = "") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out / f"{name}.png"


def _labels(events: pd.DataFrame) -> list[str]:
    lab = events.drop_duplicates("label")[["label", "policy"]]
    lab["_o"] = lab["policy"].map({p: i for i, p in enumerate(ORDER)}).fillna(99)
    return lab.sort_values(["_o", "label"])["label"].tolist()


def _color(events: pd.DataFrame, label: str) -> str:
    return COLORS.get(events.loc[events["label"] == label, "policy"].iloc[0], INK2)


def _nice(label: str, long: bool = False) -> str:
    pol, _, model = label.partition(" · ")
    name = LABELS.get(pol, pol) if long else LABELS.get(pol, pol).split(" ")[0]
    return name + (f"\n{model}" if model else "")


def plot_latency(events: pd.DataFrame, out: Path, threshold_ms: float) -> Path:
    import matplotlib.pyplot as plt

    labels = _labels(events)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, col, title in [(axes[0], "lat_decision_ms", "Entscheidung (t2 - t1)"),
                           (axes[1], "lat_e2e_ms", "Ende-zu-Ende (t3 - t0)")]:
        data = [events.loc[events["label"] == lb, col].dropna().clip(lower=0.01).values for lb in labels]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=INK2,
                                        markeredgecolor="none", alpha=0.5))
        for patch, lb in zip(bp["boxes"], labels):
            patch.set_facecolor(_color(events, lb))
            patch.set_edgecolor("white")
            patch.set_alpha(0.9)
        ax.set_yscale("log")
        ax.axhline(threshold_ms, color=INK2, linestyle="--", linewidth=1)
        ax.text(len(labels) + 0.45, threshold_ms, f"{threshold_ms:.0f} ms", va="bottom", ha="right",
                fontsize=7, color=INK2)
        ax.set_xticks(range(1, len(labels) + 1), [_nice(lb) for lb in labels], fontsize=7)
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax, "Latenz [ms], log" if ax is axes[0] else "")
    return _save(fig, out, "latency")


def plot_timeline(events: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    xcol = "video_time" if events["video_time"].notna().any() else "t_event"
    for lb in _labels(events):
        g = events[events["label"] == lb]
        agg = g.groupby("event_key").agg(x=(xcol, "median"), y=("lat_e2e_ms", "median")).dropna().sort_values("x")
        ax.plot(agg["x"], agg["y"], marker="o", markersize=4, linewidth=2, color=_color(events, lb),
                label=_nice(lb, long=True).replace("\n", " "))
    ax.set_yscale("log")
    ax.set_xlabel("Zeit im Stimulus [s]", color=INK2, fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _style(ax, "Ende-zu-Ende [ms], Median")
    return _save(fig, out, "latency_timeline")


def plot_resources(summary: pd.DataFrame, out: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if "proc_cpu_mean" not in summary:
        return None
    metrics = [("proc_cpu_mean", "CPU Prozess [%]"), ("llm_cpu_mean", "CPU LLM-Server [%]"),
               ("proc_rss_peak_mb", "RAM Prozess [MB]"), ("llm_rss_peak_mb", "RAM LLM-Server [MB]")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.6))
    x = np.arange(len(summary))
    for ax, (col, title) in zip(axes, metrics):
        vals = summary[col].fillna(0).values
        ax.bar(x, vals, color=[COLORS.get(p, INK2) for p in summary["policy"]], width=0.7,
               edgecolor="white", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x, [_nice(lb).split("\n")[0] + ("\n" + lb.split(" · ")[-1] if lb.count(" · ") >= 2 else "") for lb in summary["label"]], fontsize=7)
        ax.set_title(title, fontsize=8, color=INK)
        _style(ax)
    return _save(fig, out, "resources")


def plot_quality(summary: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    metrics = [("invalid_rate", "Ungueltig / Fehler"), ("consistency", "Konsistenz ueber Laeufe"),
               ("agreement_with_rules", "Uebereinstimmung mit R"), ("appropriateness", "Angemessen (Annotation)")]
    metrics = [m for m in metrics if summary[m[0]].notna().any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(summary))
    for ax, (col, title) in zip(axes, metrics):
        vals = summary[col].values * 100
        ax.bar(x, np.nan_to_num(vals), color=[COLORS.get(p, INK2) for p in summary["policy"]], width=0.7,
               edgecolor="white", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, 0 if np.isnan(v) else v, "n/a" if np.isnan(v) else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x, [_nice(lb).split("\n")[0] + ("\n" + lb.split(" · ")[-1] if lb.count(" · ") >= 2 else "") for lb in summary["label"]], fontsize=7)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_ylim(0, 110)
        _style(ax, "%" if ax is axes[0] else "")
    return _save(fig, out, "quality")


def plot_actions(events: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = _labels(events)
    types = sorted(events["event_type"].unique())
    acts = sorted(events["first_action"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(max(7.2, 2.4 * len(labels)), 0.5 + 0.45 * len(types)),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lb in zip(axes, labels):
        g = events[events["label"] == lb]
        m = pd.crosstab(g["event_type"], g["first_action"], normalize="index").reindex(
            index=types, columns=acts, fill_value=0)
        ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(types)):
            for j in range(len(acts)):
                v = m.values[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(acts)), acts, rotation=60, fontsize=7, ha="right")
        ax.set_yticks(range(len(types)), types, fontsize=7)
        ax.set_title(_nice(lb, long=True).replace("\n", " "), fontsize=8, color=INK)
        ax.tick_params(colors=INK2, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    return _save(fig, out, "actions")


# --- Tabellen & Report ---------------------------------------------------------
PAPER_COLS = [("label", "Cfg"), ("decision_median_ms", "Dec. med [ms]"), ("decision_p95_ms", "Dec. p95 [ms]"),
              ("e2e_median_ms", "E2E med [ms]"), ("invalid_rate", "Invalid [\\%]"),
              ("consistency", "Consist. [\\%]"), ("agreement_with_rules", "Agree R [\\%]")]


def to_latex(summary: pd.DataFrame) -> str:
    cols = [c for c in PAPER_COLS if c[0] in summary]
    lines = ["\\begin{tabular}{|" + "|".join(["l"] + ["r"] * (len(cols) - 1)) + "|}", "\\hline",
             " & ".join(f"\\textbf{{{h}}}" for _, h in cols) + " \\\\ \\hline"]
    for _, r in summary.iterrows():
        cells = []
        for c, _ in cols:
            v = r[c]
            if c == "label":
                cells.append(LABELS.get(r["policy"], r["policy"]).split(" ")[0])
            elif pd.isna(v):
                cells.append("--")
            elif c in ("invalid_rate", "consistency", "agreement_with_rules"):
                cells.append(f"{v * 100:.0f}")
            else:
                cells.append(f"{v:.0f}" if v >= 10 else f"{v:.1f}")
        lines.append(" & ".join(cells) + " \\\\ \\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def to_html(summary: pd.DataFrame, images: list[Path], sessions, threshold_ms: float,
            stats: pd.DataFrame | None = None, checks: list[str] | None = None) -> str:
    fmt = summary.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            pct = c in ("invalid_rate", "consistency", "agreement_with_rules", "appropriateness") or c.startswith("share_")
            fmt[c] = fmt[c].map(lambda v, pct=pct: "–" if pd.isna(v) else (f"{v * 100:.0f} %" if pct else f"{v:.1f}"))
    table = fmt.T.to_html(header=False, border=0, classes="tbl")
    imgs = "\n".join(
        f'<figure><img alt="{p.stem}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'
        f"<figcaption>{p.stem}</figcaption></figure>" for p in images if p)
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Policy-Vergleich</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} p.sub{{color:var(--ink2);margin:0 0 24px}}
.tbl{{border-collapse:collapse;background:var(--card);font-variant-numeric:tabular-nums;width:100%}}
.tbl td,.tbl th{{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right}}
.tbl tr:first-child td{{font-weight:600}} .tbl th{{text-align:left;color:var(--ink2);font-weight:500}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}
figure{{margin:24px 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}}
figure img{{max-width:100%;height:auto;display:block;margin:auto}}
figcaption{{color:var(--ink2);font-size:12px;margin-top:6px}}
dl{{color:var(--ink2);font-size:13px}} dt{{font-weight:600;color:var(--ink)}}
</style></head><body><main>
<h1>Regeln vs. LLM-Agent – Vergleich</h1>
<p class="sub">Sessions: {", ".join(Path(s).name for s in sessions)} · Kontingenzschwelle {threshold_ms:.0f} ms</p>
<div class="wrap">{table}</div>
{"" if stats is None or stats.empty else "<h2>Signifikanztests</h2><div class='wrap'>" + stats.to_html(index=False, border=0, classes="tbl", float_format=lambda v: f"{v:.3g}") + "</div>"}
{"" if not checks else "<h2>Datenpruefung</h2><ul>" + "".join(f"<li>{c}</li>" for c in checks) + "</ul>"}
{imgs}
<dl>
<dt>decision_*</dt><dd>Zeit von Policy-Start bis zur ersten Aktion (t2 - t1).</dd>
<dt>wait_*</dt><dd>Wartezeit in der Ereignis-Queue, waehrend die Policy noch beschaeftigt war.</dd>
<dt>e2e_*</dt><dd>Frame mit dem Ereignis bis zum ersten Motorbefehl (t3 - t0).</dd>
<dt>consistency</dt><dd>Anteil der Laeufe mit der haeufigsten Aktion je Ereignis.</dd>
<dt>agreement_with_rules</dt><dd>Anteil der Ereignisse, bei denen dieselbe Aktion wie R gewaehlt wurde.</dd>
<dt>appropriateness</dt><dd>Anteil der Reaktionen in der annotierten Menge akzeptabler Aktionen.</dd>
<dt>p / r_rank_biserial</dt><dd>Mann-Whitney-U-Test (zweiseitig) und Effektstaerke; |r| &gt; 0.5 = grosser Effekt.</dd>
</dl></main></body></html>"""


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--threshold-ms", type=float, default=1000.0,
                    help="Kontingenzschwelle fuer die Auswertung (mit Literatur begruenden!)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--by-session", action="store_true",
                    help="gleiche Policy aus verschiedenen Sessions getrennt darstellen (z. B. Sim vs. Roboter)")
    args = ap.parse_args()

    events, resources = load(args.sessions, args.by_session)
    ann = None
    if args.annotations:
        data = json.loads(args.annotations.read_text(encoding="utf-8"))
        ann = data["events"] if isinstance(data, dict) else data
    summary = summarize(events, resources, ann, args.threshold_ms)

    out = args.out or (args.sessions[0] / "report")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary.to_markdown(index=False, floatfmt=".2f")
                                    if _has_tabulate() else summary.to_string(index=False), encoding="utf-8")
    (out / "summary_table.tex").write_text(to_latex(summary), encoding="utf-8")

    st = stats_tests(events)
    if not st.empty:
        st.to_csv(out / "stats.csv", index=False)
    checks = data_checks(events)
    (out / "checks.txt").write_text("\n".join(checks) or "keine Auffaelligkeiten", encoding="utf-8")

    images = [plot_latency(events, out, args.threshold_ms), plot_timeline(events, out),
              plot_quality(summary, out), plot_resources(summary, out), plot_actions(events, out)]
    (out / "report.html").write_text(to_html(summary, images, args.sessions, args.threshold_ms, st, checks), encoding="utf-8")

    cols = ["label", "events", "decision_median_ms", "decision_p95_ms", "e2e_median_ms",
            "invalid_rate", "consistency", "agreement_with_rules"]
    print(summary[[c for c in cols if c in summary]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if not st.empty:
        print("\nSignifikanztests (Mann-Whitney-U):")
        print(st[["metric", "a", "b", "median_a", "median_b", "p", "r_rank_biserial"]].to_string(
            index=False, float_format=lambda v: f"{v:.3g}"))
    for m in checks:
        print("WARNUNG:", m)
    print(f"\nReport: {out / 'report.html'}")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
