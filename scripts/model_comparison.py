"""What each input adds to SkillCorner's context, in one figure.

    python scripts/model_comparison.py

Needs ``cache/oof_deep_<label>.csv`` from ``scripts/train_deep.py``.

Every model here is a neural network trained the same way (leave one match
out, the same early stopping), so the only thing that differs between them is
what they are given. The baseline sees SkillCorner's context and geometry
only; each other model adds one thing:

    the raw scene     DeepSets, Transformer, graph network
    our metrics       the 32 hand-made pressing metrics
    both              the graph network and the metrics model, averaged

Gains are measured against the baseline on the same rows, with a match level
bootstrap for the interval. Placebos (the same input shuffled within each
match) are drawn in gray: whatever a real input gains beyond its placebo comes
from the scenes, not from having more columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, modelling  # noqa: E402

BASE = "mlp_context"

#: (column, label, group, is placebo); groups are drawn top to bottom
ROWS = [
    ("deepsets", "DeepSets", "the raw scene", False),
    ("transformer", "Transformer", "the raw scene", False),
    ("transformer_placebo", "Transformer, shuffled scenes", "the raw scene", True),
    ("gnn", "Graph network", "the raw scene", False),
    ("gnn_placebo", "Graph network, shuffled scenes", "the raw scene", True),
    ("mlp_metrics", "Our 32 pressing metrics", "our metrics", False),
    ("mlp_metrics_placebo", "Our metrics, shuffled", "our metrics", True),
    ("gnn+mlp_metrics", "Graph network and our metrics", "both", False),
]

BLUE = "#2a78d6"
GRAY = "#9a9893"
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def compare(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    y, groups = df["y"].astype(int), df["match_id"]

    def result(name):
        return modelling.CVResult(name, df[f"p_{name}"], y, groups)

    base = result(BASE)
    rows = []
    for col, label, group, placebo in ROWS:
        if f"p_{col}" not in df:
            continue
        rich = result(col)
        delta = modelling.bootstrap_delta(base, rich, n_samples=n_boot, seed=seed)
        won, total = modelling.matches_improved(base, rich)
        m = rich.metrics()
        rows.append({
            "model": col, "label": label, "group": group, "placebo": placebo,
            "roc_auc": m["roc_auc"], "log_loss": m["log_loss"],
            "auc_gain": delta["roc_auc"][0], "auc_lo": delta["roc_auc"][1],
            "auc_hi": delta["roc_auc"][2],
            "ll_gain": delta["log_loss"][0], "ll_lo": delta["log_loss"][1],
            "ll_hi": delta["log_loss"][2],
            "matches_improved": won, "matches": total,
        })
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, base_auc: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n = len(table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * n + 1.9), dpi=200, sharey=True,
                             gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(n)[::-1]
    panels = [("auc_gain", "auc_lo", "auc_hi", "gain in AUC", 1000, "{:+.3f}"),
              ("ll_gain", "ll_lo", "ll_hi", "gain in log loss (lower loss = gain)", 1000, "{:+.4f}")]
    for ax, (mid, lo, hi, title, _, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=INK_MUTED, linewidth=0.9, alpha=0.7)
        for yi, (_, r) in zip(y, table.iterrows()):
            colour = GRAY if r["placebo"] else BLUE
            ax.hlines(yi, r[lo], r[hi], color=colour, linewidth=2, alpha=0.5)
            ax.scatter(r[mid], yi, s=55, color=colour, edgecolors=SURFACE, linewidths=2,
                       zorder=3)
            value = fmt.format(r[mid])
            if float(value) == 0:  # no "+0.000" or "-0.000"
                value = value.lstrip("+-")
            ax.annotate(value, (r[hi], yi), xytext=(5, -3),
                        textcoords="offset points", fontsize=7.5, color=INK_MUTED)
        ax.set_title(title, fontsize=9, color=INK_MUTED, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8, length=0)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=INK_MUTED, alpha=0.15, linewidth=0.8)
        for s in ax.spines.values():
            s.set_visible(False)
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + (right - left) * 0.18)  # room for the value labels
    axes[0].set_yticks(y, table["label"], fontsize=8.5)
    for tick, placebo in zip(axes[0].get_yticklabels(), table["placebo"]):
        tick.set_color(INK_MUTED if placebo else INK)

    # Group names above each block of rows.
    for group, g in table.groupby("group", sort=False):
        top = y[g.index.min()]
        axes[0].annotate(f"adding {group}", (0, top), xycoords=("axes fraction", "data"),
                         xytext=(-178, 11), textcoords="offset points", fontsize=8,
                         color=INK, fontweight="bold", annotation_clip=False)

    fig.suptitle("What each input adds to SkillCorner's context", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.01, 0.01,
             f"Every model is the same kind of neural network; the baseline sees SkillCorner's "
             f"context and geometry only (AUC {base_auc:.3f}). Gains are on the same rows, with 95% "
             "intervals\nfrom resampling matches. Gray: the same input shuffled within each match. "
             "Shuffled scenes add nothing, so what a scene model gains comes from the scenes. "
             "Shuffled metrics hurt,\nbecause 32 columns of noise get in the way; for the metrics "
             "the fair comparison is with zero, not with their gray twin. One seed per model.",
             fontsize=7, color=INK_MUTED, va="bottom")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.86, bottom=0.2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = config.REPO_ROOT / "cache" / f"oof_deep_{args.label}.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run scripts/train_deep.py first")
    df = pd.read_csv(path)
    if f"p_{BASE}" not in df:
        raise SystemExit(f"no {BASE} predictions; run "
                         "scripts/train_deep.py --models mlp_context,mlp_metrics")
    if {"p_gnn", "p_mlp_metrics"} <= set(df.columns):
        df["p_gnn+mlp_metrics"] = (df["p_gnn"] + df["p_mlp_metrics"]) / 2

    table = compare(df, args.bootstrap, args.seed)
    base_auc = modelling.compute_metrics(df["y"].astype(int), df[f"p_{BASE}"])["roc_auc"]
    print(f"baseline {BASE}: AUC {base_auc:.4f}\n")
    show = table.copy()
    show["AUC gain"] = [f"{g:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                        for g, lo, hi in zip(show["auc_gain"], show["auc_lo"], show["auc_hi"])]
    show["log loss gain"] = [f"{g:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                             for g, lo, hi in zip(show["ll_gain"], show["ll_lo"], show["ll_hi"])]
    show["matches"] = [f"{w}/{t}" for w, t in zip(show["matches_improved"], show["matches"])]
    print(show[["label", "roc_auc", "AUC gain", "log loss gain", "matches"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    resdir, figdir = config.REPO_ROOT / "results", config.REPO_ROOT / "figures"
    resdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(resdir / f"model_comparison_{args.label}.csv", index=False)
    plot(table, base_auc, figdir / f"model_comparison_{args.label}.png")
    print(f"\nwritten: results/model_comparison_{args.label}.csv, "
          f"figures/model_comparison_{args.label}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
