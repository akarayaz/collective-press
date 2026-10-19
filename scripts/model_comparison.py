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
    """Sized for a README: about 880 px wide on GitHub, so text is 9.5 pt and up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n = len(table)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.56 * n + 2.7), dpi=200, sharey=True,
                             gridspec_kw={"wspace": 0.1})
    fig.patch.set_facecolor(SURFACE)
    # A gap between groups, so each block reads as one question.
    y, pos, last = [], 0.0, None
    for g in table["group"]:
        if last is not None and g != last:
            pos -= 0.55
        y.append(pos)
        pos -= 1.0
        last = g
    y = np.array(y)
    panels = [("auc_gain", "auc_lo", "auc_hi", "Gain in AUC", "{:+.3f}"),
              ("ll_gain", "ll_lo", "ll_hi", "Gain in log loss (lower loss = gain)", "{:+.4f}")]
    for ax, (mid, lo, hi, title, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=INK_MUTED, linewidth=1.0, alpha=0.7)
        for yi, (_, r) in zip(y, table.iterrows()):
            colour = GRAY if r["placebo"] else BLUE
            ax.hlines(yi, r[lo], r[hi], color=colour, linewidth=2.4, alpha=0.5)
            ax.scatter(r[mid], yi, s=80, color=colour, edgecolors=SURFACE, linewidths=2,
                       zorder=3)
            value = fmt.format(r[mid])
            if float(value) == 0:  # no "+0.000" or "-0.000"
                value = value.lstrip("+-")
            value = value.replace("-", "−")  # same minus sign as the ticks
            if r[hi] < 0:
                # The whole interval sits below zero: a label to its right would
                # cross the zero line, one to its left would hit the row names.
                ax.annotate(value, (r[mid], yi), xytext=(0, 8), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9.5, color=INK_MUTED)
            else:
                ax.annotate(value, (r[hi], yi), xytext=(6, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=9.5, color=INK_MUTED)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9.5, length=0)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=INK_MUTED, alpha=0.15, linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        left, right = ax.get_xlim()
        ax.set_xlim(left - (right - left) * 0.05, right + (right - left) * 0.22)  # labels
        ax.set_ylim(y.min() - 0.7, y.max() + 0.9)
    axes[0].set_yticks(y, table["label"], fontsize=10.5)
    for tick, placebo in zip(axes[0].get_yticklabels(), table["placebo"]):
        tick.set_color(INK_MUTED if placebo else INK)

    # Group names above each block of rows, at the left edge of the figure.
    for group, g in table.groupby("group", sort=False):
        top = y[g.index.min()]
        axes[0].annotate(f"Adding {group}", (0.012, top), xycoords=("figure fraction", "data"),
                         xytext=(0, 15), textcoords="offset points", fontsize=10.5,
                         color=INK, fontweight="bold", va="center", annotation_clip=False)

    fig.text(0.012, 0.965, "What each input adds to SkillCorner's context", fontsize=16,
             color=INK, ha="left", va="center")
    fig.text(0.012, 0.915,
             f"Gain over the same neural network fed only SkillCorner's context and geometry "
             f"(AUC {base_auc:.3f}), on the same presses,\nwith 95% intervals from resampling "
             "matches. Gray: the same input shuffled within each match.",
             fontsize=10, color=INK_MUTED, ha="left", va="center", linespacing=1.35)
    fig.text(0.012, 0.015,
             "Shuffled scenes add nothing, so a scene model's gain comes from the scenes. "
             "Shuffled metrics hurt, as 32 columns of noise get in the way,\nso for the metrics "
             "the fair comparison is with zero. One seed per model; 17,180 presses in 20 matches.",
             fontsize=8.5, color=INK_MUTED, ha="left", va="bottom", linespacing=1.35)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.8, bottom=0.14)
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
