"""Collective pressing: what the shape around the ball adds, risk held fixed.

    python scripts/collective_press.py
    python scripts/collective_press.py --label label_regain_3s
    python scripts/collective_press.py --bootstrap 500

SkillCorner already rates how likely the carrier is to lose the ball (``xloss``)
and how much pressure they are under. It does not describe the press itself:
how many defenders converge, whether the presser is supported, how compact the
block behind them is. Those are measured here from tracking, and this script
asks one question of each:

    with SkillCorner's own context held fixed, how does the chance of a regain
    change as the press becomes more collective?

Five metrics in two halves, each in a unit a coach can picture.

Getting numbers to the ball:

``press density``           defenders within 10 m of the carrier, per extra defender
``press support``           presser's distance to the nearest teammate, per 5 m
``block compactness``       depth of the defending block, per 10 m

Closing the escape routes:

``open outlets``            free teammates within 15 m of the carrier, per outlet
``presser's cover shadow``  passing lanes the presser closes, per lane

The model is a logistic regression with every SkillCorner context metric as a
control (numeric ones through splines, so ``xloss`` can bend as it needs to)
and the metrics entering linearly, so each gets one interpretable number. Each
metric is fitted on its own and then all five together; the joint fit shows
what each adds once the others are known.
Effects are reported as odds ratios and as percentage points of regain
probability, with 95% intervals from resampling whole matches.

This is association, not causation: a team may bring more players to the ball
because the situation already looked winnable. Holding SkillCorner's context
fixed removes the part of that which their metrics capture, not all of it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, modelling  # noqa: E402

#: column -> (name, what it measures, unit of change, how the unit reads)
METRICS = {
    "f_n_pressing_local": (
        "press density", "defenders within 10 m of the carrier", 1.0,
        "per extra defender",
    ),
    "f_support_dist": (
        "press support", "presser's distance to the nearest teammate", 5.0,
        "per 5 m further away",
    ),
    "f_def_block_depth": (
        "block compactness", "depth of the defending block", 10.0,
        "per 10 m deeper",
    ),
    "f_n_open_short": (
        "open outlets", "free teammates within 15 m of the carrier", 1.0,
        "per extra open outlet",
    ),
    "f_options_shadowed_by_presser": (
        "presser's cover shadow", "passing lanes the presser closes", 1.0,
        "per extra lane closed",
    ),
}

NUMERIC_CONTROLS = [
    "m0_xloss_player_possession_start",
    "m0_xshot_player_possession_start",
    "m0_possession_epv_for_start",
    "m0_possession_epv_against_start",
    "m0_minute_start",
]
CATEGORICAL_CONTROLS = [
    "m0_overall_pressure_start_id",
    "m0_passing_option_ease_start_id",
    "m0_space_constraint_start_id",
    "m0_time_to_impact_start_id",
    "m0_third_start",
    "m0_channel_start",
    "m0_game_state",
]

# Sequential blue for the three ordered risk levels (light -> dark = low ->
# high risk). The lightest step sits at 2.06:1 on the surface, so every line
# carries a direct label. Text stays in ink colours.
RISK_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


# --- design matrix ------------------------------------------------------------

def _as_level(series: pd.Series) -> np.ndarray:
    """A categorical control as plain strings, with missing values as "missing".

    The table stores these as pandas categories; turning a category column with
    gaps into strings leaves the gaps as floats, which cannot be sorted
    alongside strings. Naming the gap makes it a level of its own.
    """
    values = series.astype(object)
    return values.where(series.notna(), "missing").astype(str).to_numpy()


class Design:
    """Builds the same columns for the fit and for counterfactual shifts."""

    def __init__(self, df: pd.DataFrame, metrics: list[str]):
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        self.metrics = metrics
        self.spline = SplineTransformer(n_knots=5, degree=3, extrapolation="linear")
        self.spline.fit(df[NUMERIC_CONTROLS].to_numpy())
        self.levels = {
            c: sorted(set(_as_level(df[c]))) for c in CATEGORICAL_CONTROLS
        }
        self.scaler = StandardScaler().fit(self._controls(df))

    def _controls(self, df: pd.DataFrame) -> np.ndarray:
        parts = [self.spline.transform(df[NUMERIC_CONTROLS].to_numpy())]
        for c, levels in self.levels.items():
            values = _as_level(df[c])
            # One column per level except the first, the reference level.
            parts.append(np.column_stack([values == lv for lv in levels[1:]]).astype(float)
                         if len(levels) > 1 else np.zeros((len(df), 0)))
        return np.hstack(parts)

    def matrix(self, df: pd.DataFrame, shift: dict | None = None) -> np.ndarray:
        controls = self.scaler.transform(self._controls(df))
        cols = []
        for m in self.metrics:
            unit = METRICS[m][2]
            v = df[m].to_numpy(dtype=float) + (shift or {}).get(m, 0.0)
            cols.append(v / unit)  # coefficient = effect of one unit
        return np.hstack([np.column_stack(cols), controls])


def fit(df: pd.DataFrame, metrics: list[str]):
    from sklearn.linear_model import LogisticRegression

    design = Design(df, metrics)
    X = design.matrix(df)
    y = df["_y"].to_numpy()
    model = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    return design, model


def effects(df: pd.DataFrame, metrics: list[str]) -> dict[str, tuple[float, float]]:
    """For each metric: (odds ratio per unit, change in mean regain probability)."""
    design, model = fit(df, metrics)
    base = model.predict_proba(design.matrix(df))[:, 1].mean()
    out = {}
    for i, m in enumerate(metrics):
        odds_ratio = float(np.exp(model.coef_[0][i]))
        shifted = model.predict_proba(
            design.matrix(df, shift={m: METRICS[m][2]})
        )[:, 1].mean()
        out[m] = (odds_ratio, float(shifted - base))
    return out


def bootstrap(df: pd.DataFrame, metrics: list[str], n: int, seed: int):
    """Resample whole matches; rows inside a match are not independent."""
    rng = np.random.default_rng(seed)
    groups = {m: g for m, g in df.groupby("meta_match_id")}
    ids = np.array(list(groups))
    draws = {m: [] for m in metrics}
    for _ in range(n):
        picked = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([groups[i] for i in picked], ignore_index=True)
        if sample["_y"].nunique() < 2:
            continue
        for m, v in effects(sample, metrics).items():
            draws[m].append(v)
    return {m: np.asarray(v) for m, v in draws.items()}


# --- the picture --------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def density_table(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    risk = pd.qcut(df["m0_xloss_player_possession_start"], 3,
                   labels=["low", "medium", "high"])
    density = df["f_n_pressing_local"].clip(upper=4).astype(int)
    rows = []
    for (r, d), g in df.groupby([risk, density], observed=True):
        k, n = int(g["_y"].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"risk": r, "defenders": d, "n": n, "rate": k / n,
                     "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    return out[out["n"] >= min_n]


def plot_density(table: pd.DataFrame, path: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    names = {"low": "low risk", "medium": "medium risk", "high": "high risk"}
    risks = table["risk"].astype(str)
    for risk, color in zip(("low", "medium", "high"), RISK_COLORS):
        g = table[risks == risk].sort_values("defenders")
        if g.empty:
            continue
        x, y = g["defenders"].to_numpy(), g["rate"].to_numpy() * 100
        ax.fill_between(x, g["lo"] * 100, g["hi"] * 100, color=color, alpha=0.12,
                        linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round",
                solid_joinstyle="round", label=names[risk])
        ax.scatter(x, y, s=36, color=color, edgecolors=SURFACE, linewidths=2,
                   zorder=3)
        ax.text(x[-1] + 0.12, y[-1], f"{names[risk]}  {y[-1]:.0f}%",
                va="center", ha="left", fontsize=8.5, color=INK)

    ticks = sorted(table["defenders"].unique())
    ax.set_xticks(ticks, [f"{t}+" if t == 4 else str(t) for t in ticks])
    ax.set_xlim(min(ticks) - 0.3, max(ticks) + 1.5)
    ax.set_ylim(0, None)
    ax.set_xlabel("pressing players within 10 m of the carrier", fontsize=9,
                  color=INK_MUTED)
    ax.set_ylabel(f"regain rate, % ({label})", fontsize=9, color=INK_MUTED)
    ax.set_title("Same risk of losing the ball, more defenders around it",
                 loc="left", fontsize=11, color=INK, pad=22)
    ax.text(0, 1.02, "risk = SkillCorner's xloss, split into thirds; bands are 95% intervals",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=INK_MUTED, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.spines["bottom"].set_alpha(0.4)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --- main ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bin", type=int, default=50,
                    help="smallest cell shown in the density chart")
    args = ap.parse_args()

    df = modelling.load_table()
    needed = list(METRICS) + NUMERIC_CONTROLS
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    df["_y"] = df[args.label].astype(int)
    print(f"{len(df)} engagements ({before - len(df)} dropped for missing values), "
          f"label {args.label} ({df['_y'].mean():.1%} positive)\n")

    # Is density just SkillCorner's pressure code under another name?
    rho = df[["f_n_pressing_local", "m0_overall_pressure_start_id"]].astype(float) \
        .corr(method="spearman").iloc[0, 1]
    print(f"press density vs SkillCorner's pressure code: Spearman r = {rho:+.2f}")
    print("  (well below 1 means density measures something the code does not)")
    corr = df[list(METRICS)].corr(method="spearman")
    corr.index = corr.columns = [METRICS[c][0] for c in METRICS]
    print("\nHow the five metrics relate to each other (Spearman r)")
    print(corr.round(2).to_string())
    print()

    results = []
    for title, metrics in [("each metric on its own", None),
                           ("all five together", list(METRICS))]:
        print(f"Controls: every SkillCorner context metric. {title.capitalize()}:")
        sets = [[m] for m in METRICS] if metrics is None else [metrics]
        for ms in sets:
            t0 = time.time()
            point = effects(df, ms)
            draws = bootstrap(df, ms, args.bootstrap, args.seed)
            for m in ms:
                name, what, unit, reads = METRICS[m]
                orr, pp = point[m]
                or_lo, or_hi = np.percentile(draws[m][:, 0], [2.5, 97.5])
                pp_lo, pp_hi = np.percentile(draws[m][:, 1], [2.5, 97.5])
                print(f"  {name:<24} {reads:<24} odds x{orr:.2f} [{or_lo:.2f}, {or_hi:.2f}]"
                      f"   regain {pp * 100:+.1f} pts [{pp_lo * 100:+.1f}, {pp_hi * 100:+.1f}]")
                results.append({"model": title, "metric": name, "column": m,
                                "unit": reads, "odds_ratio": orr, "or_lo": or_lo,
                                "or_hi": or_hi, "points": pp * 100,
                                "points_lo": pp_lo * 100, "points_hi": pp_hi * 100})
            print(f"  ({len(draws[ms[0]])} match resamples, {time.time() - t0:.0f}s)")
        print()

    table = density_table(df, args.min_bin)
    print("Regain rate by risk third and defenders within 10 m (cells with n >= "
          f"{args.min_bin})")
    order = [r for r in ("low", "medium", "high") if r in set(table["risk"].astype(str))]
    rates = table.pivot(index="defenders", columns="risk", values="rate")
    counts = table.pivot(index="defenders", columns="risk", values="n")
    rates.columns, counts.columns = rates.columns.astype(str), counts.columns.astype(str)
    print(rates[order].map(lambda v: "" if pd.isna(v) else f"{v:.1%}").to_string())
    print("\nrows per cell")
    print(counts[order].fillna(0).astype(int).to_string())

    resdir = config.REPO_ROOT / "results"
    figdir = config.REPO_ROOT / "figures"
    resdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(resdir / f"collective_press_{args.label}.csv", index=False)
    table.to_csv(resdir / f"press_density_{args.label}.csv", index=False)
    plot_density(table, figdir / f"press_density_{args.label}.png", args.label)
    print(f"\nwritten: results/collective_press_{args.label}.csv, "
          f"figures/press_density_{args.label}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
