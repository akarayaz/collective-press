"""Which parts of the positioning carry the gain, and how the model uses them.

    python scripts/explain_model.py
    python scripts/explain_model.py --skip-shap              # ablation only
    python scripts/explain_model.py --skip-ablation          # SHAP only
    python scripts/explain_model.py --label label_regain_3s

Both views are exploratory: they describe the main result, they do not test it.

Group ablation
--------------
The tracking features are split into five groups a coach would recognise.
For each group two leave-one-match-out models are fitted:

``add``   the SkillCorner baseline plus that group alone. The gain is what the
          group is worth on its own.
``drop``  the full model without that group. The loss is what the group
          carries that the other groups cannot stand in for.

Correlated groups share credit, so a group can have a large ``add`` and a small
``drop`` when another group says much the same thing. Reading both is the point.

SHAP
----
One model is fitted on every usable row and SHAP values are computed for a
sample of rows. They give the direction and size of each feature's effect on the
predicted log-odds of a regain. Two figures go to ``figures/``: the usual
beeswarm of the top features, and the share of the model's attribution that
falls to each feature group, SkillCorner's own included.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, features, modelling  # noqa: E402

#: The five groups. Every v1 tracking feature appears in exactly one.
GROUPS = {
    "carrier": [
        "carrier_x", "carrier_y_abs", "carrier_dist_to_goal",
        "carrier_dist_to_sideline", "carrier_speed", "ball_dist_to_carrier",
        "ball_speed",
    ],
    "presser": [
        "presser_dist", "presser_angle", "presser_goalside", "presser_speed",
        "presser_dist_to_goal", "closing_speed",
    ],
    "around_the_ball": [
        "support_dist", "n_pressing_near", "n_pressing_local",
        "n_attacking_near", "n_attacking_local", "numerical_advantage",
    ],
    "passing_lanes": [
        "options_shadowed_by_presser", "n_open_short", "n_open_medium",
        "n_open_long", "n_open_forward", "nearest_open_dist",
    ],
    "defensive_block": [
        "def_line_x", "def_block_depth", "def_block_width", "def_spread",
        "carrier_to_def_line", "n_pressing_goalside", "n_attacking_ahead",
    ],
}
GROUP_COLUMNS = {g: [f"f_{c}" for c in cols] for g, cols in GROUPS.items()}

#: SkillCorner's own inputs, grouped the same way for the attribution chart.
SKILLCORNER_GROUPS = {
    "SkillCorner context": modelling.CONTEXT_FEATURES,
    "SkillCorner geometry": modelling.SKILLCORNER_GEOMETRY,
}

# Validated pair (scripts/validate_palette.js, light surface): blue for the
# tracking groups, orange for SkillCorner's. Text stays in ink colours.
COLOR_TRACKING = "#2a78d6"
COLOR_SKILLCORNER = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"


def _check_groups() -> None:
    listed = [c for cols in GROUPS.values() for c in cols]
    if sorted(listed) != sorted(features.FEATURE_COLUMNS) or len(set(listed)) != len(listed):
        missing = set(features.FEATURE_COLUMNS) - set(listed)
        extra = set(listed) - set(features.FEATURE_COLUMNS)
        raise SystemExit(f"groups do not match the feature list; missing {missing}, extra {extra}")


def _fmt(delta: dict, metric: str) -> str:
    point, lo, hi = delta[metric]
    return f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]"


# --- ablation ---------------------------------------------------------------

def run_ablation(df: pd.DataFrame, args) -> pd.DataFrame:
    base_cols = modelling.FEATURE_SETS["skillcorner_geometry"]
    full_cols = modelling.FEATURE_SETS["tracking_geometry"]
    for g, cols in GROUP_COLUMNS.items():
        modelling.FEATURE_SETS[f"add_{g}"] = base_cols + cols
        modelling.FEATURE_SETS[f"drop_{g}"] = [c for c in full_cols if c not in cols]

    def cv(name):
        t0 = time.time()
        res = modelling.cross_validate(df, name, label=args.label, seed=args.seed)
        print(f"  {name:<24} {len(res.feature_names):>3} features, {time.time() - t0:.0f}s",
              flush=True)
        return res

    def delta(a, b):
        return modelling.bootstrap_delta(a, b, n_samples=args.bootstrap, seed=args.seed)

    print("Fitting models (leave-one-match-out)")
    base = cv("skillcorner_geometry")
    full = cv("tracking_geometry")
    fits = {g: (cv(f"add_{g}"), cv(f"drop_{g}")) for g in GROUPS}

    whole = delta(base, full)
    print("\nAll tracking features over the SkillCorner baseline")
    print(f"  AUC {_fmt(whole, 'roc_auc')}   log loss {_fmt(whole, 'log_loss')}")

    print("\nBy group (positive means the group helps)")
    print("  add  = baseline + this group alone, gain over the baseline")
    print("  drop = full model without this group, loss against the full model\n")

    rows = []
    for g, (add, drop) in fits.items():
        d_add = delta(base, add)    # positive: the group improves the baseline
        d_drop = delta(drop, full)  # positive: the full model is better with it
        print(f"  {g}  ({len(GROUPS[g])} features)")
        print(f"    add   AUC {_fmt(d_add, 'roc_auc')}   log loss {_fmt(d_add, 'log_loss')}")
        print(f"    drop  AUC {_fmt(d_drop, 'roc_auc')}   log loss {_fmt(d_drop, 'log_loss')}")
        row = {"group": g, "n_features": len(GROUPS[g])}
        for kind, d in (("add", d_add), ("drop", d_drop)):
            for metric in ("roc_auc", "log_loss"):
                point, lo, hi = d[metric]
                row[f"{kind}_{metric}"] = point
                row[f"{kind}_{metric}_lo"] = lo
                row[f"{kind}_{metric}_hi"] = hi
        rows.append(row)

    out = pd.DataFrame(rows)
    path = config.REPO_ROOT / "results" / f"ablation_{args.label}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nablation table: {path}")
    return out


# --- SHAP -------------------------------------------------------------------

def _numeric_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """The full feature set with categoricals as integer codes (NaN if missing).

    SHAP's explainers want plain numbers; the gradient boosting model is told
    which columns are categorical, so it treats them exactly as before.
    """
    X = modelling.feature_matrix(df, "tracking_geometry").copy()
    cat_idx = []
    for i, col in enumerate(X.columns):
        if col in modelling.CATEGORICAL:
            codes = X[col].astype("category").cat.codes.astype(float)
            X[col] = codes.where(codes >= 0, np.nan)
            cat_idx.append(i)
    return X.astype(float), cat_idx


def _shap_values(model, X_sample: pd.DataFrame, X_background: pd.DataFrame):
    """SHAP values in log-odds, with a check that they add up to the model.

    Tries the exact tree explainer first. Its support for this model's
    categorical splits is not guaranteed, so the values are checked against the
    model's own log-odds; if they do not add up, the slower model-agnostic
    permutation explainer is used instead.
    """
    import shap

    raw = model.decision_function(X_sample.to_numpy())
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_sample.to_numpy(), check_additivity=False)
        if isinstance(vals, list):
            vals = vals[-1]
        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals[:, :, -1]
        base = np.ravel(explainer.expected_value)[-1]
        gap = float(np.max(np.abs(vals.sum(axis=1) + base - raw)))
        if gap < 1e-3:
            return vals, float(base), "tree"
        print(f"  tree explainer does not add up (max gap {gap:.3g}); "
              "switching to the permutation explainer")
    except Exception as exc:  # noqa: BLE001 - any failure means: use the fallback
        print(f"  tree explainer unavailable ({type(exc).__name__}: {exc}); "
              "using the permutation explainer")

    masker = shap.maskers.Independent(X_background.to_numpy())
    explainer = shap.PermutationExplainer(model.decision_function, masker)
    exp = explainer(X_sample.to_numpy())
    vals = np.asarray(exp.values)
    base = float(np.ravel(exp.base_values)[0])
    gap = float(np.max(np.abs(vals.sum(axis=1) + np.ravel(exp.base_values) - raw)))
    print(f"  permutation explainer, max additivity gap {gap:.3g}")
    return vals, base, "permutation"


def _pretty(col: str) -> str:
    if col.startswith("m0_"):
        return col[3:].replace("_start", "").replace("_id", "") + " (SC)"
    return col[2:] if col.startswith("f_") else col


def _group_of(col: str) -> str:
    for g, cols in GROUP_COLUMNS.items():
        if col in cols:
            return g
    for g, cols in SKILLCORNER_GROUPS.items():
        if col in cols:
            return g
    return "other"


def plot_group_share(share: pd.Series, path: Path, label: str) -> None:
    """Horizontal bars: share of mean |SHAP| per group, SkillCorner in orange."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    share = share.sort_values()
    colors = [COLOR_SKILLCORNER if g.startswith("SkillCorner") else COLOR_TRACKING
              for g in share.index]

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(share.index.str.replace("_", " "), share.values * 100,
                   color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, share.values):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{v:.0%}", va="center", ha="left", fontsize=9, color=INK)

    ax.set_xlabel("share of the model's attribution (mean |SHAP|, %)",
                  fontsize=9, color=INK_MUTED)
    ax.set_title(f"What the regain model leans on ({label})", loc="left",
                 fontsize=11, color=INK)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.xaxis.grid(True, color=INK_MUTED, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.spines["bottom"].set_alpha(0.4)
    ax.set_xlim(0, max(share.values) * 100 * 1.15)
    ax.legend(handles=[Patch(color=COLOR_SKILLCORNER, label="SkillCorner's own metrics"),
                       Patch(color=COLOR_TRACKING, label="measured from tracking")],
              loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run_shap(df: pd.DataFrame, args) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    try:
        import shap  # noqa: F401
    except ImportError:
        raise SystemExit("shap is not installed: pip install shap")

    X, cat_idx = _numeric_matrix(df)
    y = df[args.label].astype(int)
    model = HistGradientBoostingClassifier(
        random_state=args.seed, categorical_features=cat_idx or None,
        **modelling.MODEL_PARAMS,
    )
    t0 = time.time()
    model.fit(X.to_numpy(), y)
    print(f"\nSHAP: model fitted on all {len(X)} rows in {time.time() - t0:.0f}s")

    sample = X.sample(n=min(args.shap_rows, len(X)), random_state=args.seed)
    background = X.sample(n=min(200, len(X)), random_state=args.seed + 1)
    t0 = time.time()
    vals, base, method = _shap_values(model, sample, background)
    print(f"  {method} explainer, {len(sample)} rows, {time.time() - t0:.0f}s")

    mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=X.columns)
    by_group = mean_abs.groupby(mean_abs.index.map(_group_of)).sum()
    share = by_group / by_group.sum()

    print("\n  Share of attribution by group")
    for g, v in share.sort_values(ascending=False).items():
        print(f"    {g:<22} {v:>6.1%}")

    tracking = mean_abs[[c for c in X.columns if c.startswith("f_")]]
    print("\n  Top tracking features (mean |SHAP|, log-odds)")
    for col, v in tracking.sort_values(ascending=False).head(10).items():
        direction = np.corrcoef(sample[col].fillna(sample[col].median()), vals[:, X.columns.get_loc(col)])[0, 1]
        sign = "higher -> more regains" if direction > 0 else "higher -> fewer regains"
        print(f"    {_pretty(col):<28} {v:.4f}   {sign} (r = {direction:+.2f})")

    figdir = config.REPO_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plot_group_share(share, figdir / f"shap_groups_{args.label}.png", args.label)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(
        values=vals, base_values=np.full(len(vals), base),
        data=sample.to_numpy(), feature_names=[_pretty(c) for c in X.columns],
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6.5)
    fig.tight_layout()
    fig.savefig(figdir / f"shap_beeswarm_{args.label}.png", dpi=200)
    plt.close(fig)

    pd.DataFrame({"feature": mean_abs.index, "group": mean_abs.index.map(_group_of),
                  "mean_abs_shap": mean_abs.values}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(config.REPO_ROOT / "results" / f"shap_{args.label}.csv", index=False)
    print(f"\n  figures: {figdir}/shap_groups_{args.label}.png, "
          f"shap_beeswarm_{args.label}.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=500,
                    help="bootstrap samples per comparison (exploratory, so fewer)")
    ap.add_argument("--shap-rows", type=int, default=2000)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()

    _check_groups()
    df = modelling.load_table()
    (config.REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} engagements, label {args.label} ({df[args.label].mean():.1%} positive)\n")

    if not args.skip_ablation:
        run_ablation(df, args)
    if not args.skip_shap:
        run_shap(df, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
