"""How each team presses, and whether it wins the ball more than its situations predict.

    python scripts/team_profiles.py
    python scripts/team_profiles.py --label label_regain_3s

Needs ``cache/oof_<label>.csv`` from ``scripts/run_models.py`` and the dynamic
events files next to the tracking.

Two views of every team, as the pressing side:

Style, what the press looks like. Means over all of the team's engagements:
defenders within 10 m of the carrier, the presser's distance to the nearest
teammate, the depth of the block, passing lanes the presser closes, open short
outlets the carrier is left with, and the share of presses in the opponent's
half. None of these is good or bad on its own; they describe a way of pressing.

Result, what the press achieves. The regain rate, and the regain rate *over
expected*: actual regains minus the regains SkillCorner's context model
predicts for the same situations. The prediction for each engagement comes
from the leave-one-match-out run, so a match never helps predict itself. Also
how often the carrier lost the ball, and how often they were turned back with a
backward pass, from SkillCorner's own event data.

Uncertainty: rows inside a match are not independent, so each match counts as
one draw and the interval is a t interval over a team's matches, which widens
properly when there are only a few. A team seen in fewer than
``--min-matches`` matches gets no interval and is marked; with 20 matches and
13 teams that is several of them, and the honest reading of their numbers is
"not enough to say".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import press_outcomes as po  # noqa: E402
from pressing import config, modelling  # noqa: E402

#: (column, header in the figure, number format)
STYLE = [
    ("f_n_pressing_local", "defenders\nnear the ball", "{:.2f}"),
    ("f_support_dist", "support\ndistance (m)", "{:.1f}"),
    ("f_def_block_depth", "block\ndepth (m)", "{:.1f}"),
    ("f_options_shadowed_by_presser", "lanes closed\nby presser", "{:.2f}"),
    ("f_n_open_short", "open outlets\nleft", "{:.2f}"),
    ("high_press", "presses in\nopp. half", "{:.0%}"),
]

# Diverging pair from the palette: blue below the league average, red above,
# neutral gray at the average. Style is not quality, so neither pole means good.
BLUE, MID, RED = "#2a78d6", "#f0efec", "#e34948"
DOT = "#2a78d6"
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


# --- inputs ----------------------------------------------------------------------

def team_names(match_ids) -> dict[int, str]:
    names = {}
    for m in match_ids:
        path = Path(config.tracking_path(m)).parent / f"{m}_match.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for side in ("home_team", "away_team"):
            team = meta.get(side) or {}
            if "id" in team:
                names[int(team["id"])] = team.get("short_name") or team.get("name") or str(team["id"])
    return names


def load(label: str) -> pd.DataFrame:
    df = modelling.load_table()
    df["meta_event_id"] = df["meta_event_id"].astype(str)

    oof_path = config.REPO_ROOT / "cache" / f"oof_{label}.csv"
    if not oof_path.is_file():
        raise SystemExit(f"{oof_path} not found; run scripts/run_models.py --label {label} first")
    oof = pd.read_csv(oof_path)
    oof = oof.rename(columns={"match_id": "meta_match_id", "event_id": "meta_event_id"})
    oof["meta_event_id"] = oof["meta_event_id"].astype(str)
    df = df.merge(oof[["meta_match_id", "meta_event_id", "y", "p_context"]],
                  on=["meta_match_id", "meta_event_id"], how="left")
    missing = df["p_context"].isna().mean()
    if missing > 0.01:
        raise SystemExit(f"{missing:.1%} of engagements have no out-of-fold prediction; "
                         "rerun scripts/run_models.py so the two files match")
    agree = (df["y"].astype(float) == df[label].astype(float)).mean()
    if agree < 0.999:
        raise SystemExit(f"labels in the table and the predictions disagree on {1 - agree:.1%} "
                         "of rows; rerun scripts/run_models.py")

    endings = po.carrier_endings(df)
    df = df.merge(endings[["meta_match_id", "meta_event_id", "lost the ball", "backward pass"]],
                  on=["meta_match_id", "meta_event_id"], how="left")
    df["high_press"] = (df["f_carrier_x"] < 0).astype(float)
    df["_y"] = df[label].astype(float)
    return df


# --- per team --------------------------------------------------------------------

def over_expected_interval(g: pd.DataFrame) -> tuple[float, float]:
    """95% interval for regains over expected, treating each match as one draw.

    With three to seven matches per team, resampling matches has too few
    distinct outcomes to be trusted and gives intervals that are too narrow.
    A t interval on the per-match values is wider when the matches are few,
    which is the honest behaviour. Points of regain rate.
    """
    from scipy import stats

    by_match = g.groupby("meta_match_id").apply(
        lambda m: (m["_y"].sum() - m["p_context"].sum()) / len(m), include_groups=False
    )
    k = len(by_match)
    pooled = (g["_y"].sum() - g["p_context"].sum()) / len(g)
    half = stats.t.ppf(0.975, k - 1) * by_match.std(ddof=1) / np.sqrt(k)
    return (pooled - half) * 100, (pooled + half) * 100


def profiles(df: pd.DataFrame, names: dict, min_matches: int) -> pd.DataFrame:
    rows = []
    for team_id, g in df.groupby("meta_team_id"):
        n_matches = g["meta_match_id"].nunique()
        row = {
            "team_id": int(team_id),
            "team": names.get(int(team_id), str(int(team_id))),
            "matches": n_matches,
            "presses": len(g),
        }
        for col, _, _ in STYLE:
            row[col] = g[col].mean()
        row["regain_rate"] = g["_y"].mean() * 100
        row["expected_rate"] = g["p_context"].mean() * 100
        row["over_expected"] = row["regain_rate"] - row["expected_rate"]
        if n_matches >= min_matches:
            row["over_lo"], row["over_hi"] = over_expected_interval(g)
        else:
            row["over_lo"] = row["over_hi"] = np.nan
        row["lost_ball_rate"] = g["lost the ball"].mean() * 100
        row["backward_rate"] = g["backward pass"].mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).sort_values("over_expected", ascending=False).reset_index(drop=True)


# --- the figure ------------------------------------------------------------------

def plot(table: pd.DataFrame, league: pd.Series, path: Path, label: str,
         min_matches: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("style", [BLUE, MID, RED])
    cols = [c for c, _, _ in STYLE]
    values = table[cols].to_numpy(dtype=float)
    # Colour is the relative difference from the league average, saturating at
    # +-25%. Scaling by the spread between teams instead would paint a 1%
    # difference deep red whenever the teams happen to be alike.
    base = league[cols].to_numpy(dtype=float)
    z = np.clip((values - base) / np.where(base == 0, np.nan, np.abs(base)) / 0.25, -1, 1)

    n = len(table)
    fig = plt.figure(figsize=(11, 0.42 * n + 2.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.4], wspace=0.06)
    ax = fig.add_subplot(grid[0])
    bx = fig.add_subplot(grid[1], sharey=ax)

    # Style grid: colour says above or below the league, the number says how much.
    ax.imshow(np.nan_to_num(z), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(n):
        for j, (_, _, fmt) in enumerate(STYLE):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(STYLE)), [h for _, h, _ in STYLE], fontsize=8, color=INK_MUTED)
    ax.xaxis.tick_top()
    labels = [f"{t}  ({m})" for t, m in zip(table["team"], table["matches"])]
    ax.set_yticks(range(n), labels, fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(-0.5, len(STYLE)), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Result: regain rate over expected, with match-resampled intervals.
    y = np.arange(n)
    bx.axvline(0, color=INK_MUTED, linewidth=0.8, alpha=0.6)
    has_ci = table["over_lo"].notna().to_numpy()
    bx.hlines(y[has_ci], table.loc[has_ci, "over_lo"], table.loc[has_ci, "over_hi"],
              color=DOT, linewidth=2, alpha=0.45)
    bx.scatter(table.loc[has_ci, "over_expected"], y[has_ci], s=40, color=DOT,
               edgecolors=SURFACE, linewidths=2, zorder=3)
    bx.scatter(table.loc[~has_ci, "over_expected"], y[~has_ci], s=40, facecolors=SURFACE,
               edgecolors=DOT, linewidths=1.5, zorder=3)
    bx.set_xlabel("regains over expected\n(points of regain rate)", fontsize=8, color=INK_MUTED)
    bx.xaxis.set_label_position("top")
    bx.xaxis.tick_top()
    bx.tick_params(axis="x", colors=INK_MUTED, labelsize=8, length=0)
    bx.tick_params(axis="y", which="both", left=False, labelleft=False, length=0)
    bx.set_facecolor(SURFACE)
    bx.xaxis.grid(True, color=INK_MUTED, alpha=0.15, linewidth=0.8)
    for s in bx.spines.values():
        s.set_visible(False)

    fig.text(0.01, 0.015,
             f"Style: colour shows how far each team is from the league average, full colour at 25% "
             f"(red above, blue below; neither is better). Result: regains minus SkillCorner-context expectation, {label}. "
             f"Lines are 95% t intervals over matches; hollow dots: fewer than {min_matches} matches, "
             f"no interval. Number after the name: matches.",
             fontsize=7, color=INK_MUTED, wrap=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.1)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --- main ------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--min-matches", type=int, default=3)
    args = ap.parse_args()

    df = load(args.label)
    names = team_names(df["meta_match_id"].unique())
    league = df[[c for c, _, _ in STYLE]].mean()
    table = profiles(df, names, args.min_matches)

    print(f"{len(df)} presses by {len(table)} teams over {df['meta_match_id'].nunique()} matches, "
          f"label {args.label}, league regain rate {df['_y'].mean():.1%}\n")

    show = table.copy()
    show["over expected"] = [
        f"{o:+.1f} [{lo:+.1f}, {hi:+.1f}]" if pd.notna(lo) else f"{o:+.1f}  (too few matches)"
        for o, lo, hi in zip(show["over_expected"], show["over_lo"], show["over_hi"])
    ]
    print("Result (rates in %)")
    print(show[["team", "matches", "presses", "regain_rate", "expected_rate", "over expected",
                "lost_ball_rate", "backward_rate"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\nStyle (means per press; league average in the last row)")
    style = show[["team"] + [c for c, _, _ in STYLE]].copy()
    style.loc[len(style)] = ["league"] + list(league)
    style.columns = ["team"] + [h.replace("\n", " ") for _, h, _ in STYLE]
    print(style.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    enough = table[table["matches"] >= args.min_matches]
    if len(enough) >= 5:
        from scipy import stats
        print(f"\nAcross the {len(enough)} teams with at least {args.min_matches} matches, "
              "Spearman r between style and regains over expected")
        print(f"  ({len(enough)} teams and {len(STYLE)} comparisons: exploratory; with this few "
              "teams even a large r can be chance)")
        for col, header, _ in STYLE:
            r, pval = stats.spearmanr(enough[col], enough["over_expected"])
            print(f"  {header.replace(chr(10), ' '):<26} r = {r:+.2f}   p = {pval:.2f}")

    resdir, figdir = config.REPO_ROOT / "results", config.REPO_ROOT / "figures"
    resdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(resdir / f"team_profiles_{args.label}.csv", index=False)
    plot(table, league, figdir / f"team_profiles_{args.label}.png", args.label, args.min_matches)
    print(f"\nwritten: results/team_profiles_{args.label}.csv, "
          f"figures/team_profiles_{args.label}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
