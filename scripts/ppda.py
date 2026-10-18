"""A PPDA-style pressing intensity per team, next to the shape of its press.

    python scripts/ppda.py

PPDA (passes allowed per defensive action) is the usual measure of how much a
team presses: the passes the opponent makes in its own defensive 60% of the
pitch, divided by the defensive actions the pressing team makes there. Lower
means more intense.

It is rebuilt here from SkillCorner's dynamic events, the only data the
competition allows:

passes allowed     the opponent's player possessions that end in a pass, with
                   the pass made in the opponent's own 60% of the pitch
defensive actions  the pressing team's on-ball engagements (pressing,
                   pressure, counter-press, recovery press) on a carrier in
                   that same zone

Classic PPDA counts tackles, interceptions, challenges and fouls instead of
engagements, so the numbers are not comparable with published PPDA values;
they rank teams by the same idea, from one consistent source.

PPDA counts how often a team presses. The metrics in this project describe how
it presses. The table and figure put the two side by side: two teams with the
same PPDA can press very differently.

Direction: x in the dynamic events is already in the event team's own
attacking direction (its goalkeeper always starts near -45, its shots end near
+40, in both halves), so ``attacking_side`` is not needed. On a possession row
that team is the one in possession; on an engagement row it is the pressing
team, so the carrier's x in his own direction is ``-x_start``. Checked in every
match before anything is counted: goalkeeper possessions must start deep in
their own half, shots must end deep in the attacking half, and the engagement
x must agree with the carrier's position in the scenes, which come from the
tracking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import press_outcomes as po  # noqa: E402
import team_profiles as tp  # noqa: E402
from pressing import config, modelling  # noqa: E402

#: x below this, in the possessing team's direction, is its own 60% of the pitch
ZONE_X = -config.PITCH_LENGTH / 2 + 0.6 * config.PITCH_LENGTH

COLUMNS = ["event_id", "event_type", "end_type", "team_id", "player_position",
           "x_start", "x_end"]

BLUE = "#2a78d6"
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def match_counts(match_id, table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Passes allowed and defensive actions for both teams of one match."""
    ev = pd.read_csv(po.events_path(match_id), usecols=COLUMNS, low_memory=False)
    ev["event_id"] = ev["event_id"].astype(str)
    teams = sorted(ev.loc[ev["event_type"] == "player_possession", "team_id"].dropna()
                   .astype(int).unique())
    if len(teams) != 2:
        raise SystemExit(f"match {match_id}: expected two teams, found {teams}")

    # Possessions: x is already in the possessing team's direction; the pass is
    # made where the possession ends.
    poss = ev[ev["event_type"] == "player_possession"].copy()
    poss["x_att"] = poss["x_end"].fillna(poss["x_start"])
    passes = poss[(poss["end_type"] == "pass") & (poss["x_att"] < ZONE_X)]

    # Engagements: x is in the pressing team's direction, the carrier attacks
    # the other way.
    eng = ev[ev["event_type"] == "on_ball_engagement"].copy()
    eng["carrier_x_att"] = -eng["x_start"]
    actions = eng[eng["carrier_x_att"] < ZONE_X]

    rows = []
    for team in teams:
        opponent = teams[1] if team == teams[0] else teams[0]
        rows.append({
            "match_id": match_id, "team_id": team,
            "passes_allowed": int((passes["team_id"] == opponent).sum()),
            "defensive_actions": int((actions["team_id"] == team).sum()),
        })

    # Checks: goalkeepers start near their own goal, shots end near the other
    # one, and the engagement x agrees with the carrier position in the scenes.
    gk = poss["player_position"].astype(str).str.upper().isin(["GK", "GOALKEEPER"])
    mine = table[table["meta_match_id"] == match_id][["meta_event_id", "f_carrier_x"]]
    both = mine.merge(eng[["event_id", "carrier_x_att"]], left_on="meta_event_id",
                      right_on="event_id", how="inner").dropna()
    check = {
        "gk_x": poss.loc[gk, "x_start"].median(),
        "shot_x": poss.loc[poss["end_type"] == "shot", "x_end"].median(),
        "scene_corr": both["f_carrier_x"].corr(both["carrier_x_att"]) if len(both) > 10 else np.nan,
        "max_abs_x": float(np.nanmax(np.abs(poss["x_att"]))),
    }
    return pd.DataFrame(rows), check


def _place_labels(fig, ax, texts, points, marker_size: float = 70) -> None:
    """Place each label where it is unmistakably its own dot's.

    Every candidate spot is scored and the best is taken: leaving the plot or
    covering another label or dot is heavily penalised; a label right next to
    its dot must also be at most half as far from it as from any other dot.
    Labels further out get a thin leader line, which settles whose they are.
    """
    import math

    from matplotlib.transforms import Bbox

    near = [(7, 5, "left"), (7, -12, "left"), (-7, 5, "right"), (-7, -12, "right")]
    far = []
    for dist in (24, 36, 50):
        for angle in (60, 120, 300, 240, 30, 150, 330, 210, 90, 270):
            dx = dist * math.cos(math.radians(angle))
            dy = dist * math.sin(math.radians(angle))
            far.append((dx, dy - 4, "left" if dx > 1 else "right" if dx < -1 else "center"))
    renderer = fig.canvas.get_renderer()
    r = (marker_size ** 0.5) / 2 * fig.dpi / 72 + 3          # dot radius in pixels, plus air
    centres = ax.transData.transform(points)
    dots = [Bbox.from_bounds(x - r, y - r, 2 * r, 2 * r) for x, y in centres]
    frame = ax.get_window_extent(renderer)

    def gap(box, c):  # distance from a point to a box, in pixels
        dx = max(box.x0 - c[0], 0, c[0] - box.x1)
        dy = max(box.y0 - c[1], 0, c[1] - box.y1)
        return (dx * dx + dy * dy) ** 0.5

    placed = []
    for i, t in enumerate(texts):
        others = dots[:i] + dots[i + 1:]
        best = None
        for rank, (dx, dy, ha) in enumerate(near + far):
            t.set_position((dx, dy))
            t.set_ha(ha)
            box = t.get_window_extent(renderer)
            pad = box.expanded(1.04, 1.25)
            score = rank * 0.01
            if not (box.x0 >= frame.x0 and box.x1 <= frame.x1
                    and box.y0 >= frame.y0 and box.y1 <= frame.y1):
                score += 1000
            score += 100 * sum(pad.overlaps(o) for o in placed + others)
            if (dx, dy, ha) in near:
                own = gap(box, centres[i])
                if not all(own < 0.5 * gap(box, c) for j, c in enumerate(centres) if j != i):
                    score += 50
            else:
                score += 5
            if best is None or score < best[0]:
                best = (score, dx, dy, ha)
        _, dx, dy, ha = best
        t.set_position((dx, dy))
        t.set_ha(ha)
        if (dx, dy, ha) not in near:
            ax.annotate("", xy=t.xy, xytext=(dx, dy + 4), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.6,
                                        shrinkA=1, shrinkB=r * 72 / fig.dpi))
        placed.append(t.get_window_extent(renderer))


def plot(table: pd.DataFrame, path: Path, min_matches: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    enough = table["matches"] >= min_matches
    ax.scatter(table.loc[enough, "ppda"], table.loc[enough, "f_n_pressing_local"], s=70,
               color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.scatter(table.loc[~enough, "ppda"], table.loc[~enough, "f_n_pressing_local"], s=42,
               facecolors=SURFACE, edgecolors=BLUE, linewidths=1.5, zorder=3)
    # League medians split the plot into four ways of pressing.
    ax.axvline(table["ppda"].median(), color=INK_MUTED, linewidth=0.8, alpha=0.5, linestyle="--")
    ax.axhline(table["f_n_pressing_local"].median(), color=INK_MUTED, linewidth=0.8, alpha=0.5,
               linestyle="--")
    ax.margins(x=0.14, y=0.12)
    ax.invert_xaxis()  # more intense pressing to the right
    ax.set_xlabel("PPDA style: passes allowed per engagement  (lower = presses more often)",
                  fontsize=9, color=INK_MUTED)
    ax.set_ylabel("defenders within 10 m of the carrier, per press", fontsize=9, color=INK_MUTED)
    ax.set_title("How often a team presses, and how many it sends", fontsize=11,
                 color=INK, loc="left", pad=12)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(True, color=INK_MUTED, alpha=0.15, linewidth=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.01, 0.012,
             "PPDA rebuilt from SkillCorner events, with on-ball engagements as the defensive "
             "actions; not comparable with published PPDA values.\n"
             f"Dashed lines: medians. Hollow dots: fewer than {min_matches} matches.",
             fontsize=7, color=INK_MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    order = table.sort_values("f_n_pressing_local", ascending=False)
    texts = [ax.annotate(r["team"], (r["ppda"], r["f_n_pressing_local"]), xytext=(6, 4),
                         textcoords="offset points", fontsize=8, color=INK)
             for _, r in order.iterrows()]
    _place_labels(fig, ax, texts, order[["ppda", "f_n_pressing_local"]].to_numpy(dtype=float))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL,
                    help="which team_profiles run to put next to PPDA")
    args = ap.parse_args()

    table = modelling.load_table()
    table["meta_event_id"] = table["meta_event_id"].astype(str)

    parts, checks = [], []
    for m in table["meta_match_id"].unique():
        counts, check = match_counts(m, table)
        parts.append(counts)
        checks.append({"match_id": m, **check})
    per_match = pd.concat(parts, ignore_index=True)
    checks = pd.DataFrame(checks)

    print(f"zone: the possessing team's own 60%, x < {ZONE_X:.1f} m in its direction\n")
    print("Direction checks (every match)")
    print(f"  goalkeeper possessions start at x, median: worst {checks['gk_x'].max():.1f} m "
          "(should be deep in the own half)")
    print(f"  possessions ending in a shot end at x, median: worst {checks['shot_x'].min():.1f} m "
          "(should be deep in the attacking half)")
    print(f"  carrier x, events against tracking: correlation worst "
          f"{checks['scene_corr'].min():.3f} (should be close to +1)")
    print(f"  largest |x| {checks['max_abs_x'].max():.1f} m (pitch half length "
          f"{config.PITCH_LENGTH / 2:.1f})")
    failed = ((checks["gk_x"] > -25) | (checks["shot_x"] < 20)
              | ~(checks["scene_corr"] > 0.9))
    if failed.any():
        print("\nDIRECTION CHECK FAILED in matches:")
        print(checks[failed].to_string(index=False))
        print("nothing written; paste this output")
        return 1

    teams = per_match.groupby("team_id").agg(
        matches=("match_id", "nunique"), passes_allowed=("passes_allowed", "sum"),
        defensive_actions=("defensive_actions", "sum")).reset_index()
    teams["ppda"] = teams["passes_allowed"] / teams["defensive_actions"]
    per_match["ppda"] = per_match["passes_allowed"] / per_match["defensive_actions"]
    spread = per_match.groupby("team_id")["ppda"].agg(ppda_min="min", ppda_max="max")
    teams = teams.merge(spread, on="team_id")
    names = tp.team_names(table["meta_match_id"].unique())
    teams["team"] = teams["team_id"].map(lambda t: names.get(int(t), str(t)))

    prof_path = config.REPO_ROOT / "results" / f"team_profiles_{args.label}.csv"
    style_cols = ["f_n_pressing_local", "f_support_dist", "high_press", "over_expected"]
    if prof_path.is_file():
        prof = pd.read_csv(prof_path)[["team_id"] + style_cols]
        teams = teams.merge(prof, on="team_id", how="left")
    else:
        print(f"\n({prof_path} not found; run scripts/team_profiles.py for the style columns)")
    teams = teams.sort_values("ppda").reset_index(drop=True)

    print("\nPPDA style per team (lower = presses more often); range over its matches")
    show = teams.copy()
    show["ppda range"] = [f"{a:.1f} to {b:.1f}" for a, b in zip(show["ppda_min"], show["ppda_max"])]
    cols = ["team", "matches", "passes_allowed", "defensive_actions", "ppda", "ppda range"]
    cols += [c for c in style_cols if c in show]
    print(show[cols].rename(columns={
        "f_n_pressing_local": "defenders near ball", "f_support_dist": "support (m)",
        "high_press": "share in opp. half", "over_expected": "regains over exp."})
        .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    enough = teams[teams["matches"] >= 3]
    if "f_n_pressing_local" in teams and len(enough) >= 5:
        from scipy import stats
        print(f"\nSpearman r with PPDA, {len(enough)} teams with at least 3 matches "
              "(descriptive; too few teams for a test to mean much)")
        for c in style_cols:
            r, p = stats.spearmanr(enough["ppda"], enough[c])
            print(f"  {c:<22} r = {r:+.2f}   p = {p:.2f}")

    resdir, figdir = config.REPO_ROOT / "results", config.REPO_ROOT / "figures"
    resdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    teams.to_csv(resdir / "ppda.csv", index=False)
    per_match.to_csv(resdir / "ppda_per_match.csv", index=False)
    written = "results/ppda.csv, results/ppda_per_match.csv"
    if "f_n_pressing_local" in teams:
        plot(teams, figdir / "ppda_vs_density.png")
        written += ", figures/ppda_vs_density.png"
    print(f"\nwritten: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
