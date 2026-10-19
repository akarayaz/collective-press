"""Figure 1: what each part of a press does.

    python scripts/figure_press.py                    # picks a clear example press
    python scripts/figure_press.py --list             # the candidates it picks from
    python scripts/figure_press.py --pick 3           # the fourth candidate instead
    python scripts/figure_press.py --event 1925299 12345   # a press of your choice

Left: one press from the data, with the five pressing metrics drawn on the
pitch exactly as :mod:`pressing.features` computes them, each numbered.
Right: with the same numbers, how one more unit of each metric changes how the
carrier's possession ends, with SkillCorner's context held fixed; below, the
body pose finding from the two matches that have it.

Needs ``results/collective_press_<label>.csv`` (``collective_press.py``),
``results/press_outcomes.csv`` (``press_outcomes.py``) and, for the last row,
``results/bodypose_press.csv`` (``bodypose_press.py``). The example press is
read from the tracking, like every scene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, features, modelling  # noqa: E402

# Palette: team identity as in pressing/viz.py; the grid's polarity uses the
# diverging pair (blue less likely, red more likely) around a neutral gray.
SURFACE, LINE = "#fcfcfb", "#c9c8c1"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8c8b84"
PRESS, POSSESS = "#2a78d6", "#eb6834"
LESS, MID, MORE = "#2a78d6", "#f0efec", "#e34948"
SATURATE_AT = 4.0  # points of probability at which a cell reaches full colour

#: grid rows: (number, label, unit as read, source metric, sign to apply)
ROWS = [
    ("1", "Defenders within 10 m", "one more", "press density", 1),
    ("2", "Presser's support", "teammate 5 m closer", "press support", -1),
    ("3", "Depth of the block", "10 m deeper", "block compactness", 1),
    ("4", "Open short outlets", "one more", "open outlets", 1),
    ("5", "Lanes the presser closes", "one more", "presser's cover shadow", 1),
]
#: grid columns: (header, source outcome name)
COLUMNS = [
    ("Won\nin 5 s", "regain"),
    ("Ball\nlost", "lost the ball"),
    ("Pass\nfails", "pass failed"),
    ("Back\npass", "backward pass"),
    ("Long\npass", "long pass"),
    ("High\npass", "high pass"),
]


# --- the numbers -----------------------------------------------------------------

def effect_table(label: str) -> tuple[pd.DataFrame, dict]:
    """(rows x columns of (points, lo, hi)) and the base rate of each column."""
    res = config.REPO_ROOT / "results"
    cp = pd.read_csv(res / f"collective_press_{label}.csv")
    cp = cp[cp["model"] == "all five together"]
    po = pd.read_csv(res / "press_outcomes.csv")

    cells, base = {}, {}
    table = modelling.load_table()
    base["regain"] = table[label].mean() * 100
    for outcome, g in po.groupby("outcome"):
        base[outcome] = g["base_rate"].iloc[0] * 100
    for num, _, _, metric, sign in ROWS:
        for _, outcome in COLUMNS:
            src = (cp[cp["metric"] == metric] if outcome == "regain"
                   else po[(po["metric"] == metric) & (po["outcome"] == outcome)])
            if src.empty:
                cells[(num, outcome)] = (np.nan, np.nan, np.nan)
                continue
            r = src.iloc[0]
            p, lo, hi = r["points"], r["points_lo"], r["points_hi"]
            if sign < 0:
                p, lo, hi = -p, -hi, -lo
            cells[(num, outcome)] = (p, lo, hi)

    pose_path = res / "bodypose_press.csv"
    if pose_path.is_file():
        bp = pd.read_csv(pose_path)
        bp = bp[(bp["term"] == "carrier facing his own half")
                & (bp["kind"] == "primary, joint")]
        for _, outcome in COLUMNS:
            name = label if outcome == "regain" else outcome
            r = bp[bp["outcome"] == name]
            cells[("B", outcome)] = ((r["points"].iloc[0], r["points_lo"].iloc[0],
                                      r["points_hi"].iloc[0]) if len(r) else
                                     (np.nan, np.nan, np.nan))
    return cells, base


# --- choosing the example ----------------------------------------------------------

def candidates(label: str) -> pd.DataFrame:
    """Presses that show every metric clearly, best first.

    In the opponent's half, three or more defenders within 10 m, the presser's
    support close, at least one lane closed and one open outlet left (so both
    kinds of lane appear), the ball won back, the presser 3 to 8 m away.
    """
    t = modelling.load_table()
    ok = (
        (t["f_carrier_x"] < -5)
        & (t["f_n_pressing_local"] >= 3)
        & (t["f_support_dist"] <= 7)
        & (t["f_options_shadowed_by_presser"] >= 1)
        & (t["f_n_open_short"] >= 1)
        & (t["f_presser_dist"].between(3, 8))
        & (t[label] == 1)
    )
    c = t[ok].copy()
    # Prefer a typical presser distance and few players far from the action.
    c["score"] = (c["f_presser_dist"] - 5.5).abs() + 0.2 * c["f_n_open_short"]
    cols = ["meta_match_id", "meta_event_id", "f_n_pressing_local", "f_support_dist",
            "f_options_shadowed_by_presser", "f_n_open_short", "f_presser_dist", "score"]
    return c.sort_values("score")[cols].reset_index(drop=True)


def load_scene(match_id, event_id):
    from pressing import events, scene

    eng = events.load_engagements(match_id)
    row = eng[eng["event_id"].astype(str) == str(event_id)]
    if row.empty:
        raise SystemExit(f"event {event_id} not found in match {match_id}")
    return scene.build_scene(match_id, row.iloc[0])


# --- drawing -----------------------------------------------------------------------

def _badge(ax, x, y, text, size=10, **kw):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=SURFACE,
            fontweight="bold", zorder=9,
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=INK, edgecolor="none"), **kw)


PLAYER_S, RING_S = 95, 330          # marker areas, points squared
PLAYER_PT = (PLAYER_S ** 0.5) / 2 + 2  # radius to keep clear, points
RING_PT = (RING_S ** 0.5) / 2 + 2


class _Layout:
    """Places labels and badges on the pitch where they cover nothing.

    Sizes are reasoned in points and converted to metres with the axes' own
    scale, so the result holds whatever the zoom. Every candidate spot near the
    thing being named is scored: covering a player, a ring, a drawn line or
    another label costs a lot, distance a little; the cheapest spot wins.
    """

    def __init__(self, ax, window):
        self.ax = ax
        self.x0, self.x1, self.y0, self.y1 = window
        ax.figure.canvas.draw()  # the axes' final size, after equal aspect
        bbox = ax.get_window_extent()
        width_pt = bbox.width * 72 / ax.figure.dpi
        lo, hi = ax.get_xlim()
        self.m = abs(hi - lo) / width_pt  # metres per point
        self.points = np.zeros((0, 3))    # x, y, radius in metres
        self.boxes = []                   # (x0, x1, y0, y1) in metres

    def avoid_points(self, xy, radius_pt):
        r = np.full((len(xy), 1), radius_pt * self.m)
        self.points = np.vstack([self.points, np.hstack([xy, r])])

    def avoid_segment(self, ax_, ay_, bx_, by_):
        n = max(2, int(np.hypot(bx_ - ax_, by_ - ay_) / 1.0))
        t = np.linspace(0, 1, n)[:, None]
        pts = np.hstack([ax_ + t * (bx_ - ax_), ay_ + t * (by_ - ay_)])
        self.avoid_points(pts, 2.5)

    def _best(self, w, h, anchors, radii):
        best, best_cost = None, np.inf
        for k, (ex, ey) in enumerate(anchors):
            for r in radii:
                for a in np.radians(np.arange(0, 360, 30)):
                    bx, by = ex + r * np.cos(a), ey + r * np.sin(a)
                    box = (bx - w / 2, bx + w / 2, by - h / 2, by + h / 2)
                    if (box[0] < self.x0 + 0.5 or box[1] > self.x1 - 0.5
                            or box[2] < self.y0 + 0.5 or box[3] > self.y1 - 0.5):
                        continue
                    # distance from each obstacle centre to the box
                    dx = np.maximum(0, np.maximum(box[0] - self.points[:, 0],
                                                  self.points[:, 0] - box[1]))
                    dy = np.maximum(0, np.maximum(box[2] - self.points[:, 1],
                                                  self.points[:, 1] - box[3]))
                    hits = int((np.hypot(dx, dy) < self.points[:, 2]).sum())
                    pad = 0.6
                    hits += sum(not (box[1] + pad < b[0] or b[1] + pad < box[0]
                                     or box[3] + pad < b[2] or b[3] + pad < box[2])
                                for b in self.boxes)
                    cost = 100 * hits + r + 0.4 * k
                    if cost < best_cost:
                        best, best_cost = (bx, by, box), cost
        return best

    def badge(self, num, anchors):
        d = 17 * self.m  # badge diameter
        found = self._best(d, d, anchors, [x * self.m for x in (12, 18, 26, 36)])
        if found:
            bx, by, box = found
            self.boxes.append(box)
            _badge(self.ax, bx, by, num)

    def text(self, label, anchors):
        w, h = (len(label) * 5.6 + 6) * self.m, 14 * self.m
        found = self._best(w, h, anchors, [x * self.m for x in (14, 20, 28, 38, 50)])
        if not found:
            return
        bx, by, box = found
        self.boxes.append(box)
        ex, ey = anchors[0]
        far = np.hypot(bx - ex, by - ey) > 24 * self.m
        self.ax.annotate(
            label, (ex, ey), xytext=(bx, by), textcoords="data", ha="center",
            va="center", fontsize=9, color=INK_2, zorder=8,
            arrowprops=(dict(arrowstyle="-", color=INK_3, linewidth=0.7, shrinkA=0,
                             shrinkB=RING_PT * 0.8) if far else None),
            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none",
                      alpha=0.9))


def draw_press(ax, sc) -> dict:
    """The example press, zoomed in, with the five metrics drawn on it."""
    from matplotlib.patches import Circle

    from pressing import viz

    carrier, presser = sc.carrier, sc.presser
    cx, cy = float(carrier["x"]), float(carrier["y"])
    pressing, attacking = sc.pressing, sc.attacking
    viz.draw_pitch(ax)

    # The window: more room towards the goal the pressing team defends (+x).
    hx, hy = config.PITCH_LENGTH / 2, config.PITCH_WIDTH / 2
    def window(lo, hi, bound):
        """Keep the window's size; slide it back inside the pitch at a line."""
        if lo < -bound:
            lo, hi = -bound, hi + (-bound - lo)
        if hi > bound:
            lo, hi = lo - (hi - bound), bound
        return lo, hi

    x0, x1 = window(cx - 21, cx + 31, hx + 2)
    y0, y1 = window(cy - 19, cy + 19, hy + 2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # 1. density: the 10 m circle, with the defenders inside it counted.
    ax.add_patch(Circle((cx, cy), features.LOCAL_M, facecolor=PRESS, alpha=0.06,
                        edgecolor="none", zorder=1))
    ax.add_patch(Circle((cx, cy), features.LOCAL_M, facecolor="none", edgecolor=PRESS,
                        linewidth=1.1, linestyle=(0, (5, 3)), zorder=2))
    n_local = int((np.hypot(pressing["x"].astype(float) - cx,
                            pressing["y"].astype(float) - cy) <= features.LOCAL_M).sum())

    # 4 and 5. lanes, as features.py defines them.
    receivers = attacking.drop(index=carrier.name, errors="ignore")
    rx, ry = receivers["x"].astype(float).to_numpy(), receivers["y"].astype(float).to_numpy()
    rdist = np.hypot(rx - cx, ry - cy)
    blockers = pressing[["x", "y"]].astype(float).to_numpy()
    presser_xy = presser[["x", "y"]].astype(float).to_numpy()[None, :]
    reach = rdist <= features.LONG_PASS_M
    shadowed = features._lane_blocked(cx, cy, receivers, presser_xy) & reach
    is_open = features._open(cx, cy, receivers, blockers, rdist)
    open_short = is_open & (rdist <= features.SHORT_PASS_M)
    lane_mid = {}
    for i in np.where(shadowed)[0]:
        ax.plot([cx, rx[i]], [cy, ry[i]], color=INK_3, linewidth=1.6,
                linestyle=(0, (2, 2.5)), zorder=2, solid_capstyle="round")
        # The presser stands in this lane, so the badge goes past him.
        lane_mid.setdefault("5", ((float(presser["x"]) + rx[i]) / 2,
                                  (float(presser["y"]) + ry[i]) / 2))
    for i in np.where(open_short)[0]:
        ax.plot([cx, rx[i]], [cy, ry[i]], color=POSSESS, linewidth=2.2, alpha=0.9,
                zorder=2, solid_capstyle="round")
        lane_mid.setdefault("4", ((cx + rx[i]) / 2, (cy + ry[i]) / 2))

    # 2. support: the presser's nearest teammate.
    others = pressing.drop(index=presser.name, errors="ignore")
    d = np.hypot(others["x"].astype(float) - float(presser["x"]),
                 others["y"].astype(float) - float(presser["y"]))
    mate = others.loc[d.idxmin()]
    sx, sy = float(mate["x"]), float(mate["y"])
    px, py = float(presser["x"]), float(presser["y"])
    ax.plot([px, sx], [py, sy], color=INK, linewidth=1.6, zorder=3)

    # 3. the back line: the mean x of the four deepest outfield defenders.
    outfield = features._outfield(pressing)
    back = float(outfield["x"].astype(float).nlargest(features.DEFENSIVE_LINE_PLAYERS).mean())
    depth = float(outfield["x"].astype(float).max() - outfield["x"].astype(float).min())
    if x0 < back < x1:
        ax.plot([back, back], [y0 + 1, y1 - 1], color=INK_3, linewidth=1.1,
                linestyle=(0, (6, 4)), zorder=1)

    # Players: pressing team circles, team in possession squares, as in viz.py.
    # Small enough that a marker and the man he marks stay apart.
    for team, colour, marker in ((pressing, PRESS, "o"), (attacking, POSSESS, "s")):
        ax.scatter(team["x"].astype(float), team["y"].astype(float), s=PLAYER_S,
                   marker=marker, facecolor=colour, edgecolor=SURFACE, linewidth=1.3,
                   zorder=5)
    for who, colour in ((presser, PRESS), (carrier, POSSESS)):
        ax.scatter([float(who["x"])], [float(who["y"])], s=RING_S, facecolor="none",
                   edgecolor=colour, linewidth=1.8, zorder=6)
    if sc.ball is not None:
        ax.scatter([sc.ball[0]], [sc.ball[1]], s=34, facecolor=INK, edgecolor=SURFACE,
                   linewidth=1.0, zorder=7)

    # Labels and numbered badges: each goes to the free spot nearest what it names.
    lay = _Layout(ax, (x0, x1, y0, y1))
    players_xy = np.vstack([pressing[["x", "y"]].astype(float).to_numpy(),
                            attacking[["x", "y"]].astype(float).to_numpy()])
    lay.avoid_points(players_xy, PLAYER_PT)
    lay.avoid_points(np.array([[cx, cy], [px, py]]), RING_PT)
    if sc.ball is not None:
        lay.avoid_points(np.array([sc.ball]), 5)
    lay.avoid_segment(px, py, sx, sy)
    for i in np.where(shadowed | open_short)[0]:
        lay.avoid_segment(cx, cy, rx[i], ry[i])

    lay.text("carrier", [(cx, cy)])
    lay.text("presser", [(px, py)])

    def segment(ax_, ay_, bx_, by_, n=5):
        return [(ax_ + t * (bx_ - ax_), ay_ + t * (by_ - ay_)) for t in np.linspace(0.3, 0.7, n)]

    circle = [(cx + features.LOCAL_M * np.cos(a), cy + features.LOCAL_M * np.sin(a))
              for a in np.radians(np.arange(90, 450, 30))]
    lay.badge("1", circle)
    lay.badge("2", segment(px, py, sx, sy))
    if shadowed.any():
        i = np.where(shadowed)[0][0]
        lay.badge("5", segment(px, py, rx[i], ry[i]))
    if open_short.any():
        i = np.where(open_short)[0][0]
        lay.badge("4", segment(cx, cy, rx[i], ry[i]))
    if x0 < back < x1:
        lay.badge("3", [(back, y) for y in np.linspace(y1 - 3, y0 + 3, 7)])
    return {"n_local": n_local, "support": float(d.min()), "depth": depth,
            "shadowed": int(shadowed.sum()), "open_short": int(open_short.sum())}


def _cell_colour(v):
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    cmap = LinearSegmentedColormap.from_list("div", [LESS, MID, MORE])
    rgb = cmap(0.5 + 0.5 * float(np.clip(v / SATURATE_AT, -1, 1)))[:3]
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return rgb, (SURFACE if lum < 0.5 else INK)


def draw_grid(ax, cells: dict, base: dict, has_pose: bool) -> None:
    from matplotlib.patches import FancyBboxPatch

    rows = [r[:3] for r in ROWS]
    if has_pose:
        rows.append(("B", "Carrier facing own half", "body pose, 2 matches"))
    n_cols = len(COLUMNS)
    # y positions top to bottom, with a gap before the body pose row
    ys = [-(i + (0.45 if r[0] == "B" else 0)) for i, r in enumerate(rows)]
    ax.set_xlim(-4.15, n_cols - 0.45)
    ax.set_ylim(min(ys) - 0.62, 1.55)
    ax.axis("off")

    for j, (head, outcome) in enumerate(COLUMNS):
        ax.text(j, 1.02, head, ha="center", va="center", fontsize=10.5, color=INK,
                linespacing=1.1)
        if outcome in base:
            ax.text(j, 0.52, f"{base[outcome]:.0f}%", ha="center", va="center",
                    fontsize=9, color=INK_3)
    ax.text(-4.1, 0.52, "how often it happens", ha="left", va="center", fontsize=9,
            color=INK_3)

    for (num, label, unit), y in zip(rows, ys):
        if num == "B":
            ax.text(-4.08, y, "B", ha="left", va="center", fontsize=10, color=SURFACE,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=INK_2, edgecolor="none"))
        else:
            _badge(ax, -3.95, y, num)
        ax.text(-3.6, y + 0.13, label, ha="left", va="center", fontsize=10.5, color=INK)
        ax.text(-3.6, y - 0.21, unit, ha="left", va="center", fontsize=9, color=INK_2,
                style="italic" if num == "B" else "normal")
        for j, (_, outcome) in enumerate(COLUMNS):
            p, lo, hi = cells.get((num, outcome), (np.nan, np.nan, np.nan))
            if np.isnan(p):
                continue
            sig = lo > 0 or hi < 0
            fill, text = _cell_colour(p) if sig else (SURFACE, INK_3)
            ax.add_patch(FancyBboxPatch((j - 0.44, y - 0.36), 0.88, 0.72,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=fill, linewidth=0 if sig else 0.8,
                                        edgecolor=LINE, zorder=2))
            value = f"{p:+.1f}" if round(p, 1) != 0 else "0.0"
            ax.text(j, y, value, ha="center", va="center", fontsize=11, color=text,
                    fontweight="bold" if sig else "normal", zorder=3)

    # Legend for the colours, under the grid.
    ly = min(ys) - 0.95
    keys = ((_cell_colour(SATURATE_AT)[0], 0, "more likely"),
            (_cell_colour(-SATURATE_AT)[0], 0, "less likely"),
            (SURFACE, 0.8, "no clear effect"))
    for k, (fill, edge, word) in enumerate(keys):
        x = -0.3 + 1.95 * k
        ax.add_patch(FancyBboxPatch((x - 0.2, ly - 0.15), 0.4, 0.3,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=fill, edgecolor=LINE, linewidth=edge,
                                    zorder=2, clip_on=False))
        ax.text(x + 0.32, ly, word, ha="left", va="center", fontsize=9, color=INK_2)
    ax.set_ylim(ly - 0.35, 1.55)


def make_figure(sc, cells, base, path: Path, label: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(11, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.25], wspace=0.03,
                            left=0.01, right=0.99, top=0.82, bottom=0.12)
    ax = fig.add_subplot(grid[0])
    gx = fig.add_subplot(grid[1])
    gx.set_facecolor(SURFACE)

    info = draw_press(ax, sc)
    has_pose = ("B", "regain") in cells
    draw_grid(gx, cells, base, has_pose)

    fig.text(0.015, 0.955, "What each part of a press does", fontsize=16, color=INK,
             ha="left", va="center")
    fig.text(0.015, 0.905,
             "Left: a press from the data, the five metrics drawn as they are measured. "
             "Right: what one more unit of each does to how the\ncarrier's possession ends, "
             "in percentage points, SkillCorner's context held fixed; coloured where the 95% "
             "interval excludes zero.",
             fontsize=10, color=INK_2, ha="left", va="center", linespacing=1.35)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=PRESS,
               markeredgecolor=PRESS, label="pressing team"),
        Line2D([], [], marker="s", linestyle="", markersize=7.5, markerfacecolor=POSSESS,
               markeredgecolor=POSSESS, label="in possession"),
        Line2D([], [], color=POSSESS, linewidth=2.2, label="open outlet"),
        Line2D([], [], color=INK_3, linewidth=1.4, linestyle=(0, (2, 2.5)),
               label="lane closed by presser"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=2,
              frameon=False, fontsize=9, labelcolor=INK_2, handlelength=1.8,
              columnspacing=1.4)
    foot = ("17,180 presses in 20 A-League matches. Row B: the two matches with body pose, "
            "the carrier facing more than 90 degrees away from the goal he attacks.")
    fig.text(0.015, 0.02, foot, fontsize=8.5, color=INK_3, ha="left", va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--list", action="store_true", help="print the candidate presses")
    ap.add_argument("--pick", type=int, default=0, help="which candidate to draw")
    ap.add_argument("--event", nargs=2, metavar=("MATCH", "EVENT"),
                    help="draw this press instead")
    args = ap.parse_args()

    if args.event:
        match_id, event_id = int(args.event[0]), args.event[1]
    else:
        c = candidates(args.label)
        if c.empty:
            raise SystemExit("no press matches the example criteria; pass --event")
        if args.list:
            print(c.head(15).to_string(float_format=lambda v: f"{v:.1f}"))
            return 0
        r = c.iloc[min(args.pick, len(c) - 1)]
        match_id, event_id = int(r["meta_match_id"]), str(r["meta_event_id"])
    print(f"example press: match {match_id}, event {event_id}")

    cells, base = effect_table(args.label)
    sc = load_scene(match_id, event_id)
    figdir = config.REPO_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / "figure1_press.png"
    info = make_figure(sc, cells, base, path, args.label)
    print(f"drawn: {info}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
