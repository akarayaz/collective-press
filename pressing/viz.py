"""Drawing a pressing scene on a pitch.

One figure per engagement: both teams, the ball, and the two players the
engagement is about. Used to sanity-check the data by eye, and later to show
what the model sees.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import config
from .scene import Scene

# Categorical identity: two teams, two hues, two marker shapes. Colours are
# slots 1 and 2 of the reference palette; the pair passes the colour-vision
# checks, and the marker shape repeats the distinction so identity never rests
# on colour alone.
SURFACE = "#fcfcfb"
LINE = "#b8b7b0"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
PRESSING_COLOR = "#2a78d6"
ATTACKING_COLOR = "#eb6834"
UNKNOWN_COLOR = "#8c8b84"
BALL_COLOR = "#0b0b0b"

SIDE_STYLE = {
    "pressing": (PRESSING_COLOR, "o"),
    "attacking": (ATTACKING_COLOR, "s"),
    "unknown": (UNKNOWN_COLOR, "^"),
}

#: Velocity arrows show where a player would be after this many seconds.
ARROW_SECONDS = 0.5


def draw_pitch(ax=None, length: float = None, width: float = None):
    """Plain pitch markings, with the origin at the centre spot."""
    length = config.PITCH_LENGTH if length is None else length
    width = config.PITCH_WIDTH if width is None else width
    hx, hy = length / 2, width / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    ax.set_facecolor(SURFACE)
    opts = dict(color=LINE, linewidth=1.2, zorder=1)

    ax.add_patch(mpatches.Rectangle((-hx, -hy), length, width, fill=False, **opts))
    ax.plot([0, 0], [-hy, hy], **opts)
    ax.add_patch(mpatches.Circle((0, 0), 9.15, fill=False, **opts))
    ax.plot([0], [0], marker="o", markersize=2.5, color=LINE, zorder=1)

    for sign in (-1, 1):
        # penalty area, six-yard box, penalty spot, goal
        ax.add_patch(
            mpatches.Rectangle(
                (sign * hx - (16.5 if sign > 0 else 0), -20.16),
                16.5, 40.32, fill=False, **opts,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (sign * hx - (5.5 if sign > 0 else 0), -9.16),
                5.5, 18.32, fill=False, **opts,
            )
        )
        ax.plot([sign * (hx - 11)], [0], marker="o", markersize=2.5,
                color=LINE, zorder=1)
        ax.add_patch(
            mpatches.Rectangle(
                (sign * hx if sign > 0 else -hx - 2, -3.66),
                2, 7.32, fill=False, **opts,
            )
        )

    ax.set_xlim(-hx - 4, hx + 4)
    ax.set_ylim(-hy - 4, hy + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_scene(scene: Scene, ax=None, show_velocity: bool = True, title: bool = True):
    """Draw one :class:`~pressing.scene.Scene`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8.5))
    draw_pitch(ax)

    players = scene.players
    for side, (color, marker) in SIDE_STYLE.items():
        grp = players[players["side"] == side]
        if grp.empty:
            continue
        # Extrapolated positions are drawn hollow: the camera did not see those
        # players in this frame, so their coordinates are model output.
        for detected, sub in grp.groupby(grp["is_detected"].fillna(True).astype(bool)):
            ax.scatter(
                sub["x"], sub["y"],
                s=330, marker=marker, zorder=3,
                facecolors=color if detected else SURFACE,
                edgecolors=color,
                linewidths=1.4,
            )

    for pid, p in players.iterrows():
        label = "" if p["number"] is None or _isnan(p["number"]) else str(int(p["number"]))
        detected = bool(p["is_detected"]) if p["is_detected"] is not None else True
        color, _ = SIDE_STYLE[p["side"]]
        ax.text(
            p["x"], p["y"], label,
            ha="center", va="center", fontsize=7.5, zorder=4,
            color=SURFACE if detected else color,
        )

    if show_velocity:
        for _, p in players.iterrows():
            if _isnan(p.get("vx")) or _isnan(p.get("vy")):
                continue
            dx, dy = p["vx"] * ARROW_SECONDS, p["vy"] * ARROW_SECONDS
            if np.hypot(dx, dy) < 0.35:
                continue
            color, _ = SIDE_STYLE[p["side"]]
            ax.annotate(
                "", xy=(p["x"] + dx, p["y"] + dy), xytext=(p["x"], p["y"]),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3,
                                alpha=0.75, shrinkA=9, shrinkB=0),
                zorder=2,
            )

    presser = scene.presser
    carrier = scene.carrier
    _ring(ax, presser, PRESSING_COLOR)
    if carrier is not None:
        _ring(ax, carrier, ATTACKING_COLOR)
        ax.plot(
            [presser["x"], carrier["x"]], [presser["y"], carrier["y"]],
            linestyle=(0, (4, 3)), color=TEXT_SECONDARY, linewidth=1.2, zorder=2,
        )
        dx, dy = carrier["x"] - presser["x"], carrier["y"] - presser["y"]
        dist = float(np.hypot(dx, dy))
        # Offset the distance label perpendicular to the line, so it clears
        # both markers however they are oriented.
        nx, ny = (-dy / dist, dx / dist) if dist > 1e-6 else (0.0, 1.0)
        mx = (presser["x"] + carrier["x"]) / 2 + nx * 2.4
        my = (presser["y"] + carrier["y"]) / 2 + ny * 2.4
        ax.text(
            mx, my, f"{dist:.1f} m",
            ha="center", va="center", fontsize=9, color=TEXT_SECONDARY, zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE, edgecolor="none"),
        )

    # The two protagonists are often a few metres apart, so push their labels
    # to opposite sides rather than stacking both above the markers.
    if carrier is None:
        _annotate(ax, presser, "presser", above=True)
    else:
        presser_above = presser["y"] >= carrier["y"]
        _annotate(ax, presser, "presser", above=presser_above)
        _annotate(ax, carrier, "carrier", above=not presser_above)

    if scene.ball is not None:
        ax.scatter(
            [scene.ball[0]], [scene.ball[1]],
            s=85, marker="o", facecolors=BALL_COLOR, edgecolors=SURFACE,
            linewidths=1.6, zorder=6,
        )

    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=PRESSING_COLOR, markeredgecolor=PRESSING_COLOR,
                   label="pressing team"),
            Line2D([], [], marker="s", linestyle="", markersize=9,
                   markerfacecolor=ATTACKING_COLOR, markeredgecolor=ATTACKING_COLOR,
                   label="team in possession"),
            Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=SURFACE, markeredgecolor=TEXT_SECONDARY,
                   label="position extrapolated"),
            Line2D([], [], marker="o", linestyle="", markersize=6,
                   markerfacecolor=BALL_COLOR, markeredgecolor=SURFACE, label="ball"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=4,
        frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
    )

    if title:
        ax.set_title(_title(scene), fontsize=12, color=TEXT_PRIMARY, loc="left", pad=14)
        ax.text(
            0, 1.005, _subtitle(scene),
            transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom",
        )
    return ax


def save_scene(scene: Scene, path=None, dpi: int = 150, **kwargs):
    """Draw a scene and write it to ``figures/`` as a PNG."""
    ax = plot_scene(scene, **kwargs)
    fig = ax.get_figure()
    fig.patch.set_facecolor(SURFACE)
    if path is None:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = config.FIGURES_DIR / f"scene_{scene.match_id}_{scene.event_id}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _ring(ax, player, color):
    ax.scatter(
        [player["x"]], [player["y"]],
        s=760, marker="o", facecolors="none", edgecolors=color,
        linewidths=2.0, zorder=5,
    )


def _annotate(ax, player, role, above: bool = True):
    name = player["short_name"] or str(player.name)
    offset = 22 if above else -22
    ax.annotate(
        f"{role}: {name}",
        xy=(player["x"], player["y"]), xytext=(0, offset),
        textcoords="offset points", ha="center",
        va="bottom" if above else "top",
        fontsize=9, color=TEXT_PRIMARY, zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                  edgecolor=LINE, linewidth=0.6),
    )


def _title(scene: Scene) -> str:
    from .metadata import load_match

    match = load_match(scene.match_id)
    pressing = match.teams.get(scene.pressing_team_id, scene.pressing_team_id)
    attacking = match.teams.get(scene.attacking_team_id, scene.attacking_team_id)
    subtype = scene.info.get("event_subtype", "engagement")
    return f"{pressing} pressing {attacking} — {subtype}"


def _subtitle(scene: Scene) -> str:
    bits = [
        f"match {scene.match_id}",
        f"event {scene.event_id}",
        f"period {scene.period} {scene.time_start}",
        f"frame {scene.frame}",
    ]
    end_type = scene.info.get("end_type")
    bits.append(f"end_type: {end_type}" if not _isnan(end_type) and end_type
                else "end_type: not recorded")
    bits.append("team in possession attacks right")
    return "  ·  ".join(str(b) for b in bits)


def _isnan(value) -> bool:
    try:
        return value is None or bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
