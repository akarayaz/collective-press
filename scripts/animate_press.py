"""Animate one press: how it forms, the frame it is measured on, what follows.

    python scripts/animate_press.py                 # the Figure 1 press, GIF only
    python scripts/animate_press.py --mp4           # also an MP4 for the video
    python scripts/animate_press.py --event 1925299 12_345
    python scripts/animate_press.py --pause 0 --mp4 --no-gif --name clip_hook

Three acts, slowed down so the eye can follow:

1. the two seconds before the press starts, players moving with short trails;
2. a pause on the start frame, the only frame the metrics read, with the five
   metrics drawn exactly as in Figure 1 and named one by one on the right;
3. play resumes until the ball is won back, or for five seconds if it is not.

The ring marks whoever the tracking says has the ball at his feet (none while
the ball travels): orange while the team in possession keeps it, blue once the
pressing team wins it.

Writes ``figures/press_animation_2d.gif`` and with ``--mp4``
``figures/press_animation_2d.mp4`` (1920 x 1080, needs ffmpeg:
``conda install -c conda-forge ffmpeg``). Coordinates are rotated like every
scene, so the team in possession attacks to the right.

This top-down version works for any press in the 20 matches. The README's
animation is the 3D one (``viz/press3d.html``), which needs body pose and so
exists only for the two matches that have it; the default name here keeps the
two from overwriting each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figure_press as fp  # noqa: E402
from pressing import config, events, metadata, scene, tracking, viz  # noqa: E402

FIGSIZE = (12.8, 7.2)        # 16:9; 1920 x 1080 at 150 dpi
TRAIL_S = 0.8                # seconds of trail behind each player
HOLDER_RADIUS_M = 2.0        # nearest player this close to the ball has it
AT_FEET_M = 1.5              # ring a holder only with the ball this close, not in flight

METRICS = [  # (badge, name, what it does), in Figure 1's numbering
    ("1", "Press density", "more defenders win the ball"),
    ("2", "Press support", "close support wins the ball"),
    ("3", "Block depth", "a compact block makes passes fail"),
    ("4", "Open outlets", "the carrier's way out"),
    ("5", "Cover shadow", "pushes the play backwards"),
]


# --- the data ----------------------------------------------------------------------

class Clip:
    """Every tracking frame around one press, rotated like its scene."""

    def __init__(self, match_id, event_id, before, after):
        eng = events.load_engagements(match_id)
        rows = eng[eng["event_id"].astype(str) == str(event_id)]
        if rows.empty:
            raise SystemExit(f"event {event_id} not found in match {match_id}")
        self.row = row = rows.iloc[0]
        self.scene = scene.build_scene(match_id, row)
        self.match = metadata.load_match(match_id)

        self.f0 = int(row["frame_start"])
        self.pressing_team = int(row["team_id"])
        self.attacking_team = next(t for t in self.match.teams if t != self.pressing_team)
        self.side = metadata.opposite_side(str(row["attacking_side"]))
        self.regain = _seconds_to_regain(match_id, event_id)

        # Stop a second after the regain, or after `after` seconds without one.
        if np.isfinite(self.regain) and self.regain <= after:
            after = min(after, self.regain + 1.0)
        self.before, self.after = before, after
        lo = self.f0 - int(round(before * config.FPS))
        hi = self.f0 + int(round(after * config.FPS))
        records = tracking.read_frames(match_id, range(lo, hi + 1))
        if self.f0 not in records:
            raise SystemExit(f"frame {self.f0} is not in the tracking file")

        team_side = self.match.players["team_id"].map(
            {self.pressing_team: "pressing", self.attacking_team: "attacking"})
        self.frames, self.balls, self.holders = {}, {}, {}
        for f, rec in sorted(records.items()):
            df = tracking.frame_to_dataframe(rec).join(team_side.rename("side"))
            df = df[df["side"].notna() & df["x"].notna() & df["y"].notna()]
            if df.empty:
                continue
            self.frames[f] = tracking.flip_for_attacking_side(df[["x", "y", "side"]],
                                                              self.side)
            self.balls[f] = tracking.flip_point(tracking.ball_xy(rec), self.side)
            self.holders[f] = (rec.get("possession") or {}).get("player_id")
        self.keys = np.array(sorted(self.frames))

    def t_of(self, f) -> float:
        return (f - self.f0) / config.FPS

    def at(self, t: float):
        """Players, ball and ball holder at time ``t`` (s from the start frame)."""
        f = self.f0 + t * config.FPS
        i = int(np.clip(np.searchsorted(self.keys, f), 1, len(self.keys) - 1))
        a, b = self.keys[i - 1], self.keys[i]
        w = float(np.clip((f - a) / (b - a), 0, 1))
        A, B = self.frames[a], self.frames[b]
        both = A.index.intersection(B.index)
        pos = A.loc[both, ["x", "y"]] * (1 - w) + B.loc[both, ["x", "y"]] * w
        pos["side"] = A.loc[both, "side"]

        ba, bb = self.balls.get(a), self.balls.get(b)
        ball = (None if ba is None and bb is None else ba if bb is None else
                bb if ba is None else (ba[0] * (1 - w) + bb[0] * w,
                                       ba[1] * (1 - w) + bb[1] * w))
        holder = self.holders[a if w < 0.5 else b]
        if (holder is None or holder not in pos.index) and ball is not None and len(pos):
            d = np.hypot(pos["x"] - ball[0], pos["y"] - ball[1])
            holder = d.idxmin() if d.min() <= HOLDER_RADIUS_M else None
        # The tracking keeps naming the passer while the ball travels, and many
        # engagements start just before the ball reaches the carrier. No ring
        # while the ball is in flight, so the ring never sits on the wrong man.
        if holder is not None and holder in pos.index and ball is not None:
            if np.hypot(pos.at[holder, "x"] - ball[0],
                        pos.at[holder, "y"] - ball[1]) > AT_FEET_M:
                holder = None
        elif ball is None:
            holder = None
        return pos, ball, holder

    def trail(self, t: float, pos: pd.DataFrame) -> dict:
        """Recent positions of each player, oldest first, ending where he is now."""
        keys = [f for f in self.keys if t - TRAIL_S < self.t_of(f) < t]
        out = {}
        for pid in pos.index:
            pts = [self.frames[f].loc[pid, ["x", "y"]].to_numpy(float)
                   for f in keys if pid in self.frames[f].index]
            if pts:
                out[pid] = np.vstack(pts + [pos.loc[pid, ["x", "y"]].to_numpy(float)])
        return out

    def camera(self, t: float) -> tuple[float, float]:
        """Window centre: the ball, smoothed, settling on the carrier at the start.

        At ``t = 0`` it is exactly where Figure 1 centres, so the pause does not
        jump.
        """
        ts = np.array([self.t_of(f) for f in self.keys])
        bx = np.array([np.nan if self.balls[f] is None else self.balls[f][0] for f in self.keys])
        by = np.array([np.nan if self.balls[f] is None else self.balls[f][1] for f in self.keys])
        near = (np.abs(ts - t) <= 0.5) & np.isfinite(bx)
        c = self.scene.carrier
        cx, cy = float(c["x"]), float(c["y"])
        if not near.any():
            return cx, cy
        sx, sy = float(bx[near].mean()), float(by[near].mean())
        k = float(np.exp(-abs(t) / 0.6))  # 1 at the start frame, fading out
        return sx * (1 - k) + cx * k, sy * (1 - k) + cy * k


def _seconds_to_regain(match_id, event_id) -> float:
    """From the modelling table, so it is the same label the analysis used."""
    try:
        from pressing import modelling
        t = modelling.load_table(only_ok=False)
    except Exception:  # no cache yet: animate without the outcome
        return float("nan")
    hit = t[(t["meta_match_id"].astype(str) == str(match_id))
            & (t["meta_event_id"].astype(str) == str(event_id))]
    if hit.empty or "label_seconds_to_regain" not in hit:
        return float("nan")
    return float(hit["label_seconds_to_regain"].iloc[0])


# --- drawing -----------------------------------------------------------------------

def _window(cx, cy):
    """Figure 1's window around a point, slid back inside the pitch."""
    hx, hy = config.PITCH_LENGTH / 2 + 2, config.PITCH_WIDTH / 2 + 2

    def slide(lo, hi, bound):
        if lo < -bound:
            lo, hi = -bound, hi + (-bound - lo)
        if hi > bound:
            lo, hi = lo - (hi - bound), bound
        return lo, hi

    return slide(cx - 21, cx + 31, hx) + slide(cy - 19, cy + 19, hy)


def draw_play(ax, clip: Clip, t: float) -> None:
    pos, ball, holder = clip.at(t)
    ax.cla()
    viz.draw_pitch(ax)
    x0, x1, y0, y1 = _window(*clip.camera(t))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    colour = {"pressing": fp.PRESS, "attacking": fp.POSSESS}
    for pid, pts in clip.trail(t, pos).items():
        ax.plot(pts[:, 0], pts[:, 1], color=colour[pos.at[pid, "side"]], alpha=0.35,
                linewidth=1.4, solid_capstyle="round", zorder=3)
    for side, marker in (("pressing", "o"), ("attacking", "s")):
        team = pos[pos["side"] == side]
        ax.scatter(team["x"], team["y"], s=fp.PLAYER_S, marker=marker,
                   facecolor=colour[side], edgecolor=fp.SURFACE, linewidth=1.3, zorder=5)
    if holder is not None and holder in pos.index:
        ax.scatter([pos.at[holder, "x"]], [pos.at[holder, "y"]], s=fp.RING_S,
                   facecolor="none", edgecolor=colour[pos.at[holder, "side"]],
                   linewidth=2.0, zorder=6)
    if ball is not None:
        ax.scatter([ball[0]], [ball[1]], s=34, facecolor=fp.INK, edgecolor=fp.SURFACE,
                   linewidth=1.0, zorder=7)


def draw_panel(px, clip: Clip, t: float, act: str, reveal: int, values: dict) -> None:
    px.cla()
    px.set_xlim(0, 1)
    px.set_ylim(0, 1)
    px.axis("off")
    teams = clip.match.teams
    # The team names in their colours double as the legend.
    px.text(0, 0.975, f"{teams[clip.pressing_team]} press", fontsize=17, color=fp.PRESS,
            va="top", fontweight="bold")
    px.text(0, 0.915, f"{teams[clip.attacking_team]} in possession", fontsize=12.5,
            color=fp.POSSESS, va="top", fontweight="bold")

    # The heading says what is on screen; at the end it becomes the outcome.
    heading, colour, weight = {"before": "The press forms",
                               "pause": "The frame every metric is measured on",
                               "after": "What happens next"}[act], fp.INK, "normal"
    if act == "after":
        if np.isfinite(clip.regain) and clip.regain <= t + 1e-9:
            heading, colour, weight = (f"Ball won back after {clip.regain:.1f} s",
                                       fp.PRESS, "bold")
        elif t >= clip.after - 1e-9:
            heading, colour, weight = (f"No regain within {clip.after:.0f} s",
                                       fp.POSSESS, "bold")
    px.text(0, 0.835, heading, fontsize=12.5, color=colour, va="top", fontweight=weight)

    lines = [
        f"{values['n_local']} defenders within 10 m",
        f"nearest teammate {values['support']:.1f} m away",
        f"{values['depth']:.0f} m from front to back",
        f"{values['open_short']} free within 15 m",
        f"{values['shadowed']} lane{'s' if values['shadowed'] != 1 else ''} closed",
    ]
    if reveal:
        # The values are this press; the grey lines are averages over all of
        # them, and must not read as a description of what happens here.
        px.text(0, 0.787, "This press, and in grey what each does on average",
                fontsize=9.5, color=fp.INK_2, va="center")
    y = 0.725
    for k, ((badge, name, job), value) in enumerate(zip(METRICS, lines)):
        if k >= reveal:
            break
        px.text(0.035, y, badge, ha="center", va="center", fontsize=10.5, color=fp.SURFACE,
                fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", facecolor=fp.INK, edgecolor="none"))
        px.text(0.1, y, name, fontsize=11.5, color=fp.INK, va="center", fontweight="bold")
        px.text(0.1, y - 0.04, value, fontsize=10.5, color=fp.INK, va="center")
        px.text(0.1, y - 0.077, job, fontsize=10, color=fp.INK_2, va="center")
        y -= 0.12

    # Timeline: where we are, the start frame and the regain.
    a, b = -clip.before, clip.after
    def X(v):
        return 0.02 + 0.96 * (v - a) / (b - a)
    yl = 0.08
    px.plot([X(a), X(b)], [yl, yl], color=fp.LINE, linewidth=2, solid_capstyle="round")
    px.plot([X(a), X(min(t, b))], [yl, yl], color=fp.INK_3, linewidth=2,
            solid_capstyle="round")
    px.plot([X(0)] * 2, [yl - 0.02, yl + 0.02], color=fp.INK, linewidth=1.4)
    px.text(X(0), yl + 0.035, "press starts", ha="center", fontsize=9, color=fp.INK_2)
    if np.isfinite(clip.regain) and clip.regain <= b:
        px.plot([X(clip.regain)] * 2, [yl - 0.02, yl + 0.02], color=fp.PRESS, linewidth=1.4)
        px.text(X(clip.regain), yl - 0.06, "ball won", ha="center", fontsize=9,
                color=fp.PRESS)
    px.scatter([X(min(max(t, a), b))], [yl], s=40, color=fp.INK, zorder=3)
    clock = "0.0 s" if abs(t) < 0.05 else f"{t:+.1f} s"
    px.text(1.0, yl + 0.035, clock, ha="right", fontsize=9, color=fp.INK_3)


# --- the storyboard -----------------------------------------------------------------

def storyboard(clip: Clip, fps: float, slow: float, pause: float, hold: float):
    """(act, t, metrics shown) for every output frame."""
    def play(t0, t1):
        n = max(2, int(round((t1 - t0) / slow * fps)))
        return np.linspace(t0, t1, n)

    shots = [("before", t, 0) for t in play(-clip.before, 0)[:-1]]
    n = int(round(pause * fps))
    step = pause / (len(METRICS) + 1)  # one metric per step, a beat at the end
    shots += [("pause", 0.0, min(len(METRICS), int((k / fps) / step) + 1)) for k in range(n)]
    shown = len(METRICS) if n else 0  # without the pause, a clean clip
    shots += [("after", t, shown) for t in play(0, clip.after)]
    shots += [("after", clip.after, shown)] * int(round(hold * fps))
    return shots


def render(clip: Clip, path: Path, fps: float, dpi: float, slow: float, pause: float,
           hold: float) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
    fig.patch.set_facecolor(fp.SURFACE)
    ax = fig.add_axes([0.01, 0.07, 0.65, 0.91])
    px = fig.add_axes([0.685, 0.07, 0.30, 0.86])
    fig.text(0.01, 0.025, "SkillCorner open data, A-League 2024/25. Slowed to "
             f"{slow:g}x. The team in possession attacks to the right.",
             fontsize=9, color=fp.INK_3, ha="left", va="center")

    if path.suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=8000,
                              extra_args=["-pix_fmt", "yuv420p"])

    shots = storyboard(clip, fps, slow, pause, hold)
    values = None
    with writer.saving(fig, str(path), dpi=dpi):
        last = None
        for act, t, reveal in shots:
            if act == "pause":
                if last != "pause":  # Figure 1's drawing, once
                    ax.cla()
                    values = fp.draw_press(ax, clip.scene)
            else:
                draw_play(ax, clip, t)
            if values is None:  # the panel needs the numbers before the pause
                values = _values(clip)
            draw_panel(px, clip, t, act, reveal, values)
            writer.grab_frame(facecolor=fp.SURFACE)
            last = act
    plt.close(fig)
    return len(shots)


def _values(clip: Clip) -> dict:
    """The five numbers, computed as Figure 1 computes them, without drawing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    values = fp.draw_press(ax, clip.scene)
    plt.close(fig)
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", nargs=2, metavar=("MATCH", "EVENT"),
                    default=["1996435", "9_684"], help="the press to animate")
    ap.add_argument("--before", type=float, default=2.0, help="seconds before the start")
    ap.add_argument("--after", type=float, default=5.0, help="seconds after, at most")
    ap.add_argument("--slow", type=float, default=0.5, help="playback speed")
    ap.add_argument("--pause", type=float, default=6.0, help="seconds on the start frame")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds on the last frame")
    ap.add_argument("--gif-width", type=int, default=960, help="GIF width in pixels")
    ap.add_argument("--gif-fps", type=float, default=12)
    ap.add_argument("--mp4", action="store_true", help="also write the 1080p MP4")
    ap.add_argument("--no-gif", action="store_true", help="skip the GIF")
    ap.add_argument("--name", default="press_animation_2d",
                    help="file name without extension; different names keep "
                         "video cuts and the README's 3D GIF apart")
    args = ap.parse_args()

    match_id, event_id = int(args.event[0]), args.event[1]
    clip = Clip(match_id, event_id, args.before, args.after)
    outcome = (f"ball won back after {clip.regain:.1f} s" if np.isfinite(clip.regain)
               else "no regain recorded")
    print(f"press: match {match_id}, event {event_id}, frame {clip.f0}; {outcome}")
    print(f"{len(clip.keys)} tracking frames, {-clip.before:+.1f} s to {clip.after:+.1f} s")

    figdir = config.REPO_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    common = dict(slow=args.slow, pause=args.pause, hold=args.hold)

    if not args.no_gif:
        gif = figdir / f"{args.name}.gif"
        n = render(clip, gif, args.gif_fps, args.gif_width / FIGSIZE[0], **common)
        print(f"written: {gif}  ({n} frames, {gif.stat().st_size / 1e6:.1f} MB)")
    if args.mp4:
        mp4 = figdir / f"{args.name}.mp4"
        n = render(clip, mp4, 25, 150, **common)
        print(f"written: {mp4}  ({n} frames, {n / 25:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
