"""Reading the extrapolated tracking files.

Each line of ``<match_id>_tracking_extrapolated.jsonl`` is one frame at 10 Hz:

    {"frame": 4999, "timestamp": "00:08:18.90", "period": 1,
     "ball_data": {"x": .., "y": .., "z": .., "is_detected": true},
     "possession": {"player_id": 560985 | null, "group": "home team"},
     "image_corners_projection": {...},
     "player_data": [{"x": .., "y": .., "player_id": .., "is_detected": ..}, ...]}

Frames before kick-off exist but are empty (``period`` is null and
``player_data`` is an empty list), so callers must handle empty frames.

The files are roughly 90 MB each, so :func:`read_frames` streams the file and
parses only the lines it was asked for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_FRAME_RE = re.compile(rb'"frame"\s*:\s*(\d+)')


def read_frames(match_id: int | str, frames) -> dict[int, dict]:
    """Return ``{frame: record}`` for the requested frame numbers.

    Frames that are missing from the file are absent from the result. One pass
    over the file, parsing JSON only for the wanted lines.
    """
    wanted = {int(f) for f in frames}
    if not wanted:
        return {}

    path: Path = config.tracking_path(match_id)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking file not found: {path}")
    _check_not_lfs_pointer(path)

    out: dict[int, dict] = {}
    remaining = set(wanted)
    with open(path, "rb") as fh:
        for line in fh:
            m = _FRAME_RE.search(line, 0, 64)
            if m is None:  # fall back to a full parse if the layout differs
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                num = rec.get("frame")
                if num in remaining:
                    out[int(num)] = rec
                    remaining.discard(int(num))
            else:
                num = int(m.group(1))
                if num in remaining:
                    out[num] = json.loads(line)
                    remaining.discard(num)
            if not remaining:
                break
    return out


def frame_to_dataframe(record: dict) -> pd.DataFrame:
    """Players of one frame as a DataFrame indexed by ``player_id``.

    Columns: ``x, y, is_detected``. An empty frame yields an empty frame with
    those columns, so downstream code never has to special-case it.
    """
    players = record.get("player_data") or []
    df = pd.DataFrame(players)
    if df.empty:
        # Explicit dtypes matter: an empty frame built without them gets
        # object columns, and arithmetic against those silently produces
        # object results that numpy cannot take a square root of.
        return pd.DataFrame(
            {
                "x": pd.Series(dtype="float64"),
                "y": pd.Series(dtype="float64"),
                "is_detected": pd.Series(dtype="object"),
            },
            index=pd.Index([], name="player_id", dtype="int64"),
        )
    df["player_id"] = df["player_id"].astype(int)
    df = df.set_index("player_id")
    if "is_detected" not in df:
        df["is_detected"] = pd.NA
    # A null coordinate anywhere in the frame would otherwise make the whole
    # column object-typed.
    for col in ("x", "y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["x", "y", "is_detected"]]


def ball_xy(record: dict) -> tuple[float, float] | None:
    """Ball ``(x, y)``, or ``None`` when the frame has no ball position."""
    ball = record.get("ball_data") or {}
    x, y = ball.get("x"), ball.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def velocities_fit(
    history: list[tuple[float, pd.DataFrame]],
    curr: pd.DataFrame,
    min_points: int = 3,
) -> pd.DataFrame:
    """Per-player velocity in m/s as a least-squares slope over a window.

    ``history`` is a list of ``(t, frame_df)`` pairs, ``t`` in seconds relative
    to the current frame (so ``t <= 0``), and it should include the current
    frame itself at ``t = 0``. Only past and present frames go in, never later
    ones, so nothing after the engagement start can leak into the velocity.

    Compared with differencing two frames (the method used until 14 October
    2026), the slope uses every frame in the window: one noisy position moves it
    far less, and a single missing frame no longer turns everyone's velocity
    into NaN. A player seen in fewer than ``min_points`` frames of the window
    gets NaN.

    Returns columns ``vx, vy, speed`` indexed like ``curr``.
    """
    parts = []
    for t, df in history:
        if df is None or df.empty:
            continue
        part = df[["x", "y"]].astype("float64").copy()
        part["t"] = float(t)
        parts.append(part)

    nan = np.full(len(curr), np.nan)
    if not parts:
        return pd.DataFrame({"vx": nan, "vy": nan, "speed": nan}, index=curr.index)

    s = pd.concat(parts).dropna(subset=["x", "y"])
    s = s.assign(tt=s["t"] * s["t"], tx=s["t"] * s["x"], ty=s["t"] * s["y"])
    agg = s.groupby(level=0).agg(
        n=("t", "size"), St=("t", "sum"), Sx=("x", "sum"), Sy=("y", "sum"),
        Stt=("tt", "sum"), Stx=("tx", "sum"), Sty=("ty", "sum"),
    )
    denom = agg["n"] * agg["Stt"] - agg["St"] ** 2
    ok = (agg["n"] >= min_points) & (denom > 1e-12)
    safe = denom.where(ok)
    vx = ((agg["n"] * agg["Stx"] - agg["St"] * agg["Sx"]) / safe).reindex(curr.index)
    vy = ((agg["n"] * agg["Sty"] - agg["St"] * agg["Sy"]) / safe).reindex(curr.index)
    return pd.DataFrame(
        {"vx": vx, "vy": vy, "speed": np.hypot(vx.to_numpy(), vy.to_numpy())},
        index=curr.index,
    )


def point_velocity_fit(
    points: list[tuple[float, tuple[float, float] | None]],
    min_points: int = 3,
) -> tuple[float, float] | None:
    """Least-squares velocity of a single point (the ball) over a window.

    Same idea as :func:`velocities_fit`. ``points`` are ``(t, (x, y))`` pairs
    with ``(x, y)`` possibly ``None``; returns ``None`` with too few points.
    """
    pts = [(t, xy) for t, xy in points if xy is not None]
    if len(pts) < min_points:
        return None
    t = np.array([p[0] for p in pts], dtype=float)
    xy = np.array([p[1] for p in pts], dtype=float)
    tc = t - t.mean()
    denom = float((tc * tc).sum())
    if denom <= 1e-12:
        return None
    vx = float((tc * (xy[:, 0] - xy[:, 0].mean())).sum() / denom)
    vy = float((tc * (xy[:, 1] - xy[:, 1].mean())).sum() / denom)
    return vx, vy


def flip_for_attacking_side(df: pd.DataFrame, attacking_side: str) -> pd.DataFrame:
    """Rotate coordinates so the reference team always attacks towards +x.

    Teams swap ends at half time, so raw coordinates are not comparable across
    periods. Rotating by 180 degrees (negating both x and y) puts every scene in
    one frame of reference. Columns ``x, y, vx, vy`` are flipped when present.
    """
    if attacking_side not in {"left_to_right", "right_to_left"}:
        raise ValueError(f"unknown attacking_side: {attacking_side!r}")
    if attacking_side == "left_to_right":
        return df.copy()
    out = df.copy()
    for col in ("x", "y", "vx", "vy"):
        if col in out:
            out[col] = -out[col]
    return out


def flip_point(xy, attacking_side: str):
    """Same rotation as :func:`flip_for_attacking_side`, for a single point."""
    if xy is None:
        return None
    x, y = xy
    if attacking_side == "left_to_right":
        return float(x), float(y)
    return -float(x), -float(y)


def _check_not_lfs_pointer(path: Path) -> None:
    """Fail loudly when a Git LFS pointer was cloned instead of the real file."""
    if path.stat().st_size > 4096:
        return
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the tracking data.\n"
            "Fetch the real files with 'git lfs pull' inside the opendata "
            "clone, or download them directly from "
            "https://media.githubusercontent.com/media/SkillCorner/opendata/master/..."
        )
