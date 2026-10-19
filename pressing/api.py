"""Press metrics from any tracking data: one frame in, 32 numbers out.

The rest of the package reads SkillCorner's open data. This module needs none
of it. Give it where the players are at the moment a press starts and who is
pressing whom, and it returns the metrics the analysis is built on::

    from pressing.api import press_metrics

    players = [
        {"id": 10, "x": 12.0, "y": -4.0, "side": "attacking"},
        {"id": 4, "x": 15.5, "y": -3.0, "side": "pressing", "vx": -3.0, "vy": 0.0},
        ...
    ]
    m = press_metrics(players, presser=4, carrier=10, ball=(12.3, -4.1))
    m["n_pressing_local"]   # defenders within 10 m of the carrier

Conventions
-----------
Metres, origin at the centre spot, x along the pitch, on a 105 x 68 m pitch.

``attack`` is the direction the team in possession attacks: ``"right"``
(towards +x, the default) or ``"left"``. With ``"left"`` the frame is rotated
by 180 degrees first, so the metrics do not depend on which way the team plays.

Velocities (``vx, vy`` in m/s) are optional. Without them the metrics that
need them (speeds, closing speed) come back as NaN and the rest are unaffected.

A player with ``position == "GK"`` is left out of the shape of the block. When
no pressing player is marked as keeper, the deepest one is taken to be him.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, scene_features
from .scene import Scene

__all__ = ["press_metrics", "FEATURE_COLUMNS", "HEADLINE"]

#: The five metrics of Figure 1, in its order: readable name to column.
HEADLINE = {
    "press density": "n_pressing_local",
    "press support": "support_dist",
    "block compactness": "def_block_depth",
    "open outlets": "n_open_short",
    "presser's cover shadow": "options_shadowed_by_presser",
}

_SIDES = {"pressing", "attacking"}


def press_metrics(
    players: pd.DataFrame | Iterable[Mapping],
    presser,
    carrier,
    ball: tuple[float, float] | None = None,
    ball_velocity: tuple[float, float] | None = None,
    attack: str = "right",
) -> dict[str, float]:
    """The 32 press metrics for one frame.

    ``players`` is a DataFrame or a list of dicts with ``id, x, y, side``
    (``side`` is ``"pressing"`` or ``"attacking"``) and optionally ``vx, vy,
    position``. A DataFrame may carry the ids in its index instead of an
    ``id`` column. ``presser`` and ``carrier`` are ids from that table.

    Returns a dict in the order of :data:`FEATURE_COLUMNS`.
    """
    table = _player_table(players)
    _check(table, presser, carrier)

    if attack not in {"right", "left"}:
        raise ValueError(f'attack must be "right" or "left", not {attack!r}')
    if attack == "left":
        for col in ("x", "y", "vx", "vy"):
            table[col] = -table[col]
        ball = None if ball is None else (-float(ball[0]), -float(ball[1]))
        ball_velocity = (None if ball_velocity is None
                         else (-float(ball_velocity[0]), -float(ball_velocity[1])))

    table["is_presser"] = table.index == presser
    table["is_carrier"] = table.index == carrier
    scene = Scene(
        match_id="", event_id="", frame=0, period=0, time_start="",
        players=table,
        ball=None if ball is None else (float(ball[0]), float(ball[1])),
        ball_velocity=(None if ball_velocity is None
                       else (float(ball_velocity[0]), float(ball_velocity[1]))),
        pressing_team_id=1, attacking_team_id=2,
        attacking_side="left_to_right", info={},
    )
    return scene_features(scene)


def _player_table(players) -> pd.DataFrame:
    df = pd.DataFrame(players).copy() if not isinstance(players, pd.DataFrame) \
        else players.copy()
    if "id" in df.columns:
        df = df.set_index("id")
    df.index.name = "player_id"

    missing = {"x", "y", "side"} - set(df.columns)
    if missing:
        raise ValueError(f"players needs columns {sorted(missing)}")
    if df.index.has_duplicates:
        raise ValueError("player ids must be unique")
    bad = set(df["side"]) - _SIDES
    if bad:
        raise ValueError(f'side must be "pressing" or "attacking", got {sorted(map(str, bad))}')

    for col in ("x", "y", "vx", "vy"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df \
            else np.nan
    df["speed"] = np.hypot(df["vx"], df["vy"])
    if "position" not in df:
        df["position"] = None
    df["team_id"] = df["side"].map({"pressing": 1, "attacking": 2})
    return df


def _check(table: pd.DataFrame, presser, carrier) -> None:
    for role, pid, side in (("presser", presser, "pressing"),
                            ("carrier", carrier, "attacking")):
        if pid not in table.index:
            raise ValueError(f"{role} {pid!r} is not in players")
        if table.at[pid, "side"] != side:
            raise ValueError(f"{role} {pid!r} should be on the {side} side")
    if table[["x", "y"]].isna().any().any():
        raise ValueError("every player needs a position (x, y)")
