"""Hand-built positioning features: the M1 half of the comparison.

Everything here is measured from the tracking frame the engagement starts on,
using only what is visible at that instant. Nothing reads the outcome, and
nothing reads a later frame. Velocities look backwards over half a second.

These are the features a person can describe in words, which is the point:
M1 asks whether geometry anyone could write down adds to SkillCorner's own
summaries, and M2 then asks whether a graph network finds more than that.

Orientation (set in :mod:`pressing.scene`): the team in possession attacks
towards +x, so the carrier is running towards the goal at ``(+52.5, 0)`` and
the pressing team is defending it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

GOAL_X = config.PITCH_LENGTH / 2
HALF_WIDTH = config.PITCH_WIDTH / 2

#: Half-width of the passing lane a defender is considered to block.
SHADOW_HALF_WIDTH_M = 1.5

#: Radii for the local density counts.
NEAR_M = 5.0
LOCAL_M = 10.0

#: How deep the "last line" is taken to be, in outfield players.
DEFENSIVE_LINE_PLAYERS = 4

#: Pass-length bands for counting open teammates: short, medium, long. Three
#: bands rather than one cut-off, so the model can learn which range matters
#: instead of a single threshold deciding who counts as an option.
SHORT_PASS_M = 15.0
LONG_PASS_M = 30.0

#: A receiver is open when no pressing player is this close to them, and, for
#: short and medium passes, no pressing player stands in the lane. Long passes
#: usually travel in the air, so a defender under the line does not cut them.
OPEN_RADIUS_M = 3.0


def scene_features(scene) -> dict:
    """Positioning features for one :class:`~pressing.scene.Scene`."""
    carrier = scene.carrier
    presser = scene.presser
    pressing = scene.pressing
    attacking = scene.attacking

    out: dict[str, float] = {}

    if carrier is None:
        return {name: np.nan for name in FEATURE_COLUMNS}

    cx, cy = float(carrier["x"]), float(carrier["y"])
    px, py = float(presser["x"]), float(presser["y"])

    # --- where the attack is -------------------------------------------------
    out["carrier_x"] = cx
    out["carrier_y_abs"] = abs(cy)
    out["carrier_dist_to_goal"] = float(np.hypot(GOAL_X - cx, cy))
    out["carrier_dist_to_sideline"] = HALF_WIDTH - abs(cy)
    out["carrier_speed"] = _f(carrier.get("speed"))

    if scene.ball is None:
        out["ball_dist_to_carrier"] = np.nan
    else:
        out["ball_dist_to_carrier"] = float(
            np.hypot(scene.ball[0] - cx, scene.ball[1] - cy)
        )
    out["ball_speed"] = (
        np.nan if scene.ball_velocity is None
        else float(np.hypot(*scene.ball_velocity))
    )

    # --- the engaging defender ----------------------------------------------
    dx, dy = px - cx, py - cy
    dist = float(np.hypot(dx, dy))
    out["presser_dist"] = dist
    # 0 degrees: the defender stands between the carrier and the goal.
    # 180 degrees: the defender is chasing from behind.
    out["presser_angle"] = abs(float(np.degrees(np.arctan2(dy, dx))))
    out["presser_goalside"] = float(px > cx)
    out["presser_speed"] = _f(presser.get("speed"))
    out["presser_dist_to_goal"] = float(np.hypot(GOAL_X - px, py))
    out["closing_speed"] = _closing_speed(presser, carrier, dist)

    # --- who else is around --------------------------------------------------
    out["support_dist"] = _nearest_teammate_distance(pressing, presser.name)

    d_press = _distances(pressing, cx, cy)
    d_att = _distances(attacking.drop(index=carrier.name, errors="ignore"), cx, cy)
    out["n_pressing_near"] = float((d_press <= NEAR_M).sum())
    out["n_pressing_local"] = float((d_press <= LOCAL_M).sum())
    out["n_attacking_near"] = float((d_att <= NEAR_M).sum())
    out["n_attacking_local"] = float((d_att <= LOCAL_M).sum())
    out["numerical_advantage"] = (
        out["n_pressing_local"] - out["n_attacking_local"]
    )

    # --- passing options ----------------------------------------------------
    # Every teammate is a potential outlet, counted by distance band; a long
    # ball over the press is an escape too. (The first version counted every
    # teammate with no notion of distance, which made its option count a
    # constant 10, since every usable row has 22 players.)
    receivers = attacking.drop(index=carrier.name, errors="ignore")
    rdist = np.hypot(receivers["x"].astype(float).to_numpy() - cx,
                     receivers["y"].astype(float).to_numpy() - cy)
    blockers = pressing[["x", "y"]].astype(float).to_numpy()
    presser_xy = presser[["x", "y"]].astype(float).to_numpy()[None, :]

    reachable = rdist <= LONG_PASS_M
    out["options_shadowed_by_presser"] = float(
        _lane_blocked(cx, cy, receivers[reachable], presser_xy).sum()
    )
    is_open = _open(cx, cy, receivers, blockers, rdist)
    out["n_open_short"] = float((is_open & (rdist <= SHORT_PASS_M)).sum())
    out["n_open_medium"] = float(
        (is_open & (rdist > SHORT_PASS_M) & (rdist <= LONG_PASS_M)).sum()
    )
    out["n_open_long"] = float((is_open & (rdist > LONG_PASS_M)).sum())
    out["n_open_forward"] = float(
        (is_open & (receivers["x"].astype(float).to_numpy() > cx)).sum()
    )
    out["nearest_open_dist"] = float(rdist[is_open].min()) if is_open.any() else np.nan

    # --- shape of the defending team ----------------------------------------
    outfield = _outfield(pressing)
    if outfield.empty:
        for name in ("def_line_x", "def_block_depth", "def_block_width",
                     "def_spread", "carrier_to_def_line",
                     "n_pressing_goalside"):
            out[name] = np.nan
    else:
        xs = outfield["x"].astype(float)
        ys = outfield["y"].astype(float)
        # The defending team's own goal is at +x, so its last line is the
        # players with the largest x.
        deepest = xs.nlargest(min(DEFENSIVE_LINE_PLAYERS, len(xs)))
        out["def_line_x"] = float(deepest.mean())
        out["def_block_depth"] = float(xs.max() - xs.min())
        out["def_block_width"] = float(ys.max() - ys.min())
        out["def_spread"] = float(np.hypot(xs.std(ddof=0), ys.std(ddof=0)))
        out["carrier_to_def_line"] = float(out["def_line_x"] - cx)
        out["n_pressing_goalside"] = float((xs > cx).sum())

    out["n_attacking_ahead"] = float(
        (attacking["x"].astype(float) > cx).sum() - 1  # the carrier is not ahead
    )

    return {name: out.get(name, np.nan) for name in FEATURE_COLUMNS}


#: Canonical order; also the model input list.
FEATURE_COLUMNS = [
    "carrier_x",
    "carrier_y_abs",
    "carrier_dist_to_goal",
    "carrier_dist_to_sideline",
    "carrier_speed",
    "ball_dist_to_carrier",
    "ball_speed",
    "presser_dist",
    "presser_angle",
    "presser_goalside",
    "presser_speed",
    "presser_dist_to_goal",
    "closing_speed",
    "support_dist",
    "n_pressing_near",
    "n_pressing_local",
    "n_attacking_near",
    "n_attacking_local",
    "numerical_advantage",
    "options_shadowed_by_presser",
    "n_open_short",
    "n_open_medium",
    "n_open_long",
    "n_open_forward",
    "nearest_open_dist",
    "def_line_x",
    "def_block_depth",
    "def_block_width",
    "def_spread",
    "carrier_to_def_line",
    "n_pressing_goalside",
    "n_attacking_ahead",
]


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _distances(players: pd.DataFrame, x: float, y: float) -> np.ndarray:
    if players.empty:
        return np.array([])
    return np.hypot(
        players["x"].astype(float).to_numpy() - x,
        players["y"].astype(float).to_numpy() - y,
    )


def _closing_speed(presser, carrier, dist: float) -> float:
    """How fast the gap is shrinking, in m/s. Positive means closing."""
    try:
        rvx = float(presser["vx"]) - float(carrier["vx"])
        rvy = float(presser["vy"]) - float(carrier["vy"])
    except (TypeError, ValueError, KeyError):
        return np.nan
    if not np.isfinite([rvx, rvy]).all() or dist < 1e-6:
        return np.nan
    ux = (float(carrier["x"]) - float(presser["x"])) / dist
    uy = (float(carrier["y"]) - float(presser["y"])) / dist
    return float(rvx * ux + rvy * uy)


def _nearest_teammate_distance(team: pd.DataFrame, player_id) -> float:
    others = team.drop(index=player_id, errors="ignore")
    if others.empty or player_id not in team.index:
        return np.nan
    me = team.loc[player_id]
    d = _distances(others, float(me["x"]), float(me["y"]))
    return float(d.min()) if len(d) else np.nan


def _lane_blocked(cx: float, cy: float, receivers: pd.DataFrame,
                  blockers: np.ndarray) -> np.ndarray:
    """For each receiver, whether a defender stands in the passing lane.

    A lane is the straight segment from the carrier to a receiver. A defender
    blocks it when they are within :data:`SHADOW_HALF_WIDTH_M` of that segment
    and between the two players rather than past either end.
    """
    if receivers.empty or len(blockers) == 0:
        return np.zeros(len(receivers), dtype=bool)

    targets = receivers[["x", "y"]].astype(float).to_numpy()
    lanes = targets - np.array([cx, cy])
    lengths = np.hypot(lanes[:, 0], lanes[:, 1])
    rel = blockers - np.array([cx, cy])  # (n_blockers, 2)
    blocked = np.zeros(len(targets), dtype=bool)
    for i, (lane, length) in enumerate(zip(lanes, lengths)):
        if length <= 1e-6:
            continue
        unit = lane / length
        along = rel @ unit
        across = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
        blocked[i] = np.any(
            (along > 0) & (along < length) & (across < SHADOW_HALF_WIDTH_M)
        )
    return blocked


def _open(cx: float, cy: float, receivers: pd.DataFrame,
          blockers: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """For each receiver, whether they are a free outlet.

    Open means no defender within :data:`OPEN_RADIUS_M` of the receiver and,
    when the pass is no longer than :data:`LONG_PASS_M`, a clear lane. ``dist``
    is each receiver's distance from the carrier.
    """
    if receivers.empty:
        return np.zeros(0, dtype=bool)
    if len(blockers) == 0:
        return np.ones(len(receivers), dtype=bool)
    targets = receivers[["x", "y"]].astype(float).to_numpy()
    gaps = np.hypot(targets[:, None, 0] - blockers[None, :, 0],
                    targets[:, None, 1] - blockers[None, :, 1]).min(axis=1)
    lane_clear = ~_lane_blocked(cx, cy, receivers, blockers) | (dist > LONG_PASS_M)
    return lane_clear & (gaps > OPEN_RADIUS_M)


def _outfield(team: pd.DataFrame) -> pd.DataFrame:
    """Team without its goalkeeper.

    Metadata marks starting keepers as ``GK`` but labels substitutes ``SUB``,
    so when no ``GK`` is on the pitch the deepest player stands in for one.
    That is a guess, and it is why the defensive-line features are treated as
    approximate.
    """
    if team.empty:
        return team
    keepers = team[team["position"] == "GK"]
    if not keepers.empty:
        return team.drop(index=keepers.index)
    return team.drop(index=team["x"].astype(float).idxmax())
