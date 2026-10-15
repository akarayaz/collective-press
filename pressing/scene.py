"""Turning engagement rows into scenes: who is where, moving how fast.

A :class:`Scene` is the input the models consume. The tabular models read
summary numbers off it (see :mod:`pressing.features`); the graph model reads
the player table directly, one node per player.

Orientation
-----------
Coordinates are rotated so the **team in possession always attacks towards +x**.
Teams swap ends at half time, so raw coordinates are not comparable across
periods, and this is the usual convention in football visualisation: play flows
left to right. On an engagement row ``attacking_side`` belongs to the pressing
team, so the carrying team's direction is its opposite.

After rotation the pressing team is defending the goal at ``+x``.

Reading cost
------------
A tracking file is about 90 MB and :func:`pressing.tracking.read_frames` streams
it once per call, so building scenes one at a time would re-read the file for
every engagement. :func:`build_scenes` collects every frame a match needs and
reads them in a single pass; :func:`build_scene` is the one-off wrapper.

Velocity
--------
Velocities are least-squares slopes of position on time over the
``lookback_frames`` before the engagement start plus the start frame itself
(0.5 s and six frames by default), see :func:`pressing.tracking.velocities_fit`.
Only frames up to the start are used. Until 14 October 2026 the velocity was
the difference between the start frame and the frame 0.5 s earlier; the window
is unchanged, only the estimator is, because a two-point difference is moved by
noise in either frame and is lost entirely when the earlier frame is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from . import config, metadata, tracking

DEFAULT_LOOKBACK_FRAMES = 5


@dataclass
class Scene:
    """One frozen moment: the frame an engagement starts on.

    ``players`` is indexed by ``player_id`` with columns

        x, y, vx, vy, speed, is_detected, team_id, side, is_presser,
        is_carrier, number, position, short_name

    where ``side`` is ``"pressing"``, ``"attacking"`` or ``"unknown"`` (a
    tracked player absent from the metadata). Positions are in metres with the
    centre spot at the origin, oriented as described in the module docstring.
    """

    match_id: str
    event_id: str
    frame: int
    period: int
    time_start: str
    players: pd.DataFrame
    ball: tuple[float, float] | None
    ball_velocity: tuple[float, float] | None
    pressing_team_id: int
    attacking_team_id: int
    attacking_side: str  # of the team in possession, before rotation
    info: dict

    @property
    def presser(self) -> pd.Series:
        return self.players[self.players["is_presser"]].iloc[0]

    @property
    def carrier(self) -> pd.Series | None:
        rows = self.players[self.players["is_carrier"]]
        return None if rows.empty else rows.iloc[0]

    @property
    def pressing(self) -> pd.DataFrame:
        return self.players[self.players["side"] == "pressing"]

    @property
    def attacking(self) -> pd.DataFrame:
        return self.players[self.players["side"] == "attacking"]


class SceneError(Exception):
    """A scene could not be built, with the reason in the message."""


INFO_KEYS = (
    "event_subtype",
    "third_start",
    "game_state",
    "interplayer_distance_start",
    "angle_of_engagement",
    "xloss_player_possession_start",
    "end_type",
)


def build_scenes(
    match_id: int | str,
    engagements: pd.DataFrame,
    lookback_frames: int = DEFAULT_LOOKBACK_FRAMES,
    normalise: bool = True,
    skip_errors: bool = True,
) -> Iterator[tuple[pd.Series, Scene | None, str | None]]:
    """Build scenes for many engagements of one match, reading the file once.

    Yields ``(event_row, scene, error)``. When a scene cannot be built the
    scene is ``None`` and ``error`` says why, so a caller can record the row as
    unusable instead of losing it silently. With ``skip_errors=False`` the
    error is raised instead.
    """
    match = metadata.load_match(match_id)

    frames = set()
    for frame in engagements["frame_start"]:
        frames.update(_window(int(frame), lookback_frames))
    records = tracking.read_frames(match_id, frames)

    for _, event in engagements.iterrows():
        try:
            yield event, _scene_from_records(
                match, event, records, lookback_frames, normalise
            ), None
        except SceneError as exc:
            if not skip_errors:
                raise
            yield event, None, str(exc)


def build_scene(
    match_id: int | str,
    event: pd.Series,
    lookback_frames: int = DEFAULT_LOOKBACK_FRAMES,
    normalise: bool = True,
) -> Scene:
    """Build the scene for a single engagement row.

    Reads the tracking file, so prefer :func:`build_scenes` for more than a
    handful of rows.
    """
    match = metadata.load_match(match_id)
    frame = int(event["frame_start"])
    records = tracking.read_frames(match_id, _window(frame, lookback_frames))
    return _scene_from_records(match, event, records, lookback_frames, normalise)


def _window(frame: int, lookback_frames: int) -> range:
    """The start frame and the ``lookback_frames`` before it, nothing later."""
    return range(int(frame) - int(lookback_frames), int(frame) + 1)


def _scene_from_records(
    match: metadata.Match,
    event: pd.Series,
    records: dict[int, dict],
    lookback_frames: int,
    normalise: bool,
) -> Scene:
    frame = int(event["frame_start"])
    if frame not in records:
        raise SceneError(f"frame {frame} missing from tracking")

    curr = tracking.frame_to_dataframe(records[frame])
    if curr.empty:
        raise SceneError(f"frame {frame} has no tracked players")

    history, ball_history = [], []
    for k in range(int(lookback_frames), -1, -1):
        rec = records.get(frame - k)
        if rec is None:
            continue
        t = -k / config.FPS
        history.append((t, curr if k == 0 else tracking.frame_to_dataframe(rec)))
        ball_history.append((t, tracking.ball_xy(rec)))
    players = curr.join(tracking.velocities_fit(history, curr))

    presser_id = int(event["player_id"])
    carrier_raw = event.get("player_in_possession_id")
    carrier_id = None if pd.isna(carrier_raw) else int(carrier_raw)

    pressing_team = int(event["team_id"])
    attacking_team = next(t for t in match.teams if t != pressing_team)

    players = players.join(
        match.players[["team_id", "number", "position", "short_name"]], how="left"
    )
    players["side"] = players["team_id"].map(
        {pressing_team: "pressing", attacking_team: "attacking"}
    ).fillna("unknown")
    players["is_presser"] = players.index == presser_id
    players["is_carrier"] = (
        False if carrier_id is None else players.index == carrier_id
    )

    if not players["is_presser"].any():
        raise SceneError(f"presser {presser_id} not tracked in frame {frame}")

    ball = tracking.ball_xy(records[frame])
    ball_velocity = tracking.point_velocity_fit(ball_history)

    # ``attacking_side`` on an engagement row is the pressing team's direction.
    carrier_side = metadata.opposite_side(str(event["attacking_side"]))
    if normalise:
        players = tracking.flip_for_attacking_side(players, carrier_side)
        ball = tracking.flip_point(ball, carrier_side)
        ball_velocity = tracking.flip_point(ball_velocity, carrier_side)

    info = {k: event[k] for k in INFO_KEYS if k in event.index}

    return Scene(
        match_id=str(match.match_id),
        event_id=str(event["event_id"]),
        frame=frame,
        period=int(event["period"]),
        time_start=str(event.get("time_start", "")),
        players=players,
        ball=ball,
        ball_velocity=ball_velocity,
        pressing_team_id=pressing_team,
        attacking_team_id=attacking_team,
        attacking_side=carrier_side,
        info=info,
    )
