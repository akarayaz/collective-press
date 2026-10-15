"""Match metadata: players, teams, periods and attacking directions.

Everything here comes from ``<match_id>_match.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from . import config


@dataclass(frozen=True)
class Match:
    """Metadata for one match.

    Attributes
    ----------
    match_id:
        SkillCorner match id.
    players:
        One row per player with index ``player_id`` and columns
        ``team_id, short_name, number, position, position_group``.
    teams:
        ``{team_id: short_name}``.
    home_team_id, away_team_id:
        Team ids.
    home_side:
        ``{period: "left_to_right" | "right_to_left"}`` for the home team.
    periods:
        ``{period: (start_frame, end_frame)}``.
    """

    match_id: str
    players: pd.DataFrame
    teams: dict[int, str]
    home_team_id: int
    away_team_id: int
    home_side: dict[int, str]
    periods: dict[int, tuple[int, int]]

    def team_of(self, player_id: int) -> int:
        return int(self.players.at[player_id, "team_id"])

    def was_on_pitch(self, player_id: int, frame: int) -> bool | None:
        """Whether this player was on the pitch at this frame.

        ``None`` when the metadata records no playing time for them.
        """
        if player_id not in self.players.index:
            return None
        start = self.players.at[player_id, "start_frame"]
        end = self.players.at[player_id, "end_frame"]
        if pd.isna(start) or pd.isna(end):
            return None
        return bool(start <= frame <= end)

    def attacking_side(self, team_id: int, period: int) -> str:
        """Direction this team attacks in, for the given period.

        Matches the ``attacking_side`` column of the dynamic events file, where
        each row carries the attacking direction of that row's own team.
        """
        side = self.home_side[period]
        if team_id == self.home_team_id:
            return side
        return "left_to_right" if side == "right_to_left" else "right_to_left"


def opposite_side(side: str) -> str:
    """The other attacking direction."""
    if side == "left_to_right":
        return "right_to_left"
    if side == "right_to_left":
        return "left_to_right"
    raise ValueError(f"unknown attacking_side: {side!r}")


@lru_cache(maxsize=None)
def load_match(match_id: int | str) -> Match:
    path = config.metadata_path(match_id)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    rows = []
    for p in raw["players"]:
        role = p.get("player_role") or {}
        total = (p.get("playing_time") or {}).get("total") or {}
        rows.append(
            {
                "player_id": int(p["id"]),
                "team_id": int(p["team_id"]),
                "short_name": p.get("short_name") or "",
                "number": p.get("number"),
                "position": role.get("acronym") or "",
                "position_group": role.get("position_group") or "",
                # The span this player was actually on the pitch, used to tell
                # a tracking error apart from a player who had been
                # substituted. Missing for an unused substitute.
                "start_frame": total.get("start_frame"),
                "end_frame": total.get("end_frame"),
            }
        )
    players = pd.DataFrame(rows).set_index("player_id").sort_index()
    if players.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate player ids in metadata")

    home, away = raw["home_team"], raw["away_team"]
    sides = raw["home_team_side"]  # one entry per period, in order
    home_side = {i + 1: s for i, s in enumerate(sides)}
    periods = {
        int(p["period"]): (int(p["start_frame"]), int(p["end_frame"]))
        for p in raw["match_periods"]
    }

    return Match(
        match_id=str(match_id),
        players=players,
        teams={int(home["id"]): home["short_name"], int(away["id"]): away["short_name"]},
        home_team_id=int(home["id"]),
        away_team_id=int(away["id"]),
        home_side=home_side,
        periods=periods,
    )

