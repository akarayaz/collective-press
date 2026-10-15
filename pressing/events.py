"""Dynamic events, and the on-ball engagements we model.

An ``on_ball_engagement`` row is one defender starting to engage the player in
possession. Verified on all 20 matches of the open dataset:

* ``team_id`` is always the engaging (defending) player's team;
* the engaging and the carrying player are never on the same team;
* ``attacking_side`` is the attacking direction of that row's own team, so on
  an engagement row it belongs to the defending team.

``end_type`` is only populated on about a quarter of the engagements, which is
why the label is built separately in :mod:`pressing.labels`.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import config

ENGAGEMENT = "on_ball_engagement"
POSSESSION = "player_possession"

#: Columns kept for an engagement. Everything here is known at the moment the
#: engagement starts; outcome columns are deliberately excluded.
ENGAGEMENT_COLUMNS = [
    "event_id",
    "match_id",
    "period",
    "frame_start",
    "time_start",
    "minute_start",
    "player_id",  # the presser
    "player_in_possession_id",  # the carrier
    "team_id",  # the pressing team
    "attacking_side",  # of the pressing team
    "event_subtype",
    "x_start",
    "y_start",
    "third_start",
    "channel_start",
    "game_state",
    "interplayer_distance_start",
    "angle_of_engagement",
    "goal_side_start",
    "delta_to_last_defensive_line_start",
    "last_defensive_line_height_start",
    "xloss_player_possession_start",
    "xshot_player_possession_start",
    "associated_player_possession_event_id",
    # When the carrier's possession began. Engagements are often logged before
    # the ball actually reaches the carrier, so this is needed to reason about
    # what the scene looks like at frame_start.
    "associated_player_possession_frame_start",
    "end_type",  # reference label, mostly missing; never a model input
]


@lru_cache(maxsize=None)
def load_events(match_id: int | str) -> pd.DataFrame:
    """All dynamic events of one match."""
    return pd.read_csv(config.events_path(match_id), low_memory=False)


def load_engagements(match_id: int | str) -> pd.DataFrame:
    """On-ball engagements of one match, with the columns listed above.

    Rows are sorted by ``frame_start``. A ``match_id`` column is guaranteed.
    """
    df = load_events(match_id)
    eng = df[df["event_type"] == ENGAGEMENT].copy()
    if "match_id" not in eng:
        eng["match_id"] = int(match_id)

    missing = [c for c in ENGAGEMENT_COLUMNS if c not in eng.columns]
    if missing:
        raise KeyError(f"match {match_id}: missing expected columns {missing}")

    eng = eng[ENGAGEMENT_COLUMNS].sort_values("frame_start")
    return eng.reset_index(drop=True)


#: SkillCorner's own read of the situation, stored on the carrier's
#: ``player_possession`` row rather than on the engagement. These are the M0
#: inputs: the summary a subscriber already has without touching tracking.
#: The ``_id`` columns are ordinal codes (1 = easiest / least pressure).
POSSESSION_CONTEXT_COLUMNS = [
    "overall_pressure_start_id",
    "passing_option_ease_start_id",
    "space_constraint_start_id",
    "time_to_impact_start_id",
    "possession_epv_for_start",
    "possession_epv_against_start",
]


def load_possession_context(match_id: int | str) -> pd.DataFrame:
    """Possession-level context, indexed by ``event_id``.

    Join onto an engagement through ``associated_player_possession_event_id``.

    Caveat worth carrying into the analysis: these values are measured when the
    *possession* starts, and 57% of engagements are logged before the ball
    reaches the carrier. For those rows the context is a moment in the future
    of the engagement. ``seconds_to_possession_start`` in the feature table
    records the gap so the models can be re-run on the rows where it is not an
    issue.
    """
    df = load_events(match_id)
    poss = df[df["event_type"] == POSSESSION]
    cols = [c for c in POSSESSION_CONTEXT_COLUMNS if c in poss.columns]
    return poss.set_index("event_id")[cols]


def load_all_engagements(match_ids=None) -> pd.DataFrame:
    """Engagements of several matches stacked into one frame."""
    ids = list(match_ids) if match_ids is not None else config.available_matches()
    return pd.concat([load_engagements(m) for m in ids], ignore_index=True)
