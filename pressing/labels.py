"""Did the press win the ball back?

The events file ships an ``end_type`` for engagements, but it is populated on
only about a quarter of them, and the gaps are not random: roughly half of the
possessions that actually end in a turnover have no engagement labelled at all.
Training on that subset alone would learn which outcomes SkillCorner chose to
record.

So the label is rebuilt from the event sequence instead: an engagement counts
as a recovery when the **pressing team starts a possession within a few seconds
of the engagement starting**. That is defined for every engagement, and it
agrees with ``end_type`` on about nine out of ten of the rows where ``end_type``
does exist (see ``scripts/label_report.py``).

What this label is not
----------------------
It records that the ball changed hands, not that the press caused it. The
carrier may simply have misplaced a pass. Two companion columns exist to probe
that: ``regain_after_interruption`` marks recoveries that began from a restart
rather than open play, and ``carrier_possession_end_type`` separates a direct
turnover from a recovery that arrived later in the move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, events as events_mod

#: Default window, in seconds, between the engagement starting and the pressing
#: team gaining possession. Sensitivity to this choice is part of the analysis,
#: so nothing downstream should hard-code it.
DEFAULT_WINDOW_SECONDS = 5.0

TIMING_COLUMNS = [
    "seconds_to_regain",
    "regain_event_id",
    "regain_after_interruption",
    "carrier_possession_end_type",
]


def add_recovery_timing(engagements: pd.DataFrame) -> pd.DataFrame:
    """Add how long after each engagement its team next had the ball.

    Adds the columns in :data:`TIMING_COLUMNS`:

    ``seconds_to_regain``
        Seconds from the engagement starting to the pressing team's next
        possession **in the same period**; ``inf`` when there is none.
    ``regain_event_id``
        The possession event that recovery refers to.
    ``regain_after_interruption``
        True when that possession followed a stoppage (throw-in, free kick,
        goal kick), so the ball came back from a restart rather than open play.
    ``carrier_possession_end_type``
        How the carrier's own possession ended, for the stricter label.

    Input must carry ``match_id``; matches are handled independently.
    """
    if not engagements.index.is_unique:
        raise ValueError(
            "engagements index must be unique; call reset_index() first"
        )
    out = []
    for match_id, grp in engagements.groupby("match_id", sort=False):
        out.append(_timing_for_match(match_id, grp))
    return pd.concat(out).loc[engagements.index]


def label_regain(
    df: pd.DataFrame, window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> pd.Series:
    """The main label: ball recovered within ``window_seconds``."""
    _require_timing(df)
    return df["seconds_to_regain"] <= float(window_seconds)


def label_regain_open_play(
    df: pd.DataFrame, window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> pd.Series:
    """Recovery within the window that did not come from a restart."""
    _require_timing(df)
    return label_regain(df, window_seconds) & ~df["regain_after_interruption"]


def label_regain_strict(
    df: pd.DataFrame, window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> pd.Series:
    """Recovery within the window *and* the carrier lost the ball directly.

    Closer to the press forcing the turnover, at the cost of a much smaller
    positive class. Used as a robustness check, not as the headline label.
    """
    _require_timing(df)
    return label_regain(df, window_seconds) & (
        df["carrier_possession_end_type"] == "possession_loss"
    )


def _timing_for_match(match_id, engagements: pd.DataFrame) -> pd.DataFrame:
    all_events = events_mod.load_events(match_id)

    poss = all_events[all_events["event_type"] == events_mod.POSSESSION]
    poss = poss[["event_id", "frame_start", "team_id", "period",
                 "game_interruption_before", "end_type"]].copy()
    poss = poss.sort_values("frame_start")

    # The carrier's own possession, joined through the id the engagement
    # already carries, gives how that possession ended.
    end_by_event = dict(zip(poss["event_id"], poss["end_type"]))

    seconds = np.full(len(engagements), np.inf)
    regain_id = np.array([None] * len(engagements), dtype=object)
    after_stop = np.zeros(len(engagements), dtype=bool)

    positions = {pid: i for i, pid in enumerate(engagements.index)}

    # Possessions are looked up per team and per period: frame numbers run
    # continuously across half time, so without the period split an engagement
    # late in the first half could match a possession in the second.
    for (team_id, period), block in poss.groupby(["team_id", "period"], sort=False):
        mask = (engagements["team_id"] == team_id) & (engagements["period"] == period)
        rows = engagements[mask]
        if rows.empty:
            continue

        frames = block["frame_start"].to_numpy()
        idx = np.searchsorted(frames, rows["frame_start"].to_numpy(), side="left")
        found = idx < len(frames)
        safe = np.minimum(idx, len(frames) - 1)

        delta = (frames[safe] - rows["frame_start"].to_numpy()) / config.FPS
        ids = block["event_id"].to_numpy()[safe]
        stops = block["game_interruption_before"].notna().to_numpy()[safe]

        for j, label_index in enumerate(rows.index):
            i = positions[label_index]
            if found[j]:
                seconds[i] = float(delta[j])
                regain_id[i] = ids[j]
                after_stop[i] = bool(stops[j])

    result = engagements.copy()
    result["seconds_to_regain"] = seconds
    result["regain_event_id"] = regain_id
    result["regain_after_interruption"] = after_stop
    result["carrier_possession_end_type"] = (
        engagements["associated_player_possession_event_id"].map(end_by_event)
    )
    return result


def _require_timing(df: pd.DataFrame) -> None:
    missing = [c for c in TIMING_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"missing {missing}; call pressing.labels.add_recovery_timing first"
        )
