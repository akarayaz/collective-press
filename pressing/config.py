"""Paths and pitch constants.

The SkillCorner open data is NOT part of this repository. Set the environment
variable ``PRESSING_DATA_DIR`` to the ``data`` folder of your local clone of
https://github.com/SkillCorner/opendata, or keep the default layout:

    some_parent/
        analytics-cup/      <- this repository
        opendata/           <- SkillCorner clone
            data/
                matches/
                aggregates/
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_default_data = REPO_ROOT.parent / "opendata" / "data"
DATA_DIR = Path(os.environ.get("PRESSING_DATA_DIR", _default_data)).expanduser()

MATCHES_DIR = DATA_DIR / "matches"
AGGREGATES_DIR = DATA_DIR / "aggregates"
FIGURES_DIR = REPO_ROOT / "figures"

# Pitch dimensions in metres, as given in each match.json. Origin is the centre
# spot, so x runs over [-52.5, 52.5] and y over [-34, 34].
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Tracking sampling rate of the extrapolated tracking files.
FPS = 10


def match_dir(match_id: int | str) -> Path:
    return MATCHES_DIR / str(match_id)


def tracking_path(match_id: int | str) -> Path:
    return match_dir(match_id) / f"{match_id}_tracking_extrapolated.jsonl"


def events_path(match_id: int | str) -> Path:
    return match_dir(match_id) / f"{match_id}_dynamic_events.csv"


def metadata_path(match_id: int | str) -> Path:
    return match_dir(match_id) / f"{match_id}_match.json"


def available_matches() -> list[str]:
    """Match ids present in the local data directory, sorted."""
    if not MATCHES_DIR.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {MATCHES_DIR}\n"
            "Set PRESSING_DATA_DIR to the 'data' folder of your SkillCorner "
            "opendata clone."
        )
    return sorted(p.name for p in MATCHES_DIR.iterdir() if p.is_dir())
