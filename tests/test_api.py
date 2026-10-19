"""The public entry point, :func:`pressing.api.press_metrics`.

Run directly (``python tests/test_api.py``) or under pytest. No data needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pressing.api import FEATURE_COLUMNS, HEADLINE, press_metrics  # noqa: E402
from pressing.features import scene_features  # noqa: E402
from test_features import make_scene  # noqa: E402

PLAYERS = [
    {"id": 10, "x": 12.0, "y": -4.0, "side": "attacking", "vx": 2.0, "vy": 0.5},
    {"id": 11, "x": 20.0, "y": 6.0, "side": "attacking", "vx": 1.0, "vy": 0.0},
    {"id": 12, "x": 2.0, "y": -12.0, "side": "attacking"},
    {"id": 13, "x": 35.0, "y": -20.0, "side": "attacking"},
    {"id": 4, "x": 15.5, "y": -3.0, "side": "pressing", "vx": -3.0, "vy": 0.2},
    {"id": 5, "x": 17.0, "y": 2.0, "side": "pressing", "vx": -1.0, "vy": 0.0},
    {"id": 6, "x": 30.0, "y": -8.0, "side": "pressing"},
    {"id": 7, "x": 31.0, "y": 8.0, "side": "pressing"},
    {"id": 1, "x": 50.0, "y": 0.0, "side": "pressing", "position": "GK"},
]
BALL = (12.3, -4.1)


def same(a: dict, b: dict) -> None:
    assert list(a) == list(b) == FEATURE_COLUMNS
    for k in FEATURE_COLUMNS:
        if np.isnan(a[k]) and np.isnan(b[k]):
            continue
        assert abs(a[k] - b[k]) < 1e-9, f"{k}: {a[k]} != {b[k]}"


def test_matches_the_pipeline():
    """The entry point gives exactly what the analysis computed from scenes."""
    rows = [dict(p, presser=p["id"] == 4, carrier=p["id"] == 10) for p in PLAYERS]
    expected = scene_features(make_scene(rows, ball=BALL))
    same(press_metrics(PLAYERS, presser=4, carrier=10, ball=BALL), expected)


def test_direction_does_not_matter():
    """A team attacking to the left, rotated, gives the same numbers."""
    mirrored = [dict(p, x=-p["x"], y=-p["y"],
                     **({"vx": -p["vx"], "vy": -p["vy"]} if "vx" in p else {}))
                for p in PLAYERS]
    same(press_metrics(mirrored, 4, 10, ball=(-BALL[0], -BALL[1]), attack="left"),
         press_metrics(PLAYERS, 4, 10, ball=BALL))


def test_dataframe_input_with_ids_in_the_index():
    import pandas as pd

    df = pd.DataFrame(PLAYERS).set_index("id")
    same(press_metrics(df, 4, 10, ball=BALL), press_metrics(PLAYERS, 4, 10, ball=BALL))


def test_without_velocities_only_the_speeds_are_missing():
    still = [{k: v for k, v in p.items() if k not in ("vx", "vy")} for p in PLAYERS]
    m = press_metrics(still, 4, 10, ball=BALL)
    speeds = {"carrier_speed", "presser_speed", "closing_speed", "ball_speed"}
    assert all(np.isnan(m[k]) for k in speeds)
    assert not any(np.isnan(m[k]) for k in FEATURE_COLUMNS if k not in speeds)


def test_bad_input_is_explained():
    for kwargs, words in (
        (dict(presser=10, carrier=4), "pressing side"),
        (dict(presser=99, carrier=10), "not in players"),
        (dict(presser=4, carrier=10, attack="up"), "attack must be"),
    ):
        try:
            press_metrics(PLAYERS, **kwargs)
        except ValueError as exc:
            assert words in str(exc), str(exc)
        else:
            raise AssertionError(f"no error for {kwargs}")


def test_headline_metrics_exist():
    assert set(HEADLINE.values()) <= set(FEATURE_COLUMNS)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
