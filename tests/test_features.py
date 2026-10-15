"""Geometry checks on hand-built scenes, where the answer is known in advance.

Run directly (``python tests/test_features.py``) or under pytest. No data
needed: the scenes are constructed in code, which is the point. Real tracking
can tell you a feature runs; only a scene you laid out yourself can tell you it
measures what its name says.

Orientation: the team in possession attacks towards +x, so the pressing team
defends the goal at (+52.5, 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing.features import scene_features  # noqa: E402
from pressing.scene import Scene  # noqa: E402


def make_scene(players: list[dict], ball=(0.0, 0.0), ball_velocity=None) -> Scene:
    """Build a Scene from a list of player dicts.

    Each needs ``id, x, y, side``; ``vx, vy, position`` are optional.
    Exactly one player must be the presser and one the carrier.
    """
    rows = []
    for p in players:
        vx, vy = p.get("vx", 0.0), p.get("vy", 0.0)
        rows.append(
            {
                "player_id": p["id"],
                "x": p["x"],
                "y": p["y"],
                "vx": vx,
                "vy": vy,
                "speed": float(np.hypot(vx, vy)),
                "is_detected": True,
                "team_id": 1 if p["side"] == "pressing" else 2,
                "side": p["side"],
                "is_presser": p.get("presser", False),
                "is_carrier": p.get("carrier", False),
                "number": p["id"],
                "position": p.get("position", "CM"),
                "short_name": f"P{p['id']}",
            }
        )
    frame = pd.DataFrame(rows).set_index("player_id")
    return Scene(
        match_id="test", event_id="t1", frame=0, period=1, time_start="00:00.0",
        players=frame, ball=ball, ball_velocity=ball_velocity,
        pressing_team_id=1, attacking_team_id=2,
        attacking_side="left_to_right", info={},
    )


def close(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"expected {b}, got {a}"


def test_presser_in_front_of_carrier():
    """A defender standing between carrier and goal: angle 0, goal side."""
    sc = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 5.0, "y": 0.0, "side": "pressing", "presser": True},
    ])
    f = scene_features(sc)
    close(f["presser_dist"], 5.0)
    close(f["presser_angle"], 0.0)
    close(f["presser_goalside"], 1.0)
    close(f["carrier_dist_to_goal"], 52.5)


def test_presser_chasing_from_behind():
    """Same distance, opposite side: angle 180, not goal side."""
    sc = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": -5.0, "y": 0.0, "side": "pressing", "presser": True},
    ])
    f = scene_features(sc)
    close(f["presser_angle"], 180.0)
    close(f["presser_goalside"], 0.0)


def test_closing_speed_signs():
    """Positive when the gap shrinks, negative when it grows."""
    closing = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 5.0, "y": 0.0, "vx": -2.0, "side": "pressing",
         "presser": True},
    ])
    close(scene_features(closing)["closing_speed"], 2.0)

    retreating = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 5.0, "y": 0.0, "vx": 2.0, "side": "pressing",
         "presser": True},
    ])
    close(scene_features(retreating)["closing_speed"], -2.0)

    # Both running the same way at the same speed: the gap is constant.
    parallel = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "vx": 3.0, "side": "attacking",
         "carrier": True},
        {"id": 2, "x": 5.0, "y": 0.0, "vx": 3.0, "side": "pressing",
         "presser": True},
    ])
    close(scene_features(parallel)["closing_speed"], 0.0)


def test_cover_shadow():
    """A defender on the passing lane blocks it; one off the lane does not."""
    sc = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 10.0, "y": 0.0, "side": "attacking"},   # blocked
        {"id": 3, "x": 0.0, "y": 10.0, "side": "attacking"},   # open
        {"id": 4, "x": 5.0, "y": 0.0, "side": "pressing", "presser": True},
    ])
    f = scene_features(sc)
    close(f["n_options"], 2.0)
    close(f["options_shadowed_by_presser"], 1.0)
    close(f["options_shadowed_by_team"], 1.0)


def test_defender_past_the_receiver_does_not_block():
    """Standing beyond the receiver is not standing in the lane."""
    sc = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 10.0, "y": 0.0, "side": "attacking"},
        {"id": 3, "x": 15.0, "y": 0.0, "side": "pressing", "presser": True},
    ])
    close(scene_features(sc)["options_shadowed_by_presser"], 0.0)


def test_team_blocks_more_lanes_than_the_presser_alone():
    sc = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 10.0, "y": 0.0, "side": "attacking"},
        {"id": 3, "x": 0.0, "y": 10.0, "side": "attacking"},
        {"id": 4, "x": 5.0, "y": 0.0, "side": "pressing", "presser": True},
        {"id": 5, "x": 0.0, "y": 5.0, "side": "pressing"},
    ])
    f = scene_features(sc)
    close(f["options_shadowed_by_presser"], 1.0)
    close(f["options_shadowed_by_team"], 2.0)


def test_local_counts_and_advantage():
    sc = make_scene([
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 3.0, "y": 0.0, "side": "pressing", "presser": True},
        {"id": 3, "x": 8.0, "y": 0.0, "side": "pressing"},
        {"id": 4, "x": 30.0, "y": 0.0, "side": "pressing"},   # far away
        {"id": 5, "x": 4.0, "y": 0.0, "side": "attacking"},
    ])
    f = scene_features(sc)
    close(f["n_pressing_near"], 1.0)      # within 5 m
    close(f["n_pressing_local"], 2.0)     # within 10 m
    close(f["n_attacking_near"], 1.0)     # the carrier is excluded
    close(f["numerical_advantage"], 1.0)
    close(f["support_dist"], 5.0)         # presser to nearest teammate


def test_goalkeeper_excluded_from_the_defensive_line():
    """The keeper sits deepest; leaving them in would drag the line back."""
    players = [
        {"id": 1, "x": 0.0, "y": 0.0, "side": "attacking", "carrier": True},
        {"id": 2, "x": 5.0, "y": 0.0, "side": "pressing", "presser": True},
        {"id": 9, "x": 50.0, "y": 0.0, "side": "pressing", "position": "GK"},
    ]
    players += [
        {"id": 10 + i, "x": 20.0, "y": float(i * 5), "side": "pressing"}
        for i in range(4)
    ]
    f = scene_features(make_scene(players))
    # Four outfielders at x=20 and the presser at x=5; the keeper at 50 is out.
    close(f["def_line_x"], 20.0)
    close(f["carrier_to_def_line"], 20.0)


def test_scene_without_a_carrier_is_all_nan():
    sc = make_scene([
        {"id": 2, "x": 5.0, "y": 0.0, "side": "pressing", "presser": True},
    ])
    f = scene_features(sc)
    assert all(pd.isna(v) for v in f.values()), "expected every feature to be NaN"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
