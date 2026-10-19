"""Why do some engagements disagree with the tracking by tens of metres?

    python scripts/diagnostics/check_alignment.py --all     # writes results/diagnostics/alignment_check.csv
    python scripts/diagnostics/diagnose_tail.py

About 1.4% of engagements put the two named players far apart in the tracking
frame while the events file reports them within a few metres. Extrapolated
positions account for roughly half of those. This script tests what is behind
the rest, because a structural error that is filtered without being understood
tends to come back later as a feature that quietly means something else.

Two candidates, both measured against a control sample of engagements that
agree:

``teleport``
    An identity swap in the tracking: the track belonging to one player is
    briefly assigned to another, so a player appears to jump across the pitch.
    A real player covers at most about a metre in a 0.1 s frame, so a step of
    several metres is physically impossible and marks the frame as suspect.

``another opponent at the reported distance``
    If the events file's distance matches some *other* opponent rather than
    the named carrier, then the identity, not the geometry, is what is wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, events, metadata, tracking  # noqa: E402

#: Where the diagnostics write; results/ is not tracked by git.
OUT_DIR = config.REPO_ROOT / "results" / "diagnostics"

#: A step larger than this between consecutive frames (0.1 s) is not football.
TELEPORT_M_PER_FRAME = 3.0


def _player_track(records, frames, player_id):
    """One player's positions across a window, with the frame each belongs to.

    Returns ``(frame_numbers, points)``. Frames where the player is absent are
    skipped rather than filled, so the caller must account for the gap.
    """
    nums, pts = [], []
    for f in frames:
        rec = records.get(f)
        if rec is None:
            continue
        for p in rec.get("player_data") or []:
            if int(p["player_id"]) == player_id:
                nums.append(f)
                pts.append((p["x"], p["y"]))
                break
    return np.asarray(nums, dtype=float), np.asarray(pts, dtype=float)


def _max_step(nums: np.ndarray, pts: np.ndarray) -> float:
    """Largest movement per frame, in metres.

    Distances are divided by the gap in frames, so a missing frame does not
    masquerade as a teleport.
    """
    if len(pts) < 2:
        return np.nan
    steps = np.hypot(*np.diff(pts, axis=0).T)
    gaps = np.diff(nums)
    return float((steps / gaps).max())


def analyse_match(match_id, rows: pd.DataFrame, half_window: int) -> list[dict]:
    match = metadata.load_match(match_id)
    eng = events.load_engagements(match_id).set_index("event_id")

    needed = set()
    windows = {}
    for event_id in rows["event_id"]:
        frame = int(eng.at[event_id, "frame_start"])
        window = range(frame - half_window, frame + half_window + 1)
        windows[event_id] = (frame, list(window))
        needed.update(window)

    records = tracking.read_frames(match_id, needed)

    out = []
    for _, row in rows.iterrows():
        event_id = row["event_id"]
        frame, window = windows[event_id]
        e = eng.loc[event_id]
        presser = int(e["player_id"])
        carrier = int(e["player_in_possession_id"])

        rec = records.get(frame)
        nearest = np.nan
        if rec is not None:
            players = tracking.frame_to_dataframe(rec)
            if presser in players.index:
                me = players.loc[presser]
                others = players.join(match.players[["team_id"]], how="left")
                opponents = others[others["team_id"] == match.team_of(carrier)]
                if not opponents.empty:
                    d = np.hypot(opponents["x"] - me["x"], opponents["y"] - me["y"])
                    nearest = float(d.min())

        out.append(
            {
                "match_id": match_id,
                "event_id": event_id,
                "group": row["group"],
                "abs_error": row["abs_error"],
                "reported": row["reported"],
                "nearest_opponent": nearest,
                "presser_max_step": _max_step(*_player_track(records, window, presser)),
                "carrier_max_step": _max_step(*_player_track(records, window, carrier)),
                "lead_seconds": (
                    frame - e["associated_player_possession_frame_start"]
                ) / config.FPS,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="disagreement in metres that counts as bad")
    ap.add_argument("--window", type=float, default=2.0,
                    help="seconds either side of the engagement to inspect")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = OUT_DIR / "alignment_check.csv"
    if not path.is_file():
        print(f"{path} not found; run scripts/diagnostics/check_alignment.py --all first",
              file=sys.stderr)
        return 1

    check = pd.read_csv(path)
    bad = check[check["abs_error"] > args.threshold].copy()
    good = check[check["abs_error"] <= 0.5].copy()
    if bad.empty:
        print("nothing above the threshold")
        return 0
    good = good.sample(n=min(len(bad), len(good)), random_state=args.seed)
    bad["group"] = "disagrees"
    good["group"] = "control"
    rows = pd.concat([bad, good], ignore_index=True)

    half_window = int(args.window * config.FPS)
    print(f"{len(bad)} disagreeing and {len(good)} control engagements, "
          f"+/-{args.window:g} s around each\n")

    results = []
    for i, (match_id, grp) in enumerate(rows.groupby("match_id", sort=True), 1):
        print(f"[{i}] match {match_id}: {len(grp)}", flush=True)
        results.extend(analyse_match(match_id, grp, half_window))
    res = pd.DataFrame(results)

    res["teleport"] = (
        res[["presser_max_step", "carrier_max_step"]].max(axis=1)
        > TELEPORT_M_PER_FRAME
    )
    res["matches_another_opponent"] = (
        res["nearest_opponent"] - res["reported"]
    ).abs() < 0.5

    print("\nBiggest single-frame step by either player (metres per 0.1 s)")
    print(
        res.groupby("group")[["presser_max_step", "carrier_max_step"]]
        .median()
        .to_string(float_format=lambda v: f"{v:.2f}")
    )

    print(f"\nShare showing a physically impossible step "
          f"(> {TELEPORT_M_PER_FRAME:g} m per frame)")
    print(res.groupby("group")["teleport"].mean()
          .to_string(float_format=lambda v: f"{v:.1%}"))

    print("\nShare where some other opponent sits at exactly the reported "
          "distance")
    print(res.groupby("group")["matches_another_opponent"].mean()
          .to_string(float_format=lambda v: f"{v:.1%}"))

    print("\nSeconds between the carrier gaining the ball and the engagement")
    print(res.groupby("group")["lead_seconds"]
          .describe()[["mean", "50%", "max"]]
          .to_string(float_format=lambda v: f"{v:.2f}"))

    worst = res[res["group"] == "disagrees"]
    explained = worst["teleport"] | worst["matches_another_opponent"]
    print(f"\n{explained.mean():.1%} of the disagreeing engagements show at "
          "least one of the two signatures")

    by_match = (
        res[res["group"] == "disagrees"].groupby("match_id").size()
        .sort_values(ascending=False)
    )
    print(f"\nConcentration: {len(by_match)} of 20 matches contribute; "
          f"worst three\n{by_match.head(3).to_string()}")

    out = OUT_DIR / "tail_diagnosis.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nfull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
