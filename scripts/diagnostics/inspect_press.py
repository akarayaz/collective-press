"""Frame by frame: where the ball is around one press, and who is near it.

    python scripts/diagnostics/inspect_press.py
    python scripts/diagnostics/inspect_press.py --event 1925299 12_345 --before 1 --after 3

Prints the events logged around the press, then for every tracking frame the
ball's height, whether the camera saw it, the nearest player and how far the
carrier and the presser are from it. Written to tell a ball that was never
controlled apart from a dribble or an estimated ball position.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from pressing import config, events, metadata, tracking  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", nargs=2, metavar=("MATCH", "EVENT"),
                    default=["1996435", "9_684"])
    ap.add_argument("--before", type=float, default=0.5)
    ap.add_argument("--after", type=float, default=1.5)
    args = ap.parse_args()
    match_id, event_id = int(args.event[0]), args.event[1]

    ev = events.load_events(match_id)
    row = ev[ev["event_id"].astype(str) == event_id].iloc[0]
    f0 = int(row["frame_start"])
    fps = config.FPS
    start = row.get("associated_player_possession_frame_start")
    print(f"engagement {event_id}: {row['event_subtype']}, frame {f0}")
    if start == start:  # not NaN
        print(f"carrier's possession starts at {(start - f0) / fps:+.1f} s")

    near = ev[(ev["frame_start"] >= f0 - 3 * fps) & (ev["frame_start"] <= f0 + 3 * fps)].copy()
    near["t"] = (near["frame_start"] - f0) / fps
    cols = [c for c in ("t", "event_type", "event_subtype", "player_name",
                        "team_shortname", "end_type") if c in near]
    print("\nevents within 3 s")
    print(near[cols].to_string(index=False))

    names = metadata.load_match(match_id).players["short_name"]
    carrier, presser = int(row["player_in_possession_id"]), int(row["player_id"])
    lo, hi = f0 - int(args.before * fps), f0 + int(args.after * fps)
    recs = tracking.read_frames(match_id, range(lo, hi + 1))

    print("\n   t     ball z  seen   nearest player       carrier  presser  possession")
    for f in sorted(recs):
        r = recs[f]
        t = (f - f0) / fps
        ball = r.get("ball_data") or {}
        if ball.get("x") is None:
            print(f"{t:+5.1f} s  no ball position")
            continue
        df = tracking.frame_to_dataframe(r)
        d = np.hypot(df["x"] - ball["x"], df["y"] - ball["y"]).sort_values()
        nearest = names.get(d.index[0], d.index[0])
        holder = (r.get("possession") or {}).get("player_id")
        z = ball.get("z")
        print(f"{t:+5.1f} s  {'' if z is None else f'{z:4.1f}':>6}  "
              f"{str(ball.get('is_detected')):<5}  {str(nearest)[:16]:<16} {d.iloc[0]:4.1f} m  "
              f"{d.get(carrier, np.nan):5.1f} m  {d.get(presser, np.nan):5.1f} m  "
              f"{names.get(holder, '-') if holder is not None else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
