"""Which frame does the events file actually describe?

    python scripts/check_alignment.py --all     # writes alignment_check.csv
    python scripts/find_reference_frame.py

On about 1.4% of engagements the two named players sit tens of metres apart in
the tracking frame while the events file reports them a few metres apart. The
tracks through those moments are smooth and both players are on the pitch, so
the tracking is not at fault and the two players are the right ones. What is
left is that ``interplayer_distance_start`` describes a different instant from
``frame_start``.

That would matter well beyond the tail. If SkillCorner's notion of "the start
of the engagement" is not the frame this code reads, every feature is built a
moment early or late, on every row, and the tail is only where that gap grows
large enough to notice.

So: scan the frames around each engagement, find the one where the measured
distance matches the reported one, and see whether the offset lands on a frame
the events file already names.

    frame_start                               the engagement, as read now
    associated_player_possession_frame_start  the carrier receiving the ball
    frame_physical_start                      the physical-tracking frame
    frame_end                                 the engagement ending

A control sample of engagements that already agree is scanned the same way;
their best frame should be the engagement frame itself, which is what makes
the comparison worth anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, events, tracking  # noqa: E402

#: How close a frame's distance has to be to the reported one to count.
TOLERANCE_M = 0.25

CANDIDATES = {
    "possession_start": "associated_player_possession_frame_start",
    "physical_start": "frame_physical_start",
    "engagement_end": "frame_end",
}


def _distance_at(records, frame, presser, carrier):
    rec = records.get(frame)
    if rec is None:
        return np.nan
    pos = {}
    for p in rec.get("player_data") or []:
        pid = int(p["player_id"])
        if pid in (presser, carrier):
            pos[pid] = (p["x"], p["y"])
    if len(pos) < 2:
        return np.nan
    (ax, ay), (bx, by) = pos[presser], pos[carrier]
    return float(np.hypot(ax - bx, ay - by))


def analyse_match(match_id, rows, half_window: int):
    raw = events.load_events(match_id)
    eng = raw[raw["event_type"] == events.ENGAGEMENT].set_index("event_id")

    needed = set()
    for event_id in rows["event_id"]:
        frame = int(eng.at[event_id, "frame_start"])
        needed.update(range(frame - half_window, frame + half_window + 1))
        for col in CANDIDATES.values():
            value = eng.at[event_id, col]
            if pd.notna(value):
                needed.add(int(value))

    records = tracking.read_frames(match_id, needed)

    out = []
    for _, row in rows.iterrows():
        event_id = row["event_id"]
        e = eng.loc[event_id]
        frame = int(e["frame_start"])
        presser = int(e["player_id"])
        carrier = int(e["player_in_possession_id"])
        reported = float(e["interplayer_distance_start"])

        window = np.arange(frame - half_window, frame + half_window + 1)
        dists = np.array(
            [_distance_at(records, int(f), presser, carrier) for f in window]
        )
        errs = np.abs(dists - reported)

        record = {
            "match_id": match_id,
            "event_id": event_id,
            "group": row["group"],
            "reported": reported,
            "abs_error_at_frame_start": row["abs_error"],
        }

        if np.all(np.isnan(errs)):
            record["best_offset_s"] = np.nan
            record["best_error"] = np.nan
        else:
            best = int(np.nanargmin(errs))
            record["best_offset_s"] = float(window[best] - frame) / config.FPS
            record["best_error"] = float(errs[best])

        for name, col in CANDIDATES.items():
            value = e[col]
            if pd.isna(value):
                record[f"err_at_{name}"] = np.nan
                continue
            d = _distance_at(records, int(value), presser, carrier)
            record[f"err_at_{name}"] = abs(d - reported)

        out.append(record)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=5.0,
                    help="seconds either side of the engagement to scan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = config.REPO_ROOT / "alignment_check.csv"
    if not path.is_file():
        print(f"{path} not found; run scripts/check_alignment.py --all first",
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
    print(f"scanning +/-{args.window:g} s around {len(bad)} disagreeing and "
          f"{len(good)} control engagements\n")

    results = []
    for i, (match_id, grp) in enumerate(rows.groupby("match_id", sort=True), 1):
        print(f"[{i}] match {match_id}: {len(grp)}", flush=True)
        results.extend(analyse_match(match_id, grp, half_window))
    res = pd.DataFrame(results)

    res["found"] = res["best_error"] < TOLERANCE_M
    print(f"\nA frame within +/-{args.window:g} s where the distance matches "
          f"the reported one to {TOLERANCE_M} m")
    print(res.groupby("group")["found"].mean()
          .to_string(float_format=lambda v: f"{v:.1%}"))

    print("\nHow far that frame sits from frame_start, in seconds")
    found = res[res["found"]]
    if not found.empty:
        print(
            found.groupby("group")["best_offset_s"]
            .describe()[["count", "mean", "25%", "50%", "75%"]]
            .to_string(float_format=lambda v: f"{v:.2f}")
        )
        print("\n  offset distribution for the disagreeing group")
        offsets = found.loc[found["group"] == "disagrees", "best_offset_s"]
        print(offsets.round(1).value_counts().head(10).to_string())

    print(f"\nDoes a frame the events file already names fit better?")
    print(f"  (share within {TOLERANCE_M} m of the reported distance)")
    cols = ["abs_error_at_frame_start"] + [f"err_at_{n}" for n in CANDIDATES]
    summary = pd.DataFrame(
        {c: res.groupby("group")[c].apply(lambda s: (s < TOLERANCE_M).mean())
         for c in cols}
    )
    summary.columns = ["frame_start"] + list(CANDIDATES)
    print(summary.T.to_string(float_format=lambda v: f"{v:.1%}"))

    out = config.REPO_ROOT / "reference_frame.csv"
    res.to_csv(out, index=False)
    print(f"\nfull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
