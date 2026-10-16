"""What each part of the press does to the carrier, not just whether it wins the ball.

    python scripts/press_outcomes.py
    python scripts/press_outcomes.py --bootstrap 300

A regain within five seconds is one way a press can succeed. Another is to
take away the short option and force a hurried long ball, or to turn the
carrier back towards their own goal. A regain label cannot credit those, so
this script looks at how the carrier's possession actually ended, from
SkillCorner's own event data, and asks how each press metric moves the odds of
each ending, with SkillCorner's context held fixed exactly as in
:mod:`collective_press`.

Endings, read from the carrier's possession that the engagement belongs to:

``lost the ball``     the possession ended in a possession loss
``long pass``         ended in a pass SkillCorner calls long
``high pass``         ended in a pass through the air
``backward pass``     ended in a pass SkillCorner calls backward
``pass failed``       ended in a pass that did not reach a teammate

All five metrics enter together, so each number is what that metric adds once
the others are known. Association, not causation, as before.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import collective_press as cp  # noqa: E402
from pressing import config, modelling  # noqa: E402

POSSESSION_COLUMNS = ["event_id", "end_type", "pass_range", "high_pass",
                      "pass_direction", "pass_outcome"]

OUTCOMES = {
    "lost the ball": lambda p: p["end_type"] == "possession_loss",
    "long pass": lambda p: (p["end_type"] == "pass") & (p["pass_range"] == "long"),
    "high pass": lambda p: (p["end_type"] == "pass") & (p["high_pass"].astype(str) == "True"),
    "backward pass": lambda p: (p["end_type"] == "pass") & (p["pass_direction"] == "backward"),
    "pass failed": lambda p: (p["end_type"] == "pass") & (p["pass_outcome"] != "successful"),
}


def events_path(match_id) -> Path:
    """The dynamic events file sits next to the tracking file of the same match."""
    tracking = Path(config.tracking_path(match_id))
    return tracking.parent / f"{match_id}_dynamic_events.csv"


def carrier_endings(df: pd.DataFrame) -> pd.DataFrame:
    """For every engagement row, how the carrier's possession ended."""
    parts = []
    for match_id in df["meta_match_id"].unique():
        events = pd.read_csv(events_path(match_id), low_memory=False)
        events["event_id"] = events["event_id"].astype(str)
        link = events[["event_id", "associated_player_possession_event_id"]].rename(
            columns={"event_id": "meta_event_id",
                     "associated_player_possession_event_id": "possession_id"}
        )
        link["possession_id"] = link["possession_id"].astype(str)
        poss = events.loc[events["event_type"] == "player_possession", POSSESSION_COLUMNS]
        poss = poss.rename(columns={"event_id": "possession_id"})
        rows = df.loc[df["meta_match_id"] == match_id, ["meta_match_id", "meta_event_id"]].copy()
        rows["meta_event_id"] = rows["meta_event_id"].astype(str)
        rows = rows.merge(link, on="meta_event_id", how="left").merge(
            poss, on="possession_id", how="left"
        )
        parts.append(rows)
    out = pd.concat(parts, ignore_index=True)
    for name, rule in OUTCOMES.items():
        known = out["end_type"].notna()
        out[name] = np.where(known, rule(out).astype(float), np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = modelling.load_table()
    df = df.dropna(subset=list(cp.METRICS) + cp.NUMERIC_CONTROLS).reset_index(drop=True)
    df["meta_event_id"] = df["meta_event_id"].astype(str)

    endings = carrier_endings(df)
    df = df.merge(endings[["meta_match_id", "meta_event_id", *OUTCOMES]],
                  on=["meta_match_id", "meta_event_id"], how="left")
    matched = df["lost the ball"].notna()
    print(f"{len(df)} engagements, {matched.sum()} linked to the carrier's possession "
          f"({matched.mean():.1%})\n")
    df = df[matched].reset_index(drop=True)

    metrics = list(cp.METRICS)
    rows = []
    for outcome in OUTCOMES:
        sub = df.copy()
        sub["_y"] = sub[outcome].astype(int)
        if min(sub["_y"].sum(), (1 - sub["_y"]).sum()) < 50:
            print(f"{outcome}: too few cases either way to model, skipped\n")
            continue
        t0 = time.time()
        point = cp.effects(sub, metrics)
        draws = cp.bootstrap(sub, metrics, args.bootstrap, args.seed)
        print(f"{outcome}  (happens in {sub['_y'].mean():.1%} of presses, "
              f"{time.time() - t0:.0f}s)")
        for m in metrics:
            name, _, _, reads = cp.METRICS[m]
            pp = point[m][1] * 100
            lo, hi = np.percentile(draws[m][:, 1], [2.5, 97.5]) * 100
            flag = "  *" if lo > 0 or hi < 0 else ""
            print(f"  {name:<24} {reads:<24} {pp:+.1f} pts [{lo:+.1f}, {hi:+.1f}]{flag}")
            rows.append({"outcome": outcome, "base_rate": sub["_y"].mean(),
                         "metric": name, "unit": reads, "points": pp,
                         "points_lo": lo, "points_hi": hi})
        print()
    print("* the 95% interval excludes zero")

    resdir = config.REPO_ROOT / "results"
    resdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(resdir / "press_outcomes.csv", index=False)
    print("\nwritten: results/press_outcomes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
