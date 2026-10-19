"""Do the tracking frames and the events file describe the same moment?

    python scripts/diagnostics/check_alignment.py                # one match
    python scripts/diagnostics/check_alignment.py --all          # every match

Writes ``results/diagnostics/alignment_check.csv``, which
``diagnose_tail.py`` and ``find_reference_frame.py`` read.

Everything downstream assumes that ``frame_start`` on an engagement points at
the tracking frame where that engagement begins, and that ``player_id`` and
``player_in_possession_id`` identify the two players involved. If any of that
is wrong, the features and the graph are built on sand.

The check does not take those assumptions on trust: it measures the distance
between the two players in the tracking frame and compares it with the
``interplayer_distance_start`` the events file reports. The two numbers come
from different places, so close agreement across thousands of engagements is
real evidence, where three hand-picked examples are not.

Also reported: how often a player the events file names is missing from the
frame, and how many players are tracked at all.
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


def check_match(match_id, limit: int | None, seed: int) -> pd.DataFrame:
    match = metadata.load_match(match_id)
    eng = events.load_engagements(match_id)
    if limit is not None and limit < len(eng):
        eng = eng.sample(n=limit, random_state=seed)

    # One streaming pass over the ~90 MB file for every frame we need.
    records = tracking.read_frames(match_id, eng["frame_start"].unique())

    rows = []
    for _, e in eng.iterrows():
        frame = int(e["frame_start"])
        rec = records.get(frame)
        if rec is None:
            rows.append({"match_id": match_id, "event_id": e["event_id"], "frame_missing": True})
            continue

        players = tracking.frame_to_dataframe(rec)
        presser = int(e["player_id"])
        carrier = e["player_in_possession_id"]
        carrier = None if pd.isna(carrier) else int(carrier)

        row = {
            "match_id": match_id,
            "event_id": e["event_id"],
            "frame_missing": False,
            "n_tracked": len(players),
            "presser_missing": presser not in players.index,
            "carrier_missing": carrier is None or carrier not in players.index,
            "reported": e["interplayer_distance_start"],
            "measured": np.nan,
            # Two candidate explanations for a mismatch, recorded so the tail
            # can be attributed rather than guessed at: a position the camera
            # did not see (so SkillCorner extrapolated it), and a player the
            # metadata says was not on the pitch at that frame.
            "presser_detected": pd.NA,
            "carrier_detected": pd.NA,
            "presser_on_pitch": match.was_on_pitch(presser, frame),
            "carrier_on_pitch": (
                None if carrier is None else match.was_on_pitch(carrier, frame)
            ),
        }
        if not row["presser_missing"] and not row["carrier_missing"]:
            a = players.loc[presser]
            b = players.loc[carrier]
            row["measured"] = float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))
            row["presser_detected"] = a["is_detected"]
            row["carrier_detected"] = b["is_detected"]
        rows.append(row)

    return pd.DataFrame(rows)


def _explain_tail(ok: pd.DataFrame, threshold: float = 2.0) -> None:
    """Attribute the badly-disagreeing engagements to a cause, or fail to."""
    both_seen = (
        ok["presser_detected"].fillna(False).astype(bool)
        & ok["carrier_detected"].fillna(False).astype(bool)
    )
    both_on = (
        ok["presser_on_pitch"].fillna(True).astype(bool)
        & ok["carrier_on_pitch"].fillna(True).astype(bool)
    )

    print("\nWhere the disagreement comes from")
    print("  group                         n      median   over 2 m")
    for name, mask in (
        ("both players seen by camera", both_seen),
        ("at least one extrapolated", ~both_seen),
        ("both on the pitch per metadata", both_on),
        ("at least one already substituted", ~both_on),
    ):
        sub = ok[mask]
        if sub.empty:
            print(f"  {name:<32} 0")
            continue
        over = (sub["abs_error"] > threshold).mean()
        print(f"  {name:<32}{len(sub):>6}   {sub['abs_error'].median():>7.3f} m"
              f"   {over:>7.2%}")

    bad = ok[ok["abs_error"] > threshold]
    if bad.empty:
        print(f"\n  nothing disagrees by more than {threshold} m")
        return
    extrapolated = (~both_seen).loc[bad.index].mean()
    substituted = (~both_on).loc[bad.index].mean()
    unexplained = (both_seen & both_on).loc[bad.index].mean()
    print(f"\n  of the {len(bad)} engagements disagreeing by more than "
          f"{threshold} m:")
    print(f"    {extrapolated:>6.1%} have an extrapolated position")
    print(f"    {substituted:>6.1%} name a player already substituted")
    print(f"    {unexplained:>6.1%} are explained by neither")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default=None)
    ap.add_argument("--all", action="store_true", help="every available match")
    ap.add_argument("--limit", type=int, default=None,
                    help="sample this many engagements per match")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = (
        config.available_matches() if args.all
        else [args.match or config.available_matches()[0]]
    )

    frames = []
    for i, mid in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] match {mid}", flush=True)
        frames.append(check_match(mid, args.limit, args.seed))
    df = pd.concat(frames, ignore_index=True)

    n = len(df)
    print(f"\n{len(ids)} match(es), {n} engagements checked\n")

    print("Coverage")
    for name, col in (
        ("frame absent from tracking", "frame_missing"),
        ("presser not in frame", "presser_missing"),
        ("carrier not in frame", "carrier_missing"),
    ):
        bad = int(df[col].fillna(False).sum())
        print(f"  {name:<28} {bad:>6}  ({bad / n:.2%})")

    tracked = df["n_tracked"].dropna()
    if not tracked.empty:
        print(f"  players tracked per frame     "
              f"median {tracked.median():.0f}, "
              f"min {tracked.min():.0f}, max {tracked.max():.0f}")
        short = int((tracked < 22).sum())
        print(f"  frames with fewer than 22     {short:>6}  ({short / n:.2%})")

    ok = df.dropna(subset=["measured", "reported"]).copy()
    ok["abs_error"] = (ok["measured"] - ok["reported"]).abs()
    err = ok["abs_error"]
    print(f"\nDistance: tracking vs events file, on {len(err)} engagements")
    print(f"  median      {err.median():.3f} m")
    print(f"  90th pct    {err.quantile(0.90):.3f} m")
    print(f"  99th pct    {err.quantile(0.99):.3f} m")
    for limit in (0.5, 1.0, 2.0):
        print(f"  over {limit:g} m   {(err > limit).mean():.2%}")

    _explain_tail(ok)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "alignment_check.csv"
    df["abs_error"] = ok["abs_error"]  # same index; NaN where a distance is missing
    df.to_csv(out, index=False)
    print(f"\nfull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
