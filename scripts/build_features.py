"""Build the modelling table: one row per engagement, cached to disk.

    python scripts/build_features.py            # every match
    python scripts/build_features.py --match 1925299

Reads each match's tracking file exactly once and writes
``cache/engagements.csv`` with four kinds of column:

``meta_*``    identity and timing; never a model input
``m0_*``      SkillCorner's own summaries, the baseline the project measures against
``f_*``       positioning measured from the tracking frame (:mod:`pressing.features`)
``q_*``       data quality, including the tracking-versus-events check
``label_*``   the rebuilt recovery label at several windows

Only ``m0_*`` and ``f_*`` are ever fed to a model. Keeping the prefixes means
a stray outcome column cannot quietly become an input.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, events, features, labels, scene  # noqa: E402

LABEL_WINDOWS = (3.0, 5.0, 8.0)

#: Beyond this disagreement between tracking and the events file, the row is
#: not trusted. Justified in the README: the median disagreement is 5 mm and
#: 96% of rows are under half a metre, so this cuts a tail, not a shoulder.
MAX_TRACKING_DISAGREEMENT_M = 2.0

META_FROM_EVENTS = [
    "match_id", "event_id", "period", "frame_start", "time_start",
    "player_id", "player_in_possession_id", "team_id", "event_subtype",
    "end_type",
]

M0_FROM_ENGAGEMENT = [
    "minute_start",
    "xloss_player_possession_start",
    "xshot_player_possession_start",
    "interplayer_distance_start",
    "angle_of_engagement",
    "delta_to_last_defensive_line_start",
    "last_defensive_line_height_start",
]

M0_CATEGORICAL = ["third_start", "channel_start", "game_state"]


def build_match(match_id) -> pd.DataFrame:
    eng = events.load_engagements(match_id).reset_index(drop=True)
    eng = labels.add_recovery_timing(eng)
    context = events.load_possession_context(match_id)

    rows = []
    for event, sc, error in scene.build_scenes(match_id, eng):
        row = {f"meta_{c}": event.get(c) for c in META_FROM_EVENTS}
        row["meta_seconds_to_possession_start"] = (
            event["associated_player_possession_frame_start"]
            - event["frame_start"]
        ) / config.FPS

        for c in M0_FROM_ENGAGEMENT + M0_CATEGORICAL:
            row[f"m0_{c}"] = event.get(c)
        ctx_id = event.get("associated_player_possession_event_id")
        if ctx_id in context.index:
            for c, v in context.loc[ctx_id].items():
                row[f"m0_{c}"] = v

        for w in LABEL_WINDOWS:
            row[f"label_regain_{w:g}s"] = bool(
                event["seconds_to_regain"] <= w
            )
        row["label_seconds_to_regain"] = event["seconds_to_regain"]
        row["label_regain_strict_5s"] = bool(
            event["seconds_to_regain"] <= 5.0
            and event["carrier_possession_end_type"] == "possession_loss"
        )
        row["label_regain_after_interruption"] = bool(
            event["regain_after_interruption"]
        )

        row["q_scene_error"] = error
        if sc is None:
            row.update({f"f_{c}": np.nan for c in features.FEATURE_COLUMNS})
            row.update(
                q_n_tracked=np.nan, q_presser_detected=pd.NA,
                q_carrier_detected=pd.NA, q_tracking_disagreement=np.nan,
            )
        else:
            for c, v in features.scene_features(sc).items():
                row[f"f_{c}"] = v
            carrier = sc.carrier
            presser = sc.presser
            row["q_n_tracked"] = len(sc.players)
            row["q_presser_detected"] = presser.get("is_detected")
            row["q_carrier_detected"] = (
                pd.NA if carrier is None else carrier.get("is_detected")
            )
            # The same independent check scripts/check_alignment.py runs, kept
            # per row so the filter travels with the data.
            reported = event.get("interplayer_distance_start")
            measured = row.get("f_presser_dist")
            row["q_tracking_disagreement"] = (
                abs(measured - reported)
                if pd.notna(measured) and pd.notna(reported) else np.nan
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df["q_both_detected"] = (
        df["q_presser_detected"].fillna(False).astype(bool)
        & df["q_carrier_detected"].fillna(False).astype(bool)
    )
    df["q_ok"] = (
        df["q_scene_error"].isna()
        & (df["q_n_tracked"] == 22)
        & (df["q_tracking_disagreement"] <= MAX_TRACKING_DISAGREEMENT_M)
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default=None, help="one match instead of all")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()

    ids = [args.match] if args.match else config.available_matches()

    parts = []
    for i, match_id in enumerate(ids, 1):
        t0 = time.time()
        part = build_match(match_id)
        parts.append(part)
        print(f"[{i}/{len(ids)}] {match_id}: {len(part)} engagements, "
              f"{int(part['q_ok'].sum())} usable, {time.time() - t0:.0f}s",
              flush=True)
    df = pd.concat(parts, ignore_index=True)

    out = Path(args.out) if args.out else config.REPO_ROOT / "cache" / "engagements.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{len(df)} engagements, {df.shape[1]} columns")
    print(f"usable: {int(df['q_ok'].sum())} ({df['q_ok'].mean():.1%})")

    print("\nDropped because")
    reasons = {
        "scene could not be built": df["q_scene_error"].notna(),
        "fewer than 22 players tracked": df["q_n_tracked"] != 22,
        f"disagrees with events by > {MAX_TRACKING_DISAGREEMENT_M:g} m":
            df["q_tracking_disagreement"] > MAX_TRACKING_DISAGREEMENT_M,
    }
    for name, mask in reasons.items():
        print(f"  {name:<46} {int(mask.sum()):>6}")

    usable = df[df["q_ok"]]
    print("\nLabel rate on usable rows")
    for w in LABEL_WINDOWS:
        col = f"label_regain_{w:g}s"
        print(f"  {w:g}s window   {usable[col].mean():.1%}")

    feat = [f"f_{c}" for c in features.FEATURE_COLUMNS]
    missing = usable[feat].isna().mean().sort_values(ascending=False)
    print("\nFeatures with missing values on usable rows")
    worst = missing[missing > 0]
    print("  none" if worst.empty
          else worst.head(8).to_string(float_format=lambda v: f"{v:.1%}"))

    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
