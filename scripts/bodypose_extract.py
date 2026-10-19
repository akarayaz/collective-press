"""Which way the carrier and the presser are facing when a press starts.

    python scripts/bodypose_extract.py

Reads SkillCorner's body pose files for the two matches that have them
(1925299 Brisbane Roar v Perth Glory, 1996435 Sydney FC v Adelaide United),
downloaded from huggingface.co/datasets/SkillCorner/opendata-bodypose into
``data/bodypose/`` (see ``--pose-dir``). Each file is a zip of about 600 MB; it
is streamed, never extracted.

For every usable engagement of those matches (``q_ok`` in
``cache/engagements.csv``) it finds the pose frame at the engagement start and
records, for the carrier and the presser, the direction the upper body faces,
in the same rotated frame as the scenes (the carrier attacks towards +x); where
the ball is relative to the carrier; and a few angles of posture (how far the
presser leans and bends his knees, how far the carrier's head is turned).

Facing direction
----------------
The shoulders give a line across the body; the direction the player faces is
that line turned a quarter towards the front: with ``d`` the vector from the
right shoulder to the left, facing is ``(d_y, -d_x)``. Hips are the fallback
when a shoulder is missing. On the sample in SkillCorner's repository this
agrees with the nose's direction from the neck on 99.9% of player frames and
with the running direction of players faster than 5 m/s on 100%; both checks
are repeated here on the press frames and printed.

Body pose exists only for players the camera sees, about half of player frames
near the ball. When either player has no joints at the start frame, the
nearest frame within ``--max-offset`` seconds where both have them is used,
and the offset is recorded.

Writes ``cache/bodypose_engagements.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import press_outcomes as po  # noqa: E402
from pressing import config, modelling  # noqa: E402

MATCHES = (1925299, 1996435)
POSE_FPS, TRACKING_FPS = 25, 10
FRAME_RE = re.compile(rb'^\{\s*"frame"\s*:\s*(\d+)')
EVENT_COLUMNS = ["event_id", "event_type", "frame_start", "player_id",
                 "player_in_possession_id", "attacking_side"]


def facing(joints: dict | None) -> tuple[np.ndarray | None, float]:
    """Unit vector the upper body faces, on the pitch, and the joints' p90 error (cm)."""
    if not joints:
        return None, np.nan
    for left, right in (("lShoulder", "rShoulder"), ("lHip", "rHip")):
        a, b = joints.get(left), joints.get(right)
        if a and b:
            d = np.asarray(a["xyz"][:2], float) - np.asarray(b["xyz"][:2], float)
            f = np.array([d[1], -d[0]])
            n = np.linalg.norm(f)
            if n > 1e-6:
                return f / n, max(a.get("p90_mae_cm", np.nan), b.get("p90_mae_cm", np.nan))
    return None, np.nan


def nose_direction(joints: dict | None) -> np.ndarray | None:
    if not joints or not joints.get("nose") or not joints.get("neck"):
        return None
    v = (np.asarray(joints["nose"]["xyz"][:2], float)
         - np.asarray(joints["neck"]["xyz"][:2], float))
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _xyz(joints, name):
    j = joints.get(name) if joints else None
    return np.asarray(j["xyz"], float) if j else None


def _angle3(a, b) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / n, -1, 1)))) if n > 1e-9 else np.nan


def posture(joints: dict | None) -> dict:
    """How the body is held, from angles between joints only.

    The z of a joint is accurate relative to the player but not to the pitch,
    so only angles inside one body are used, never heights.

    ``trunk_lean``  degrees the line from mid hip to neck leans away from
                    vertical: near 0 upright, larger when bent or sprinting.
    ``knee``        the smaller of the two knee angles (hip, knee, ankle):
                    180 a straight leg, lower a bent one.
    ``head_turn``   degrees between where the nose points and where the upper
                    body faces: how far the head is turned.
    """
    out = {"trunk_lean": np.nan, "knee": np.nan, "head_turn": np.nan}
    if not joints:
        return out
    hip, neck = _xyz(joints, "midHip"), _xyz(joints, "neck")
    if hip is not None and neck is not None:
        out["trunk_lean"] = _angle3(neck - hip, np.array([0.0, 0.0, 1.0]))
    knees = []
    for side in "lr":
        h, k, a = (_xyz(joints, f"{side}Hip"), _xyz(joints, f"{side}Knee"),
                   _xyz(joints, f"{side}Ankle"))
        if h is not None and k is not None and a is not None:
            knees.append(_angle3(h - k, a - k))
    if knees:
        out["knee"] = min(knees)
    f, _ = facing(joints)
    nose = nose_direction(joints)
    if f is not None and nose is not None:
        out["head_turn"] = float(np.degrees(np.arccos(np.clip(f @ nose, -1, 1))))
    return out


def stream_frames(archive: Path, wanted: set[int]):
    """Yield (frame, record) for the wanted pose frames, parsing nothing else."""
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
        with zf.open(name) as fh:
            for line in fh:
                m = FRAME_RE.match(line)
                if m and int(m.group(1)) in wanted:
                    yield int(m.group(1)), json.loads(line)


def extract_match(match_id, archive: Path, table: pd.DataFrame, max_offset: int):
    ev = pd.read_csv(po.events_path(match_id), usecols=EVENT_COLUMNS, low_memory=False)
    ev["event_id"] = ev["event_id"].astype(str)
    keep = set(table.loc[table["meta_match_id"] == match_id, "meta_event_id"])
    eng = ev[(ev["event_type"] == "on_ball_engagement") & ev["event_id"].isin(keep)].copy()
    eng = eng.dropna(subset=["frame_start", "player_id", "player_in_possession_id"])

    centre = {r.event_id: int(round(POSE_FPS / TRACKING_FPS * r.frame_start))
              for r in eng.itertuples()}
    wanted = {c + k for c in centre.values() for k in range(-max_offset, max_offset + 1)}
    frames = {}
    for f, rec in stream_frames(archive, wanted):
        frames[f] = ({int(p["player_id"]): p for p in rec.get("player_data") or []},
                     rec.get("ball_data") or {})
    print(f"  {len(eng)} engagements, {len(frames)} of {len(wanted)} pose frames found",
          flush=True)

    rows = []
    for r in eng.itertuples():
        carrier, presser = int(r.player_in_possession_id), int(r.player_id)
        # The scenes rotate so the carrier attacks +x; attacking_side on an
        # engagement row is the pressing team's, so the carrier's is the other.
        flip = -1.0 if r.attacking_side == "left_to_right" else 1.0
        out = {"match_id": match_id, "event_id": r.event_id, "offset_frames": np.nan}
        order = sorted(range(-max_offset, max_offset + 1), key=abs)
        for k in order:
            players, ball = frames.get(centre[r.event_id] + k, ({}, {}))
            if not players or carrier not in players or presser not in players:
                continue
            cf, cerr = facing(players[carrier].get("joints"))
            pf, perr = facing(players[presser].get("joints"))
            if cf is None or pf is None:
                continue
            c_xy = np.array([players[carrier]["x"], players[carrier]["y"]], float)
            p_xy = np.array([players[presser]["x"], players[presser]["y"]], float)
            nose = nose_direction(players[carrier].get("joints"))
            cpost = posture(players[carrier].get("joints"))
            ppost = posture(players[presser].get("joints"))
            has_ball = ball.get("x") is not None and ball.get("y") is not None
            b_rel = (np.array([ball["x"], ball["y"]], float) - c_xy) if has_ball else (np.nan, np.nan)
            out.update({
                "offset_frames": k,
                "carrier_face_x": flip * cf[0], "carrier_face_y": flip * cf[1],
                "presser_face_x": flip * pf[0], "presser_face_y": flip * pf[1],
                "carrier_x": flip * c_xy[0],
                "presser_rel_x": flip * (p_xy - c_xy)[0],
                "presser_rel_y": flip * (p_xy - c_xy)[1],
                "carrier_err_cm": cerr, "presser_err_cm": perr,
                "carrier_nose_agrees": float(cf @ nose > 0) if nose is not None else np.nan,
                "ball_rel_x": flip * b_rel[0], "ball_rel_y": flip * b_rel[1],
                "ball_detected": float(bool(ball.get("is_detected"))) if has_ball else np.nan,
                "carrier_head_turn": cpost["head_turn"],
                "carrier_trunk_lean": cpost["trunk_lean"],
                "presser_trunk_lean": ppost["trunk_lean"],
                "presser_knee": ppost["knee"],
            })
            break
        rows.append(out)
    return pd.DataFrame(rows)


def add_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Angles in degrees, 0 to 180.

    ``approach_angle``  between where the carrier faces and where the presser
                        is: 0 means the presser is straight in front of him,
                        180 straight behind, out of sight.
    ``carrier_to_goal`` between where the carrier faces and the goal he
                        attacks (+x): 0 facing it, 180 facing his own.
    ``presser_to_carrier`` between where the presser faces and the carrier:
                        0 means square on to him.
    ``ball_presser_angle`` around the carrier, between the ball and the
                        presser: 0 means the ball is on the presser's side,
                        exposed; 180 means the carrier is between them,
                        shielding it. Read it only when the ball is close
                        (``ball_dist``).
    """
    cf = df[["carrier_face_x", "carrier_face_y"]].to_numpy()
    pf = df[["presser_face_x", "presser_face_y"]].to_numpy()
    rel = df[["presser_rel_x", "presser_rel_y"]].to_numpy()
    rel_u = rel / np.linalg.norm(rel, axis=1, keepdims=True)

    def angle(a, b):
        return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))

    ball = df[["ball_rel_x", "ball_rel_y"]].to_numpy()
    ball_n = np.linalg.norm(ball, axis=1, keepdims=True)
    df["ball_dist"] = ball_n[:, 0]
    df["ball_presser_angle"] = angle(np.divide(ball, ball_n, out=np.full_like(ball, np.nan),
                                               where=ball_n > 1e-6), rel_u)
    df["approach_angle"] = angle(cf, rel_u)
    df["carrier_to_goal"] = angle(cf, np.tile([1.0, 0.0], (len(df), 1)))
    df["presser_to_carrier"] = angle(pf, -rel_u)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-dir", default=None,
                    help="folder with 1925299.jsonl.zip and 1996435.jsonl.zip "
                         "(default: data/bodypose in the repository)")
    ap.add_argument("--max-offset", type=float, default=0.2,
                    help="seconds either side of the start to look for a pose")
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir) if args.pose_dir else config.REPO_ROOT / "data" / "bodypose"
    table = modelling.load_table()
    table["meta_event_id"] = table["meta_event_id"].astype(str)
    max_offset = int(round(args.max_offset * POSE_FPS))

    parts = []
    for m in MATCHES:
        archive = pose_dir / f"{m}.jsonl.zip"
        if not archive.is_file():
            raise SystemExit(f"{archive} not found; download it from "
                             "huggingface.co/datasets/SkillCorner/opendata-bodypose")
        print(f"match {m}: streaming {archive.name}", flush=True)
        parts.append(extract_match(m, archive, table, max_offset))
    df = pd.concat(parts, ignore_index=True)

    found = df["offset_frames"].notna()
    print(f"\n{found.sum()} of {len(df)} engagements have a pose for both players "
          f"({found.mean():.0%})")
    print(f"  at the start frame itself: {(df['offset_frames'] == 0).mean():.0%}; "
          f"within {args.max_offset:g} s: {found.mean():.0%}")
    df = df[found].reset_index(drop=True)
    df = add_angles(df)

    print("\nChecks")
    print(f"  carrier facing agrees with nose direction: "
          f"{df['carrier_nose_agrees'].mean():.1%} (should be close to 100%)")
    print(f"  shoulder error (p90, cm): carrier median {df['carrier_err_cm'].median():.1f}, "
          f"presser median {df['presser_err_cm'].median():.1f}")
    both = df.merge(table[["meta_match_id", "meta_event_id", "f_carrier_x"]],
                    left_on=["match_id", "event_id"], right_on=["meta_match_id", "meta_event_id"])
    r = both["carrier_x"].corr(both["f_carrier_x"])
    print(f"  carrier x, pose file against our scenes: correlation {r:.3f} "
          "(close to +1 means the rotation is right)")
    # Correlation catches a wrong rotation but not a wrong frame: x spans the
    # whole pitch, so even a second's shift keeps it near 1. The gap itself does
    # catch it: at the same instant both files should put the carrier within
    # centimetres of each other; a second apart, several metres.
    same = both[both["offset_frames"] == 0]
    gap = (same["carrier_x"] - same["f_carrier_x"]).abs().median()
    print(f"  carrier x, pose against scenes at the start frame: median gap {gap:.2f} m "
          "(should be well under 0.25 m if the frames line up)")
    if not r > 0.9:
        print("ROTATION CHECK FAILED; nothing written, paste this output")
        return 1
    if not gap < 0.25:
        print("FRAME ALIGNMENT CHECK FAILED; nothing written, paste this output")
        return 1
    dist = np.hypot(df["presser_rel_x"], df["presser_rel_y"])
    print(f"  presser distance from the pose file: median {dist.median():.1f} m "
          "(the scenes say about 6 m)")

    print("\nWhere the presser comes from, relative to where the carrier faces")
    bins = pd.cut(df["approach_angle"], [0, 60, 120, 180], include_lowest=True,
                  labels=["in front (0-60)", "side (60-120)", "behind (120-180)"])
    print(bins.value_counts().sort_index().to_string())
    close = df["ball_dist"] < 2
    print(f"\nball within 2 m of the carrier in {close.mean():.0%} of presses; there, the carrier "
          f"shields it (ball more than 120 degrees from the presser) in "
          f"{(df.loc[close, 'ball_presser_angle'] > 120).mean():.0%}")
    print("posture, medians: presser trunk lean "
          f"{df['presser_trunk_lean'].median():.0f} deg, presser knee {df['presser_knee'].median():.0f} deg, "
          f"carrier head turn {df['carrier_head_turn'].median():.0f} deg")
    print(f"\ncarrier facing his own half (more than 90 degrees from the goal he "
          f"attacks): {(df['carrier_to_goal'] > 90).mean():.0%}")

    out = config.REPO_ROOT / "cache" / "bodypose_engagements.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
