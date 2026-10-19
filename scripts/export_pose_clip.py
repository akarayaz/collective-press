"""Cut one press out of SkillCorner's body pose data, for the 3D viewer.

    python scripts/export_pose_clip.py                       # the Figure 1 press
    python scripts/export_pose_clip.py --event 1996435 9_684 --before 2 --after 5

Then open ``viz/press3d.html`` in Chrome (double-click is enough).

Writes ``cache/press3d_clip.js``: every pose frame (25 fps) from ``--before``
seconds before the press starts until a second after the ball is won back (or
``--after`` seconds), with the 29 joints of every player the camera saw,
rotated like every scene so the team in possession attacks to the right. It
also carries what the viewer draws on the start frame: the same five metrics
as Figure 1, worked out from the tracking scene exactly as there.

The clip is SkillCorner data, so it is written to ``cache/``, which is never
committed. Only matches with body pose work: 1925299 and 1996435.

Heights: SkillCorner's joint heights are right relative to the player but not
to the pitch, so each player is lowered until his lowest foot joint touches
the ground. Players without joints in a frame are kept as a position only, and
the viewer draws them as a disc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pressing import config, features  # noqa: E402

POSE_FPS = 25
JOINTS = ["nose", "neck", "lEye", "rEye", "lEar", "rEar", "lShoulder", "rShoulder",
          "lElbow", "rElbow", "lWrist", "rWrist", "lThumb", "rThumb", "lPinky", "rPinky",
          "midHip", "lHip", "rHip", "lKnee", "rKnee", "lAnkle", "rAnkle", "lHeel", "rHeel",
          "lBigToe", "rBigToe", "lSmallToe", "rSmallToe"]
FEET = [JOINTS.index(j) for j in ("lAnkle", "rAnkle", "lHeel", "rHeel",
                                  "lBigToe", "rBigToe", "lSmallToe", "rSmallToe")]


def _r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def player_frame(p: dict, flip: float) -> dict | None:
    """One player in one pose frame: position, and joints when there are any."""
    if p.get("x") is None or p.get("y") is None:
        return None
    out = {"x": _r(flip * p["x"]), "y": _r(flip * p["y"]), "j": None}
    joints = p.get("joints")
    if joints:
        arr = np.full((len(JOINTS), 3), np.nan)
        for k, name in enumerate(JOINTS):
            j = joints.get(name)
            if j and j.get("xyz") is not None:
                arr[k] = j["xyz"]
        if np.isfinite(arr).all(axis=1).sum() >= 10:  # a body, not a few stray points
            arr[:, :2] *= flip
            feet = arr[FEET, 2]
            feet = feet[np.isfinite(feet)]
            arr[:, 2] -= feet.min() if len(feet) else np.nanmin(arr[:, 2])
            out["j"] = [None if not np.isfinite(a).all() else [_r(a[0]), _r(a[1]), _r(a[2])]
                        for a in arr]
    return out


def build_frames(records: dict[int, dict], p0: int, flip: float, side_of: dict) -> list:
    """Every pose frame as {t, ball, players}, ``t`` in seconds from the start."""
    frames = []
    for f in sorted(records):
        rec = records[f]
        players = {}
        for p in rec.get("player_data") or []:
            pid = int(p["player_id"])
            if pid not in side_of:
                continue  # referees, or players not on the pitch
            pf = player_frame(p, flip)
            if pf is not None:
                players[str(pid)] = pf
        b = rec.get("ball_data") or {}
        ball = (None if b.get("x") is None or b.get("y") is None else
                [_r(flip * b["x"]), _r(flip * b["y"]), _r(b.get("z") or 0.0)])
        frames.append({"t": round((f - p0) / POSE_FPS, 3), "ball": ball, "p": players})
    return frames


def overlay(scene) -> dict:
    """What Figure 1 draws on the start frame, as player ids and numbers.

    Lines are given by the players they join, so the viewer can attach them to
    the bodies even though pose and tracking are produced separately.
    """
    carrier, presser = scene.carrier, scene.presser
    cx, cy = float(carrier["x"]), float(carrier["y"])
    pressing, attacking = scene.pressing, scene.attacking

    receivers = attacking.drop(index=carrier.name, errors="ignore")
    rx = receivers["x"].astype(float).to_numpy()
    ry = receivers["y"].astype(float).to_numpy()
    rdist = np.hypot(rx - cx, ry - cy)
    blockers = pressing[["x", "y"]].astype(float).to_numpy()
    presser_xy = presser[["x", "y"]].astype(float).to_numpy()[None, :]
    shadowed = features._lane_blocked(cx, cy, receivers, presser_xy) & (rdist <= features.LONG_PASS_M)
    is_open = features._open(cx, cy, receivers, blockers, rdist)
    open_short = is_open & (rdist <= features.SHORT_PASS_M)

    others = pressing.drop(index=presser.name, errors="ignore")
    d = np.hypot(others["x"].astype(float) - float(presser["x"]),
                 others["y"].astype(float) - float(presser["y"]))
    mate = d.idxmin()
    outfield = features._outfield(pressing)
    xs = outfield["x"].astype(float)
    n_local = int((np.hypot(pressing["x"].astype(float) - cx,
                            pressing["y"].astype(float) - cy) <= features.LOCAL_M).sum())
    return {
        "carrier": str(int(carrier.name)), "presser": str(int(presser.name)),
        "mate": str(int(mate)),
        "shadowed": [str(int(i)) for i in receivers.index[shadowed]],
        "open": [str(int(i)) for i in receivers.index[open_short]],
        "back_line_x": _r(xs.nlargest(features.DEFENSIVE_LINE_PLAYERS).mean()),
        "radius": features.LOCAL_M,
        "values": {"n_local": n_local, "support": _r(d.min(), 1),
                   "depth": _r(xs.max() - xs.min(), 0),
                   "open_short": int(open_short.sum()), "shadowed": int(shadowed.sum())},
    }


def facing_at_start(frames: list, carrier: str) -> float | None:
    """Degrees between where the carrier faces and the goal he attacks (+x).

    From the shoulders (hips as a fallback), at the pose frame nearest the
    start where he has them; the same rule as scripts/bodypose_extract.py.
    """
    ls, rs, lh, rh = (JOINTS.index(n) for n in ("lShoulder", "rShoulder", "lHip", "rHip"))
    for fr in sorted(frames, key=lambda fr: abs(fr["t"])):
        if abs(fr["t"]) > 0.2:
            break
        j = (fr["p"].get(carrier) or {}).get("j")
        if not j:
            continue
        for a, b in ((ls, rs), (lh, rh)):
            if j[a] and j[b]:
                dx, dy = j[a][0] - j[b][0], j[a][1] - j[b][1]
                face = np.array([dy, -dx])
                n = np.linalg.norm(face)
                if n > 1e-6:
                    return round(float(np.degrees(np.arccos(np.clip(face[0] / n, -1, 1)))), 0)
    return None


def main() -> int:
    import animate_press as ap
    from bodypose_extract import stream_frames

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event", nargs=2, metavar=("MATCH", "EVENT"),
                        default=["1996435", "9_684"])
    parser.add_argument("--before", type=float, default=2.0)
    parser.add_argument("--after", type=float, default=5.0)
    parser.add_argument("--pose-dir", default=None,
                        help="folder with <match>.jsonl.zip (default: data/bodypose)")
    args = parser.parse_args()
    match_id, event_id = int(args.event[0]), args.event[1]

    pose_dir = Path(args.pose_dir) if args.pose_dir else config.REPO_ROOT / "data" / "bodypose"
    archive = pose_dir / f"{match_id}.jsonl.zip"
    if not archive.is_file():
        raise SystemExit(f"{archive} not found: body pose exists for 1925299 and 1996435 only")

    clip = ap.Clip(match_id, event_id, args.before, args.after)
    flip = -1.0 if clip.side == "right_to_left" else 1.0
    p0 = int(round(POSE_FPS / config.FPS * clip.f0))
    lo, hi = p0 - int(round(clip.before * POSE_FPS)), p0 + int(round(clip.after * POSE_FPS))
    print(f"press {match_id} {event_id}: tracking frame {clip.f0}, pose frames {lo} to {hi}")
    print(f"streaming {archive.name} (a few minutes: the file is read up to the press)")
    records = {}
    for f, rec in stream_frames(archive, set(range(lo, hi + 1))):
        records[f] = rec
        if len(records) == hi - lo + 1:
            break
    if p0 not in records:
        raise SystemExit(f"pose frame {p0} not in {archive.name}")

    side_of = {int(pid): ("pressing" if tid == clip.pressing_team else "attacking")
               for pid, tid in clip.match.players["team_id"].items()
               if tid in (clip.pressing_team, clip.attacking_team)}
    frames = build_frames(records, p0, flip, side_of)
    ov = overlay(clip.scene)
    names = clip.match.players["short_name"]

    seen = [sum(1 for p in fr["p"].values() if p["j"]) for fr in frames]
    start = next(fr for fr in frames if fr["t"] == 0)
    # Pose and tracking are made separately: at the start frame the carrier
    # should sit where the scene puts him, give or take centimetres.
    c = start["p"].get(ov["carrier"])
    gap = (np.hypot(c["x"] - float(clip.scene.carrier["x"]), c["y"] - float(clip.scene.carrier["y"]))
           if c else float("nan"))
    print(f"{len(frames)} pose frames; players with joints per frame: "
          f"median {int(np.median(seen))}, fewest {min(seen)}")
    print(f"carrier at the start: pose against tracking scene {gap:.2f} m apart "
          "(should be well under 0.5 m)")
    if not gap < 0.5:
        print("ALIGNMENT CHECK FAILED; nothing written, paste this output")
        return 1

    teams = clip.match.teams
    data = {
        "match_id": match_id, "event_id": event_id,
        "pressing_team": teams[clip.pressing_team],
        "attacking_team": teams[clip.attacking_team],
        "regain": _r(clip.regain, 1), "before": clip.before, "after": clip.after,
        "joints": JOINTS,
        "side": {str(k): v for k, v in side_of.items()},
        "names": {pid: str(names.get(int(pid), "")) for pid in
                  {ov["carrier"], ov["presser"], ov["mate"]}},
        "overlay": ov,
        "carrier_to_goal": facing_at_start(frames, ov["carrier"]),
        "frames": frames,
    }
    out = config.REPO_ROOT / "cache" / "press3d_clip.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.PRESS3D = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"carrier faces {data['carrier_to_goal']} degrees away from the goal he attacks")
    print(f"written: {out} ({out.stat().st_size / 1e6:.1f} MB); now open viz/press3d.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
