"""Export every usable engagement as a set of tokens for the deep models.

    python scripts/export_scenes.py

Writes ``cache/scenes.npz``. Each engagement becomes 23 tokens, the 22 players
and the ball, each described by the same features:

    x, y                    position, pitch-normalised (carrier attacks +x)
    dx, dy                  position relative to the carrier
    vx, vy                  velocity, over the half second before the start
    pressing, attacking,    which side the token belongs to (exactly one is 1)
    ball
    presser, carrier        the two players the engagement is about
    detected                seen by the camera rather than extrapolated

Nothing here is a hand-built pressing metric: no counts, no distances to a
teammate, no lanes. The deep models get the raw scene and have to find the
structure themselves, which is the point of comparing them with the tabular
model that is given those metrics.

Only rows the tabular models use (``q_ok`` in ``cache/engagements.csv``) are
exported, in that table's order, so the two can be compared row for row.
Reads each tracking file once, like ``build_features.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, events, modelling, scene  # noqa: E402

N_TOKENS = 23  # 22 players and the ball
FEATURES = ["x", "y", "dx", "dy", "vx", "vy", "pressing", "attacking", "ball",
            "presser", "carrier", "detected"]

# Scales that bring every feature to roughly unit size.
HALF_LENGTH = config.PITCH_LENGTH / 2
HALF_WIDTH = config.PITCH_WIDTH / 2
REL_SCALE = 20.0   # metres
VEL_SCALE = 8.0    # m/s, about a fast run


def _num(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def scene_tokens(sc) -> tuple[np.ndarray, np.ndarray]:
    """One scene as a (23, n_features) array and a (23,) mask of real tokens.

    Order is fixed so the export is reproducible (carrier, presser, the rest
    of the pressing side by distance to the carrier, then the attacking side),
    but the models treat the tokens as a set and do not rely on it.
    """
    X = np.zeros((N_TOKENS, len(FEATURES)), dtype=np.float32)
    mask = np.zeros(N_TOKENS, dtype=bool)
    carrier = sc.carrier
    if carrier is None:
        return X, mask
    cx, cy = float(carrier["x"]), float(carrier["y"])

    players = sc.players[sc.players["side"].isin(["pressing", "attacking"])].copy()
    players["_d"] = np.hypot(players["x"].astype(float) - cx, players["y"].astype(float) - cy)
    players["_rank"] = np.select(
        [players["is_carrier"], players["is_presser"], players["side"] == "pressing"],
        [0, 1, 2], default=3,
    )
    players = players.sort_values(["_rank", "_d"]).head(N_TOKENS - 1)

    for i, (_, p) in enumerate(players.iterrows()):
        x, y = float(p["x"]), float(p["y"])
        X[i] = [
            x / HALF_LENGTH, y / HALF_WIDTH,
            (x - cx) / REL_SCALE, (y - cy) / REL_SCALE,
            _num(p.get("vx")) / VEL_SCALE, _num(p.get("vy")) / VEL_SCALE,
            float(p["side"] == "pressing"), float(p["side"] == "attacking"), 0.0,
            float(bool(p["is_presser"])), float(bool(p["is_carrier"])),
            float(bool(p.get("is_detected")) if pd.notna(p.get("is_detected")) else 0.0),
        ]
        mask[i] = True

    if sc.ball is not None:
        bx, by = sc.ball
        bvx, bvy = sc.ball_velocity if sc.ball_velocity is not None else (0.0, 0.0)
        X[N_TOKENS - 1] = [
            bx / HALF_LENGTH, by / HALF_WIDTH,
            (bx - cx) / REL_SCALE, (by - cy) / REL_SCALE,
            _num(bvx) / VEL_SCALE, _num(bvy) / VEL_SCALE,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ]
        mask[N_TOKENS - 1] = True
    return X, mask


def main() -> int:
    table = modelling.load_table()
    table["meta_event_id"] = table["meta_event_id"].astype(str)
    wanted = {(m, e) for m, e in zip(table["meta_match_id"], table["meta_event_id"])}

    found: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    match_ids = table["meta_match_id"].unique()
    for i, match_id in enumerate(match_ids, 1):
        t0 = time.time()
        eng = events.load_engagements(match_id).reset_index(drop=True)
        eng = eng[[(match_id, str(e)) in wanted for e in eng["event_id"]]]
        n = 0
        for event, sc, _ in scene.build_scenes(match_id, eng):
            if sc is None:
                continue
            found[(match_id, str(event["event_id"]))] = scene_tokens(sc)
            n += 1
        print(f"[{i}/{len(match_ids)}] {match_id}: {n} scenes, {time.time() - t0:.0f}s",
              flush=True)

    keys = list(zip(table["meta_match_id"], table["meta_event_id"]))
    missing = [k for k in keys if k not in found]
    if missing:
        print(f"\n{len(missing)} usable rows had no scene and are left out; "
              "the training script aligns on match and event id")
    keys = [k for k in keys if k in found]

    X = np.stack([found[k][0] for k in keys])
    mask = np.stack([found[k][1] for k in keys])
    out = config.REPO_ROOT / "cache" / "scenes.npz"
    np.savez_compressed(
        out, X=X, mask=mask,
        match_id=np.array([k[0] for k in keys]),
        event_id=np.array([k[1] for k in keys]),
        features=np.array(FEATURES),
    )
    print(f"\n{len(keys)} scenes, {X.shape[1]} tokens x {X.shape[2]} features, "
          f"{mask.sum(1).mean():.1f} real tokens per scene on average")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
