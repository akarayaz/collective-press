"""Does the way bodies face change what a press does? A pilot on two matches.

    python scripts/bodypose_press.py

Needs ``cache/bodypose_engagements.csv`` (``scripts/bodypose_extract.py``) and
``cache/oof_<label>.csv`` (``scripts/run_models.py``).

Body pose exists for two matches only, so this is a pilot, kept apart from the
models trained on twenty. Two questions were fixed before looking:

1. Blind side. Does a press that comes from the side or from behind the
   carrier, relative to where his upper body faces, win the ball more often
   than one he can see coming?
2. Facing his own half. When the carrier faces away from the goal he attacks,
   how does his possession end: does he lose the ball more often, play
   backwards more often?

Everything else (shielding the ball, the presser's posture, the carrier's head)
is reported as exploratory.

What is held fixed
------------------
The leave-one-match-out prediction of the ``tracking_geometry`` model for the
same press: SkillCorner's context and geometry and our 32 pressing metrics,
from a model that never saw the match. It enters as a logit, with its own
coefficient, so its calibration does not matter. The carrier's xloss enters as
well, because it speaks to the endings other than a regain. A body pose effect
here is what the pose adds beyond everything the twenty-match model knows.

Uncertainty
-----------
Two matches cannot be resampled as matches. Presses are grouped into blocks of
five minutes of one match, and the blocks are resampled: presses close in time
share a game state, so this is wider and more honest than resampling presses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import press_outcomes as po  # noqa: E402
from pressing import config, modelling  # noqa: E402

BLOCK_MINUTES = 5

#: name -> (how it is built, what one unit means); "primary" is fixed in advance
TERMS = {
    "presser from the side": ("approach", "60 to 120 degrees, against in front"),
    "presser from behind": ("approach", "over 120 degrees, against in front"),
    "carrier facing his own half": ("binary", "facing more than 90 degrees from goal"),
    "ball shielded": ("binary", "ball within 2 m and over 120 degrees from the presser"),
    "presser trunk lean": ("per", 10.0),
    "presser knee angle": ("per", 10.0),
    "carrier head turn": ("per", 10.0),
}
PRIMARY = {"presser from the side", "presser from behind", "carrier facing his own half"}


def load(label: str) -> pd.DataFrame:
    path = config.REPO_ROOT / "cache" / "bodypose_engagements.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run scripts/bodypose_extract.py first")
    pose = pd.read_csv(path)
    pose["event_id"] = pose["event_id"].astype(str)

    table = modelling.load_table()
    table["meta_event_id"] = table["meta_event_id"].astype(str)
    df = pose.merge(table, left_on=["match_id", "event_id"],
                    right_on=["meta_match_id", "meta_event_id"], how="inner")

    oof = pd.read_csv(config.REPO_ROOT / "cache" / f"oof_{label}.csv")
    oof["event_id"] = oof["event_id"].astype(str)
    df = df.merge(oof[["match_id", "event_id", "p_tracking_geometry"]],
                  on=["match_id", "event_id"], how="left")
    if df["p_tracking_geometry"].isna().any():
        raise SystemExit("some presses have no out-of-fold prediction; rerun scripts/run_models.py")

    endings = po.carrier_endings(df)
    df = df.merge(endings[["meta_match_id", "meta_event_id", *po.OUTCOMES]],
                  on=["meta_match_id", "meta_event_id"], how="left")

    df = df.copy()  # many columns were added by the merges; defragment once
    p = df["p_tracking_geometry"].clip(1e-4, 1 - 1e-4)
    df["ctl_logit"] = np.log(p / (1 - p))
    x = df["m0_xloss_player_possession_start"].astype(float)
    df["ctl_xloss"] = (x.fillna(x.median()) - x.mean()) / x.std()

    a = df["approach_angle"]
    df["presser from the side"] = ((a > 60) & (a <= 120)).astype(float)
    df["presser from behind"] = (a > 120).astype(float)
    df["carrier facing his own half"] = (df["carrier_to_goal"] > 90).astype(float)
    df["ball shielded"] = ((df["ball_dist"] < 2) & (df["ball_presser_angle"] > 120)).astype(float)
    df["presser trunk lean"] = df["presser_trunk_lean"]
    df["presser knee angle"] = df["presser_knee"]
    df["carrier head turn"] = df["carrier_head_turn"]

    df["block"] = (df["match_id"].astype(str) + "_"
                   + (df["m0_minute_start"].astype(float) // BLOCK_MINUTES).astype(int).astype(str))
    return df


def _fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)


def effect(df: pd.DataFrame, y: str, term: str, joint: bool = False) -> float:
    """Average change in the outcome's probability, in points, for one unit of the term.

    Approach angle enters as two indicators together (side, behind; front is
    the reference) so each is read against a press from in front.
    """
    kind, unit = TERMS[term]
    terms = ["presser from the side", "presser from behind"] if kind == "approach" else [term]
    if joint:  # all three primary terms together, so each is read net of the others
        terms = ["presser from the side", "presser from behind", "carrier facing his own half"]
    d = df.dropna(subset=terms + [y])
    cols = ["ctl_logit", "ctl_xloss"] + terms
    X = d[cols].to_numpy(dtype=float)
    yv = d[y].to_numpy(dtype=int)
    if yv.min() == yv.max():
        return np.nan
    m = _fit(X, yv)
    j = cols.index(term)
    lo, hi = X.copy(), X.copy()
    if kind == "per":
        hi[:, j] = hi[:, j] + unit
    else:
        lo[:, j], hi[:, j] = 0.0, 1.0
        if kind == "approach":  # the other approach indicator off: compare with in front
            name = ("presser from behind" if term == "presser from the side"
                    else "presser from the side")
            other = cols.index(name)
            lo[:, other] = hi[:, other] = 0.0
    return float((m.predict_proba(hi)[:, 1] - m.predict_proba(lo)[:, 1]).mean() * 100)


def bootstrap(df, y, term, n, seed, joint: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = {b: g for b, g in df.groupby("block")}
    ids = np.array(list(blocks))
    out = []
    for _ in range(n):
        sample = pd.concat([blocks[b] for b in rng.choice(ids, len(ids))], ignore_index=True)
        out.append(effect(sample, y, term, joint))
    return np.asarray(out, dtype=float)


def cross_match_check(df: pd.DataFrame, y: str) -> list[tuple]:
    """Train on one match, test on the other: does adding the pose help predict?"""
    from sklearn.metrics import log_loss, roc_auc_score
    terms = ["presser from the side", "presser from behind", "carrier facing his own half"]
    rows = []
    for test in df["match_id"].unique():
        tr, te = df[df["match_id"] != test], df[df["match_id"] == test]
        base = ["ctl_logit", "ctl_xloss"]
        p0 = _fit(tr[base].to_numpy(), tr[y]).predict_proba(te[base].to_numpy())[:, 1]
        p1 = _fit(tr[base + terms].to_numpy(), tr[y]).predict_proba(te[base + terms].to_numpy())[:, 1]
        rows.append((test, len(te), log_loss(te[y], p0) - log_loss(te[y], p1),
                     roc_auc_score(te[y], p1) - roc_auc_score(te[y], p0)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load(args.label)
    outcomes = [args.label] + list(po.OUTCOMES)
    print(f"{len(df)} presses with body pose in {df['match_id'].nunique()} matches, "
          f"{df['block'].nunique()} blocks of {BLOCK_MINUTES} minutes\n")

    print("Raw rates by where the presser comes from (no controls)")
    side = pd.cut(df["approach_angle"], [0, 60, 120, 180], include_lowest=True,
                  labels=["in front", "side", "behind"])
    raw = df.groupby(side, observed=True)[outcomes].mean().mul(100)
    raw.insert(0, "presses", df.groupby(side, observed=True).size())
    print(raw.to_string(float_format=lambda v: f"{v:.1f}"))
    facing = pd.Series(np.where(df["carrier facing his own half"] == 1, "facing own half",
                                "facing goal"), index=df.index, name="carrier")
    raw2 = df.groupby(facing)[outcomes].mean().mul(100)
    raw2.insert(0, "presses", df.groupby(facing).size())
    print("\nRaw rates by where the carrier faces")
    print(raw2.to_string(float_format=lambda v: f"{v:.1f}"))

    print("\nEffects with the twenty-match model's prediction and xloss held fixed "
          "(points of probability, 95% block bootstrap)")
    rows, joint_rows = [], []
    for term, (kind, unit) in TERMS.items():
        tag = "primary" if term in PRIMARY else "exploratory"
        reads = f"per {unit:g} degrees" if kind == "per" else unit
        print(f"\n  {term}  [{tag}; {reads}]")
        for y in outcomes:
            point = effect(df, y, term)
            draws = bootstrap(df, y, term, args.bootstrap, args.seed)
            lo, hi = np.nanpercentile(draws, [2.5, 97.5])
            flag = "  *" if lo > 0 or hi < 0 else ""
            print(f"    {y:<18} {point:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]{flag}")
            rows.append({"term": term, "kind": tag, "unit": reads, "outcome": y,
                         "base_rate": df[y].mean() * 100, "points": point,
                         "points_lo": lo, "points_hi": hi})
    res = pd.DataFrame(rows)
    stars = (res["points_lo"] > 0) | (res["points_hi"] < 0)
    for tag in ("primary", "exploratory"):
        sel = res["kind"] == tag
        print(f"\n* the interval excludes zero. {tag}: {int(stars[sel].sum())} stars in "
              f"{int(sel.sum())} tests; with no real effect anywhere about "
              f"{0.05 * sel.sum():.1f} would appear by chance.")

    print("\nWhere the presser comes from, by where the carrier faces (share of each column)")
    ct = pd.crosstab(side, facing, normalize="columns").mul(100)
    print(ct.to_string(float_format=lambda v: f"{v:.0f}%"))
    print("\nThe three primary terms in one model, each net of the other two")
    for term in ("presser from the side", "presser from behind", "carrier facing his own half"):
        print(f"\n  {term}")
        for y in outcomes:
            point = effect(df, y, term, joint=True)
            lo, hi = np.nanpercentile(bootstrap(df, y, term, args.bootstrap, args.seed, True),
                                      [2.5, 97.5])
            flag = "  *" if lo > 0 or hi < 0 else ""
            print(f"    {y:<18} {point:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]{flag}")
            joint_rows.append({"term": term, "kind": "primary, joint", "outcome": y,
                               "points": point, "points_lo": lo, "points_hi": hi})

    print("\nDoes the body pose help predict a regain in the other match?")
    for test, n, dll, dauc in cross_match_check(df, args.label):
        print(f"  test on {test} ({n} presses): log loss gain {dll:+.4f}, AUC gain {dauc:+.4f}")

    resdir = config.REPO_ROOT / "results"
    resdir.mkdir(parents=True, exist_ok=True)
    pd.concat([res, pd.DataFrame(joint_rows)], ignore_index=True).to_csv(
        resdir / "bodypose_press.csv", index=False)
    print("\nwritten: results/bodypose_press.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
