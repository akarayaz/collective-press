"""Does the comparison machinery reach the right verdict?

Run directly (``python tests/test_modelling.py``) or under pytest. No data
needed.

The models will be pointed at real engagements, where nobody knows the answer
in advance. That is exactly why they are first pointed at tables where the
answer is known: one where the outcome genuinely depends on the tracking
features, one where it does not, and one where the tracking features have been
shuffled. The machinery should say "real gain", "no gain", and "no gain".

A pipeline that reports a gain on the second or third case would report one on
the real data too, and there would be no way to tell.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import modelling  # noqa: E402

N_MATCHES = 20
ROWS_PER_MATCH = 110

# Small and fast: this is a test of the procedure, not of model quality.
FAST_PARAMS = dict(
    max_iter=60,
    learning_rate=0.1,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
)
FAST_BOOTSTRAP = 300


def make_table(signal: str, seed: int = 0) -> pd.DataFrame:
    """A table shaped like cache/engagements.csv, with a known cause.

    ``signal="tracking"``  the outcome depends on two tracking features
    ``signal="context"``   it depends only on context, so tracking adds nothing
    """
    rng = np.random.default_rng(seed)
    n = N_MATCHES * ROWS_PER_MATCH

    df = pd.DataFrame(
        {
            "meta_match_id": np.repeat(np.arange(N_MATCHES), ROWS_PER_MATCH),
            "meta_event_id": [f"e{i}" for i in range(n)],
            "q_ok": True,
        }
    )
    for col in modelling.CONTEXT_FEATURES + modelling.SKILLCORNER_GEOMETRY:
        df[col] = rng.normal(size=n)
    for col in modelling.TRACKING_GEOMETRY:
        df[col] = rng.normal(size=n)
    for col in modelling.CATEGORICAL:
        df[col] = pd.Series(
            rng.choice(["a", "b", "c"], size=n), dtype="category"
        )

    context_term = 0.9 * df["m0_xloss_player_possession_start"]
    tracking_term = (
        1.4 * df["f_presser_dist"] - 1.1 * df["f_options_shadowed_by_team"]
    )
    logit = -1.3 + context_term
    if signal == "tracking":
        logit = logit + tracking_term

    p = 1 / (1 + np.exp(-logit))
    df[modelling.DEFAULT_LABEL] = rng.random(n) < p
    return df


def run(df: pd.DataFrame):
    base = modelling.cross_validate(
        df, "skillcorner_geometry", params=FAST_PARAMS
    )
    rich = modelling.cross_validate(
        df, "tracking_geometry", params=FAST_PARAMS
    )
    delta = modelling.bootstrap_delta(base, rich, n_samples=FAST_BOOTSTRAP)
    improved = modelling.matches_improved(base, rich)
    return modelling.verdict(delta, improved), delta, improved


def test_detects_a_real_signal():
    """When tracking genuinely drives the outcome, say so."""
    got, delta, improved = run(make_table("tracking"))
    assert got == "real gain", f"expected a real gain, got {got!r} ({delta['log_loss']})"
    assert delta["log_loss"][1] > 0, "lower bound should clear zero"
    assert improved[0] >= modelling.MIN_MATCHES_IMPROVED


def test_reports_nothing_when_there_is_nothing():
    """When the outcome owes nothing to tracking, do not invent a gain."""
    got, delta, _ = run(make_table("context"))
    assert got != "real gain", f"claimed a gain on noise ({delta['log_loss']})"


def test_placebo_removes_a_real_gain():
    """Shuffling within matches should destroy a gain that was real."""
    df = make_table("tracking")
    shuffled = modelling.shuffle_within_match(df, modelling.TRACKING_GEOMETRY)
    got, delta, _ = run(shuffled)
    assert got != "real gain", (
        f"a gain survived the shuffle ({delta['log_loss']}), which means it is "
        "not coming from the scenes"
    )


def test_shuffle_keeps_the_column_intact():
    """The placebo must move values around, not change what they are."""
    df = make_table("tracking")
    shuffled = modelling.shuffle_within_match(df, modelling.TRACKING_GEOMETRY)
    col = "f_presser_dist"
    assert np.allclose(
        np.sort(df[col].to_numpy()), np.sort(shuffled[col].to_numpy())
    ), "the shuffled column should hold the same values"
    assert not np.allclose(df[col].to_numpy(), shuffled[col].to_numpy()), \
        "the shuffled column should not be in the same order"
    by_match = df.groupby("meta_match_id")[col].mean()
    by_match_shuffled = shuffled.groupby("meta_match_id")[col].mean()
    assert np.allclose(by_match, by_match_shuffled), \
        "shuffling inside a match must not move values between matches"


def test_feature_sets_are_nested():
    for smaller, larger in zip(
        ["context", "skillcorner_geometry"],
        ["skillcorner_geometry", "tracking_geometry"],
    ):
        assert set(modelling.FEATURE_SETS[smaller]) < set(
            modelling.FEATURE_SETS[larger]
        ), f"{smaller} should be contained in {larger}"


def test_no_outcome_column_can_reach_a_model():
    """Only m0_ and f_ columns are model inputs."""
    for name, cols in modelling.FEATURE_SETS.items():
        bad = [c for c in cols if not c.startswith(("m0_", "f_"))]
        assert not bad, f"{name} contains non-input columns: {bad}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
