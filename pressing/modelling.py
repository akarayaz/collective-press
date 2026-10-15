"""The nested models, the cross-validation, and the decision rule.

Three feature sets, each adding to the one before, so that any gain can be
attributed to what was added rather than to the model that happened to be run:

``context``
    What SkillCorner's own metrics say about the situation: the carrier's
    ``xloss`` and ``xshot``, the pressure and space codes on their possession,
    EPV, the zone, the score, the clock. No geometry.
``skillcorner_geometry``
    ``context`` plus the geometry SkillCorner already ships on the engagement
    row: the distance and angle to the carrier, and the defensive line. A
    subscriber has these without touching tracking, so crediting them to this
    project would be claiming the obvious.
``tracking_geometry``
    plus everything measured from the tracking frame in
    :mod:`pressing.features`: support, local counts, blocked passing lanes,
    closing speed, the shape of the defensive block.

The question the project asks is the step from ``skillcorner_geometry`` to
``tracking_geometry``. The first step is reported too, because knowing how much
the basic geometry is worth is what makes the second number interpretable.

Validation is leave-one-match-out: every match takes a turn as the test set and
the other nineteen train. Engagements inside a match are not independent, so a
random split would put near-duplicate moments on both sides and flatter every
model.

Hyperparameters are fixed in advance and never tuned against a test fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config, features

CONTEXT_FEATURES = [
    "m0_minute_start",
    "m0_xloss_player_possession_start",
    "m0_xshot_player_possession_start",
    "m0_overall_pressure_start_id",
    "m0_passing_option_ease_start_id",
    "m0_space_constraint_start_id",
    "m0_time_to_impact_start_id",
    "m0_possession_epv_for_start",
    "m0_possession_epv_against_start",
    "m0_third_start",
    "m0_channel_start",
    "m0_game_state",
]

SKILLCORNER_GEOMETRY = [
    "m0_interplayer_distance_start",
    "m0_angle_of_engagement",
    "m0_delta_to_last_defensive_line_start",
    "m0_last_defensive_line_height_start",
]

TRACKING_GEOMETRY = [f"f_{c}" for c in features.FEATURE_COLUMNS]

FEATURE_SETS = {
    "context": CONTEXT_FEATURES,
    "skillcorner_geometry": CONTEXT_FEATURES + SKILLCORNER_GEOMETRY,
    "tracking_geometry": CONTEXT_FEATURES + SKILLCORNER_GEOMETRY + TRACKING_GEOMETRY,
}

CATEGORICAL = ["m0_third_start", "m0_channel_start", "m0_game_state"]

DEFAULT_LABEL = "label_regain_5s"

#: Fixed in advance. No tuning against a test fold happened, and none should.
#: scikit-learn's histogram gradient boosting is used rather than LightGBM or
#: XGBoost: it is the same family of model, it takes missing values and
#: categorical columns natively, and it keeps the repository installable from
#: scikit-learn alone.
MODEL_PARAMS = dict(
    max_iter=400,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=60,
    l2_regularization=1.0,
    early_stopping=False,
)

# --- the decision rule, written down before the numbers were seen -----------
#: The gain is called real only when its interval clears zero and it shows up
#: in most matches, not just a lucky few.
MIN_MATCHES_IMPROVED = 14
N_MATCHES = 20
BOOTSTRAP_SAMPLES = 2000


@dataclass
class CVResult:
    name: str
    oof: pd.Series
    y: pd.Series
    match_id: pd.Series
    feature_names: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return compute_metrics(self.y, self.oof)

    def per_match_log_loss(self) -> pd.Series:
        return (
            pd.DataFrame({"y": self.y, "p": self.oof, "m": self.match_id})
            .groupby("m")
            .apply(lambda g: _safe_log_loss(g["y"], g["p"]), include_groups=False)
        )


def load_table(path=None, only_ok: bool = True) -> pd.DataFrame:
    path = path or config.REPO_ROOT / "cache" / "engagements.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run scripts/build_features.py first"
        )
    df = pd.read_csv(path, low_memory=False)
    if only_ok:
        df = df[df["q_ok"]].reset_index(drop=True)
    for col in CATEGORICAL:
        if col in df:
            df[col] = df[col].astype("category")
    return df


def feature_matrix(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    names = [c for c in FEATURE_SETS[feature_set] if c in df.columns]
    missing = set(FEATURE_SETS[feature_set]) - set(names)
    if missing:
        raise KeyError(f"feature set {feature_set!r} missing columns: {sorted(missing)}")
    return df[names]


def cross_validate(
    df: pd.DataFrame,
    feature_set: str,
    label: str = DEFAULT_LABEL,
    params: dict | None = None,
    seed: int = 0,
) -> CVResult:
    """Leave-one-match-out predictions for one feature set."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = feature_matrix(df, feature_set)
    y = df[label].astype(int)
    groups = df["meta_match_id"]
    oof = pd.Series(np.nan, index=df.index, dtype=float)

    for match_id in groups.unique():
        test = groups == match_id
        model = HistGradientBoostingClassifier(
            random_state=seed,
            categorical_features="from_dtype",
            **(params or MODEL_PARAMS),
        )
        model.fit(X[~test], y[~test])
        oof[test] = model.predict_proba(X[test])[:, 1]

    return CVResult(feature_set, oof, y, groups, list(X.columns))


def compute_metrics(y, p) -> dict[str, float]:
    return {
        "log_loss": _safe_log_loss(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(np.mean(y)),
        "n": int(len(y)),
    }


def bootstrap_delta(
    base: CVResult,
    richer: CVResult,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Match-level bootstrap of the difference between two models.

    Matches, not rows, are resampled: rows within a match share a game state
    and are not independent, so resampling rows would understate the spread.

    Returns ``{metric: (point_estimate, low, high)}`` with the gain signed so
    that a positive number always means the richer model is better.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "match": base.match_id.to_numpy(),
            "y": base.y.to_numpy(),
            "p_base": base.oof.to_numpy(),
            "p_rich": richer.oof.to_numpy(),
        }
    )
    by_match = {m: g for m, g in frame.groupby("match")}
    match_ids = np.array(list(by_match))

    def gains(sample: pd.DataFrame) -> dict[str, float]:
        y = sample["y"]
        return {
            "log_loss": _safe_log_loss(y, sample["p_base"])
            - _safe_log_loss(y, sample["p_rich"]),
            "roc_auc": (
                roc_auc_score(y, sample["p_rich"]) - roc_auc_score(y, sample["p_base"])
                if y.nunique() > 1 else np.nan
            ),
            "brier": brier_score_loss(y, sample["p_base"])
            - brier_score_loss(y, sample["p_rich"]),
        }

    point = gains(frame)
    draws = {k: [] for k in point}
    for _ in range(n_samples):
        picked = rng.choice(match_ids, size=len(match_ids), replace=True)
        sample = pd.concat([by_match[m] for m in picked], ignore_index=True)
        for k, v in gains(sample).items():
            draws[k].append(v)

    out = {}
    for k, values in draws.items():
        arr = np.asarray(values, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[k] = (point[k], float(lo), float(hi))
    return out


def matches_improved(base: CVResult, richer: CVResult) -> tuple[int, int]:
    """How many matches the richer model predicts better, by log loss."""
    a = base.per_match_log_loss()
    b = richer.per_match_log_loss()
    aligned = pd.concat([a.rename("base"), b.rename("rich")], axis=1).dropna()
    return int((aligned["rich"] < aligned["base"]).sum()), int(len(aligned))


def verdict(delta: dict, improved: tuple[int, int]) -> str:
    """Apply the pre-registered rule to one comparison."""
    _, low, _ = delta["log_loss"]
    won, total = improved
    if low > 0 and won >= MIN_MATCHES_IMPROVED:
        return "real gain"
    if low > 0:
        return f"interval clears zero but only {won}/{total} matches improve"
    point = delta["log_loss"][0]
    if point > 0:
        return "points the right way, but the interval includes zero"
    return "no gain"


def shuffle_within_match(
    df: pd.DataFrame, columns: list[str], seed: int = 0
) -> pd.DataFrame:
    """Permute these columns among the rows of each match.

    The placebo. Shuffling inside a match keeps every column's distribution and
    each match's composition intact while destroying the link between a
    particular scene and its outcome. A model that still gains from the
    shuffled columns is reading something it should not be able to read.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _, idx in df.groupby("meta_match_id").groups.items():
        order = rng.permutation(len(idx))
        for col in columns:
            out.loc[idx, col] = df.loc[idx, col].to_numpy()[order]
    return out


def _safe_log_loss(y, p) -> float:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(log_loss(y, p, labels=[0, 1]))
