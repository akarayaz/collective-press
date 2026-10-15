"""Run the nested models and apply the decision rule.

    python scripts/run_models.py
    python scripts/run_models.py --label label_regain_3s     # window sensitivity
    python scripts/run_models.py --skip-placebo              # faster

Prints the decision rule first, so that it is on the record before the numbers
are, then the model comparison, the per-match win counts, and the placebos.

The headline is not any model's score. It is the step from
``skillcorner_geometry`` to ``tracking_geometry``: what positioning measured
off the tracking adds to what SkillCorner already ships.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from pressing import config, modelling  # noqa: E402

ORDER = ["context", "skillcorner_geometry", "tracking_geometry"]

#: (title, baseline model, model under test, columns the placebo shuffles)
PLACEBOS = [
    ("tracking features (v1)", "skillcorner_geometry", "tracking_geometry",
     modelling.TRACKING_GEOMETRY),
]



def _show(delta: dict, improved=None) -> None:
    for metric, (point, lo, hi) in delta.items():
        print(f"    {metric:<9} {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if improved is not None:
        print(f"    matches improved: {improved[0]}/{improved[1]}")


def _placebo_section(df, results, args, title, base_name, real_name,
                     columns) -> None:
    """Shuffle some features and measure what that alone buys.

    Two comparisons come out of this, and the second is the one that counts.

    The first asks what a placebo gains over the baseline. If the answer is
    "about as much as the real features", then beating the baseline is not
    evidence about positioning: a model with 45 columns differs from one with
    16 in capacity and in how much match-level context it can infer, whether
    or not those columns are attached to the right rows.

    The second compares the real model with the placebo directly. Both have
    the same columns, the same capacity and the same match-level information;
    the only difference is whether each scene's features belong to that scene.
    Whatever separates them is the scenes.
    """
    base = results[base_name]
    real = results[real_name]

    print(f"\nPlacebo: {title} permuted among the rows of each match "
          f"({base_name} -> {real_name})")
    print("  distributions and match composition are kept; only the link")
    print("  between a scene and its outcome is broken")

    for seed in range(args.placebo_seeds):
        shuffled = modelling.shuffle_within_match(df, columns, seed=seed)
        fake = modelling.cross_validate(
            shuffled, real_name, label=args.label, seed=args.seed
        )

        print(f"\n  shuffle {seed}: log loss {fake.metrics()['log_loss']:.4f} "
              f"(real {real.metrics()['log_loss']:.4f}, "
              f"baseline {base.metrics()['log_loss']:.4f})")

        print("    what the placebo alone gains over the baseline")
        _show(
            modelling.bootstrap_delta(
                base, fake, n_samples=args.bootstrap, seed=args.seed
            ),
            modelling.matches_improved(base, fake),
        )

        print("    real minus placebo, same columns and capacity on both sides")
        delta = modelling.bootstrap_delta(
            fake, real, n_samples=args.bootstrap, seed=args.seed
        )
        improved = modelling.matches_improved(fake, real)
        _show(delta, improved)
        print(f"    verdict: {modelling.verdict(delta, improved)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--table", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=modelling.BOOTSTRAP_SAMPLES)
    ap.add_argument("--skip-placebo", action="store_true")
    ap.add_argument("--placebo-seeds", type=int, default=3,
                    help="how many different shuffles to try")
    ap.add_argument("--all-rows", action="store_true",
                    help="include rows flagged q_ok = False")
    args = ap.parse_args()

    order = ORDER
    placebos = PLACEBOS

    print("Decision rule, fixed before the numbers were seen")
    print(f"  a gain counts as real when the 95% interval of the log-loss")
    print(f"  improvement is entirely above zero AND at least "
          f"{modelling.MIN_MATCHES_IMPROVED} of {modelling.N_MATCHES} matches "
          "improve")
    print("  validation is leave-one-match-out; hyperparameters are fixed")
    print()

    table = Path(args.table) if args.table else None
    df = modelling.load_table(table, only_ok=not args.all_rows)
    print(f"{len(df)} engagements, {df['meta_match_id'].nunique()} matches, "
          f"label {args.label} ({df[args.label].mean():.1%} positive)\n")

    results = {}
    for name in order:
        t0 = time.time()
        results[name] = modelling.cross_validate(
            df, name, label=args.label, seed=args.seed
        )
        n_feat = len(results[name].feature_names)
        print(f"  {name:<22} {n_feat:>3} features, {time.time() - t0:.0f}s",
              flush=True)

    print("\nOut-of-fold performance")
    rows = []
    for name in order:
        m = results[name].metrics()
        rows.append({"model": name, **m})
    table_out = pd.DataFrame(rows).set_index("model")
    print(table_out.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nWhat each step adds")
    for base_name, rich_name in zip(order, order[1:]):
        base, rich = results[base_name], results[rich_name]
        delta = modelling.bootstrap_delta(
            base, rich, n_samples=args.bootstrap, seed=args.seed
        )
        improved = modelling.matches_improved(base, rich)
        print(f"\n  {base_name}  ->  {rich_name}")
        _show(delta, improved)
        print(f"    verdict: {modelling.verdict(delta, improved)}")

    if not args.skip_placebo:
        for title, base_name, real_name, columns in placebos:
            _placebo_section(df, results, args, title, base_name, real_name,
                             columns)

    out = config.REPO_ROOT / "cache" / f"oof_{args.label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "match_id": results[order[0]].match_id,
            "event_id": df["meta_event_id"],
            "y": results[order[0]].y,
            **{f"p_{name}": results[name].oof for name in order},
        }
    ).to_csv(out, index=False)
    print(f"\nout-of-fold predictions: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())