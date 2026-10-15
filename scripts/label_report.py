"""Check the rebuilt recovery label against the one the events file ships.

    python scripts/label_report.py

Prints, over every available match:

1. how often SkillCorner's own ``end_type`` is present at all;
2. the positive rate of the rebuilt label at several windows;
3. a cross-tabulation of the rebuilt label against ``end_type``, on the rows
   where ``end_type`` exists. The two are measured independently, so agreement
   there is the evidence that the rebuilt label means what it claims to mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from pressing import config, events, labels  # noqa: E402

WINDOWS = (3.0, 5.0, 8.0)
REGAIN_TYPES = {"direct_regain", "indirect_regain"}


def main() -> int:
    match_ids = config.available_matches()
    eng = events.load_all_engagements(match_ids)
    eng = labels.add_recovery_timing(eng)
    n = len(eng)

    print(f"{len(match_ids)} matches, {n} on-ball engagements\n")

    recorded = eng["end_type"].notna()
    print("1. SkillCorner's own end_type")
    print(f"   recorded on {recorded.sum()} of {n} engagements "
          f"({recorded.mean():.1%})")
    print(eng.loc[recorded, "end_type"].value_counts().to_string(header=False))

    print("\n2. Rebuilt label: pressing team regains possession within X s")
    print("   window   positive   rate    from a restart")
    for w in WINDOWS:
        lab = labels.label_regain(eng, w)
        restart = (lab & eng["regain_after_interruption"]).sum()
        print(f"   {w:>4.0f} s   {lab.sum():>8}   {lab.mean():>5.1%}   "
              f"{restart:>6} ({restart / max(lab.sum(), 1):.1%})")

    w = labels.DEFAULT_WINDOW_SECONDS
    lab = labels.label_regain(eng, w)

    print(f"\n3. Agreement with end_type at the {w:.0f} s window")
    sub = eng[recorded].copy()
    sub["rebuilt"] = lab[recorded]
    table = pd.crosstab(sub["end_type"], sub["rebuilt"])
    table.columns = ["no regain", "regain"]
    table["agrees"] = [
        row["regain"] / row.sum() if idx in REGAIN_TYPES
        else row["no regain"] / row.sum()
        for idx, row in table.iterrows()
    ]
    print(table.to_string(float_format=lambda v: f"{v:.1%}"))

    agree = (
        sub["rebuilt"] == sub["end_type"].isin(REGAIN_TYPES)
    ).mean()
    print(f"\n   overall agreement where end_type exists: {agree:.1%}")
    print(f"   positive rate where end_type is missing:  "
          f"{lab[~recorded].mean():.1%}")

    print("\n4. Stricter variants at the default window")
    for name, series in (
        ("open play only", labels.label_regain_open_play(eng, w)),
        ("carrier lost it directly", labels.label_regain_strict(eng, w)),
    ):
        print(f"   {name:<26} {series.sum():>6}  ({series.mean():.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
