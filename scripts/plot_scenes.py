"""Draw a few pressing scenes from one match, as a first look at the data.

    python scripts/plot_scenes.py --match 1925299 --n 3

With no --match the first available match is used. Figures land in figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pressing import config, events, scene, viz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default=None, help="match id (default: the first one)")
    ap.add_argument("--n", type=int, default=3, help="how many scenes to draw")
    ap.add_argument(
        "--subtype", default=None,
        help="only engagements of this subtype (pressure, pressing, "
             "counter_press, recovery_press, other)",
    )
    ap.add_argument(
        "--labelled", action="store_true",
        help="only engagements whose end_type is recorded",
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    args = ap.parse_args()

    match_id = args.match or config.available_matches()[0]
    eng = events.load_engagements(match_id)
    print(f"match {match_id}: {len(eng)} on-ball engagements")

    sel = eng
    if args.subtype:
        sel = sel[sel["event_subtype"] == args.subtype]
    if args.labelled:
        sel = sel[sel["end_type"].notna()]
    if sel.empty:
        print("no engagements match those filters", file=sys.stderr)
        return 1

    sel = sel.sample(n=min(args.n, len(sel)), random_state=args.seed)
    print(f"drawing {len(sel)} of {len(eng)} engagements\n")

    print(
        "  event      frame    time       subtype         players  "
        "distance m (tracking / events file)"
    )
    for _, row in sel.iterrows():
        sc = scene.build_scene(match_id, row)
        path = viz.save_scene(sc)

        # Consistency check: the presser-to-carrier distance measured in the
        # tracking frame should match the distance the events file reports. A
        # mismatch means the frames and the events are not aligned the way this
        # code assumes, and nothing built on top of it would be trustworthy.
        carrier = sc.carrier
        measured = float("nan")
        if carrier is not None:
            presser = sc.presser
            measured = float(
                ((presser["x"] - carrier["x"]) ** 2
                 + (presser["y"] - carrier["y"]) ** 2) ** 0.5
            )
        reported = sc.info.get("interplayer_distance_start")

        print(
            f"  {sc.event_id:>8}  {sc.frame:>6}  {sc.time_start:>9}  "
            f"{str(sc.info.get('event_subtype')):<15} {len(sc.players):>7}  "
            f"{measured:>8.2f} / {reported}"
        )
        print(f"            {path}")

    print(f"\nfigures written to {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
