"""How do x coordinates in the dynamic events relate to attacking_side?

    python scripts/diagnostics/check_event_direction.py 1899585 1874553

For each match, period and team: the attacking_side on its possession rows,
where its goalkeeper's possessions start (raw x, should be near its own goal)
and where its possessions that end in a shot end (raw x, near the other goal).
Then, for engagements, how the event's x columns compare with the carrier's x
in our scenes, which are correctly oriented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import press_outcomes as po  # noqa: E402
from pressing import modelling  # noqa: E402


def main() -> int:
    table = modelling.load_table()
    table["meta_event_id"] = table["meta_event_id"].astype(str)
    for m in [int(a) for a in sys.argv[1:]]:
        ev = pd.read_csv(po.events_path(m), low_memory=False)
        ev["event_id"] = ev["event_id"].astype(str)
        print(f"\n=== match {m} ===")
        print("event types:", ev["event_type"].value_counts().to_dict())
        print("end types:", ev["end_type"].value_counts().head(8).to_dict())
        print("positions:", sorted(ev["player_position"].dropna().astype(str).unique())[:20])

        poss = ev[ev["event_type"] == "player_possession"].copy()
        gk = poss["player_position"].astype(str).str.upper().isin(["GK", "GOALKEEPER"])
        rows = []
        for (period, team), g in poss.groupby(["period", "team_id"]):
            rows.append({
                "period": period, "team": team,
                "attacking_side": "/".join(sorted(g["attacking_side"].astype(str).unique())),
                "n": len(g),
                "gk_x_start": g.loc[gk.loc[g.index], "x_start"].median(),
                "shot_x_end": g.loc[g["end_type"] == "shot", "x_end"].median(),
                "n_shots": int((g["end_type"] == "shot").sum()),
                "all_x_end": g["x_end"].median(),
            })
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

        eng = ev[ev["event_type"] == "on_ball_engagement"]
        print(f"engagements: {len(eng)}, non-missing share: "
              f"x_start {eng['x_start'].notna().mean():.2f}, "
              f"player_in_possession_x_start {eng['player_in_possession_x_start'].notna().mean():.2f}")
        both = table[table["meta_match_id"] == m][["meta_event_id", "f_carrier_x"]].merge(
            eng, left_on="meta_event_id", right_on="event_id", how="inner")
        print(f"engagements matched to our scenes: {len(both)}")
        sign = both["attacking_side"].map({"left_to_right": 1.0, "right_to_left": -1.0})
        for col in ("x_start", "player_in_possession_x_start"):
            raw = both[col].astype(float)
            for name, val in (("raw", raw), ("-sign*raw", -sign * raw), ("sign*raw", sign * raw)):
                r = both["f_carrier_x"].corr(val)
                print(f"  corr(scene carrier x, {name} {col}) = {r:+.3f}" if pd.notna(r)
                      else f"  corr(scene carrier x, {name} {col}) = n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
