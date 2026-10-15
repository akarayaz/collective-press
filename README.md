# Pressing and ball recovery on the SkillCorner A-League open data

Work in progress for the SkillCorner x PySport Analytics Cup 2.0, on the
football theme of defensive positioning.

**Question.** When a defender engages the player in possession, does the
pressing side's *positioning* explain ball recovery beyond what SkillCorner's
own context metrics (pressure level, `xloss`) already say?

Three nested models answer it on the same match-level cross-validation folds:

| Model | Inputs | What the comparison isolates |
|---|---|---|
| M0 | SkillCorner's own summaries: `xloss`, pressure level, zone, game state | the baseline these metrics already reach |
| M1 | M0 + hand-built positioning features | whether geometry adds anything |
| M2 | M0 + a graph network over the tracking frame | whether player-to-player structure adds more |

The headline result is the *gain* from M0 to M1 and M2, with match-level
bootstrap intervals, not any single model's score.

> The label records that the ball was recovered, not that the press caused it.
> Results are reported as association, never causation.

## Status

Data layer, the label, the feature table, and the tabular models. The graph
model comes next.

### The decision rule, fixed before the numbers

A gain counts as real only when **both** hold:

* the 95% interval of the log-loss improvement is entirely above zero, from a
  bootstrap that resamples **matches**, not rows, since engagements inside a
  match are not independent;
* at least **14 of the 20 matches** improve.

Validation is leave-one-match-out. Hyperparameters are fixed in
`pressing.modelling.MODEL_PARAMS` and were never tuned against a test fold.

Every comparison also runs a placebo: the tracking features are permuted among
the rows of each match, which keeps each column's distribution and each
match's composition but destroys the link between a scene and its outcome. A
gain that survives that is not coming from the scenes, and means there is a
leak to find.

`tests/test_modelling.py` points the whole procedure at tables whose answer is
known in advance: one where the outcome really does depend on the tracking
features, one where it does not, and one that has been shuffled. It has to say
"real gain", "no gain", "no gain".

### Data quality

Tracking and the events file agree closely: across all 17,445 engagements the
presser-to-carrier distance measured in the tracking frame differs from the
reported `interplayer_distance_start` by a median of **5 mm**, with a median
signed difference of exactly zero, so there is no systematic offset and
`frame_start` is the right frame to read.

1.4% of rows disagree by more than 2 m, some by tens of metres. Four
explanations were tested and the first is the only one that holds:

| | |
|---|---|
| one position extrapolated rather than seen | present in 54% of them |
| a player already substituted | 0% |
| a tracking identity swap | 0%, no impossible steps in either group |
| the distance describing a different frame | no, offsets are scattered and no named frame fits |

The remainder is unexplained and concentrates in three matches, which the
events files may have been refreshed for separately. Those rows are flagged
`q_ok = False` and excluded rather than silently repaired.

### The label

### The label

`end_type` in the events file marks the outcome of an engagement, but it is
present on only 24.5% of them, and the gaps are not random: about half of the
possessions that end in a turnover have no engagement labelled at all. So the
label is rebuilt from the event sequence: **the pressing team starts a
possession within 5 seconds of the engagement starting.** That is defined for
all 17,445 engagements.

Over the 20 matches, measured by `scripts/label_report.py`:

| | |
|---|---|
| positive rate at 3 / 5 / 8 s | 11.3% / 21.0% / 31.2% |
| agreement with `end_type`, where it exists | 87.4% |
| of which `direct_regain` matched | 92.7% |
| positive rate where `end_type` is missing | 6.8% |
| recoveries that came from a restart (5 s) | 1 of 3671 |

The two measurements are independent, so the agreement is the evidence that
the rebuilt label means what it claims to. The restart count answers a
specific worry: at these windows, recoveries from throw-ins and free kicks are
not quietly inflating the positive class.

The window is a parameter, never hard-coded downstream, and the headline
result has to survive 3 s and 8 s as well.

## Getting the data

The dataset is **not** in this repository. Clone SkillCorner's open data next to
this folder and fetch the Git LFS objects, which hold the tracking files:

```bash
git lfs install
git clone https://github.com/SkillCorner/opendata.git
cd opendata && git lfs pull        # about 1.8 GB
```

A tracking file should be roughly 90 MB. A few hundred bytes means Git LFS did
not run and only the pointer was cloned. Without Git LFS, fetch a match
directly:

```bash
id=1925299
curl -L -o "data/matches/$id/${id}_tracking_extrapolated.jsonl" \
  "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/$id/${id}_tracking_extrapolated.jsonl"
```

The code looks for `../opendata/data` by default. Anywhere else, point
`PRESSING_DATA_DIR` at the `data` folder.

## Running it

```bash
conda create -n analytics-cup python=3.11 -y
conda activate analytics-cup
pip install -r requirements.txt

python scripts/plot_scenes.py --match 1925299 --n 3
```

Each scene is written to `figures/`.

Then the two checks that have to pass before anything is built on this data:

```bash
python scripts/check_alignment.py --all    # tracking vs events, every match
python scripts/label_report.py             # the label vs SkillCorner's own
```

`check_alignment.py` measures the presser-to-carrier distance in the tracking
frame and compares it with the `interplayer_distance_start` the events file
reports. The two come from different places, so agreement across thousands of
engagements is the evidence that `frame_start` points where this code assumes
it does and that the two named players are the right ones. It also reports how
often a named player is missing from the frame.

## Layout

```
pressing/
    config.py      paths, pitch dimensions, frame rate
    metadata.py    players, teams, periods, attacking directions
    tracking.py    streaming frame reader, velocities, orientation
    events.py      dynamic events, on-ball engagements
    labels.py      was the ball recovered, rebuilt from the event sequence
    scene.py       engagements -> oriented snapshots, one file pass per match
    features.py    positioning features measured off a scene
    modelling.py   nested feature sets, match-wise CV, bootstrap, decision rule
    viz.py         drawing a scene
scripts/
    plot_scenes.py        draw a few engagements from one match
    check_alignment.py    tracking frames vs the events file
    find_reference_frame.py  which frame the events file describes
    diagnose_tail.py      what is behind the disagreeing rows
    label_report.py       label diagnostics over every match
    build_features.py     the modelling table -> cache/engagements.csv
    run_models.py         the three models, the gains, the placebo
tests/
    test_features.py   geometry checks on hand-built scenes
    test_modelling.py  the decision rule, on tables with a known answer
```

```bash
python scripts/build_features.py      # reads every tracking file once
python scripts/run_models.py          # the comparison
python tests/test_features.py         # no data needed
python tests/test_modelling.py        # no data needed
```

The three models are nested, so each gain is attributable to what was added:

| | |
|---|---|
| `context` | SkillCorner's read of the situation: `xloss`, `xshot`, pressure and space codes, EPV, zone, score, clock. No geometry. |
| `skillcorner_geometry` | plus the geometry SkillCorner already ships: distance and angle to the carrier, the defensive line. A subscriber has these without touching tracking. |
| `tracking_geometry` | plus everything measured from the tracking frame: support, local counts, blocked passing lanes, closing speed, the shape of the block. |

The headline is the **second** step. The first is reported because knowing
what the basic geometry is worth is what makes the second number mean
anything.

Columns are prefixed by role: `meta_` (identity, never an input), `m0_`
(SkillCorner's summaries), `f_` (positioning from tracking), `q_` (quality),
`label_`. Only `m0_` and `f_` reach a model, so an outcome column cannot
quietly become an input.

### Conventions

* Coordinates are metres, origin at the centre spot, so x spans
  [-52.5, 52.5] and y spans [-34, 34].
* Scenes are rotated so the **team in possession always attacks towards +x**.
  Teams swap ends at half time, so raw coordinates are not comparable across
  periods.
* On an `on_ball_engagement` row, `team_id` is the pressing team and
  `attacking_side` is that team's attacking direction. Both verified across all
  20 matches.
* Velocities are finite differences over 0.5 s (5 frames at 10 Hz) and are NaN
  when the earlier frame is unavailable.
* A hollow marker means SkillCorner extrapolated that position rather than
  tracking it from the camera.

## Licence

MIT, see `LICENSE.md`. The SkillCorner open data is covered by its own terms.
