#!/usr/bin/env bash
# Rebuild every result and figure in the README from SkillCorner's open data.
#
#     bash scripts/reproduce.sh
#
# Needs the data in data/ (see pressing/config.py) and `pip install -e ".[deep]"`.
# Takes several hours on a laptop CPU, almost all of it in train_deep.py.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_features.py      # scenes, 32 metrics, labels, cleaning
python scripts/run_models.py          # gradient boosting, nested, with placebos
python scripts/export_scenes.py       # raw scenes for the neural models
python scripts/train_deep.py --models mlp_context,mlp_metrics,deepsets,transformer,gnn
python scripts/model_comparison.py    # Figure 2
python scripts/collective_press.py    # effects on regains
python scripts/press_outcomes.py      # effects on the other outcomes
python scripts/team_profiles.py
python scripts/ppda.py
python scripts/bodypose_extract.py    # needs data/bodypose/
python scripts/bodypose_press.py
python scripts/figure_press.py --event 1996435 9_684   # Figure 1
