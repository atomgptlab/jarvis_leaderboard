HYD local-environment histogram gradient boosting
================================================

This is a frozen contribution for the JARVIS dft_3d_magmom_oszicar
SinglePropertyClass benchmark. Its current status is recorded in
publication-state.json. Validation results do not establish a leaderboard rank.

Method
------
The input is restricted to lattice vectors, atom coordinates, element symbols,
and the Cartesian/fractional-coordinate flag. No calculated material property
is used as an input descriptor. MatMiner uses its standard elemental-property
tables (including elemental ground-state properties), not computed properties
of the benchmark material. The 434 descriptors consist of 145 MatMiner
composition descriptors, 129 element-fraction/lattice descriptors, and 160
periodic local-environment descriptors. These include radial distributions,
coordination counts, nearest distances and electronegativity differences.

The selected classifier is sklearn HistGradientBoostingClassifier. Its exact
parameters are in config.json: 127 leaf nodes, learning rate 0.06, regularization
1.0 and seed 17. The 259 boosting iterations were chosen using an internal
training holdout, followed by a refit on all 41,766 official training rows.
The official validation split was used to compare candidate methods. One ID
shared with training is excluded from internal validation; the structure-audit
score additionally excludes 24 known duplicate structures. This audit does not
identify every symmetry-equivalent structure.

The optional 128-descriptor MatMiner structural extension did not improve
validation accuracy in three seeds (95.2500% each), and was excluded from this
model. Structural featurization failures were retained as NaN in that comparison;
the failure audit records 1,267/41,766 training and 183/5,221 validation rows.

The recorded accuracy is 95.3840% on 5,221 validation examples and 95.3627% on
5,197 examples after the duplicate-structure exclusions. Two other seeds each
gave 95.3649%. The archived metrics, configuration and replay evidence specify
what was actually checked. The 2023 MatMiner benchmark is not claimed as an
exactly reproduced baseline: its structural features and library versions
differ from this model. No test targets or public test predictions were used
to train, tune, calibrate, ensemble or choose this candidate.

Frozen test evaluation
----------------------
After final-selection-freeze.json was written, predictions were generated for
all 5,222 official test IDs. The pinned upstream get_metric_value function
returned 0.9527 (4,975 correct; unrounded accuracy 0.9527001148985064). This
post-freeze score was not used to select, tune or alter the model. It is a local
check against the official scorer, not confirmation of a public leaderboard rank.

Replay
------
Use Python 3.12. Inference is CPU-only and does not require an A100.

  PYTHON_BIN=python3.12 bash run.sh

The script creates an isolated environment, installs pinned dependencies,
downloads the official 2022 JARVIS structure database, verifies its SHA-256,
retains only structural inputs, and predicts the ordered IDs in test_ids.json.
That file contains IDs only, not targets. To use an existing matching database:

  bash run.sh --raw-json /path/to/jdft_3d-12-12-2022.json

The output name follows the official leaderboard contract:
AI-SinglePropertyClass-magmom_oszicar-dft_3d-test-acc.csv.zip
The CSV inside contains only id,prediction. replay-evidence.json records timing,
input/model-selection hashes and the resulting prediction hashes. The driver
does not load the official benchmark label archive or compute a test score.

An optional container recipe is included. It has not yet been built or tested
in this environment; the verified local environment is recorded in config.json.
The container's Python 3.12 base tag is floating; the Python packages are pinned.

Training replay
---------------
After installing requirements.txt, run from this directory:

  python retrain_frozen_hgb.py --selection selection.json \
    --training-labels training-labels.json --training-config config.json \
    --output .training-cache/retrained.joblib

This rebuilds the same descriptor order and refits the already frozen
parameters using only the included official training labels. It performs no
new parameter search or validation/test selection. --raw-json can supply the
same local source as above. Training replay has been executed locally; its
prediction comparison is recorded separately from any claim of identical
serialized model bytes or cross-platform floating-point behavior.

Data and software provenance
----------------------------
Official raw source: https://ndownloader.figshare.com/files/38521619
Expected JSON member: jdft_3d-12-12-2022.json
Expected SHA-256:
72da38713e509195220771b83b07cbd2d21f5d0d5cc203e1067dd597465b0b35

The raw archive contains calculated properties, but only atomic structures are
retained by inference. Training labels are supplied separately in
training-labels.json. Test labels are never supplied to either replay program.
Model checksums and exact feature order are in selection.json. Software
versions and source hashes from the actual experiment are in config.json.

Team: HYD
Public contact authorized by the contributor: 949801198@qq.com
Source repository: https://github.com/donkeey11/jarvis_leaderboard
Target upstream branch: atomgptlab/jarvis_leaderboard, develop
