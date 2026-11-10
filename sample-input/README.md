# Sample input

## From pose-kinematics (the usual case)

Link pose-kinematics' `kinematics` output (`pose-kinematics/<session_id>_kinematics.parquet` and its
`sessions.csv`) to `kinematics`. For the agreement with expert labels, link the session NWB files
that have labelled behaviour intervals to `session`.

## Real data: DANDI:002047 (open field, WT vs 5xFAD)

1. Run pose-kinematics on DANDI:002047 NWB files (see pose-kinematics' `sample-input/README.md`; e.g.
   `sub-F01_ses-OF-F01-S1_behavior+image+ophys.nwb`, asset `52efa19d-15cf-45b6-93ae-4e16a4f4a056`,
   `https://api.dandiarchive.org/api/assets/52efa19d-15cf-45b6-93ae-4e16a4f4a056/download/`).
2. Link its output to `kinematics` and the same NWB files to `session`: their `processing/behavior`
   has the authors' labels (`rear`, `freezing`, `locomotion`, `rest`, `walk`; the spatial zones
   `center`, `middle_zone`, `walls` are skipped automatically).

The Pennsieve demo uses six day-1 open-field sessions (WT F35, F40, F48; 5xFAD F07, F14, F20),
staged by `workflow-systems-neuroscience/scripts/stage-demo-data.py openfield-5xfad`; the
pose-kinematics output is in `data/work/openfield-5xfad/pose-kinematics/`.

## Synthetic

`python -m tests.make_fixture data/fixture` writes four simulated sessions with three planted
behaviours (rest, walk, rear; `tests/simulate.py`) as kinematics.parquet, plus session NWB files with
the planted behaviours as labels (when pynwb is installed).
