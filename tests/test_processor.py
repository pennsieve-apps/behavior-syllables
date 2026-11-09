"""The synthetic fixture (three planted behaviours) through the processor (no Docker)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests import make_fixture

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixture")
    make_fixture.build(out, "both", 120.0, nwb=True)
    return out


def run(input_dir, output_dir, **params):
    env = dict(os.environ, INPUT_DIR=str(input_dir), OUTPUT_DIR=str(output_dir), MPLBACKEND="Agg")
    env.update({k: str(v) for k, v in params.items()})
    return subprocess.run([sys.executable, "-m", "processor.main"], cwd=REPO, env=env, capture_output=True, text=True)


@pytest.fixture(scope="session")
def out3(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out3")
    r = run(fixture_dir / "linked", out, N_SYLLABLES=3)
    assert r.returncode == 0, r.stdout + r.stderr
    return out / "behavior-syllables"


def test_outputs(out3):
    for sid in ("wt-01", "wt-02", "ko-01", "ko-02"):
        assert (out3 / f"{sid}_syllables.csv").exists()
        assert (out3 / f"{sid}_syllables.nwb").exists()
    for name in ("syllables.csv", "usage.csv", "transitions.csv", "group_stats.csv", "agreement.csv", "summary.json",
                 "provenance.json", "model.json", "ethogram.png", "agreement.png", "syllable_poses.png"):
        assert (out3 / name).exists(), name
    s = json.loads((out3 / "summary.json").read_text())
    assert s["n_syllables"] == 3
    # Planted behaviours last ~1-2 s; the sticky AR-HMM must not chop them into jitter-length syllables.
    assert s["median_syllable_duration_s"] > 0.5 and not s["short_syllable_warning"]
    assert all(f["caption"] for f in s["figures"])
    p = json.loads((out3 / "provenance.json").read_text())
    assert p["variant"] == "hmm" and p["citations"]


def test_recovers_planted_behaviours(out3, fixture_dir):
    from sklearn.metrics import adjusted_mutual_info_score

    for sid in ("wt-01", "ko-01"):
        z = pd.read_csv(out3 / f"{sid}_syllables.csv")
        truth = pd.read_csv(fixture_dir / "truth" / f"{sid}_states.csv")
        assert (z["time"].to_numpy() == pytest.approx(truth["time"].to_numpy()))
        ok = z["syllable"] >= 0
        assert adjusted_mutual_info_score(truth["state"][ok], z["syllable"][ok]) > 0.8
        # Each planted behaviour is mostly one syllable.
        ct = pd.crosstab(truth["state"][ok], z["syllable"][ok], normalize="index")
        assert (ct.max(axis=1) > 0.6).all(), ct


def test_agreement_with_labels(out3):
    s = json.loads((out3 / "summary.json").read_text())
    a = s["agreement"]
    assert set(a["labels"]) == {"rear", "rest", "walk"}          # the spatial "center" table is ignored
    assert a["ami"] > 0.8 and a["ami_shifted_null_mean"] < 0.05
    rows = list(csv.DictReader(open(out3 / "agreement.csv")))
    assert {"syllable", "label", "p_label_given_syllable", "jaccard"} <= set(rows[0])


def test_nwb_and_session_clock(out3):
    import pynwb

    with pynwb.NWBHDF5IO(str(out3 / "wt-01_syllables.nwb"), "r") as io:
        nwb = io.read()
        assert nwb.identifier != "fixture-wt-01" and "fixture-wt-01" in nwb.notes   # new UUID, source noted
        assert nwb.source_script_file_name
        df = nwb.processing["behavior"]["syllables"].to_dataframe()
        assert df["start_time"].iloc[0] == pytest.approx(5.0)
        assert set(df["syllable"]) <= {0, 1, 2}


def test_group_stats(out3):
    rows = list(csv.DictReader(open(out3 / "group_stats.csv")))
    assert len(rows) == 3 and rows[0]["test"] == "Welch t-test" and rows[0]["n_WT"] == "2"


def test_auto_number_and_flat_layout(fixture_dir, tmp_path):
    r = run(fixture_dir / "flat", tmp_path, MAX_SYLLABLES=8)
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads((tmp_path / "behavior-syllables" / "summary.json").read_text())
    assert 3 <= s["n_syllables"] <= 8
    assert s["model"]["heldout_loglik_per_frame"]
    assert s["agreement"]["ami"] > 0.4


def test_without_labels(tmp_path):
    make_fixture.build(tmp_path / "fx", "linked", 60.0, nwb=False)
    r = run(tmp_path / "fx" / "linked", tmp_path / "out", N_SYLLABLES=3)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (tmp_path / "out" / "behavior-syllables" / "agreement.csv").exists()


def test_bad_parameters_and_no_input(fixture_dir, tmp_path):
    assert run(fixture_dir / "linked", tmp_path / "a", METHOD="vame").returncode == 2
    assert run(fixture_dir / "linked", tmp_path / "b", N_SYLLABLES=1).returncode == 2
    assert run(fixture_dir / "linked", tmp_path / "c", USE_KEYPOINTS="nose,wing").returncode == 2
    (tmp_path / "empty").mkdir()
    assert run(tmp_path / "empty", tmp_path / "d").returncode == 1


def test_kpms_needs_kpms_variant(fixture_dir, tmp_path):
    try:
        import keypoint_moseq  # noqa: F401
        pytest.skip("keypoint-moseq installed here")
    except ImportError:
        pass
    assert run(fixture_dir / "linked", tmp_path, METHOD="kpms").returncode == 2
