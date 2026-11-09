"""
Synthetic input for behavior-syllables: four simulated open-field sessions (tests/simulate.py) with
three planted behaviours (rest, walk, rear), "WT" sessions walking more than "KO", written as
pose-kinematics would: <session_id>_kinematics.parquet (keypoints in cm, centroid, heading, speed,
...) with its `pennsieve` metadata, a sessions.csv sheet, and (when pynwb is installed) one session
NWB per session with the planted behaviours as labelled TimeIntervals (rest, walk, rear), matched
through the NWB identifier.

  python -m tests.make_fixture OUT [--layout linked|flat|both] [--seconds 180] [--no-nwb]

writes OUT/linked/{kinematics,session}/ and/or OUT/flat/, and OUT/truth/<session_id>_states.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tests.simulate import KEYPOINTS, simulate

FPS = 30.0
RADIUS = 30.0
SESSIONS = [("wt-01", "m01", "WT", 1.6, 1), ("wt-02", "m02", "WT", 1.6, 2), ("ko-01", "m03", "KO", 0.5, 3),
            ("ko-02", "m04", "KO", 0.5, 4)]
EDGES = [["nose", "headcenter"], ["leftear", "headcenter"], ["rightear", "headcenter"], ["headcenter", "bodycenter"],
         ["bodycenter", "tailbase"]]


def kinematics_table(sim, t, rng):
    """The columns pose-kinematics writes, from simulated keypoints (cm, y up) -> image orientation (y down)."""
    kp = sim["keypoints"].copy()
    kp[..., 1] *= -1
    kp += rng.normal(0, 0.03, kp.shape)
    n = len(t)
    cols = {"time": t, "frame": np.arange(n)}
    c = kp[:, KEYPOINTS.index("bodycenter")]
    cols["centroid_x"], cols["centroid_y"] = c[:, 0], c[:, 1]
    v = np.gradient(c, t, axis=0)
    cols["speed"] = np.hypot(v[:, 0], v[:, 1])
    cols["step_distance"] = np.r_[0, np.hypot(*np.diff(c, axis=0).T)]
    vec = kp[:, KEYPOINTS.index("headcenter")] - kp[:, KEYPOINTS.index("tailbase")]
    h = np.arctan2(-vec[:, 1], vec[:, 0])
    cols["heading"] = h
    cols["angular_speed"] = np.gradient(np.unwrap(h), t)
    cols["body_length"] = np.hypot(*(kp[:, KEYPOINTS.index("nose")] - kp[:, KEYPOINTS.index("tailbase")]).T)
    for j, k in enumerate(KEYPOINTS):
        cols[f"kp_{k}_x"], cols[f"kp_{k}_y"] = kp[:, j, 0], kp[:, j, 1]
        vv = np.gradient(kp[:, j], t, axis=0)
        cols[f"kp_{k}_speed"] = np.hypot(vv[:, 0], vv[:, 1])
    r = np.hypot(c[:, 0], c[:, 1])
    cols["in_center"] = r <= RADIUS / 2
    cols["in_periphery"] = r > RADIUS / 2
    cols["in_wall_zone"] = r >= RADIUS - 6
    cols["distance_to_wall"] = RADIUS - r
    cols["valid"] = np.ones(n, bool)
    # A short tracking gap the model has to skip.
    gap = slice(n // 2, n // 2 + 15)
    for k in list(cols):
        if k.startswith("kp_") or k in ("centroid_x", "centroid_y", "speed", "heading", "angular_speed", "body_length"):
            cols[k] = np.asarray(cols[k], float)
            cols[k][gap] = np.nan
    cols["valid"][gap] = False
    return cols


def write_parquet(path, cols, meta):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(cols)
    table = table.replace_schema_metadata({b"pennsieve": json.dumps(meta).encode()})
    pq.write_table(table, path)


def write_label_nwb(path, ident, sid, t, states):
    import pynwb
    from pynwb.epoch import TimeIntervals

    nwb = pynwb.NWBFile(session_description=f"synthetic {sid}", identifier=ident,
                        session_start_time=datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc))
    mod = nwb.create_processing_module("behavior", "labels")
    end = t[-1] + 1 / FPS
    for st in ("rest", "walk", "rear"):
        m = np.r_[0, (states == st).astype(int), 0]
        d = np.diff(m)
        ti = TimeIntervals(name=st, description=f"Behavioral state. Episodes of {st} (simulated truth).")
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ti.add_interval(start_time=float(t[a]), stop_time=float(t[b]) if b < len(t) else float(end))
        mod.add(ti)
    zone = TimeIntervals(name="center", description="Spatial zone. Not a behaviour; must be ignored.")
    zone.add_interval(start_time=float(t[0]), stop_time=float(t[len(t) // 3]))
    mod.add(zone)
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def have_nwb():
    try:
        import pynwb  # noqa: F401

        return True
    except ImportError:
        return False


def build(out: Path, layout: str = "both", seconds: float = 180.0, nwb: bool = True):
    nwb = nwb and have_nwb()
    stage = out / "_staging"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "kinematics" / "pose-kinematics").mkdir(parents=True)
    (stage / "session").mkdir(parents=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    sheet = []
    for sid, subject, cond, walk_bias, seed in SESSIONS:
        sim = simulate(seconds, FPS, RADIUS, seed, center_bias=1.0 if cond == "WT" else 0.0, walk_bias=walk_bias)
        rng = np.random.default_rng(50 + seed)
        t = 5.0 + sim["t"]
        cols = kinematics_table(sim, t, rng)
        ident = f"fixture-{sid}"
        meta = {"app": "pose-kinematics", "version": "0.1.0", "session_id": sid, "units": "cm",
                "px_per_cm": [8.0, 8.0], "keypoints": KEYPOINTS, "edges": EDGES, "centroid_keypoints": ["bodycenter"],
                "head_keypoints": ["headcenter"], "tail_keypoint": "tailbase", "nose_keypoint": "nose",
                "fps_effective": FPS, "time_source": "simulated", "sheet": {"subject_id": subject, "condition": cond},
                "nwb": {"identifier": ident, "session_start_time": "2026-11-09T10:00:00+00:00",
                        "session_description": f"synthetic {sid}", "subject": {"subject_id": subject,
                                                                               "species": "Mus musculus"}}}
        write_parquet(stage / "kinematics" / "pose-kinematics" / f"{sid}_kinematics.parquet", cols, meta)
        if nwb:
            write_label_nwb(stage / "session" / f"sub-{subject}_ses-{sid}.nwb", ident, sid, t, sim["state"])
        with open(out / "truth" / f"{sid}_states.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "state"])
            w.writerows([[f"{a:.6f}", s] for a, s in zip(t, sim["state"])])
        sheet.append({"session_id": sid, "path": f"{sid}_kinematics.parquet", "subject_id": subject, "condition": cond})
    with open(stage / "kinematics" / "pose-kinematics" / "sessions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["session_id", "path", "subject_id", "condition"])
        w.writeheader()
        w.writerows(sheet)
    if layout in ("linked", "both"):
        dest = out / "linked"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(stage, dest)
        if not any((dest / "session").iterdir()):
            (dest / "session").rmdir()
    if layout in ("flat", "both"):
        dest = out / "flat"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for p in stage.rglob("*"):
            if p.is_file():
                shutil.copy2(p, dest / p.name)
    shutil.rmtree(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--no-nwb", action="store_true")
    a = ap.parse_args()
    build(a.out, a.layout, a.seconds, not a.no_nwb)
    print(f"fixture written to {a.out}")


if __name__ == "__main__":
    main()
