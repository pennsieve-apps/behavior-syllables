"""
Finding the input.

Port `kinematics`: <session_id>_kinematics.parquet files as pose-kinematics writes them (time x
features, with the cleaned keypoints and a `pennsieve` JSON entry in the Parquet metadata), found
anywhere under INPUT_DIR; a sessions.csv sheet anywhere (condition / group per session_id).
Port `session` (optional): NWB files with labelled behaviour intervals (TimeIntervals), matched
to a session by the NWB `identifier` that pose-kinematics copied into the Parquet metadata, else by
session_id (sessions.csv path, sub-<a>_ses-<b>, file stem).
"""
from __future__ import annotations

import csv
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from processor.common import Sessions, read_sheet, sanitise, walk_files

log = logging.getLogger("behavior-syllables")
# TimeIntervals that are not behaviour labels.
NOT_LABELS = {"epochs", "trials", "invalid_times", "arena_regions", "syllables", "center", "periphery",
              "wall_zone", "middle_zone", "walls"}


@dataclass
class Session:
    session_id: str
    path: Path
    meta: dict
    sheet: dict = field(default_factory=dict)


@dataclass
class Found:
    sessions: list[Session]
    session_nwbs: list[Path]
    sheet: Sessions
    skipped: dict


def parquet_meta(path: Path) -> dict:
    import pyarrow.parquet as pq

    md = pq.read_schema(path).metadata or {}
    raw = md.get(b"pennsieve")
    return json.loads(raw) if raw else {}


def discover(input_dir: Path) -> Found:
    files = list(walk_files(input_dir))
    rows = []
    for p in files:
        if p.name == "sessions.csv":
            try:
                rows += read_sheet(p)
            except (OSError, csv.Error):
                pass
    # Merge sheets (pose-kinematics writes one; the user may link another): first value wins.
    merged = {}
    for r in rows:
        sid = sanitise(r["session_id"])
        cur = merged.setdefault(sid, {})
        for k, v in r.items():
            if v not in (None, "") and k not in cur:
                cur[k] = v
    sheet = Sessions([dict(v, session_id=k) for k, v in merged.items()])
    sessions, nwbs, skipped, seen = [], [], {}, {}
    for p in files:
        low = p.name.lower()
        if low.endswith("_kinematics.parquet"):
            try:
                meta = parquet_meta(p)
            except Exception as exc:
                skipped[str(p.relative_to(input_dir))] = f"not a readable Parquet file: {exc}"
                continue
            sid = sanitise(meta.get("session_id") or p.name[: -len("_kinematics.parquet")])
            if sid in seen:
                skipped[str(p.relative_to(input_dir))] = f"session {sid} already read from {seen[sid].name}"
                continue
            seen[sid] = p
            sh = dict(meta.get("sheet") or {})
            sh.update(sheet.sheet_columns(sid))
            sessions.append(Session(sid, p, meta, sh))
        elif low.endswith(".nwb"):
            nwbs.append(p)
    sessions.sort(key=lambda s: s.session_id)
    return Found(sessions, nwbs, sheet, skipped)


def nwb_identifier(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as f:
            v = f["identifier"][()]
            return v.decode() if isinstance(v, bytes) else str(v)
    except (OSError, KeyError, ValueError):
        return None


def match_label_nwbs(found: Found) -> dict:
    """session_id -> NWB path with labels."""
    by_id = {}
    for p in found.session_nwbs:
        ident = nwb_identifier(p)
        if ident:
            by_id.setdefault(ident, []).append(p)
    out = {}
    for s in found.sessions:
        ident = (s.meta.get("nwb") or {}).get("identifier")
        cands = [p for p in by_id.get(ident, [])] if ident else []
        if not cands:
            cands = [p for p in found.session_nwbs if found.sheet.session_id(p) == s.session_id]
        # Not our own outputs (behavior.nwb, syllables.nwb carry the same identifier).
        cands = [p for p in cands if not p.name.endswith(("_behavior.nwb", "_syllables.nwb"))]
        if cands:
            out[s.session_id] = sorted(cands, key=lambda p: (0 if "session" in p.parts else 1, len(str(p))))[0]
    return out


def read_labels(path: Path, tables: list[str] | None) -> dict:
    """{label: [(start, stop), ...]} from the NWB's TimeIntervals (processing/* and intervals/).

    With tables=None (LABEL_TABLES=auto): every TimeIntervals table except epochs/trials/
    invalid_times, our own tables, and spatial zones (tables named like arena regions, or whose
    description starts with "Spatial zone", as in DANDI 002047). A table with a `label` /
    `behavior` column gives one label per value.
    """
    import pynwb

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pynwb.NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            cands = []
            for mod in nwb.processing.values():
                cands += [(n, o) for n, o in mod.data_interfaces.items() if type(o).__name__ == "TimeIntervals"]
            if nwb.intervals:
                cands += list(nwb.intervals.items())
            for name, ti in cands:
                if tables is not None:
                    if name not in tables:
                        continue
                elif name in NOT_LABELS or (ti.description or "").lower().startswith("spatial zone"):
                    continue
                df = ti.to_dataframe()
                if not len(df):
                    continue
                col = next((c for c in ("label", "behavior", "behaviour", "name", "type") if c in df.columns), None)
                if col:
                    for lab, g in df.groupby(df[col].astype(str)):
                        out.setdefault(str(lab), []).extend(zip(g.start_time.astype(float), g.stop_time.astype(float)))
                else:
                    out.setdefault(name, []).extend(zip(df.start_time.astype(float), df.stop_time.astype(float)))
    return {k: sorted(v) for k, v in out.items()}


def label_matrix(labels: dict, t: np.ndarray):
    """(T, L) bool: frame time in [start, stop) of any interval of the label."""
    names = sorted(labels)
    m = np.zeros((len(t), len(names)), bool)
    for j, n in enumerate(names):
        iv = np.array(labels[n], float).reshape(-1, 2)
        if not len(iv):
            continue
        starts, stops = iv[:, 0], iv[:, 1]
        order = np.argsort(starts)
        starts, stops = starts[order], np.maximum.accumulate(stops[order])
        idx = np.searchsorted(starts, t, side="right") - 1
        ok = idx >= 0
        m[ok, j] = t[ok] < stops[idx[ok]]
    return names, m
