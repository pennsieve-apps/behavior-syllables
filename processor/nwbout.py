"""<session_id>_syllables.nwb: a TimeIntervals table `syllables` (start_time, stop_time, syllable)."""
from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np


def _dt(text):
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def write_syllables_nwb(path, sid, identity, t, z, method_note, app_version, source_name="pose-kinematics output"):
    import pynwb
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    from processor.analysis import bouts, frame_durations

    from processor.common import nwbfile_kwargs

    subj = None
    if identity.get("subject"):
        s = {k: v for k, v in identity["subject"].items() if k in ("subject_id", "species", "sex", "genotype", "strain",
                                                                   "age", "description", "date_of_birth", "weight")}
        if s.get("date_of_birth"):
            s["date_of_birth"] = _dt(s["date_of_birth"])
        try:
            subj = Subject(**{k: v for k, v in s.items() if v is not None})
        except Exception:
            subj = None
    kw = nwbfile_kwargs(identity, sid, "behavior-syllables", app_version,
                        f"Unsupervised behavioural syllables for session {sid} ({method_note}), from {source_name}.")
    nwb = pynwb.NWBFile(subject=subj, **kw)
    mod = nwb.create_processing_module("behavior", "Behavioural syllables (behavior-syllables)")
    ti = TimeIntervals(name="syllables", description=f"Syllable bouts ({method_note}); syllable 0 = the most used; "
                       "stop = time of the first frame after the bout.")
    ti.add_column(name="syllable", description="syllable id (0 = most used across all sessions)")
    dur = frame_durations(t)
    for s, a, b in bouts(z, t):
        stop = t[b] if b < len(t) else t[-1] + dur[-1]
        ti.add_interval(start_time=float(t[a]), stop_time=float(stop), syllable=int(s))
    mod.add(ti)
    mod.add(pynwb.TimeSeries(name="syllable", data=z.astype("int16"), timestamps=np.asarray(t, float), unit="n.a.",
                             description="syllable per pose frame; -1 = not modelled (missing pose)"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pynwb.NWBHDF5IO(str(path), "w") as io:
            io.write(nwb)
