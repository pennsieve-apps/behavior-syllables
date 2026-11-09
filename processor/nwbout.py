"""<session_id>_syllables.nwb: a TimeIntervals table `syllables` (start_time, stop_time, syllable)."""
from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone

import numpy as np


def _dt(text):
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def write_syllables_nwb(path, sid, identity, t, z, method_note, app_version):
    import pynwb
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    from processor.analysis import bouts, frame_durations

    start = _dt(identity.get("session_start_time")) or datetime(1970, 1, 1, tzinfo=timezone.utc)
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
    nwb = pynwb.NWBFile(session_description=identity.get("session_description") or f"session {sid}",
                        identifier=identity.get("identifier") or str(uuid.uuid4()), session_start_time=start,
                        timestamps_reference_time=_dt(identity.get("timestamps_reference_time")) or start,
                        subject=subj, source_script=f"behavior-syllables {app_version}",
                        notes=f"Unsupervised behavioural syllables for session {sid}: {method_note}")
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
