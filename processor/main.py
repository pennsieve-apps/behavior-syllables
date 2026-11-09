"""
behavior-syllables: unsupervised behavioural syllables from pose kinematics.

Reads pose-kinematics' <session_id>_kinematics.parquet (port `kinematics`) and, optionally, session
NWB files with labelled behaviour intervals (port `session`); fits one model jointly on every session
(METHOD=hmm: sticky autoregressive HMM on egocentric posture, speed and turning; METHOD=kpms: keypoint-MoSeq), and writes
under OUTPUT_DIR/behavior-syllables/ (the single output port `syllables`):

  <session_id>_syllables.csv   time, syllable (per pose frame; -1 = not modelled)
  <session_id>_syllables.nwb   TimeIntervals `syllables` (start_time, stop_time, syllable)
  syllables.csv                per syllable: bouts, usage, durations, speed, posture
  usage.csv                    per session x syllable usage (+ sessions.csv columns)
  transitions.csv              bout-to-bout transition counts and probabilities
  group_stats.csv              usage differences between conditions (when sessions.csv has them)
  agreement.csv                syllable x expert label overlap (when the session NWB has labels)
  ethogram.png, syllable_poses.png, syllable_usage.png, transitions.png, agreement.png, ...
  model.json, summary.json, provenance.json
"""
from __future__ import annotations

import csv
import logging
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from processor.common import peak_rss_mb, set_threads, writable_caches, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("behavior-syllables")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "h5py", "hdmf", "pynwb", "jax", "absl"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "behavior-syllables"
REPO_ROOT = Path(__file__).resolve().parent.parent
CITATIONS = {"hmm": ["10.1016/j.neuron.2015.11.031", "10.1214/10-AOAS395", "10.21105/joss.07069"],
             "kpms": ["10.1038/s41592-024-02318-2", "10.1016/j.neuron.2015.11.031"]}

DEFAULTS = {
    "METHOD": os.environ.get("DEFAULT_METHOD", "hmm"),
    "N_SYLLABLES": "0",
    "MAX_SYLLABLES": "12",
    "N_PCS": "0",
    "USE_KEYPOINTS": "auto",
    "MIN_DURATION_S": "0.2",
    "MAX_ITER": "100",
    "STICKINESS": "100000",
    "KAPPA": "1000000",
    "KPMS_ITERS": "200",
    "MIN_USAGE": "0.005",
    "LABEL_TABLES": "auto",
    "GROUP_BY": "condition",
    "USE_GPU": "auto",
    "SEED": "42",
    "THREADS": "0",
}


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p = {}
    try:
        for k in ("N_SYLLABLES", "MAX_SYLLABLES", "N_PCS", "MAX_ITER", "KPMS_ITERS", "SEED", "THREADS"):
            p[k] = int(raw[k])
        p["MIN_DURATION_S"] = float(raw["MIN_DURATION_S"])
        p["MIN_USAGE"] = float(raw["MIN_USAGE"])
        p["KAPPA"] = float(raw["KAPPA"])
        p["STICKINESS"] = float(raw["STICKINESS"])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    p["METHOD"] = raw["METHOD"].strip().lower()
    if p["METHOD"] not in ("hmm", "kpms"):
        raise ParameterError("METHOD must be hmm or kpms")
    p["USE_GPU"] = raw["USE_GPU"].strip().lower()
    if p["USE_GPU"] not in ("auto", "true", "false"):
        raise ParameterError("USE_GPU must be auto, true or false")
    if p["N_SYLLABLES"] < 0 or p["N_SYLLABLES"] == 1 or p["N_SYLLABLES"] > 50:
        raise ParameterError("N_SYLLABLES must be 0 (choose) or 2-50")
    if not 4 <= p["MAX_SYLLABLES"] <= 50:
        raise ParameterError("MAX_SYLLABLES must be 4-50")
    if p["STICKINESS"] < 0:
        raise ParameterError("STICKINESS must be >= 0")
    if not 0 <= p["MIN_USAGE"] < 0.2:
        raise ParameterError("MIN_USAGE must be in [0, 0.2)")
    if p["N_PCS"] < 0 or p["MAX_ITER"] < 1 or p["KPMS_ITERS"] < 1 or p["MIN_DURATION_S"] < 0 or p["KAPPA"] <= 0 \
            or p["THREADS"] < 0:
        raise ParameterError("N_PCS, MIN_DURATION_S, THREADS >= 0; MAX_ITER, KPMS_ITERS >= 1; KAPPA > 0")
    p["USE_KEYPOINTS"] = raw["USE_KEYPOINTS"].strip()
    lt = raw["LABEL_TABLES"].strip()
    p["LABEL_TABLES"] = None if lt.lower() == "auto" else [x.strip() for x in lt.split(",") if x.strip()]
    if lt.lower() == "none":
        p["LABEL_TABLES"] = []
    p["GROUP_BY"] = raw["GROUP_BY"].strip() or "condition"
    return p


def app_version():
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions(method):
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    pkgs = ["dynamax", "jax", "jaxlib", "scikit-learn", "numpy", "scipy", "pandas", "pyarrow", "pynwb", "hdmf", "matplotlib"]
    if method == "kpms":
        pkgs += ["keypoint-moseq", "jax-moseq"]
    for pkg in pkgs:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _write_csv(path, rows, columns=None):
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)) for k, v in r.items()})


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    writable_caches()
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    if params["METHOD"] == "kpms":
        try:
            import keypoint_moseq  # noqa: F401
        except ImportError:
            log.error("METHOD=kpms needs the kpms variant of this app (keypoint-MoSeq is not in this image)")
            return 2
    threads = set_threads(params["THREADS"])
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1

    import numpy as np
    import pandas as pd

    from processor import analysis as A
    from processor import figures as F
    from processor import models as Mo
    from processor.features import FeatureBuilder, choose_keypoints, segments
    from processor.inputs import discover, label_matrix, match_label_nwbs, read_labels
    from processor.nwbout import write_syllables_nwb

    found = discover(input_dir)
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.sessions:
        log.error("no <session_id>_kinematics.parquet (pose-kinematics output) under %s", input_dir)
        return 1
    top = output_dir / APP_NAME
    top.mkdir(parents=True, exist_ok=True)
    warnings_ = []
    statuses = {}
    data = {}
    for s in found.sessions:
        t = time.monotonic()
        try:
            df = pd.read_parquet(s.path)
            if len(df) < 30:
                raise ValueError(f"only {len(df)} frames")
            for col in ("time", "centroid_x", "centroid_y", "heading", "speed", "angular_speed"):
                if col not in df.columns:
                    raise ValueError(f"column {col} missing (not a pose-kinematics kinematics.parquet?)")
            if df["heading"].isna().all():
                raise ValueError("no heading (needs head and tail keypoints in pose-kinematics)")
            data[s.session_id] = (df, s.meta)
            statuses[s.session_id] = {"status": "ok", "seconds": round(time.monotonic() - t, 2)}
        except Exception as exc:
            log.error("%s: failed: %s", s.session_id, exc)
            statuses[s.session_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if not data:
        log.error("every session failed")
        return 1
    units = {m.get("units", "cm") for _, m in data.values()}
    if len(units) > 1:
        warnings_.append("sessions differ in units (cm and px): posture features are not comparable; give "
                         "pose-kinematics ARENA_SIZE_CM or PX_PER_CM for every session")
    unit = units.pop() if len(units) == 1 else "mixed"
    try:
        keypoints = choose_keypoints(list(data.values()), params["USE_KEYPOINTS"])
    except ValueError as exc:
        log.error("%s", exc)
        return 2 if params["USE_KEYPOINTS"] != "auto" else 1
    fps_all = [float(1 / np.median(np.diff(df["time"].to_numpy()))) for df, _ in data.values()]
    fps = float(np.median(fps_all))
    min_len = max(1, int(round(params["MIN_DURATION_S"] * fps)))
    sids = sorted(data)
    log.info("%d sessions, %d frames, %.2f fps, %d keypoints (%s), method %s, %d threads", len(sids),
             sum(len(data[s][0]) for s in sids), fps, len(keypoints), ", ".join(keypoints), params["METHOD"], threads)

    fb = FeatureBuilder(keypoints, params["N_PCS"], derivatives=False).fit([data[s][0] for s in sids])
    log.info("posture PCA: %d components explain %.0f%% of egocentric pose variance", fb.n_pcs,
             100 * fb.explained.sum())
    labels = {}
    model_info = {}
    t_fit = time.monotonic()
    if params["METHOD"] == "hmm":
        seqs, where = [], []
        for sid in sids:
            X = fb.transform(data[sid][0])
            valid = np.isfinite(X).all(axis=1)
            for a, b in segments(valid, 10):
                seqs.append(X[a:b])
                where.append((sid, a, b))
            data[sid] = data[sid] + (valid,)
        log.info("%d frames with a few occluded keypoints filled with their mean egocentric position",
                 fb.n_imputed)
        if not seqs:
            log.error("no frames with complete features")
            return 1
        lab, hmm_params, model_info = Mo.fit_arhmm(seqs, fps, params["N_SYLLABLES"], params["MAX_SYLLABLES"],
                                                   params["SEED"], params["MAX_ITER"], params["STICKINESS"])
        for sid in sids:
            labels[sid] = np.full(len(data[sid][0]), -1, int)
        raw_durations = []
        for (sid, a, b), z in zip(where, lab):
            t_seg = data[sid][0]["time"].to_numpy(float)[a:b]
            raw_durations += [t_seg[e - 1] - t_seg[st] + 1 / fps for _, st, e in A.bouts(np.asarray(z), t_seg)]
            labels[sid][a:b] = Mo.merge_short_bouts(z, min_len)
    else:
        sess = {}
        meta0 = data[sids[0]][1]
        for sid in sids:
            df, meta = data[sid]
            scale = np.array(meta.get("px_per_cm") or [1.0, 1.0]) if meta.get("units") == "cm" else np.ones(2)
            xy = np.stack([np.column_stack([df[f"kp_{k}_x"], df[f"kp_{k}_y"]]) for k in keypoints], axis=1) * scale
            valid = np.isfinite(xy).all(axis=2)
            sess[sid] = (xy, valid)
            data[sid] = data[sid] + (valid.all(axis=1) & np.isfinite(df["heading"].to_numpy()),)
        ant = [k for k in (meta0.get("head_keypoints") or []) + [meta0.get("nose_keypoint")] if k in keypoints] or keypoints[:1]
        post = [k for k in [meta0.get("tail_keypoint")] if k in keypoints] or keypoints[-1:]
        try:
            zs, model_info = Mo.fit_kpms(sess, keypoints, ant, post, fps, params["KAPPA"], params["KPMS_ITERS"],
                                         params["SEED"], Path(tempfile.mkdtemp(prefix="kpms-")), params["USE_GPU"])
        except RuntimeError as exc:
            log.error("%s", exc)
            return 3
        for sid in sids:
            z = np.asarray(zs[sid], int)[: len(data[sid][0])]
            z = np.r_[z, np.full(len(data[sid][0]) - len(z), -1)]
            z[~data[sid][2]] = -1
            labels[sid] = z
        raw_durations = [b - a + 1 / fps for sid in sids
                         for _, st, e in A.bouts(labels[sid], data[sid][0]["time"].to_numpy(float))
                         for a, b in [(data[sid][0]["time"].iloc[st], data[sid][0]["time"].iloc[e - 1])]]
    t_fit = time.monotonic() - t_fit
    # Median syllable (bout) duration straight from the model, before short bouts are merged: keypoint
    # jitter shows up as implausibly short syllables (Weinreb et al. 2024); MoSeq syllables last ~0.3-0.5 s.
    median_raw = float(np.median(raw_durations)) if raw_durations else None
    if median_raw is not None and median_raw < 0.2:
        warnings_.append(f"median syllable duration from the model is {1000 * median_raw:.0f} ms (< 200 ms): "
                         "syllables this short usually follow tracking jitter, not behaviour; raise STICKINESS "
                         "(hmm) or KAPPA (kpms), or check pose quality")
    log.info("median syllable duration from the model: %.0f ms", 1000 * (median_raw or 0))
    labels, mapping = A.relabel_by_usage(labels, params["MIN_USAGE"])
    n_syl = len(set(mapping.values()))
    rare_id = n_syl - 1 if len(set(mapping.values())) < len(mapping) else None
    log.info("%d syllables in use after merging bouts shorter than %d frames (%.2f s); fit %.1fs", n_syl, min_len,
             params["MIN_DURATION_S"], t_fit)

    # Per-session outputs.
    usage_rows = []
    sheet = {s.session_id: s.sheet for s in found.sessions}
    for sid in sids:
        df, meta = data[sid][0], data[sid][1]
        t = df["time"].to_numpy(float)
        z = labels[sid]
        pd.DataFrame({"time": t, "syllable": z}).to_csv(top / f"{sid}_syllables.csv", index=False, float_format="%.6f")
        write_syllables_nwb(top / f"{sid}_syllables.nwb", sid, meta.get("nwb") or {}, t, z,
                            model_info.get("method", params["METHOD"]), app_version())
        rows = A.per_session_usage(sid, z, t, n_syl)
        for r in rows:
            r.update({k: v for k, v in sheet.get(sid, {}).items() if k in ("subject_id", "condition", "group")})
        usage_rows += rows
        nb = len(A.bouts(z, t))
        log.info("%s: %d frames, %.0f%% modelled, %d bouts, %d syllables used", sid, len(z), 100 * (z >= 0).mean(), nb,
                 len(set(z[z >= 0].tolist())))

    table = A.syllable_table({sid: (labels[sid], data[sid][0]["time"].to_numpy(float), data[sid][0]) for sid in sids},
                             n_syl)
    C, P = A.transitions(labels, n_syl)
    _write_csv(top / "transitions.csv", [{"from_syllable": i, "to_syllable": j, "count": int(C[i, j]),
                                          "probability": float(P[i, j])} for i in range(n_syl) for j in range(n_syl)
                                         if i != j])
    group_col = params["GROUP_BY"]
    if not any(sheet.get(s, {}).get(group_col) for s in sids) and group_col == "condition" and \
            any(sheet.get(s, {}).get("group") for s in sids):
        group_col = "group"
    stats_rows, groups = A.group_usage_stats(usage_rows, sheet, group_col, n_syl)
    if stats_rows:
        _write_csv(top / "group_stats.csv", stats_rows)
        for r in table:
            for g in groups:
                st = next(x for x in stats_rows if x["syllable"] == r["syllable"])
                r[f"usage_{g}"] = st.get(f"mean_{g}")
            r["q_value"] = next(x for x in stats_rows if x["syllable"] == r["syllable"]).get("q_value")
    _write_csv(top / "usage.csv", usage_rows)

    # Agreement with labelled behaviour.
    agreement_summary, agree_rows, zoom = None, [], None
    if params["LABEL_TABLES"] != []:
        nwb_for = match_label_nwbs(found)
        z_all, M_all, names_all, zs_by, prim_by = [], [], None, {}, {}
        per_session = {}
        for sid in sids:
            if sid not in nwb_for:
                continue
            try:
                labs = read_labels(nwb_for[sid], params["LABEL_TABLES"])
            except Exception as exc:
                warnings_.append(f"{sid}: labels in {nwb_for[sid].name} not readable: {exc}")
                continue
            if not labs:
                continue
            t = data[sid][0]["time"].to_numpy(float)
            names, M = label_matrix(labs, t)
            per_session[sid] = (names, M, nwb_for[sid].name)
        all_names = sorted({n for v in per_session.values() for n in v[0]})
        for sid, (names, M, _) in per_session.items():
            full = np.zeros((len(M), len(all_names)), bool)
            for j, n in enumerate(names):
                full[:, all_names.index(n)] = M[:, j]
            z_all.append(labels[sid])
            M_all.append(full)
            zs_by[sid] = labels[sid]
            prim_by[sid] = A.primary_label(all_names, full)
            per_session[sid] = (all_names, full, per_session[sid][2])
        if z_all:
            agree_rows, agreement_summary = A.agreement(np.concatenate(z_all), all_names, np.concatenate(M_all),
                                                        params["SEED"], zs_by, prim_by)
            from sklearn.metrics import adjusted_mutual_info_score

            agreement_summary["per_session_ami"] = {
                sid: float(adjusted_mutual_info_score(prim_by[sid][labels[sid] >= 0], labels[sid][labels[sid] >= 0]))
                for sid in zs_by}
            agreement_summary["label_sources"] = {sid: v[2] for sid, v in per_session.items()}
            _write_csv(top / "agreement.csv", agree_rows)
            # The zoom shows the labelled session with the most modelled frames.
            zoom = max(sorted(per_session), key=lambda sid: float((labels[sid] >= 0).mean()))
            log.info("agreement with expert labels (%s) over %d sessions: AMI %.3f (time-shifted null %.3f)",
                     ", ".join(all_names), len(zs_by), agreement_summary["ami"],
                     agreement_summary["ami_shifted_null_mean"] or float("nan"))
            for r in table:
                best = max((a for a in agree_rows if a["syllable"] == r["syllable"]),
                           key=lambda a: a["p_label_given_syllable"] or 0, default=None)
                if best:
                    r["top_label"] = best["label"]
                    r["top_label_fraction"] = best["p_label_given_syllable"]
            agreement_summary["per_session_names"] = None
        else:
            log.info("no labelled behaviour intervals found in a session NWB; agreement.csv not written")
    _write_csv(top / "syllables.csv", table)

    # Figures.
    figures = []
    try:
        if agreement_summary:
            p = F.agreement_heatmap(agree_rows, agreement_summary["labels"], n_syl, agreement_summary, top / "agreement.png")
            figures.append({"path": f"{APP_NAME}/{p.name}", "caption":
                            f"How the unsupervised syllables line up with the experts' behaviour labels (no labels "
                            f"were used to find the syllables). Each row is a syllable, each column a label; the "
                            f"number is the fraction of that syllable's frames the experts gave that label. A row "
                            f"with one dark cell is a syllable that matches one behaviour. Adjusted mutual "
                            f"information over all labelled frames: {agreement_summary['ami']:.2f} (0 = chance, "
                            f"1 = identical); with the syllable sequence shifted in time it is "
                            f"{agreement_summary['ami_shifted_null_mean'] or 0:.2f}."})
            names, M, _ = per_session[zoom]
            p = F.ethogram_zoom(zoom, data[zoom][0]["time"].to_numpy(float), labels[zoom], names, M, n_syl, 120,
                                top / "ethogram_vs_labels.png")
            figures.append({"path": f"{APP_NAME}/{p.name}", "caption":
                            f"{zoom}, first two minutes: the syllable sequence (top, colour = syllable) above the "
                            f"experts' labels (grey = labelled; white in the syllable row = pose missing, not modelled). Syllable switches that coincide with label "
                            f"onsets show the model finds the same behavioural boundaries the experts marked."})
        p = F.ethogram([(sid, data[sid][0]["time"].to_numpy(float), labels[sid], sheet.get(sid, {}).get(group_col))
                        for sid in sids], n_syl, top / "ethogram.png")
        figures.append({"path": f"{APP_NAME}/{p.name}", "caption":
                        "Ethogram: the syllable the model assigns to every pose frame, one row per session (colour = "
                        "syllable, 0 = the most used overall). Long stretches of one colour are sustained "
                        "behaviours (resting, locomotion bouts); rapid alternation is fine-grained movement such as "
                        "exploration of the walls."})
        mean_poses = []
        for s in range(n_syl):
            acc = []
            for sid in sids:
                mp = fb.egocentric_mean(data[sid][0], labels[sid] == s)
                if mp is not None:
                    acc.append((mp, int((labels[sid] == s).sum())))
            mean_poses.append(np.average([a for a, _ in acc], axis=0, weights=[w for _, w in acc]) if acc else None)
        edges = [tuple(e) for e in (data[sids[0]][1].get("edges") or [])]
        if not edges:
            edges = _default_edges(keypoints, data[sids[0]][1])
        p = F.syllable_poses(mean_poses, keypoints, edges, table, unit, top / "syllable_poses.png")
        figures.append({"path": f"{APP_NAME}/{p.name}", "caption":
                        "What each syllable looks like: the average body posture during the syllable, seen from "
                        "above with the animal facing right, with its share of time, mean bout length and mean "
                        "speed. Elongated postures with high speed are locomotion; compact, slow ones are rest or "
                        "grooming; a short body with the nose near the body centre is typical of rearing."})
        if stats_rows:
            n_g = {g: stats_rows[0].get(f"n_{g}") for g in groups}
            sig = [r["syllable"] for r in stats_rows if r.get("q_value") is not None and r["q_value"] < 0.05]
            p = F.usage_by_group(usage_rows, sheet, group_col, groups, stats_rows, n_syl, top / "syllable_usage.png")
            figures.append({"path": f"{APP_NAME}/{p.name}", "caption":
                            f"Share of time in each syllable per session (dots) and group mean (bars) by {group_col} "
                            f"({', '.join(f'{g} n={n}' for g, n in n_g.items())} sessions). "
                            f"{stats_rows[0]['test']} per syllable, Benjamini-Hochberg across syllables; * = q < 0.05"
                            f" ({'syllables ' + ', '.join(map(str, sig)) if sig else 'none'}); Hedges g effect sizes in "
                            f"group_stats.csv. With this few sessions per group the comparison is underpowered: a "
                            f"real difference can easily stay non-significant, so read the effect sizes."})
        else:
            p = F.usage_by_group(usage_rows, sheet, group_col, [], [], n_syl, top / "syllable_usage.png")
            figures.append({"path": f"{APP_NAME}/{p.name}", "caption": "Share of time in each syllable per session."})
        p = F.transition_matrix(P, top / "transitions.png")
        figures.append({"path": f"{APP_NAME}/{p.name}", "caption":
                        "Which syllable follows which: probability of moving from the row syllable to the column "
                        "syllable at the end of a bout (self-transitions excluded). Strong off-diagonal cells are "
                        "stereotyped sequences, e.g. rearing followed by a turn."})
        if model_info.get("heldout_loglik_per_frame"):
            p = F.model_selection(model_info["heldout_loglik_per_frame"], model_info["n_states"],
                                  top / "model_selection.png")
            figures.append({"path": f"{APP_NAME}/{p.name}", "caption":
                            "How the number of syllables was chosen: log-likelihood of held-out data (every fifth "
                            "minute) for models with 4, 6, ... states; the dashed line is the smallest number that "
                            "reaches 90% of the best improvement over 4 states."})
    except Exception as exc:
        log.exception("figures failed: %s", exc)

    model_doc = {"method": params["METHOD"], "info": model_info, "keypoints": keypoints, "fps": fps,
                 "n_syllables": n_syl, "relabel_by_usage": mapping, "min_duration_frames": min_len,
                 "features": fb.names, "pca_components": fb.pca_components, "pca_mean": fb.pca_mean,
                 "pca_explained_variance_ratio": fb.explained, "feature_mean": fb.mu, "feature_sd": fb.sd}
    if params["METHOD"] == "hmm":
        model_doc["arhmm"] = hmm_params
    write_json(top / "model.json", model_doc)
    summary = {
        "app": APP_NAME, "version": app_version(), "method": params["METHOD"],
        "n_syllables": n_syl, "rare_syllable": rare_id,
        "median_syllable_duration_s": median_raw,
        "median_syllable_duration_after_merge_s": float(np.median([r for sid in sids for r in _bout_durations(A, labels[sid], data[sid][0]["time"].to_numpy(float))])) if sids else None,
        "short_syllable_warning": bool(median_raw is not None and median_raw < 0.2), "keypoints": keypoints, "units": unit,
        "sessions": {sid: {"n_frames": int(len(labels[sid])), "modelled_fraction": float((labels[sid] >= 0).mean()),
                           "n_bouts": len(A.bouts(labels[sid], data[sid][0]["time"].to_numpy(float))),
                           "usage_time": {r["syllable"]: r["usage_time"] for r in usage_rows if r["session_id"] == sid}}
                     | {k: v for k, v in sheet.get(sid, {}).items() if k in ("subject_id", "condition", "group")}
                     for sid in sids},
        "syllables": table,
        "agreement": agreement_summary,
        "group_column": group_col if stats_rows else None,
        "group_stats": stats_rows,
        "model": {k: v for k, v in model_info.items()},
        "figures": figures,
        "tables": [f"{APP_NAME}/{n}" for n in ("syllables.csv", "usage.csv", "transitions.csv")]
                  + ([f"{APP_NAME}/group_stats.csv"] if stats_rows else [])
                  + ([f"{APP_NAME}/agreement.csv"] if agreement_summary else []),
        "warnings": warnings_,
    }
    write_json(top / "summary.json", summary)
    variant = os.environ.get("APP_VARIANT", params["METHOD"])
    write_json(top / "provenance.json", {
        "app": APP_NAME, "version": app_version(), "variant": variant, "tools": tool_versions(params["METHOD"]),
        "parameters": params, "inputs": sorted(str(s.path.relative_to(input_dir)) for s in found.sessions),
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.monotonic() - t0, 2), "peak_rss_mb": peak_rss_mb(),
        "gpu": _gpu_info(params["METHOD"]), "sessions": statuses, "citations": CITATIONS[params["METHOD"]],
        "threads": threads, "skipped": found.skipped,
    })
    for w in warnings_:
        log.warning("%s", w)
    log.info("done: %d sessions, %d syllables, %.1fs, peak RSS %s MB -> %s", len(sids), n_syl, time.monotonic() - t0,
             peak_rss_mb(), top)
    return 0


def _bout_durations(A, z, t):
    dur = A.frame_durations(t)
    return [t[e - 1] - t[st] + dur[e - 1] for _, st, e in A.bouts(z, t)] or [float("nan")]


def _default_edges(keypoints, meta):
    chain = [k for k in [meta.get("nose_keypoint")] + list(meta.get("head_keypoints") or [])
             + list(meta.get("centroid_keypoints") or [])[:1] + [meta.get("tail_keypoint")] if k in keypoints]
    chain = list(dict.fromkeys(chain))
    return list(zip(chain[:-1], chain[1:]))


def _gpu_info(method):
    if method != "kpms":
        return None
    try:
        import jax

        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpus:
            return None
        stats = gpus[0].memory_stats() or {}
        return {"name": gpus[0].device_kind, "peak_vram_mb": round(stats.get("peak_bytes_in_use", 0) / 2 ** 20)}
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
