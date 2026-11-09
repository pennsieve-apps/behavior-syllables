"""Usage, bouts, transitions, group differences and agreement with labelled behaviour."""
from __future__ import annotations

import numpy as np


def bouts(z, t):
    """[(syllable, start_idx, end_idx_exclusive)] of runs, skipping -1 (no model)."""
    out = []
    if not len(z):
        return out
    change = np.r_[True, z[1:] != z[:-1]]
    starts = np.where(change)[0]
    ends = np.r_[starts[1:], len(z)]
    for a, b in zip(starts, ends):
        if z[a] >= 0:
            out.append((int(z[a]), int(a), int(b)))
    return out


def frame_durations(t):
    d = np.diff(t)
    return np.r_[d, np.median(d) if len(d) else 0.0]


def relabel_by_usage(labels: dict, min_usage: float = 0.0) -> tuple[dict, dict]:
    """Syllable 0 = the most used (by frames), 1 the next, ...; -1 stays -1.

    Syllables used in less than `min_usage` of all modelled frames are pooled into one last
    syllable ("rare"), as keypoint-MoSeq users usually do (kpms keeps up to 100 states, most of
    them nearly unused)."""
    allz = np.concatenate([z[z >= 0] for z in labels.values()]) if labels else np.array([], int)
    ids, counts = np.unique(allz, return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    frac = counts[np.argsort(-counts, kind="stable")] / max(1, counts.sum())
    major = [int(o) for o, f in zip(order, frac) if f >= min_usage]
    rare = [int(o) for o, f in zip(order, frac) if f < min_usage]
    mapping = {o: i for i, o in enumerate(major)}
    for o in rare:
        mapping[o] = len(major)
    out = {}
    for sid, z in labels.items():
        nz = np.full_like(z, -1)
        for o, n in mapping.items():
            nz[z == o] = n
        out[sid] = nz
    return out, mapping


def per_session_usage(sid, z, t, n_syl):
    dur = frame_durations(t)
    valid = z >= 0
    total = dur[valid].sum()
    b = bouts(z, t)
    nb = len(b)
    rows = []
    for s in range(n_syl):
        m = z == s
        sb = [x for x in b if x[0] == s]
        rows.append({"session_id": sid, "syllable": s, "seconds": float(dur[m].sum()),
                     "usage_time": float(dur[m].sum() / total) if total else 0.0, "n_bouts": len(sb),
                     "usage_bouts": len(sb) / nb if nb else 0.0,
                     "bouts_per_minute": len(sb) / (total / 60) if total else 0.0})
    return rows


def syllable_table(sessions, n_syl):
    """sessions: {sid: (z, t, df)}. One row per syllable, pooled over sessions."""
    rows = []
    allb = {s: [] for s in range(n_syl)}
    feats = {s: {"speed": [], "angular_speed": [], "body_length": []} for s in range(n_syl)}
    total_t, total_b = 0.0, 0
    sec = np.zeros(n_syl)
    for sid, (z, t, df) in sessions.items():
        dur = frame_durations(t)
        total_t += dur[z >= 0].sum()
        b = bouts(z, t)
        total_b += len(b)
        for s, a, e in b:
            allb[s].append(t[e - 1] - t[a] + dur[e - 1])
        for s in range(n_syl):
            m = z == s
            sec[s] += dur[m].sum()
            for f in feats[s]:
                if f in df.columns:
                    v = df[f].to_numpy(float)[m]
                    feats[s][f].append(np.abs(v) if f == "angular_speed" else v)
    for s in range(n_syl):
        d = np.array(allb[s])
        row = {"syllable": s, "n_bouts": len(d), "seconds": float(sec[s]),
               "usage_time": float(sec[s] / total_t) if total_t else 0.0,
               "usage_bouts": len(d) / total_b if total_b else 0.0,
               "mean_duration_s": float(d.mean()) if len(d) else None,
               "median_duration_s": float(np.median(d)) if len(d) else None}
        for f, vals in feats[s].items():
            v = np.concatenate(vals) if vals else np.array([])
            v = v[np.isfinite(v)]
            key = "mean_abs_angular_speed" if f == "angular_speed" else f"mean_{f}"
            row[key] = float(v.mean()) if len(v) else None
        rows.append(row)
    return rows


def transitions(labels: dict, n_syl):
    """Bout-to-bout transitions (self-transitions excluded by construction), pooled over sessions."""
    C = np.zeros((n_syl, n_syl), int)
    for z in labels.values():
        # Consecutive frames with different syllables; a gap (-1) between two bouts breaks the chain.
        a, b = z[:-1], z[1:]
        m = (a != b) & (a >= 0) & (b >= 0)
        np.add.at(C, (a[m], b[m]), 1)
    P = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    return C, P


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    n = len(q)
    order = np.argsort(q)
    ranked = q[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(ranked, 1.0)
    out[ok] = res
    return out


def hedges_g(a, b):
    """Standardised mean difference (a - b) with the small-sample correction (Hedges & Olkin)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))) if sp > 0 else None


def group_usage_stats(usage_rows, sheet, group_col, n_syl):
    """Per syllable: usage_time by group; Welch t (2 groups) or one-way ANOVA (3+), BH over syllables."""
    from scipy import stats

    groups = sorted({sheet.get(r["session_id"], {}).get(group_col) for r in usage_rows} - {None, ""})
    if len(groups) < 2:
        return [], groups
    out = []
    for s in range(n_syl):
        vals = {g: np.array([r["usage_time"] for r in usage_rows if r["syllable"] == s
                             and sheet.get(r["session_id"], {}).get(group_col) == g]) for g in groups}
        row = {"syllable": s, "group_column": group_col}
        for g in groups:
            row[f"n_{g}"] = len(vals[g])
            row[f"mean_{g}"] = float(vals[g].mean()) if len(vals[g]) else None
            row[f"sd_{g}"] = float(vals[g].std(ddof=1)) if len(vals[g]) > 1 else None
        p = p2 = stat = None
        test = None
        usable = [vals[g] for g in groups if len(vals[g]) >= 2]
        if len(groups) == 2 and len(usable) == 2:
            a, b = vals[groups[0]], vals[groups[1]]
            if np.ptp(np.r_[a, b]) > 0:
                r = stats.ttest_ind(a, b, equal_var=False)
                stat, p = float(r.statistic), float(r.pvalue)
                p2 = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                row["hedges_g"] = hedges_g(a, b)
            test = "Welch t-test"
        elif len(usable) == len(groups) >= 3:
            r = stats.f_oneway(*usable)
            stat, p = float(r.statistic), float(r.pvalue)
            p2 = float(stats.kruskal(*usable).pvalue)
            test = "one-way ANOVA"
        row.update(test=test, statistic=stat, p_value=p, p_value_secondary=p2,
                   test_secondary="Mann-Whitney U" if len(groups) == 2 else "Kruskal-Wallis")
        out.append(row)
    q = bh([r["p_value"] if r["p_value"] is not None else np.nan for r in out])
    for r, qq in zip(out, q):
        r["q_value"] = None if not np.isfinite(qq) else float(qq)
    return out, groups


# ----------------------------------------------------------------------------- agreement

def primary_label(names, M):
    """One label per frame: among the labels a frame has, the rarest overall (the most specific:
    'rear' over 'walk'); 'none' when it has none."""
    freq = M.sum(axis=0)
    order = np.argsort(freq)          # rarest first
    out = np.array(["none"] * len(M), dtype=object)
    for j in order[::-1]:             # write commonest first, rarest last (wins)
        out[M[:, j]] = names[j]
    return out


def agreement(z_all, names, M_all, seed, z_by_session=None, prim_by_session=None):
    """Pooled over the labelled sessions. Returns rows per (syllable, label) and a summary."""
    from sklearn.metrics import adjusted_mutual_info_score

    ok = z_all >= 0
    z, M = z_all[ok], M_all[ok]
    rows = []
    n = len(z)
    syls = sorted(set(z.tolist()))
    for s in syls:
        ms = z == s
        for j, lab in enumerate(names):
            ml = M[:, j]
            inter = int((ms & ml).sum())
            union = int((ms | ml).sum())
            p_l = ml.mean() if n else 0
            rows.append({"syllable": int(s), "label": lab, "n_frames_syllable": int(ms.sum()),
                         "n_frames_label": int(ml.sum()), "n_frames_both": inter,
                         "p_label_given_syllable": inter / ms.sum() if ms.sum() else None,
                         "p_syllable_given_label": inter / ml.sum() if ml.sum() else None,
                         "jaccard": inter / union if union else None,
                         "enrichment": (inter / ms.sum()) / p_l if ms.sum() and p_l else None})
    prim = primary_label(names, M)
    ami = float(adjusted_mutual_info_score(prim, z))
    # Temporal-structure null: syllable sequences circularly shifted within each session by >= 60 s.
    rng = np.random.default_rng(seed)
    null = []
    if z_by_session and prim_by_session:
        for _ in range(20):
            zs, ps = [], []
            for sid, zz in z_by_session.items():
                okk = zz >= 0
                zz, pp = zz[okk], prim_by_session[sid][okk]
                if len(zz) < 10:
                    continue
                shift = int(rng.integers(len(zz) // 4, max(len(zz) // 4 + 1, 3 * len(zz) // 4)))
                zs.append(np.roll(zz, shift))
                ps.append(pp)
            if zs:
                null.append(adjusted_mutual_info_score(np.concatenate(ps), np.concatenate(zs)))
    best = {}
    for lab in names:
        cand = [r for r in rows if r["label"] == lab and r["p_label_given_syllable"] is not None]
        if cand:
            r = max(cand, key=lambda r: (r["jaccard"] or 0))
            best[lab] = {"syllable": r["syllable"], "jaccard": r["jaccard"],
                         "p_label_given_syllable": r["p_label_given_syllable"],
                         "p_syllable_given_label": r["p_syllable_given_label"]}
    summary = {"ami": ami, "ami_shifted_null_mean": float(np.mean(null)) if null else None,
               "ami_shifted_null_max": float(np.max(null)) if null else None, "n_frames": int(n),
               "labels": names, "label_fraction": {lab: float(M[:, j].mean()) for j, lab in enumerate(names)},
               "best_syllable_per_label": best,
               "primary_label_rule": "per frame, the rarest of its labels (most specific); 'none' if unlabelled"}
    return rows, summary
