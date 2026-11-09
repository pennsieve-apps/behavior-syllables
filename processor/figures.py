"""PNG figures, 150 dpi."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

DPI = 150
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Syllables beyond eight: darker / lighter steps of the same hues, so every syllable has a colour; the
# ethogram also prints syllable numbers in its legend, so identity is never colour alone.
EXTRA = ["#1b4f8f", "#a8441d", "#127552", "#9e6b00", "#b04a75", "#004d00", "#3f3191", "#9c2c2b",
         "#8db8ea", "#f4ac8c", "#8bd8bb", "#f6cf7a", "#f1b3cb", "#7fc17f", "#a89ee9", "#f0a1a0"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
GROUP = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10})


def colour(s):
    pal = CAT + EXTRA
    return pal[s % len(pal)]


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]", "-", s)


def _cmap(n):
    return ListedColormap(["#ffffff"] + [colour(i) for i in range(n)])


def ethogram(sessions, n_syl, out, max_sessions=12):
    """sessions: [(sid, t, z, group)]."""
    sessions = sessions[:max_sessions]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(sessions) + 1.5))
    cmap = _cmap(n_syl)
    for i, (sid, t, z, g) in enumerate(sessions):
        ax.imshow((z + 1)[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=n_syl, interpolation="nearest",
                  extent=(t[0], t[-1], i + 0.4, i - 0.4))
    ax.set_yticks(range(len(sessions)))
    ax.set_yticklabels([f"{sid}" + (f" ({g})" if g else "") for sid, _, _, g in sessions], fontsize=7)
    ax.set_ylim(len(sessions) - 0.5, -0.5)
    ax.set_xlim(min(s[1][0] for s in sessions), max(s[1][-1] for s in sessions))
    ax.set_xlabel("time (s, session clock)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour(s)) for s in range(n_syl)]
    ax.legend(handles, [f"{s}" for s in range(n_syl)], title="syllable", ncol=min(n_syl, 16), fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.25 if len(sessions) > 3 else -0.45), frameon=False)
    ax.set_title(f"Ethogram: syllable over time, {len(sessions)} sessions (white = no pose / not modelled)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def ethogram_zoom(sid, t, z, names, M, n_syl, window_s, out):
    """The first `window_s` seconds of one session: syllables above, the expert labels below."""
    end = t[0] + window_s
    m = t <= end
    rows = 1 + len(names)
    fig, ax = plt.subplots(figsize=(10, 0.32 * rows + 1.4))
    ax.imshow((z[m] + 1)[None, :], aspect="auto", cmap=_cmap(n_syl), vmin=0, vmax=n_syl, interpolation="nearest",
              extent=(t[m][0], t[m][-1], 0.4, -0.4))
    for j, lab in enumerate(names):
        ax.imshow(np.where(M[m, j], 1, 0)[None, :], aspect="auto", cmap=ListedColormap(["#ffffff", MUTED]),
                  vmin=0, vmax=1, interpolation="nearest", extent=(t[m][0], t[m][-1], j + 1.4, j + 0.6))
    ax.set_yticks(range(rows))
    ax.set_yticklabels(["syllable"] + [f"label: {n}" for n in names], fontsize=7)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlabel("time (s, session clock)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour(s)) for s in range(n_syl)]
    ax.legend(handles, [str(s) for s in range(n_syl)], title="syllable", ncol=min(n_syl, 16), fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False)
    ax.set_title(f"{sid}: syllables and expert labels, first {window_s:.0f} s")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def agreement_heatmap(rows, names, n_syl, summary, out):
    P = np.full((n_syl, len(names)), np.nan)
    usage = np.zeros(n_syl)
    for r in rows:
        if r["p_label_given_syllable"] is not None:
            P[r["syllable"], names.index(r["label"])] = r["p_label_given_syllable"]
            usage[r["syllable"]] = r["n_frames_syllable"]
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2.6, 0.36 * n_syl + 1.8))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(n_syl):
        for j in range(len(names)):
            if np.isfinite(P[i, j]):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if P[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(n_syl))
    ax.set_yticklabels([f"syllable {i}" for i in range(n_syl)], fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("fraction of the syllable's frames\nwith this expert label")
    null = summary.get("ami_shifted_null_mean")
    ax.set_title(f"Syllables vs expert labels: AMI {summary['ami']:.2f}"
                 + (f" (time-shifted null {null:.2f})" if null is not None else ""))
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def usage_by_group(usage_rows, sheet, group_col, groups, stats_rows, n_syl, out):
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * n_syl + 1.5), 3.6))
    groups = groups or [None]
    w = 0.8 / len(groups)
    rng = np.random.default_rng(0)
    q = {r["syllable"]: r.get("q_value") for r in stats_rows}
    for gi, g in enumerate(groups):
        for s in range(n_syl):
            v = np.array([r["usage_time"] for r in usage_rows if r["syllable"] == s
                          and (g is None or sheet.get(r["session_id"], {}).get(group_col) == g)])
            x = s + (gi - (len(groups) - 1) / 2) * w
            if len(v):
                ax.bar(x, v.mean(), width=w * 0.9, color=GROUP[gi % len(GROUP)], alpha=0.35, lw=0)
                ax.scatter(x + rng.uniform(-w / 5, w / 5, len(v)), v, s=10, color=GROUP[gi % len(GROUP)], zorder=3)
    top = ax.get_ylim()[1]
    for s in range(n_syl):
        if q.get(s) is not None and q[s] < 0.05:
            ax.text(s, top * 0.97, "*", ha="center", fontsize=12)
    if groups != [None]:
        for gi, g in enumerate(groups):
            ax.bar([], [], color=GROUP[gi % len(GROUP)], label=g)
        ax.legend(frameon=False, title=group_col)
    ax.set_xticks(range(n_syl))
    ax.set_xlabel("syllable (0 = most used)")
    ax.set_ylabel("fraction of time")
    ax.set_title("Syllable usage per session" + (f" by {group_col}" if groups != [None] else ""))
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def transition_matrix(P, out):
    n = P.shape[0]
    fig, ax = plt.subplots(figsize=(0.35 * n + 2.4, 0.35 * n + 1.8))
    im = ax.imshow(np.where(np.eye(n, dtype=bool), np.nan, P), cmap="Blues", vmin=0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("to syllable")
    ax.set_ylabel("from syllable")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("P(next syllable)")
    ax.set_title("Syllable transitions (bout to bout)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def syllable_poses(mean_poses, keypoints, edges, table, units, out):
    n = len(mean_poses)
    cols = min(6, n)
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(1.9 * cols, 2.5 * rows_ + 0.4), squeeze=False)
    allp = np.concatenate([p for p in mean_poses if p is not None]) if any(p is not None for p in mean_poses) else np.zeros((1, 2))
    lim = np.nanmax(np.abs(allp)) * 1.15 if allp.size else 1
    for s, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if s >= n:
            ax.axis("off")
            continue
        p = mean_poses[s]
        if p is not None:
            for a, b in edges:
                if a in keypoints and b in keypoints:
                    i, j = keypoints.index(a), keypoints.index(b)
                    ax.plot(p[[i, j], 0], p[[i, j], 1], color=MUTED, lw=1)
            ax.scatter(p[:, 0], p[:, 1], s=14, color=colour(s), zorder=3)
            ax.annotate("", xy=(lim * 0.9, -lim * 0.85), xytext=(lim * 0.45, -lim * 0.85),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        r = table[s]
        sp = r.get("mean_speed")
        ax.set_title(f"{s}: {100 * r['usage_time']:.0f}% time\n{r['mean_duration_s'] or 0:.2f} s, "
                     f"{sp:.1f} {units}/s" if sp is not None else f"{s}", fontsize=7)
    fig.suptitle("Mean posture of each syllable (animal facing right, arrow); title: share of time, "
                 "mean bout duration, mean speed", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def model_selection(scores, chosen, out):
    ks = sorted(scores)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(ks, [scores[k] for k in ks], color=CAT[0], lw=2, marker="o")
    ax.axvline(chosen, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("number of syllables (HMM states)")
    ax.set_ylabel("held-out log-likelihood / frame")
    ax.set_title(f"Model selection: {chosen} syllables")
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out
