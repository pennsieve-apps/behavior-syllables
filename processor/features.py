"""
Per-frame features for the syllable model, from pose-kinematics' kinematics.parquet.

1. Egocentric pose: every keypoint relative to the body centroid, rotated so the animal faces +x
   (the heading pose-kinematics computed, tail base -> head). Translation and rotation in the arena
   are removed; what is left is posture.
2. PCA of the egocentric pose over all sessions together (N_PCS components, or as many as explain
   90 % of the variance, at most 8).
3. Movement: log(1 + speed) and angular speed (the AR-HMM models how the posture changes itself;
   the PCs' rate of change is added only with derivatives=True).
4. Every feature z-scored over all sessions, so no feature dominates by its units, and values beyond
   4 SD compressed logarithmically (single-frame tracking errors).

Frames missing at most a third of the keypoints get those keypoints' mean egocentric position.
Frames with any other missing value are left out of the model (syllable -1); a session is split into
segments at such frames.
"""
from __future__ import annotations

import numpy as np


def keypoint_columns(df, meta) -> list[str]:
    kps = meta.get("keypoints") or sorted({c[3:-2] for c in df.columns if c.startswith("kp_") and c.endswith("_x")})
    return [k for k in kps if f"kp_{k}_x" in df.columns and f"kp_{k}_y" in df.columns]


def choose_keypoints(dfs_meta, param: str) -> list[str]:
    """USE_KEYPOINTS=auto: keypoints present in every session, missing in < 20 % of frames, and not
    on the tail beyond its base (tails swing freely and dominate posture, as keypoint-MoSeq warns)."""
    common = None
    for df, meta in dfs_meta:
        kps = keypoint_columns(df, meta)
        common = kps if common is None else [k for k in common if k in kps]
    common = common or []
    if param and param != "auto":
        want = [k.strip() for k in param.split(",") if k.strip()]
        missing = [k for k in want if k not in common]
        if missing:
            raise ValueError(f"USE_KEYPOINTS {missing} not in every session (have {', '.join(common)})")
        return want
    out = []
    for k in common:
        low = k.lower().replace("_", "")
        if "tail" in low and not any(b in low for b in ("base", "root", "1", "start")):
            continue
        miss = np.mean([df[f"kp_{k}_x"].isna().mean() for df, _ in dfs_meta])
        if miss < 0.2:
            out.append(k)
    if len(out) < 3:
        raise ValueError(f"fewer than 3 usable keypoints ({out}); set USE_KEYPOINTS")
    return out


def egocentric(df, keypoints) -> np.ndarray:
    """(T, 2K): keypoints relative to the centroid, rotated so the heading points to +x (y up)."""
    cx, cy = df["centroid_x"].to_numpy(float), df["centroid_y"].to_numpy(float)
    h = df["heading"].to_numpy(float)
    c, s = np.cos(-h), np.sin(-h)
    out = np.empty((len(df), 2 * len(keypoints)))
    for j, k in enumerate(keypoints):
        dx = df[f"kp_{k}_x"].to_numpy(float) - cx
        dy = -(df[f"kp_{k}_y"].to_numpy(float) - cy)          # image y is down; posture in y-up
        out[:, 2 * j] = c * dx - s * dy
        out[:, 2 * j + 1] = s * dx + c * dy
    return out


def derivative(x, t):
    out = np.full_like(x, np.nan)
    if len(t) < 3:
        return out
    dt = np.gradient(t)
    out[:] = np.gradient(x, axis=0) / dt[:, None]
    return out


class FeatureBuilder:
    def __init__(self, keypoints, n_pcs: int, derivatives: bool = False):
        self.derivatives = derivatives
        self.keypoints = keypoints
        self.n_pcs = n_pcs
        self.pca_mean = self.pca_components = self.explained = None
        self.kp_mean = None
        self.n_imputed = 0
        self.mu = self.sd = None
        self.names = None

    def _ego(self, df):
        """Egocentric pose; a frame missing at most a third of its keypoints (an occluded paw) gets their
        mean egocentric position, so it is not dropped from the model."""
        ego = egocentric(df, self.keypoints)
        if self.kp_mean is None:
            return ego
        T, K = len(ego), len(self.keypoints)
        e = ego.reshape(T, K, 2)
        miss = ~np.isfinite(e).all(axis=2)
        anchored = np.isfinite(df["centroid_x"].to_numpy(float)) & np.isfinite(df["heading"].to_numpy(float))
        fill = anchored & (miss.sum(axis=1) <= K // 3) & miss.any(axis=1)
        self.n_imputed += int(fill.sum())
        for k in range(K):
            rows = fill & miss[:, k]
            e[rows, k, :] = self.kp_mean[k]
        return e.reshape(T, 2 * K)

    def fit(self, frames):
        """frames: list of (df) per session."""
        ego = np.concatenate([egocentric(df, self.keypoints) for df in frames])
        with np.errstate(all="ignore"):
            self.kp_mean = np.nanmean(ego.reshape(len(ego), -1, 2), axis=0)
        ok = np.isfinite(ego).all(axis=1)
        X = ego[ok]
        self.pca_mean = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.pca_mean, full_matrices=False)
        var = S ** 2 / max(1, len(X) - 1)
        frac = var / var.sum()
        if self.n_pcs <= 0:
            self.n_pcs = int(min(8, max(2, np.searchsorted(np.cumsum(frac), 0.90) + 1)))
        self.n_pcs = min(self.n_pcs, Vt.shape[0])
        self.pca_components = Vt[: self.n_pcs]
        self.explained = frac[: self.n_pcs]
        raw = np.concatenate([self._raw(df) for df in frames])
        okr = np.isfinite(raw).all(axis=1)
        self.mu = raw[okr].mean(axis=0)
        self.sd = raw[okr].std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def _raw(self, df):
        ego = self._ego(df)
        pcs = (ego - self.pca_mean) @ self.pca_components.T
        t = df["time"].to_numpy(float)
        dpcs = derivative(np.where(np.isfinite(pcs), pcs, np.nan), t)
        speed = np.log1p(np.clip(df["speed"].to_numpy(float), 0, None))
        ang = np.clip(df["angular_speed"].to_numpy(float), -20, 20)
        if not self.derivatives:   # the AR-HMM models the dynamics itself
            self.names = [f"pc{i + 1}" for i in range(pcs.shape[1])] + ["log_speed", "angular_speed"]
            return np.column_stack([pcs, speed, ang])
        self.names = [f"pc{i + 1}" for i in range(pcs.shape[1])] + [f"d_pc{i + 1}" for i in range(pcs.shape[1])] + \
                     ["log_speed", "angular_speed"]
        return np.column_stack([pcs, dpcs, speed, ang])

    def transform(self, df):
        # Values beyond 4 SD are compressed logarithmically (4 + log(1 + excess)): single-frame tracking
        # errors (huge pose velocities) would otherwise get a state of their own or make every state's
        # likelihood underflow; a hard clip would pile them up at the bound instead.
        z = (self._raw(df) - self.mu) / self.sd
        a = np.abs(z)
        return np.where(a > 4, np.sign(z) * (4 + np.log1p(np.maximum(a - 4, 0))), z)

    def egocentric_mean(self, df, mask):
        ego = egocentric(df, self.keypoints)[mask]
        ego = ego[np.isfinite(ego).all(axis=1)]
        return ego.mean(axis=0).reshape(-1, 2) if len(ego) else None


def segments(valid: np.ndarray, min_len: int):
    """[(a, b)] runs of valid frames of at least min_len."""
    v = np.r_[0, valid.astype(int), 0]
    d = np.diff(v)
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(i, j) for i, j in zip(a, b) if j - i >= min_len]
