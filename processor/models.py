"""
Syllable models.

hmm (default): a sticky autoregressive HMM (dynamax LinearAutoregressiveHMM, one lag, EM with a
Dirichlet "stickiness" prior on self-transitions, as in MoSeq's AR-HMM; Wiltschko et al. 2015, Fox
et al. 2011) on the egocentric posture PCs plus log speed and angular speed, fitted jointly on every
session's segments and decoded with Viterbi. Each state predicts the next frame linearly from the
last one, so a syllable is a pose *trajectory*, not a cluster of single frames, and frame-to-frame
keypoint jitter does not create spurious switches (Weinreb et al. 2024). STICKINESS is relative:
STICKINESS x (frames per state) pseudo-counts on each self-transition (pseudo_counts()). N_SYLLABLES
= 0 picks the number of states by held-out log-probability: every fifth 512-frame chunk (~17 s at
30 fps) is held out, models with 4, 6, ..., MAX_SYLLABLES states are fitted on the rest, and the smallest number
that reaches 90 % of the best held-out gain over 4 states is chosen (held-out likelihood keeps
creeping up with more states on long recordings, so the elbow, not the maximum, is the choice).

kpms (not published; see README): keypoint-MoSeq (Weinreb et al. 2024): an autoregressive HMM fitted jointly with a
keypoint-noise model and the animal's centroid and heading, on the cleaned keypoints. AR-only
fitting for 50 iterations, then the full model for KPMS_ITERS iterations, as in the kpms tutorial.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("behavior-syllables")


CHUNK = 512   # frames per fitting batch element (17 s at 30 fps)


def _chunks(seqs, size):
    """Equal-length chunks of the segments (dynamax fits a batch of equal-length sequences). The
    remainder of each segment shorter than `size` (and segments shorter than `size`) are left out of
    the fit; every frame is still decoded (_decode)."""
    out = []
    for s in seqs:
        n = len(s) // size
        out += [s[i * size:(i + 1) * size] for i in range(n)]
    return out


def _arhmm(k, d, stickiness):
    from dynamax.hidden_markov_model import LinearAutoregressiveHMM

    return LinearAutoregressiveHMM(k, d, num_lags=1, transition_matrix_stickiness=float(stickiness))


def pseudo_counts(stickiness, n_frames, k):
    """STICKINESS is relative: the self-transition prior gets STICKINESS x (frames per state) pseudo-
    counts, so the prior weighs the same against the data whatever the cohort size (a fixed number of
    pseudo-counts would dominate a small dataset and vanish in a large one)."""
    return float(stickiness) * n_frames / k


def _fit_arhmm(chunks, k, seed, max_iter, stickiness, label="", init=None):
    import jax.numpy as jnp
    import jax.random as jr

    E = jnp.asarray(np.stack(chunks), dtype=jnp.float32)
    d = E.shape[-1]
    hmm = _arhmm(k, d, pseudo_counts(stickiness, E.shape[0] * E.shape[1], k))
    inputs = jnp.stack([hmm.compute_inputs(e) for e in E])
    if init is None:
        params, props = hmm.initialize(jr.PRNGKey(seed), method="kmeans", emissions=E.reshape(-1, d))
    else:
        _, props = hmm.initialize(jr.PRNGKey(seed), method="prior")
        params = init
    t0 = time.monotonic()
    it, prev, lls = 0, -np.inf, []
    n = E.shape[0] * E.shape[1]
    while it < max_iter:
        step = min(10, max_iter - it)
        params, ll = hmm.fit_em(params, props, E, inputs=inputs, num_iters=step, verbose=False)
        it += step
        cur = float(ll[-1]) / n
        lls.append(cur)
        log.info("AR-HMM%s %d states: EM iteration %d, log-probability %.4f per frame (%.0fs)", label, k, it, cur,
                 time.monotonic() - t0)
        if cur - prev < 1e-4:
            break
        prev = cur
    return hmm, params, {"n_iter": it, "converged": bool(it < max_iter), "fit_seconds": round(time.monotonic() - t0, 1),
                         "log_prob_per_frame": lls[-1] if lls else None}


def _heldout(hmm, params, chunks):
    import jax
    import jax.numpy as jnp

    E = jnp.asarray(np.stack(chunks), dtype=jnp.float32)
    inputs = jnp.stack([hmm.compute_inputs(e) for e in E])
    lp = jax.vmap(lambda e, i: hmm.marginal_log_prob(params, e, i))(E, inputs)
    return float(lp.sum()) / (E.shape[0] * E.shape[1])


def _decode(hmm, params, seq, size):
    """Viterbi over a whole segment, padded to a multiple of `size` (few distinct lengths to compile)."""
    import jax.numpy as jnp

    n = len(seq)
    m = int(np.ceil(n / size)) * size
    pad = np.concatenate([seq, np.repeat(seq[-1:], m - n, axis=0)]) if m > n else seq
    e = jnp.asarray(pad, dtype=jnp.float32)
    z = hmm.most_likely_states(params, e, inputs=hmm.compute_inputs(e))
    return np.asarray(z)[:n]


def fit_arhmm(seqs, fps, n_syllables, max_k, seed, max_iter, stickiness):
    size = CHUNK if min(len(s) for s in seqs) >= 0 and sum(len(s) // CHUNK for s in seqs) >= 4 else 64
    info = {"method": "sticky autoregressive HMM (dynamax LinearAutoregressiveHMM, 1 lag), EM + Viterbi",
            "stickiness": stickiness, "chunk_frames": size}
    chunks = _chunks(seqs, size)
    if not chunks:
        raise RuntimeError("no segment long enough to fit")
    init = None
    if n_syllables <= 0:
        test = chunks[4::5] or chunks[-1:]
        train = [c for i, c in enumerate(chunks) if i % 5 != 4] or chunks
        scores, fits = {}, {}
        for k in range(4, max(4, max_k) + 1, 2):
            try:
                hmm, params, _ = _fit_arhmm(train, k, seed, max_iter, stickiness, label=" (selection)")
            except Exception as exc:
                log.warning("N_SYLLABLES=%d: fit failed (%s)", k, exc)
                continue
            scores[k] = _heldout(hmm, params, test)
            fits[k] = params
            log.info("model selection: %d states, held-out log-probability %.3f per frame", k, scores[k])
        if not scores:
            raise RuntimeError("no model could be fitted")
        best = max(scores, key=scores.get)
        base = scores[min(scores)]
        gain = scores[best] - base
        k = best
        if gain > 0:
            k = next(kk for kk in sorted(scores) if scores[kk] - base >= 0.9 * gain)
        init = fits[k]
        info.update(selection="held-out log-probability elbow (90 % of the gain)", heldout_loglik_per_frame=scores)
    else:
        k = n_syllables
    hmm, params, fi = _fit_arhmm(chunks, k, seed, max_iter, stickiness, init=init)
    T = np.asarray(params.transitions.transition_matrix)
    dwell = 1.0 / np.maximum(1e-9, 1.0 - np.diag(T)) / fps
    info.update(stickiness_pseudo_counts=pseudo_counts(stickiness, len(chunks) * size, k),
                expected_dwell_s={"min": float(dwell.min()), "median": float(np.median(dwell)), "max": float(dwell.max())})
    info.update(n_states=k, converged=fi["converged"], n_iter=fi["n_iter"], fit_seconds=fi["fit_seconds"],
                loglik_per_frame=fi["log_prob_per_frame"])
    log.info("AR-HMM: %d states fitted on %d frames (%d chunks of %d), %d EM iterations, %.1fs", k,
             len(chunks) * size, len(chunks), size, fi["n_iter"], fi["fit_seconds"])
    labels = [_decode(hmm, params, s, size) for s in seqs]
    p = {"initial_probs": np.asarray(params.initial.probs), "transition_matrix": np.asarray(params.transitions.transition_matrix),
         "emission_weights": np.asarray(params.emissions.weights), "emission_biases": np.asarray(params.emissions.biases),
         "emission_covariances": np.asarray(params.emissions.covs)}
    return labels, p, info


def merge_short_bouts(z, min_len):
    """Bouts shorter than min_len frames take the label of their longer neighbour (repeat until none)."""
    z = z.copy()
    if min_len <= 1 or len(z) < 2:
        return z
    for _ in range(50):
        change = np.r_[True, z[1:] != z[:-1]]
        starts = np.where(change)[0]
        ends = np.r_[starts[1:], len(z)]
        lens = ends - starts
        short = np.where(lens < min_len)[0]
        if not len(short):
            break
        changed = False
        for i in short:
            left = lens[i - 1] if i > 0 else -1
            right = lens[i + 1] if i + 1 < len(lens) else -1
            if left < 0 and right < 0:
                continue
            src = starts[i - 1] if left >= right else starts[i + 1]
            z[starts[i]:ends[i]] = z[src]
            changed = True
        if not changed:
            break
    return z


# ----------------------------------------------------------------------------- keypoint-MoSeq

def fit_kpms(sessions, keypoints, anterior, posterior, fps, kappa, iters, seed, workdir, use_gpu):
    """sessions: {sid: (coords (T, K, 2) in px or cm, valid (T, K))}. Returns {sid: syllables (T,)}."""
    import jax

    devices = jax.devices()
    on_gpu = any(d.platform == "gpu" for d in devices)
    if use_gpu == "true" and not on_gpu:
        raise RuntimeError("USE_GPU=true but JAX sees no GPU")
    if use_gpu == "auto" and not on_gpu:
        log.info("USE_GPU=auto: cuda not available, using cpu")
    import keypoint_moseq as kpms

    project = workdir / "kpms_project"
    kpms.setup_project(str(project), overwrite=True)
    config = kpms.load_config(str(project))
    config.update(bodyparts=list(keypoints), use_bodyparts=list(keypoints), anterior_bodyparts=list(anterior),
                  posterior_bodyparts=list(posterior), fps=float(fps), seed=int(seed))
    coords, confs = {}, {}
    for sid, (xy, valid) in sessions.items():
        xy = xy.copy()
        for j in range(xy.shape[1]):   # fill gaps by interpolation; their confidence is 0
            for d in range(2):
                v = xy[:, j, d]
                ok = np.isfinite(v)
                if ok.any():
                    xy[:, j, d] = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
                else:
                    xy[:, j, d] = 0.0
        coords[sid] = xy
        confs[sid] = valid.astype(float)
    data, metadata = kpms.format_data(coords, confs, bodyparts=list(keypoints), use_bodyparts=list(keypoints),
                                      seg_length=config.get("seg_length", 10000))
    pca = kpms.fit_pca(**data, **{k: v for k, v in config.items() if k in ("anterior_idxs", "posterior_idxs",
                                                                         "conf_threshold", "PCA_fitting_num_frames",
                                                                         "verbose")})
    cs = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(min(10, max(2, np.searchsorted(cs, 0.90) + 1)))
    config["latent_dim"] = latent_dim
    log.info("keypoint-MoSeq: %d sessions, %d keypoints, latent_dim %d (90%% of pose variance), kappa %g, %d iterations, "
             "devices %s", len(coords), len(keypoints), latent_dim, kappa, iters, [d.platform for d in devices])
    model = kpms.init_model(data, pca=pca, **config)
    model = kpms.update_hypparams(model, kappa=kappa)
    t = time.monotonic()
    ar_iters = min(50, max(5, iters))   # the tutorial's 50, fewer only for very short test runs
    model, name = kpms.fit_model(model, data, metadata, str(project), ar_only=True, num_iters=ar_iters,
                                 generate_progress_plots=False, save_every_n_iters=1000)
    log.info("keypoint-MoSeq: AR-HMM fitted (%d iterations, %.0fs)", ar_iters, time.monotonic() - t)
    model = kpms.update_hypparams(model, kappa=kappa / 10)
    model = fit_model_full(kpms, model, data, metadata, project, name, iters, ar_iters)
    results = kpms.extract_results(model, metadata, str(project), name, save_results=False)
    out = {sid: np.asarray(r["syllable"], int) for sid, r in results.items()}
    info = {"method": "keypoint-MoSeq (AR-HMM + keypoint SLDS, Gibbs sampling)", "kappa": kappa, "iterations": iters,
            "latent_dim": latent_dim, "fit_seconds": round(time.monotonic() - t, 1),
            "device": "gpu" if on_gpu else "cpu", "n_states_max": int(config["trans_hypparams"]["num_states"])}
    return out, info


def fit_model_full(kpms, model, data, metadata, project, name, iters, start):
    t = time.monotonic()
    model = kpms.fit_model(model, data, metadata, str(project), name, ar_only=False, start_iter=start,
                           num_iters=start + iters, generate_progress_plots=False, save_every_n_iters=1000)[0]
    log.info("keypoint-MoSeq: full model fitted (%d iterations, %.0fs)", iters, time.monotonic() - t)
    return model
