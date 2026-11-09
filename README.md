# behavior-syllables

> **Variant `kpms` (keypoint-MoSeq): non-commercial academic use only.** keypoint-MoSeq is licensed
> by Harvard University for internal, non-commercial research and academic use; commercial use,
> including fee-for-service work, core facilities serving for-profit parties and industry-sponsored
> projects, needs a licence from Harvard's Office of Technology Development (otd@harvard.edu). The
> default `hmm` variant contains no such code (all permissive licences).

This app follows the **MoSeq** idea of Sandeep Robert Datta's lab at Harvard Medical School:
behaviour is a sequence of short, stereotyped, reused motifs ("syllables") that an unsupervised
model can find in pose data (Wiltschko, Johnson, Iurilli, Peterson, Katon, Pashkovski, Abraira,
Adams & Datta 2015). Its `kpms` variant runs **[keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq)**,
developed by Caleb Weinreb, Sandeep Robert Datta and colleagues (Weinreb et al. 2024, Nature
Methods, [10.1038/s41592-024-02318-2](https://doi.org/10.1038/s41592-024-02318-2)), built on
[JAX](https://github.com/jax-ml/jax) (Google). Its default `hmm` variant fits MoSeq's core model, a
**sticky autoregressive hidden Markov model** (AR-HMM; the "sticky" prior of Fox, Sudderth, Jordan
& Willsky 2011), with **[Dynamax](https://github.com/probml/dynamax)**, the JAX state-space model
library of Scott W. Linderman, Peter Chang, Giles Harper-Donnelly, Aleyna Kara, Xinglong Li,
Gerardo Duran-Martin and Kevin Murphy (Linderman et al. 2025), plus
[scikit-learn](https://github.com/scikit-learn/scikit-learn) for the agreement scores. **Pennsieve packages these
methods as a workflow step; it did not develop them.** The demo data are DANDI:002047 by Viktor Plusnin,
Nikita Pospelov and colleagues ([10.48324/dandi.002047/0.261106.0903](https://doi.org/10.48324/dandi.002047/0.261106.0903), CC-BY-4.0),
whose expert behaviour labels make the agreement figure possible.

**What it does.** pose-kinematics turns pose tracks into clean keypoints, speed and heading. This
app asks: *what does the animal do, moment by moment?*, without anyone labelling frames. It aligns
every frame's posture to the animal's own body (so position in the arena and facing direction do
not matter), adds how fast the posture and the animal move, and fits one model over all sessions
that splits the recording into recurring states: syllables such as resting, walking, turning,
rearing or grooming. You get, per session and per condition, how often each syllable is used and
for how long, which syllable follows which, and whether a condition (e.g. 5xFAD) uses some
syllables more than another (WT). When the session NWB has expert labels (DANDI 002047: rear,
freezing, locomotion, rest, walk), it measures how well the unsupervised syllables match them: the
headline figure of the **Markerless Behaviour** workflow,
`pose-estimate → pose-kinematics → behavior-syllables → neuro-report`.

**Discovery, not classification.** The syllables are found without labels, so they are a
hypothesis about how behaviour is organised, not a classifier for named behaviours. When you need
"rearing" or "grooming" detected reliably, supervised classifiers trained on labelled frames do
better: in a comparison across supervised, unsupervised and semi-supervised action segmentation
(Blau et al. 2024) supervised methods were the most accurate. That is why this app always measures
agreement with expert labels when they exist, and shows it first.

For a computational analyst: features are egocentric keypoints (centroid-subtracted, rotated to the
tail-base → head axis; already smoothed by pose-kinematics) reduced by PCA to the components
explaining 90 % of variance (≤ 8), plus log(1 + speed) and angular speed, z-scored over all
sessions with values beyond 4 SD compressed logarithmically. A Dynamax `LinearAutoregressiveHMM`
(one lag: each state predicts the next frame's features linearly from the current frame, with its
own noise covariance) with `transition_matrix_stickiness` = `STICKINESS` is fitted by EM on all
sessions jointly (segments cut into 512-frame chunks, k-means initialisation seeded by `SEED`) and
decoded with Viterbi. Modelling *trajectories* rather than single frames is what keeps syllables
from following frame-to-frame keypoint jitter, which otherwise yields implausibly short syllables
(Weinreb et al. 2024); `summary.json` reports the median syllable duration straight from the
model and warns when it is under 200 ms. Why Dynamax: MIT, maintained, an AR-HMM with a sticky
prior out of the box, CPU-only JAX in the default image; hmmlearn (BSD) has no autoregressive
HMM, and `ssm` (Linderman lab, MIT) is not maintained on PyPI (0.0.1) and must be built from git.
keypoint-MoSeq goes further (it also models keypoint noise, centroid and heading jointly with the
AR-HMM, by Gibbs sampling), hence the `kpms` variant.

## Variants

| Variant | Default | Compute | What |
|---|---|---|---|
| `hmm` | yes | standard (4 vCPU, 4 GB) | Sticky AR-HMM (Dynamax, CPU-only JAX) on egocentric posture, speed and turning. Permissive licences only. |
| `kpms` | | gpu (T4, 16 GB) | keypoint-MoSeq 0.8.6 on JAX 0.12 with CUDA 12 (falls back to CPU with `USE_GPU=auto`, logged, but CPU fitting of a real dataset takes hours). **Non-commercial academic use only** (Harvard licence; `licenses/keypoint-moseq-LICENSE.md`, kept in the image). |

Build with `docker build --build-arg METHOD=hmm|kpms`. Both variants have the same ports and
parameters; `METHOD` defaults to the variant.

### keypoint-MoSeq licence check

The licence (`licenses/keypoint-moseq-LICENSE.md`, read 2026-11-09) grants non-commercial research
and academic use; section 3 allows creating and distributing derivative works restricted to
non-commercial research and academic use; section 2 requires keeping Harvard's notices; nothing
forbids redistributing the software in a container image. So the variant is built, with the
licence in the image, in `THIRD_PARTY.md`, in the variant's description and at the top of this
README. jax-moseq (keypoint-MoSeq's model code) carries the same licence. Open question for the
licence decision: the grant covers an end user's *internal* use at their own academic institution,
so running kpms as a hosted step for users of other institutions (and keeping for-profit users of
the platform away from the variant) is something Harvard OTD should confirm.

## Method

1. **Read** every `<session_id>_kinematics.parquet` (pose-kinematics; keypoints in cm, centroid,
   heading, speed, angular speed) and `sessions.csv` (condition / group).
2. **Keypoints** (`USE_KEYPOINTS=auto`): those in every session, missing in < 20 % of frames, no tail
   tips. Frames missing at most a third of them get those keypoints' mean egocentric position;
   frames missing more, or the centroid / heading, are not modelled (syllable −1) and split the
   session into segments.
3. **Features** (above), **model** (`METHOD`): `hmm`: number of states `N_SYLLABLES`, or 0 = chosen
   by held-out log-probability: every fifth 512-frame chunk held out, models with 4, 6, …,
   `MAX_SYLLABLES` states fitted on the rest; the smallest number that reaches 90 % of the best
   held-out gain over 4 states is chosen (held-out likelihood keeps rising slowly with more states on
   long recordings, so the elbow, not the maximum), and the final fit on all chunks starts from it. `kpms`: keypoint-MoSeq's default
   configuration, latent dimension from 90 % of the PCA variance, 50 AR-HMM iterations (fewer only
   when `KPMS_ITERS` < 50) at `KAPPA`,
   then `KPMS_ITERS` full-model iterations at `KAPPA`/10, as in the keypoint-MoSeq tutorial.
4. **Clean-up**: bouts shorter than `MIN_DURATION_S` take the label of their longer neighbour;
   syllables are renumbered by usage (0 = most used); syllables used in < `MIN_USAGE` of frames
   are pooled into one last "rare" syllable.
5. **Tables**: usage (fraction of time, fraction of bouts, bouts/min), bout durations, mean speed,
   angular speed and body length per syllable; bout-to-bout transition counts/probabilities.
6. **Groups** (`GROUP_BY`): per syllable, session usage compared with Welch's t-test (Mann–Whitney
   U and Hedges' g alongside), Benjamini–Hochberg FDR across syllables.
7. **Agreement** (session NWB with labelled `TimeIntervals`): per syllable × label, the fraction of
   the syllable's frames inside the label's intervals, the reverse, Jaccard index and enrichment;
   the **adjusted mutual information** (Vinh et al. 2009) between syllables and a per-frame label
   (the most specific of a frame's labels: the rarest; `none` when unlabelled), with a null from
   the syllable sequences circularly shifted in time (20 shifts, seeded) to show what temporal
   autocorrelation alone gives.

## Ports

| Port | Kind | Files | Notes |
|---|---|---|---|
| `kinematics` | input (subfolder) | `<session_id>_kinematics.parquet`, `sessions.csv` | pose-kinematics output |
| `session` | input (subfolder, optional) | `*.nwb` with `TimeIntervals` | matched by the NWB `identifier` pose-kinematics recorded in the Parquet metadata, else by session_id |
| `syllables` | output | `behavior-syllables/...` | the whole OUTPUT_DIR |

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `METHOD` | `hmm` (`kpms` in the kpms variant) | `hmm` or `kpms`. `kpms` in the hmm image exits 2. |
| `N_SYLLABLES` | `0` | hmm: number of states; 0 = chosen by held-out likelihood. |
| `MAX_SYLLABLES` | `12` | hmm, N_SYLLABLES=0: the largest number tried. |
| `N_PCS` | `0` | Posture PCs; 0 = 90 % of variance (≤ 8). |
| `USE_KEYPOINTS` | `auto` | Keypoints modelled (comma-separated). |
| `MIN_DURATION_S` | `0.2` | Shortest bout (s); shorter bouts are merged. |
| `MIN_USAGE` | `0.005` | Syllables below this share of frames are pooled into one "rare" syllable. |
| `MAX_ITER` | `100` | hmm: maximum EM iterations (stops earlier when the gain is < 1e-4 per frame). |
| `STICKINESS` | `100000` | hmm: pseudo-counts on each state's self-transition (sticky prior); larger = longer syllables. |
| `KAPPA` | `1000000` | kpms: stickiness (larger = longer syllables). |
| `KPMS_ITERS` | `200` | kpms: full-model Gibbs iterations (the paper uses ~500). |
| `LABEL_TABLES` | `auto` | Label tables in the session NWB; `auto` = all TimeIntervals except epochs/trials/invalid_times and spatial zones; `none` = skip. |
| `GROUP_BY` | `condition` | sessions.csv column compared. |
| `USE_GPU` | `auto` | kpms: `auto` (GPU if JAX sees one, else CPU, logged), `true` (exit 3 without GPU), `false`. |
| `SEED` | `42` | AR-HMM k-means initialisation, kpms sampling, agreement null. |
| `THREADS` | `0` | numpy/BLAS threads; 0 = cgroup CPU quota. |

## Outputs (`OUTPUT_DIR/behavior-syllables/`)

| File | Contents |
|---|---|
| `<session_id>_syllables.csv` | `time` (session clock, s), `syllable` (−1 = not modelled). |
| `<session_id>_syllables.nwb` | Session fields copied from the source NWB (via pose-kinematics), a new UUID `identifier` with the source noted in `notes`; passes nwb-inspect (schema, NWB Inspector, DANDI) on the demo outputs; `processing/behavior/syllables` (`TimeIntervals`: start_time, stop_time, syllable) and a `syllable` TimeSeries per frame. |
| `syllables.csv` | Per syllable: `n_bouts, seconds, usage_time, usage_bouts, mean_duration_s, median_duration_s, mean_speed, mean_abs_angular_speed, mean_body_length`, `usage_<group>`, `q_value`, `top_label`, `top_label_fraction`. |
| `usage.csv` | Per session × syllable: `seconds, usage_time, n_bouts, usage_bouts, bouts_per_minute` + sheet columns. |
| `transitions.csv` | `from_syllable, to_syllable, count, probability` (bout to bout, pooled). |
| `group_stats.csv` | Per syllable: `n_`, `mean_`, `sd_` per group, `test, statistic, p_value, p_value_secondary, hedges_g, q_value`. |
| `agreement.csv` | Per syllable × label: `n_frames_syllable, n_frames_label, n_frames_both, p_label_given_syllable, p_syllable_given_label, jaccard, enrichment`. |
| `model.json` | Keypoints, PCA (components, mean, explained variance), feature scaling, AR-HMM parameters (initial probabilities, transition matrix, AR weights, biases, noise covariances), renumbering, model-selection scores. |
| `agreement.png`, `ethogram_vs_labels.png`, `ethogram.png`, `syllable_poses.png`, `syllable_usage.png`, `transitions.png`, `model_selection.png` | Figures; captions in `summary.json`. |
| `summary.json`, `provenance.json` | `median_syllable_duration_s` (from the model, before merging) and `short_syllable_warning` (< 200 ms), per-session usage, syllable table, agreement (AMI, null, per-session AMI, best syllable per label), group tests, model info; versions, parameters, timings, peak memory. |

## Results on the demo data

DANDI 002047, six 12-min day-1 open-field sessions (3 WT, 3 5xFAD; 129 484 frames at 30 fps),
pose-kinematics output from the authors' DeepLabCut pose, the session NWB files for the labels,
`hmm` variant, defaults. Output in
`workflow-systems-neuroscience/data/work/openfield-5xfad/behavior-syllables/`.

- **Keypoints**: 11 of 14 (`auto` dropped the nose and the two miniscope points, missing or below
  confidence in > 20 % of frames). Posture: 7 PCs = 90 % of egocentric variance; 9 features with
  speed and turning. 80–99 % of frames modelled per session (the rest: centroid or more than a
  third of the keypoints missing, mostly where the miniscope cable hides the head).
- **Number of syllables**: held-out log-probability per frame 5.31, 5.60, 5.66, 5.68, 5.70 for 4,
  6, 8, 10, 12 states; the elbow rule picked 10.
- **Syllable durations**: median **367 ms** straight from the sticky AR-HMM (MoSeq's syllables last
  ~300–500 ms), 500 ms after merging bouts < 0.2 s; mean bout durations 0.45–1.2 s per syllable.
  (The earlier per-frame Gaussian HMM needed the merging to reach plausible durations.)
- **What they are** (`syllables.csv`, `syllable_poses.png`): two locomotion syllables (14–16 cm/s,
  elongated body, 91–96 % of their frames labelled *locomotion* by the experts), a fast turn
  (syllable 9, 4.2 rad/s), slow exploration / walking syllables (1–7 cm/s, 58–95 % *walk*) and
  three still ones (< 1 cm/s; syllable 6: 1.2-s bouts at 0.25 cm/s, 99.9 % *rest*).
- **Agreement with the experts' labels** (`agreement.csv`, `agreement.png`): adjusted mutual
  information **0.36** over all labelled frames (per session 0.29–0.42), against **0.016** for the
  same syllable sequences shifted in time. Locomotion, walking and rest are recovered (best
  syllable Jaccard 0.38, 0.35, 0.40; each behaviour spread over 2–3 syllables); **rearing** (best
  Jaccard 0.13) and **freezing** (0.4 % of frames; 0.04) are not separated from slow walking /
  rest from above with these keypoints. These are discoveries to inspect, not a classifier: a
  supervised classifier trained on the labels would do better on rearing (Blau et al. 2024).
- **WT vs 5xFAD usage** (n = 3 sessions per group; Welch t-test per syllable, BH across 10
  syllables): no syllable differs (all q ≥ 0.83). Largest effects: 5xFAD used locomotion syllable 5
  more (16 vs 6 % of time, Hedges g 1.0) and the long-rest syllable 6 less (1.4 vs 12.6 %; WT usage
  dominated by one barely moving animal, F48). **With three animals per group this comparison is
  underpowered**; it demonstrates the analysis, not a phenotype.

On the synthetic fixture (four 120-s sessions, three planted behaviours, `N_SYLLABLES=3`): the
planted behaviours are recovered almost exactly (AMI with the truth 0.996, time-shifted null 0.003;
median syllable 1.3 s, as simulated). keypoint-MoSeq (`METHOD=kpms`, 20 iterations, CPU, native
arm64) on the same fixture: AMI 0.54 before pooling rare syllables (too few iterations and frames
for kpms to settle; the kpms FAQ asks for hours of data).

## Runtime

`hmm` (default): `computeTypes: [standard]`, 4 vCPU / 4096 MiB, 1 h timeout. Measured on the demo
data (6 sessions, 129 484 frames, defaults including model selection over 4–12 states):

| Where | Wall time | Peak RSS | cgroup `memory.peak` |
|---|---|---|---|
| linux/amd64 image under emulation on an M-series Mac, `--cpus 4 --memory 4g`, run as the brief's Lambda test (`--read-only --user 65534:65534 --tmpfs /tmp`, HOME/TMPDIR under `/tmp/w`) | 653 s | 1.65 GB | 1.90 GB |
| native arm64 Python (same code, for comparison) | 124 s | 1.81 GB | |

Results are identical in both (10 syllables, median 367 ms, AMI 0.363). Most of the time is the
five model-selection fits; `N_SYLLABLES` > 0 skips them. Memory is JAX's working set over all
sessions' 512-frame chunks; it grows with the total number of frames (about 1.5 GB per 100 k frames
above the ~0.4 GB baseline), so large cohorts need more memory, and more than ~1 h of video in total
needs a longer timeout. These values are within the platform's Lambda limits for a dataset this
size, so the platform may run it as a Lambda step. Image: 0.97 GB unpacked (`/usr/local`), 0.33 GB
compressed.

`kpms`: `computeTypes: [gpu]`, 16 GB, 4 h timeout; GPU numbers **to be measured on the dev compute
node**. On this Mac (no GPU) the CUDA image starts, JAX logs that it cannot initialise CUDA, the app
logs `USE_GPU=auto: cuda not available, using cpu` and finishes on CPU (fixture, 4 × 40 s,
`KPMS_ITERS=3`: 487 s under emulation, peak RSS 1.85 GB, also as the Lambda test). keypoint-MoSeq
on CPU is far too slow for real data; `USE_GPU=true` exits 3 without a GPU. Image: 3.8 GB
compressed (JAX + CUDA 12.9 runtime wheels).

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python    # pytest on the synthetic fixture, no Docker
make run PYTHON=.venv/bin/python     # Docker (hmm variant) over data/fixture/linked -> data/output/
docker build --platform linux/amd64 --build-arg METHOD=kpms -t behavior-syllables:kpms .
```

## Limitations

- **Syllables are statistical states, not named behaviours.** The model finds what recurs in the
  pose; whether syllable 3 "is" rearing is a judgement the agreement table and the posture figure
  help with, not something the model knows. Different seeds, numbers of states or recordings give
  different (usually similar) parcellations; syllable numbers are not comparable between runs.
- **Top-view 2D pose limits what can be told apart.** Rearing, grooming and freezing look alike
  from above in a few keypoints (a short, still body); the demo's labels for these are matched
  least well. Sniffing, whisking and fine paw movements need more keypoints or a side view.
- **Pose quality bounds everything.** Frames with too many missing keypoints are not modelled (8–20 %
  of frames in some demo sessions, where the miniscope cable hides the head).
- The AR-HMM has geometric bout durations (the sticky prior only lengthens them) and treats the
  cleaned keypoints as exact; keypoint-MoSeq also models keypoint noise, at GPU cost and under an
  academic licence.
- With N_SYLLABLES=0 the held-out likelihood still improves at 12 states on 12-minute sessions; the
  elbow rule is a practical choice, not a "true" number of behaviours.
- Group comparisons are per session, FDR-corrected across syllables, and underpowered with a few
  animals per group.

## Licences

Apache-2.0 for this app (see `LICENSE`). The `hmm` image contains only permissively licensed
components. The `kpms` image also contains keypoint-MoSeq and jax-moseq under Harvard's
non-commercial research and academic use licence (`licenses/keypoint-moseq-LICENSE.md`). Details:
`THIRD_PARTY.md`.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full
author lists in `CITATION.cff`. The entries below are generated from the registry records.

MoSeq (behavioural syllables from an AR-HMM; the default `hmm` variant is this model) and the sticky HMM prior:

> Wiltschko AB, Johnson MJ, Iurilli G, Peterson RE, Katon JM, Pashkovski SL, Abraira VE, Adams RP, Datta SR. **Mapping Sub-Second Structure in Mouse Behavior.** Neuron 88(6):1121-1135 (2015).
> https://doi.org/10.1016/j.neuron.2015.11.031, PMID 26687221.

> Fox EB, Sudderth EB, Jordan MI, Willsky AS. **A sticky HDP-HMM with application to speaker diarization.** The Annals of Applied Statistics 5(2A):1020-1056 (2011).
> https://doi.org/10.1214/10-AOAS395.

Dynamax (the AR-HMM implementation; its CITATION.cff asks for this article) and JAX (https://github.com/jax-ml/jax; Bradbury et al. 2018, software):

> Linderman SW, Chang P, Harper-Donnelly G, Kara A, Li X, Duran-Martin G, Murphy K. **Dynamax: A Python package for probabilistic state space modeling with JAX.** Journal of Open Source Software 10(108):7069 (2025).
> https://doi.org/10.21105/joss.07069.

keypoint-MoSeq (the `kpms` variant; its licence asks users to cite Wiltschko et al. 2015, above, and this):

> Weinreb C, Pearl JE, Lin S, Osman MAM, Zhang L, Annapragada S, Conlin E, Hoffmann R, Makowska S, Gillis WF, Jay M, Ye S, Mathis A, Mathis MW, Pereira T, Linderman SW, Datta SR. **Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics.** Nature Methods 21(7):1329-1339 (2024).
> https://doi.org/10.1038/s41592-024-02318-2, PMID 38997595.

Hidden Markov models and Viterbi decoding:

> Rabiner LR. **A tutorial on hidden Markov models and selected applications in speech recognition.** Proceedings of the IEEE 77(2):257-286 (1989).
> https://doi.org/10.1109/5.18626.

> Viterbi A. **Error bounds for convolutional codes and an asymptotically optimum decoding algorithm.** IEEE Transactions on Information Theory 13(2):260-269 (1967).
> https://doi.org/10.1109/TIT.1967.1054010.

Supervised vs unsupervised action segmentation (why the agreement with expert labels matters):

> Blau A, Schaffer ES, Mishra N, Miska NJ, International Brain Laboratory, Paninski L, Whiteway MR. **A study of animal action segmentation algorithms across supervised, unsupervised, and semi-supervised learning paradigms.** Neurons, Behavior, Data analysis, and Theory (2024).
> https://doi.org/10.51628/001c.127770, PMID 40843338.

Agreement and statistics (scikit-learn, https://github.com/scikit-learn/scikit-learn, computes AMI):

> Vinh NX, Epps J, Bailey J. **Information theoretic measures for clusterings comparison.** Proceedings of the 26th Annual International Conference on Machine Learning:1073-1080 (2009).
> https://doi.org/10.1145/1553374.1553511.

> Benjamini Y, Hochberg Y. **Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.** Journal of the Royal Statistical Society Series B: Statistical Methodology 57(1):289-300 (1995).
> https://doi.org/10.1111/j.2517-6161.1995.tb02031.x.

> WELCH BL. **THE GENERALIZATION OF ‘STUDENT'S’ PROBLEM WHEN SEVERAL DIFFERENT POPULATION VARLANCES ARE INVOLVED.** Biometrika 34(1-2):28-35 (1947).
> https://doi.org/10.1093/biomet/34.1-2.28, PMID 20287819.

> Mann HB, Whitney DR. **On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other.** The Annals of Mathematical Statistics 18(1):50-60 (1947).
> https://doi.org/10.1214/aoms/1177730491.

> Hedges LV. **Distribution Theory for Glass's Estimator of Effect size and Related Estimators.** Journal of Educational Statistics 6(2):107-128 (1981).
> https://doi.org/10.3102/10769986006002107.

Neurodata Without Borders (PyNWB):

> Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K, Frank L, Bouchard KE. **The Neurodata Without Borders ecosystem for neurophysiological data science.** eLife 11:e78362 (2022).
> https://doi.org/10.7554/eLife.78362, PMID 36193886.

Demo data (DataCite creators; DataCite lists as contributors Viktor Plusnin, Olga Rogozhnikova, Vladimir
Sotskov, Nikita Pospelov, Anna Ivanova, Nikita Savelev, Ksenia Toropova, Olga Ivashkina and Konstantin
Anokhin; CC-BY-4.0):

> Plusnin V, Pospelov N. **Hippocampal CA1 one-photon calcium imaging and pose-tracked behavior in wild-type and 5xFAD mice across four behavioral paradigms.** DANDI Archive (2026).
> https://doi.org/10.48324/dandi.002047/0.261106.0903.

## Acknowledgements

MoSeq and keypoint-MoSeq are developed by the Datta lab (Harvard Medical School) with collaborators
including the Linderman lab (Stanford) and the Mathis labs (EPFL); Dynamax by the probml
community (Linderman and Murphy labs), scikit-learn by its open-source community; NWB by the Neurodata Without Borders community. Thanks to the
DANDI:002047 authors for sharing pose, expert behaviour labels and imaging together under CC-BY on
the DANDI Archive.
