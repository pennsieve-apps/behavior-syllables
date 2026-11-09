# Third-party software

This app is Apache-2.0 (see `LICENSE`). Each variant's image contains the third-party components
below, installed from PyPI at the pinned versions in `requirements.txt` (`hmm`) or
`requirements-kpms.txt` (`kpms`); their licences apply to the images. Licences were read from each
installed package's metadata in the built image (`importlib.metadata`). No GPL code is imported in either variant;
`imageio-ffmpeg` (which ships a GPL ffmpeg executable) is left out of the kpms image.

## keypoint-MoSeq: non-commercial academic use only (kpms variant)

| Component | Authors / copyright holder | Licence | Source |
|---|---|---|---|
| keypoint-moseq 0.8.6 | President and Fellows of Harvard College; developed by Caleb Weinreb and Sandeep Robert Datta | **Non-Commercial Research and Academic Use Software License** (Harvard). Commercial use, including fee-for-service, core facilities serving for-profit parties and industry-sponsored projects, needs a separate licence from Harvard OTD (otd@harvard.edu). Derivative works may be distributed for non-commercial research and academic use only; Harvard's notices must be kept. Full text: `licenses/keypoint-moseq-LICENSE.md` (also in the image at `/processor/licenses/`). | https://github.com/dattalab/keypoint-moseq |
| jax-moseq 0.3.9 | Datta lab, Harvard Medical School | `LicenseRef-Harvard-Academic-Non-Commercial` (same terms) | https://github.com/dattalab/jax-moseq |

Users of the `kpms` variant must use it for non-commercial research and academic purposes only and
should cite Wiltschko et al. 2015 and Weinreb et al. 2024, as the licence asks (README, Citations).

## Both variants

| Component | Authors / copyright holder | What | Licence | Source |
|---|---|---|---|---|
| dynamax 1.0.3 (hmm) / 1.0.2 (kpms, as jax-moseq requires) | Scott W. Linderman, Peter Chang, Giles Harper-Donnelly, Aleyna Kara, Xinglong Li, Gerardo Duran-Martin, Kevin Murphy | Sticky autoregressive HMM (EM, Viterbi, marginal likelihood) | MIT | https://github.com/probml/dynamax |
| JAX 0.12.0.1, jaxlib 0.12.0.1 | Google (JAX authors) | Array computing for dynamax (CPU-only in the hmm image) | Apache-2.0 | https://github.com/jax-ml/jax |
| tfp-nightly 0.26.0.dev20261109, jaxtyping, optax, fastprogress, ml_dtypes, opt_einsum | Google (TensorFlow Probability) and others | Dependencies of dynamax (tfp-nightly is pinned to the exact build resolved on 2026-11-09) | Apache-2.0, MIT | PyPI |
| scikit-learn 1.9.1 | The scikit-learn developers | Adjusted mutual information | BSD-3-Clause | https://scikit-learn.org |
| NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1 | NumPy, SciPy, pandas developers | Arrays, PCA (SVD), statistics, tables | BSD-3-Clause (NumPy: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0) | numpy.org, scipy.org, pandas.pydata.org |
| pyarrow 26.0.1 | Apache Software Foundation | Reading Parquet | Apache-2.0 | https://arrow.apache.org |
| pynwb 4.2.0, hdmf 6.2.0, h5py 3.16.0 | Neurodata Without Borders contributors; h5py developers | NWB reading and writing | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb |
| Matplotlib 3.11.2 | Matplotlib Development Team | PNG figures | Matplotlib licence (PSF-based) | https://matplotlib.org |
| joblib, threadpoolctl, cloudpickle, narwhals, attrs, jsonschema, ruamel.yaml, pillow, ... | respective developers | Dependencies (versions in the requirements files) | BSD-3-Clause, MIT, Apache-2.0, PSF-2.0, MIT-CMU (Pillow) | PyPI |
| python:3.12.16-slim-bookworm | Python Software Foundation, Debian | Base image | PSF-2.0; Debian packages under their own free licences | https://hub.docker.com/_/python |

## kpms variant only (besides keypoint-MoSeq above)

| Component | Authors / copyright holder | What | Licence | Source |
|---|---|---|---|---|
| jax-cuda12-plugin, jax-cuda12-pjrt 0.12.0.1 | Google (JAX authors) | JAX on NVIDIA GPUs | Apache-2.0 | https://github.com/jax-ml/jax |
| NVIDIA CUDA 12.9 runtime wheels (cublas, cudnn 9, cufft, cusolver, cusparse, nccl, nvjitlink, nvrtc, cupti, nvcc, nvshmem) | NVIDIA | GPU libraries used by JAX | NVIDIA software licence (redistributable as pip wheels; LicenseRef-NVIDIA-Proprietary) | https://pypi.org/user/nvidia |
| dm-tree, absl-py, optree | Google / DeepMind and others | Dependencies of jax-moseq | Apache-2.0 | PyPI |
| numba, llvmlite | Anaconda and contributors | jax-moseq kernels | BSD-2-Clause; llvmlite BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | https://numba.pydata.org |
| sleap-io, ndx-pose, opencv-python-headless, vidio, imageio, holoviews, bokeh, panel, plotly, statsmodels, seaborn, ipykernel, ipywidgets, ... | respective developers | keypoint-MoSeq's I/O, plotting and notebook dependencies (installed, not used by this app); opencv bundles FFmpeg libraries under LGPL-2.1 | BSD-3-Clause, MIT, Apache-2.0 (bundled libraries under their own licences) | PyPI |

## Not bundled

- **Demo data** (not in the images or the repository): DANDI:002047 (Plusnin, Pospelov et al.,
  doi:10.48324/dandi.002047/0.261106.0903), CC-BY-4.0.
