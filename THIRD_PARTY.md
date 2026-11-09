# Third-party software

This app is Apache-2.0 (see `LICENSE`). The published image contains the third-party components
below, installed from PyPI at the pinned versions in `requirements.txt` and `requirements-tfp.txt`;
their licences apply to the image. Licences were read from each
installed package's metadata in the built image (`importlib.metadata`). No GPL code is imported, and nothing in the
image is under a non-commercial licence.

## Components

| Component | Authors / copyright holder | What | Licence | Source |
|---|---|---|---|---|
| dynamax 1.0.3 | Scott W. Linderman, Peter Chang, Giles Harper-Donnelly, Aleyna Kara, Xinglong Li, Gerardo Duran-Martin, Kevin Murphy | Sticky autoregressive HMM (EM, Viterbi, marginal likelihood) | MIT | https://github.com/probml/dynamax |
| JAX 0.12.0.1, jaxlib 0.12.0.1 | The JAX authors (James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, Qiao Zhang and contributors; JAX says it is not an official Google product) | Array computing for dynamax (CPU-only build) | Apache-2.0 | https://github.com/jax-ml/jax |
| tfp-nightly 0.26.0.dev20261109 (TensorFlow Probability's JAX substrate), jaxtyping, optax, fastprogress, ml_dtypes, opt_einsum | The TensorFlow Probability authors and others | Dependencies of dynamax. dynamax requires tfp-nightly, and the last stable tensorflow-probability (0.25.0, 2024) fails to import with JAX 0.12, so the nightly is pinned by version and wheel sha256 in `requirements-tfp.txt` | Apache-2.0, MIT | PyPI |
| scikit-learn 1.9.1 | The scikit-learn developers | Adjusted mutual information | BSD-3-Clause | https://scikit-learn.org |
| NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1 | NumPy, SciPy, pandas developers | Arrays, PCA (SVD), statistics, tables | BSD-3-Clause (NumPy: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0) | numpy.org, scipy.org, pandas.pydata.org |
| pyarrow 26.0.1 | Apache Software Foundation | Reading Parquet | Apache-2.0 | https://arrow.apache.org |
| pynwb 4.2.0, hdmf 6.2.0, h5py 3.16.0 | Neurodata Without Borders contributors; h5py developers | NWB reading and writing | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb |
| Matplotlib 3.11.2 | Matplotlib Development Team | PNG figures | Matplotlib licence (PSF-based) | https://matplotlib.org |
| joblib, threadpoolctl, cloudpickle, narwhals, attrs, jsonschema, ruamel.yaml, pillow, ... | respective developers | Dependencies (versions in the requirements files) | BSD-3-Clause, MIT, Apache-2.0, PSF-2.0, MIT-CMU (Pillow) | PyPI |
| python:3.12.16-slim-bookworm | Python Software Foundation, Debian | Base image | PSF-2.0; Debian packages under their own free licences | https://hub.docker.com/_/python |

## Not bundled

- **keypoint-MoSeq and jax-moseq** (Harvard, Non-Commercial Research and Academic Use licence,
  `licenses/keypoint-moseq-LICENSE.md`) are **not in the published image**. The repository can
  build an image with them (`--build-arg METHOD=kpms`, `requirements-kpms.txt`) for when Harvard OTD
  grants permission for hosted use; that image is not published (README).
- **Demo data** (not in the images or the repository): DANDI:002047 (Plusnin, Pospelov et al.,
  doi:10.48324/dandi.002047/0.261106.0903), CC-BY-4.0.
