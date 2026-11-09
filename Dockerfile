FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# METHOD=hmm (the published image): sticky autoregressive HMM (dynamax, CPU-only JAX), permissive
# licences. METHOD=kpms builds an image that also has keypoint-MoSeq with JAX for CUDA 12; it is NOT
# published (Harvard's licence does not cover hosted use for other institutions; see README).
ARG METHOD=hmm

# The complete pinned dependency set of the variant (no resolver; wheels, and pure-Python sdists such
# as commentjson for kpms).
COPY requirements.txt requirements-tfp.txt requirements-kpms.txt /processor/
RUN if [ "$METHOD" = "kpms" ]; then req=requirements-kpms.txt; else req=requirements.txt; fi \
    && pip install --no-deps --prefer-binary -r /processor/$req \
    && if [ "$METHOD" != "kpms" ]; then pip install --no-deps --require-hashes -r /processor/requirements-tfp.txt; fi \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -type d -name tests -prune -exec rm -rf {} +

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION
COPY licenses/ /processor/licenses/

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    USER=app \
    LOGNAME=app \
    MPLBACKEND=Agg \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    DEFAULT_METHOD=${METHOD} \
    APP_VARIANT=${METHOD}

# Check that the stack imports (for kpms also keypoint-MoSeq and JAX). Nothing is baked into /tmp or
# HOME: on Lambda both are empty and only INPUT_DIR, OUTPUT_DIR, HOME and TMPDIR are writable;
# processor/common.py points matplotlib / numba / XDG caches at $TMPDIR at startup.
RUN export MPLCONFIGDIR=/dev/shm/mpl HOME=/dev/shm NUMBA_CACHE_DIR=/dev/shm/numba && python -c "import matplotlib.pyplot, dynamax.hidden_markov_model, pynwb, pyarrow.parquet, sklearn.metrics" \
    && if [ "$METHOD" = "kpms" ]; then python -c "import keypoint_moseq, jax; print('jax', jax.__version__)"; fi

# Runs as uid 1000 on ECS and as an arbitrary unprivileged user on Lambda; nothing here needs a
# particular uid (USER=app above answers getpass).
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
