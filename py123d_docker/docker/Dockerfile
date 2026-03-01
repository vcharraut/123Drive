ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG EXTRAS=nuplan
ARG DEVKIT=""
ARG PY123D_REF=HEAD

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

RUN git clone --depth 1 https://github.com/autonomousvision/py123d.git /opt/py123d && \
    if [ "${PY123D_REF}" != "HEAD" ]; then git -C /opt/py123d fetch --depth 1 origin "${PY123D_REF}" && git -C /opt/py123d checkout FETCH_HEAD; fi

# waymo-open-dataset-tf-2-12-0==1.6.7 pins jaxlib==0.4.13 and jax==0.4.13 which don't
# exist on PyPI (min available: 0.4.17). Use uv --override to replace transitive pins.
RUN printf 'jaxlib>=0.4.17\njax>=0.4.13\n' > /tmp/overrides.txt && \
    uv pip install --system --no-cache --override /tmp/overrides.txt -e "/opt/py123d[${EXTRAS}]"

RUN if [ -n "${DEVKIT}" ]; then \
    uv pip install --system --no-cache "${DEVKIT}" pytest; \
    fi

ENTRYPOINT ["py123d-conversion"]
