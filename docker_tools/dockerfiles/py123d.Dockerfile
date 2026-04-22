ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG DATASET
ARG EXTRAS

ENV DATASET=${DATASET}
ENV HYDRA_FULL_ERROR=1
ENV CUDA_VISIBLE_DEVICES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG PY123D_VERSION=v0.2.1

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --branch ${PY123D_VERSION} https://github.com/autonomousvision/py123d.git /tmp/py123d && \
    pip install --no-cache-dir "/tmp/py123d[${EXTRAS}]" && \
    if [ "${EXTRAS}" = "nuplan" ]; then pip install --no-cache-dir pytest "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"; fi && \
    rm -rf /tmp/py123d

COPY py123d_entrypoint.py /app/

VOLUME ["/input", "/output"]

ENTRYPOINT ["python", "/app/py123d_entrypoint.py"]
