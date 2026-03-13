ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG EXTRAS=nuplan
ARG PY123D_REF=2a71af776ee47392f548c7cd327afefa995faa67

ENV HYDRA_FULL_ERROR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/autonomousvision/py123d.git /tmp/py123d && \
    git -C /tmp/py123d checkout "${PY123D_REF}" && \
    pip install --no-cache-dir "/tmp/py123d[${EXTRAS}]" && \
    if [ "${EXTRAS}" = "nuplan" ]; then pip install --no-cache-dir pytest "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"; fi && \
    rm -rf /tmp/py123d

ARG DATASET
ENV DATASET=${DATASET}

COPY py123d_run_conversion.py py123d_config.py /app/

ENTRYPOINT ["python", "/app/py123d_run_conversion.py"]
