ARG PYTHON_VERSION=3.13
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

# py123d build context: the local checkout that 123Drive is locked to (see docker_tools/cli.py)
COPY --from=py123d pyproject.toml README.md LICENSE /tmp/py123d/
COPY --from=py123d src /tmp/py123d/src

# Pin shared dependencies to 123Drive's uv.lock: newer numpy/shapely change the large CARLA maps.
# wrapt stays free: the nuplan extra's aiobotocore needs wrapt<2, and the lock pins 2.x only for ray.
COPY --from=ghcr.io/astral-sh/uv:0.8.21 /uv /bin/
COPY --from=123drive pyproject.toml uv.lock /tmp/123Drive/
RUN cd /tmp/123Drive && \
    uv export --frozen --no-hashes --no-emit-project --no-emit-package py123d --no-emit-package wrapt \
    -o /tmp/constraints.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -c /tmp/constraints.txt "/tmp/py123d${EXTRAS:+[${EXTRAS}]}" && \
    if [ "${EXTRAS}" = "nuplan" ]; then pip install --no-cache-dir -c /tmp/constraints.txt "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"; fi && \
    rm -rf /tmp/py123d /tmp/123Drive

COPY py123d_entrypoint.py /app/

VOLUME ["/input", "/output"]

ENTRYPOINT ["python", "/app/py123d_entrypoint.py"]
