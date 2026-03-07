ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG EXTRAS=nuplan
ARG PY123D_REF=dev_v0.1.0

COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/autonomousvision/py123d.git /tmp/py123d && \
    git -C /tmp/py123d checkout "${PY123D_REF}" && \
    uv pip install --system "/tmp/py123d[${EXTRAS}]" && \
    if [ "${EXTRAS}" = "nuplan" ]; then uv pip install --system pytest "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"; fi && \
    rm -rf /tmp/py123d

ENTRYPOINT ["py123d-conversion"]
