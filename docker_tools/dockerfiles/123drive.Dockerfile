ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/123Drive

WORKDIR /app/123Drive

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra convert --frozen --no-dev

COPY docker_tools/123drive_entrypoint.py /app/123drive_entrypoint.py

VOLUME ["/input", "/output"]

ENTRYPOINT ["python", "/app/123drive_entrypoint.py"]
