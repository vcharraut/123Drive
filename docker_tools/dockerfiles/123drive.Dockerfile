ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG DRIVE123_REF=main

COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "123Drive @ git+https://github.com/vcharraut/123Drive@${DRIVE123_REF}"

ENTRYPOINT ["convert"]
