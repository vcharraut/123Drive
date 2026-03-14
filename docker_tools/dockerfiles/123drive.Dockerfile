ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG DRIVE123_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir "123Drive @ git+https://github.com/vcharraut/123Drive@${DRIVE123_REF}"

COPY 123drive_entrypoint.py /app/123drive_entrypoint.py

VOLUME ["/input", "/output"]

ENTRYPOINT ["python", "/app/123drive_entrypoint.py"]
