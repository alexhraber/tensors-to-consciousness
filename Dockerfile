FROM rust:slim AS rust-builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build -p explorer --release

FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_CACHE_DIR=/tmp/uv-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

COPY --from=rust-builder /build/target/release/explorer /usr/local/bin/explorer

WORKDIR /workspace

CMD ["explorer"]

FROM runtime AS ci

COPY .github/ci/requirements-test.txt /tmp/requirements-test.txt
RUN uv pip install --system -r /tmp/requirements-test.txt

FROM ci AS render

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    xterm \
    xdotool \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
