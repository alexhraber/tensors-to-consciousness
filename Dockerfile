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

# CI containers must be able to build/test Rust crates and build the optional PyO3 extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Minimal Rust toolchain for running `cargo test` inside the container.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable && \
    rustc --version && cargo --version

COPY .github/ci/requirements-test.txt /tmp/requirements-test.txt
RUN uv pip install --system -r /tmp/requirements-test.txt

FROM ci AS render

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    xterm \
    xdotool \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
