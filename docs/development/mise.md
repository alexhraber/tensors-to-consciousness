# Mise Task Runtime

`mise` is the canonical local task/runtime interface for this repository.

CI calls `mise run ...` tasks directly, so running the same tasks locally reproduces CI behavior.

## Install and Bootstrap

```bash
mise install
mise run install-test-deps
```

## Core Task Surface

```bash
mise tasks ls
mise run compile
mise run test-unit
mise run test-integration
mise run test-all
mise run cov-unit
mise run cov-integration
mise run cov-report
mise run docs-generate
mise run docs-verify
mise run contract-transforms
FRAMEWORK=jax mise run contract-framework
mise run assets-regenerate
mise run pre-push
```

## Local Gate (Docker-First)

The pre-push hook runs a path-scoped gate that executes CI-equivalent commands in a local Docker image.

Full gate used by the pre-push hook:

```bash
mise run pre-push
```

Pre-push behavior:

- detects changed files relative to upstream
- selects only relevant CI job scopes
- runs selected scopes in parallel across local CPU cores by default (`CI_GATE_JOBS=nproc`)
- runs those scopes in a local Docker image (same as CI)
- caches successful local gate runs in `.git/explorer-cache/ci-gate.json` to speed repeated loops on unchanged signatures

Pre-commit behavior:

- contributor bootstrap only (`explorer ops bootstrap`)
- full validation intentionally runs at pre-push to avoid duplicate local pain

Emergency bypass for one push:

```bash
SKIP_ACT=1 git push
```

Force cache bypass for a single gate run:

```bash
CI_GATE_NO_CACHE=1 mise run pre-push
```

## Render and Headless Capture

Linux-only system dependency task:

```bash
mise run install-render-system-deps
```

Smoke checks:

```bash
mise run render-verify-assets
mise run render-smoke-progressions
mise run render-smoke-tui
```
