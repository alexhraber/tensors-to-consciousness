# Contributing

This project is maintained as an architecture-first exploration platform. Contributions are expected to preserve contract clarity, runtime reliability, and operator ergonomics.

## System Model

- `transforms/`: canonical transform catalog and metadata definitions.
- `frameworks/<framework>/`: backend execution adapters.
- `tools/`: runtime, diagnostics, rendering, and TUI surfaces.
- `explorer` (Rust): primary user-facing entrypoint and TUI host.
- `crates/core/`: optional acceleration kernels consumed by Python bridges.

## Local Workflow

```bash
mise install
mise run install-test-deps
python tools/setup_contributor.py
python tools/install_githooks.py
python -m pip install pre-commit
pre-commit install
mise run rust-build
./target/debug/explorer
./target/debug/explorer run --framework jax --transforms default
mise run test-all
```

## Commit Hygiene

- Keep changes scoped and reviewable.
- Use conventional commit subjects: `type(scope): summary` (enforced by `.githooks/commit-msg`).
- Branch-first policy: commits and pushes from `main`/`master` are blocked by hooks (override only with `ALLOW_MAIN_COMMIT=1` or `ALLOW_MAIN_PUSH=1`).
- Branch naming policy is enforced by hooks via `tools/git_policy.py`.
- Required branch format: `type/scope-short-topic` (lowercase kebab-case).
- Allowed branch types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`.
- Valid examples: `fix/ci-act-container-collision`, `docs/readme-minimal-refresh`, `chore/pre-push-policy-hardening`.
- Regenerate generated references when catalog/framework contracts change.
- Do not commit runtime state (`.config/`, virtual environments, caches, output scratch files).

## Hooks and Automation

- Repo hook entry: `.githooks/pre-commit`
- Repo hook entry: `.githooks/commit-msg`
- Repo hook entry: `.githooks/pre-push`
- Installer: `python tools/install_githooks.py`
- Contributor bootstrap: `python tools/setup_contributor.py` (auto-invoked by `.githooks/pre-commit` when needed)
- Pre-commit hook: lightweight bootstrap only
- Catalog docs generator: `mise run docs-generate`
- Render asset generator: `mise run assets-regenerate`
- Core build: `./tools/build_core.sh`
- Local Actions simulation with `act` (workflow-driven, executes `mise` tasks): `mise run act-ci`
- Full pre-push gate (single validation choke point: hook -> `act` -> workflow -> `mise`): `mise run pre-push` (also runs automatically via `.githooks/pre-push`, and only runs jobs for changed paths)
- PR submission helper: `mise run submit-pr` (pushes current feature branch and opens/reuses a PR via GitHub API with DNS override fallback)
- Optional DNS override list for GitHub API: `GH_API_RESOLVE_IPS=140.82.114.6,140.82.113.6,140.82.112.6`
- Pre-push cache: successful gate jobs are cached per change signature in `.git/explorer-cache/act-gate.json` to keep repeat loops fast
- Pre-push parallelism: default is all local cores (`CI_GATE_JOBS=nproc`); set `CI_GATE_JOBS=<n|nproc>` (or run `python tools/pre_push_gate.py --jobs <n|nproc>`) to tune concurrency

Commit message examples:

- `fix(tui): handle ctrl+c cleanly`
- `chore(ci): consolidate python target to 3.14`
- `docs: refine operator workflow`

## Adding or Updating Framework Backends

Each framework backend must include:

- `frameworks/<framework>/utils.py`
- `frameworks/<framework>/test_setup.py`

Optional backend-specialized helpers may live under `frameworks/<framework>/transforms/`.

Requirements:

- preserve conceptual behavior across frameworks
- prefer native autodiff where supported
- explicitly document numerical approximation tradeoffs

## Transform and Runtime Checklist

1. Add or modify transform definitions in `transforms/transforms.json`.
2. Verify registry/catalog resolution remains valid.
3. Validate setup and run paths through `explorer`.
4. Run both unit and integration test suites.

Validation commands:

```bash
./target/debug/explorer list-transforms
./target/debug/explorer validate --framework jax
./target/debug/explorer run --framework jax --transforms default
mise run test-unit
mise run test-integration
```

## Documentation Expectations

- Keep `README.md` concise and operator-oriented.
- Keep implementation detail in `docs/` reference and usage guides.
- Ensure every user-facing command in docs is executable as written.
- Keep `docs/reference/transforms.md` and `docs/reference/frameworks.md` synchronized with source metadata.
