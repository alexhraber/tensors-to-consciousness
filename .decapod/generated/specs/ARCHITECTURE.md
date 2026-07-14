# Architecture


<!-- decapod:capability-overlay:persistent-state:start -->


## Persistent State Architecture Overlay

### State Ownership
- Each entity type MUST have a designated state owner
- State ownership boundaries MUST be explicitly documented
- Cross-boundary state access MUST go through defined interfaces

### Transaction Boundaries
- All multi-entity mutations MUST occur within explicit transactions
- Transaction boundaries MUST be documented in ARCHITECTURE.md
- Compensating transactions for distributed operations

### Storage Abstraction
- Storage ownership, consistency behavior, and access boundaries MUST be explicit
- Portability or swappable implementations are project decisions, not universal requirements
- Migration and rollback treatment MUST match the selected storage technology
<!-- decapod:capability-overlay:persistent-state:end -->
## Direction
TUI/CLI tensor exploration tool with hybrid Python/Rust architecture. Rust host manages CLI/TUI lifecycle, Python provides framework-abstracted tensor transforms via maturin extension.

## What This Project Is
tensors-to-consciousness is a CLI+TUI application with a Rust host that drives Python tensor transform pipelines via PyO3/maturin.

Architectural principles:
- **Simplicity**: Framework adapters with a unified interface contract.
- **Modularity**: Clear separation between Rust CLI/TUI and Python transform engine.
- **Reliability**: Graceful degradation when framework backends are unavailable.

## Current Facts
- Runtime/languages: Python (transforms), Rust (CLI/TUI host)
- Detected surfaces/framework hints: cargo, maturin, python, pyproject
- Product type: CLI/TUI application

## Architecture Map
This project's architecture consists of the following key layers/directories:
- `frameworks/`: Framework-specific adapters (numpy, pytorch, jax, keras, mlx, cupy)
- `transforms/`: Transform catalog, registry, and definition system
- `tools/`: Core tooling, diagnostics, and headless utilities
- `crates/explorer/`: Rust TUI/CLI host (maturin/PyO3)
- `crates/core/`: Rust core library (PyO3 extension)
- `tests/`: Integration and unit test suite

## Data Flows
- User invokes Rust CLI/TUI (`explorer run`, `explorer tui`).
- Rust host spawns Python engine via PyO3 RPC.
- Python `FrameworkEngine` executes transform pipeline against selected backend (MLX/JAX/PyTorch/NumPy/Keras/CuPy).
- Results returned as tensors; Rust renders ASCII heatmap or TUI.

## Strongest Existing Primitives
- Define the strongest existing primitives in the codebase (e.g., helper utilities, base controllers, data access layers).

## Topology
```text
Host Application -> Library API -> Domain Core -> Adapters (Store / Network)
```

## Store Boundaries
```mermaid
flowchart LR
  I[Inbound Requests] --> C[Core Logic]
  C --> W[(Write Store)]
  C --> R[(Read Store)]
```

## Happy Path Sequence
```text
Client request -> API validation -> domain execution -> persistence -> response with trace id
```

## Error Path
```mermaid
sequenceDiagram
  participant Client
  participant Service
  participant Store
  Client->>Service: Request
  Service->>Store: Database Query
  Store--xService: Error/Timeout
  Service-->>Client: Typed Error / Recovery Instructions
```

## Execution Path
- Ingress parse + validation:
- Policy/interlock checks:
- Core execution + persistence:
- Verification and artifact emission:

## Concurrency and Runtime Model
- Execution model:
- Isolation boundaries:
- Backpressure strategy:
- Shared state synchronization:

## Deployment Topology
- Runtime units:
- Region/zone model:
- Rollout strategy (blue/green/canary):
- Rollback trigger and blast-radius scope:

## Data and Contracts
- Inbound contracts (CLI/API/events):
- Outbound dependencies (datastores/queues/external APIs):
- Data ownership boundaries:
- Schema evolution + migration policy:

## ADR Register
| ADR | Title | Status | Rationale | Date |
|---|---|---|---|---|
| ADR-001 | Initial topology choice | Proposed | Define first stable architecture | YYYY-MM-DD |

## Delivery Plan (first 3 slices)
- Slice 1 (ship first):
- Slice 2:
- Slice 3:

## Risks and Mitigations
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Contract drift across components | Medium | High | Spec + schema checks in CI |
| Runtime saturation under peak load | Medium | High | Capacity model + load tests |

<!-- decapod:codebase-attestation:start -->
## Codebase Attestation

- Repository signal fingerprint: `1e9f42833a8b5f6f33cf5478f761f0e7406b2d391779f229609a1635898af2f5`
- Significant implementation surfaces: `.github/` (3 files), `Cargo.lock/` (1 files), `Cargo.toml/` (1 files), `Dockerfile/` (1 files), `README.md/` (1 files), `crates/` (2 files), `docker-compose.yml/` (1 files), `docs/` (1 files), `examples/` (1 files), `pyproject.toml/` (1 files)
- Refreshed from the current codebase by `decapod specs.refresh`
<!-- decapod:codebase-attestation:end -->
