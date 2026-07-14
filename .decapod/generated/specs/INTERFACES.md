# Interfaces


<!-- decapod:capability-overlay:public-api:start -->


## Public API Capability Overlay

### API Contract Requirements
- All public endpoints MUST define explicit request/response schemas
- Versioning strategy MUST be documented (URL path or header-based)
- All public endpoints MUST implement idempotency for mutating operations
- Rate limiting and pagination MUST be implemented for list endpoints

### Compatibility Guarantees
- Backward-compatible changes ONLY within a version
- Breaking changes require new version (v1, v2, etc.)
- Deprecation and removal policy MUST be selected for this project and proven against its consumers

### Security Requirements
- All public endpoints MUST implement authentication
- Abuse-control enforcement point MUST be a documented project decision
- Input validation MUST reject malformed requests with typed errors
<!-- decapod:capability-overlay:public-api:end -->
## Contract Principles
- Prefer explicit schemas over implicit behavior.
- Every mutating interface defines idempotency semantics.
- Every failure path maps to a typed, documented error code.

## Generated Contract Depth
Generated interface specs should include:
- API/CLI contracts with request/response schemas.
- Read/write ownership for each storage path.
- Idempotency and retry behavior for mutations.
- Typed failure classes and recovery instructions.

## API / RPC Contracts
| Interface | Method | Request Schema | Response Schema | Errors | Idempotency |
|---|---|---|---|---|---|
| `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

## Event Consumers
| Consumer | Event | Ordering Requirement | Retry Policy | DLQ Policy |
|---|---|---|---|---|
| `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

## Outbound Dependencies
| Dependency | Purpose | SLA | Timeout | Circuit-Breaker |
|---|---|---|---|---|
| `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

## Inbound Contracts
- API / RPC entrypoints: Rust CLI commands (run, tui, list-transforms, validate, render-tensor)
- CLI surfaces: `explorer [run|tui|list-transforms|validate|render-tensor|ops]`
- Event/webhook consumers: None (CLI tool, not a service)
- Repository-detected surfaces: cargo, maturin, python, pyproject

## Data Ownership
- Source-of-truth tables/collections:
- Cross-boundary read models:
- Consistency expectations:

## Error Taxonomy Example (service_or_library)
```python
class ApiError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
```

## Failure Semantics
| Failure Class | Retry/Backoff | Client Contract | Observability |
|---|---|---|---|
| Validation | No retry | 4xx typed error | warn log + metric |
| Dependency timeout | Exponential backoff | 503 with retryable code | error log + alert |
| Conflict | Conditional retry | 409 with conflict detail | info log + metric |

## Timeout Budget
| Hop | Budget (ms) | Notes |
|---|---|---|
| Client -> Edge/API | 500 | Includes auth + routing |
| API -> Domain | 300 | Includes validation |
| Domain -> Store/Dependency | 200 | Includes retry overhead |

## Interface Versioning
- Version strategy (`v1`, date-based, semver):
- Backward-compatibility guarantees:
- Deprecation window and removal policy:

<!-- decapod:codebase-attestation:start -->
## Codebase Attestation

- Repository signal fingerprint: `1e9f42833a8b5f6f33cf5478f761f0e7406b2d391779f229609a1635898af2f5`
- Significant implementation surfaces: `.github/` (3 files), `Cargo.lock/` (1 files), `Cargo.toml/` (1 files), `Dockerfile/` (1 files), `README.md/` (1 files), `crates/` (2 files), `docker-compose.yml/` (1 files), `docs/` (1 files), `examples/` (1 files), `pyproject.toml/` (1 files)
- Refreshed from the current codebase by `decapod specs.refresh`
<!-- decapod:codebase-attestation:end -->
