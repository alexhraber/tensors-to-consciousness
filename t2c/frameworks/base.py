"""Backend protocol and shared exceptions for framework integrations."""

from dataclasses import dataclass
from types import ModuleType


class BackendConfigurationError(RuntimeError):
    """Raised when a requested backend is unavailable or invalid."""


@dataclass(frozen=True)
class Backend:
    """A loaded framework backend."""

    name: str
    core: ModuleType
    nn: ModuleType | None = None
