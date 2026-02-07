from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TensorField:
    tensor: Any
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


TransformFn = Callable[[TensorField, Any, dict[str, float]], TensorField]


@dataclass(frozen=True)
class AlgorithmDefinition:
    key: str
    defaults: dict[str, float]
    transform: TransformFn
