from __future__ import annotations

from algos.contracts import AlgorithmDefinition
from algos.contracts import TensorField


def _base(field: TensorField, ops, params: dict[str, float], *, sign: float = 1.0) -> TensorField:
    tensor = field.tensor
    noise = ops.normal_like(tensor)
    mixed = ops.matmul(tensor, ops.transpose(tensor))
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    out = ops.add(ops.mul(beta, tensor), ops.mul(sign * alpha, mixed))
    out = ops.add(out, ops.mul(gamma, noise))
    field.tensor = out
    return field


def _quadratic(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    quad = ops.mul(tensor, tensor)
    out = ops.add(ops.mul(params["beta"], tensor), ops.mul(params["alpha"], quad))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _momentum(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    velocity = field.memory.get("velocity")
    if velocity is None:
        velocity = ops.zeros_like(tensor)
    grad = ops.matmul(tensor, ops.transpose(tensor))
    velocity = ops.add(ops.mul(0.85, velocity), ops.mul(params["alpha"], grad))
    out = ops.sub(ops.mul(params["beta"], tensor), velocity)
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.memory["velocity"] = velocity
    field.tensor = out
    return field


def _adam(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    m = field.memory.get("m")
    v = field.memory.get("v")
    if m is None:
        m = ops.zeros_like(tensor)
    if v is None:
        v = ops.zeros_like(tensor)
    grad = ops.matmul(tensor, ops.transpose(tensor))
    m = ops.add(ops.mul(0.9, m), ops.mul(0.1, grad))
    v = ops.add(ops.mul(0.99, v), ops.mul(0.01, ops.mul(grad, grad)))
    out = ops.sub(ops.mul(params["beta"], tensor), ops.mul(params["alpha"], m))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.memory["m"] = m
    field.memory["v"] = v
    field.tensor = out
    return field


def _flow(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    mixed = ops.matmul(tensor, ops.transpose(tensor))
    quad = ops.mul(tensor, tensor)
    out = ops.add(ops.mul(params["beta"], tensor), ops.mul(params["alpha"], mixed))
    out = ops.add(out, ops.mul(params["alpha"], quad))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


COMMON = {"alpha": 0.004, "beta": 0.95, "gamma": 0.03}


ALGORITHM_DEFINITIONS: dict[str, AlgorithmDefinition] = {
    "tensor_ops": AlgorithmDefinition("tensor_ops", COMMON, lambda f, o, p: _base(f, o, p, sign=1.0)),
    "jacobian": AlgorithmDefinition("jacobian", COMMON, lambda f, o, p: _base(f, o, p, sign=1.0)),
    "attention_surface": AlgorithmDefinition("attention_surface", COMMON, lambda f, o, p: _base(f, o, p, sign=1.0)),
    "gradient_descent": AlgorithmDefinition("gradient_descent", COMMON, lambda f, o, p: _base(f, o, p, sign=-1.0)),
    "information_bound": AlgorithmDefinition("information_bound", COMMON, lambda f, o, p: _base(f, o, p, sign=-1.0)),
    "momentum": AlgorithmDefinition("momentum", COMMON, _momentum),
    "thermo_learning": AlgorithmDefinition("thermo_learning", COMMON, _momentum),
    "adam": AlgorithmDefinition("adam", COMMON, _adam),
    "grokking": AlgorithmDefinition("grokking", COMMON, _adam),
    "forward_pass": AlgorithmDefinition("forward_pass", COMMON, _flow),
    "activation_flow": AlgorithmDefinition("activation_flow", COMMON, _flow),
    "scaling_laws": AlgorithmDefinition("scaling_laws", COMMON, _flow),
    "chain_rule": AlgorithmDefinition("chain_rule", COMMON, _quadratic),
    "norms": AlgorithmDefinition("norms", COMMON, _quadratic),
    "manifold_field": AlgorithmDefinition("manifold_field", COMMON, _quadratic),
}


def get_algorithm_definition(key: str) -> AlgorithmDefinition:
    if key not in ALGORITHM_DEFINITIONS:
        raise KeyError(f"Algorithm definition not found: {key}")
    return ALGORITHM_DEFINITIONS[key]
