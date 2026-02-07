from __future__ import annotations

from algos.catalog import catalog_transforms
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


def _laplacian_diffusion(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    lap = ops.sub(ops.matmul(tensor, ops.transpose(tensor)), tensor)
    out = ops.add(tensor, ops.mul(params["alpha"], lap))
    out = ops.add(ops.mul(params["beta"], out), ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _advection_transport(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    forward = ops.matmul(tensor, ops.transpose(tensor))
    backward = ops.matmul(ops.transpose(tensor), tensor)
    drift = ops.sub(forward, backward)
    out = ops.sub(ops.mul(params["beta"], tensor), ops.mul(params["alpha"], drift))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _reaction_diffusion(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    a = field.tensor
    b = field.memory.get("rd_b")
    if b is None:
        b = ops.mul(0.1, ops.normal_like(a))
    lap_a = ops.sub(ops.matmul(a, ops.transpose(a)), a)
    lap_b = ops.sub(ops.matmul(b, ops.transpose(b)), b)
    reaction = ops.mul(a, b)
    a_next = ops.sub(ops.add(a, ops.mul(params["alpha"], lap_a)), ops.mul(params["gamma"], reaction))
    b_next = ops.sub(
        ops.add(b, ops.add(ops.mul(params["alpha"], lap_b), ops.mul(params["gamma"], reaction))),
        ops.mul(0.1, b),
    )
    field.memory["rd_b"] = b_next
    field.tensor = ops.add(ops.mul(params["beta"], a_next), ops.mul(params["gamma"], ops.normal_like(a)))
    return field


def _spectral_filter(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    low_band = ops.mul(0.5, ops.add(tensor, ops.transpose(tensor)))
    high_band = ops.sub(tensor, low_band)
    out = ops.add(ops.mul(params["beta"], low_band), ops.mul(params["alpha"], high_band))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _wave_propagation(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    prev = field.memory.get("wave_prev")
    if prev is None:
        prev = tensor
    velocity = ops.sub(tensor, prev)
    lap = ops.sub(ops.matmul(tensor, ops.transpose(tensor)), tensor)
    out = ops.add(tensor, ops.add(ops.mul(params["beta"], velocity), ops.mul(params["alpha"], lap)))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.memory["wave_prev"] = tensor
    field.tensor = out
    return field


def _hamiltonian_step(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    q = field.tensor
    p = field.memory.get("hamiltonian_p")
    if p is None:
        p = ops.zeros_like(q)
    grad = ops.matmul(q, ops.transpose(q))
    p = ops.sub(ops.mul(params["beta"], p), ops.mul(params["alpha"], grad))
    q_next = ops.add(q, p)
    q_next = ops.add(q_next, ops.mul(params["gamma"], ops.normal_like(q)))
    field.memory["hamiltonian_p"] = p
    field.tensor = q_next
    return field


def _attention_message_passing(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    affinity = ops.matmul(tensor, ops.transpose(tensor))
    messages = ops.matmul(affinity, tensor)
    out = ops.add(ops.mul(params["beta"], tensor), ops.mul(params["alpha"], messages))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _tensor_decomposition(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    left = ops.matmul(tensor, ops.transpose(tensor))
    right = ops.matmul(ops.transpose(tensor), tensor)
    recon = ops.add(ops.matmul(left, tensor), ops.matmul(tensor, right))
    out = ops.add(ops.mul(params["beta"], tensor), ops.mul(params["alpha"], recon))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _topology_regularization(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    symmetric = ops.mul(0.5, ops.add(tensor, ops.transpose(tensor)))
    curvature = ops.sub(ops.matmul(symmetric, ops.transpose(symmetric)), symmetric)
    out = ops.sub(ops.mul(params["beta"], symmetric), ops.mul(params["alpha"], curvature))
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _entropy_flow(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    energy = ops.mul(tensor, tensor)
    coupling = ops.matmul(tensor, ops.transpose(tensor))
    out = ops.add(
        ops.sub(ops.mul(params["beta"], tensor), ops.mul(params["alpha"], energy)),
        ops.mul(0.5 * params["alpha"], coupling),
    )
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _stochastic_process(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    mean = field.memory.get("ou_mean")
    if mean is None:
        mean = ops.zeros_like(tensor)
    restoring = ops.sub(tensor, mean)
    drift = ops.mul(-params["alpha"], restoring)
    out = ops.add(ops.mul(params["beta"], tensor), drift)
    out = ops.add(out, ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


def _constraint_projection(field: TensorField, ops, params: dict[str, float]) -> TensorField:
    tensor = field.tensor
    symmetric = ops.mul(0.5, ops.add(tensor, ops.transpose(tensor)))
    gram = ops.matmul(symmetric, ops.transpose(symmetric))
    projected = ops.sub(symmetric, ops.mul(params["alpha"], ops.sub(gram, symmetric)))
    out = ops.add(ops.mul(params["beta"], projected), ops.mul(params["gamma"], ops.normal_like(tensor)))
    field.tensor = out
    return field


COMMON = {"alpha": 0.004, "beta": 0.95, "gamma": 0.03}


TRANSFORM_IMPLS = {
    "base_pos": lambda f, o, p: _base(f, o, p, sign=1.0),
    "base_neg": lambda f, o, p: _base(f, o, p, sign=-1.0),
    "quadratic": _quadratic,
    "momentum": _momentum,
    "adam": _adam,
    "flow": _flow,
    "laplacian_diffusion": _laplacian_diffusion,
    "advection_transport": _advection_transport,
    "reaction_diffusion": _reaction_diffusion,
    "spectral_filter": _spectral_filter,
    "wave_propagation": _wave_propagation,
    "hamiltonian_step": _hamiltonian_step,
    "attention_message_passing": _attention_message_passing,
    "tensor_decomposition": _tensor_decomposition,
    "topology_regularization": _topology_regularization,
    "entropy_flow": _entropy_flow,
    "stochastic_process": _stochastic_process,
    "constraint_projection": _constraint_projection,
}


def _definition_for_entry(entry: dict[str, object]) -> AlgorithmDefinition:
    key = str(entry["key"])
    transform_name = str(entry.get("transform", ""))
    transform = TRANSFORM_IMPLS.get(transform_name)
    if transform is None:
        raise KeyError(f"Unknown transform implementation '{transform_name}' for key '{key}'")
    return AlgorithmDefinition(key, COMMON, transform)


TRANSFORM_DEFINITIONS: dict[str, AlgorithmDefinition] = {
    str(entry["key"]): _definition_for_entry(entry) for entry in catalog_transforms()
}

# Backward-compatible alias.
ALGORITHM_DEFINITIONS = TRANSFORM_DEFINITIONS


def get_transform_definition(key: str) -> AlgorithmDefinition:
    if key not in TRANSFORM_DEFINITIONS:
        raise KeyError(f"Transform definition not found: {key}")
    return TRANSFORM_DEFINITIONS[key]


def get_algorithm_definition(key: str) -> AlgorithmDefinition:
    return get_transform_definition(key)
