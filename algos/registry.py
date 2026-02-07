from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmPreset:
    samples: int
    freq: float
    amplitude: float
    damping: float
    noise: float
    phase: float
    grid: int


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    title: str
    description: str
    formula: str
    complexity: int
    source_module: str
    preset: AlgorithmPreset


ALGORITHM_SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        "tensor_ops",
        "Tensor Operations",
        "Elementwise and matrix operations over structured tensors.",
        "C = A ⊙ B,  M = A @ B",
        0,
        "computational_primitives",
        AlgorithmPreset(900, 1.4, 0.9, 0.05, 0.06, 0.2, 72),
    ),
    AlgorithmSpec(
        "norms",
        "Norm Geometry",
        "Magnitude and stability readout across tensor trajectories.",
        "||x||₂ = sqrt(sum_i x_i²)",
        0,
        "computational_primitives",
        AlgorithmPreset(980, 1.6, 1.0, 0.06, 0.05, 0.3, 76),
    ),
    AlgorithmSpec(
        "chain_rule",
        "Chain Rule Field",
        "Derivative amplification through nested nonlinearities.",
        "d/dx sin(x²) = 2x cos(x²)",
        1,
        "automatic_differentiation",
        AlgorithmPreset(1100, 1.9, 1.1, 0.08, 0.04, 0.55, 88),
    ),
    AlgorithmSpec(
        "jacobian",
        "Jacobian Sensitivity",
        "Multivariate gradient coupling across dimensions.",
        "J_ij = ∂f_i/∂x_j",
        1,
        "automatic_differentiation",
        AlgorithmPreset(1180, 2.05, 1.05, 0.09, 0.05, 0.7, 90),
    ),
    AlgorithmSpec(
        "gradient_descent",
        "Gradient Descent",
        "Iterative descent dynamics over a curved objective.",
        "xₜ₊₁ = xₜ - η∇f(xₜ)",
        2,
        "optimization_theory",
        AlgorithmPreset(1200, 2.2, 1.0, 0.13, 0.10, 0.9, 96),
    ),
    AlgorithmSpec(
        "momentum",
        "Momentum Descent",
        "Velocity-augmented traversal with inertia memory.",
        "vₜ₊₁ = βvₜ + ∇f(xₜ),  xₜ₊₁ = xₜ - ηvₜ₊₁",
        2,
        "optimization_theory",
        AlgorithmPreset(1280, 2.35, 1.1, 0.11, 0.12, 1.05, 98),
    ),
    AlgorithmSpec(
        "adam",
        "Adam Dynamics",
        "Adaptive first/second moment optimization geometry.",
        "xₜ₊₁ = xₜ - η m̂ₜ / (sqrt(v̂ₜ)+ε)",
        2,
        "optimization_theory",
        AlgorithmPreset(1320, 2.5, 1.05, 0.12, 0.13, 1.15, 102),
    ),
    AlgorithmSpec(
        "forward_pass",
        "Forward Composition",
        "Layered nonlinear transformation and feature shaping.",
        "y = σ(W₂ σ(W₁x + b₁) + b₂)",
        3,
        "neural_theory",
        AlgorithmPreset(1300, 2.5, 1.2, 0.09, 0.12, 1.2, 104),
    ),
    AlgorithmSpec(
        "activation_flow",
        "Activation Flow",
        "Activation distribution drift across depth.",
        "a_l = φ(W_l a_{l-1} + b_l)",
        3,
        "neural_theory",
        AlgorithmPreset(1360, 2.7, 1.18, 0.10, 0.13, 1.35, 108),
    ),
    AlgorithmSpec(
        "manifold_field",
        "Manifold Field",
        "Curved latent field with coupled oscillatory terms.",
        "z = exp(-λ||x||²) · sin(ωx) · cos(ωy)",
        4,
        "advanced_theory",
        AlgorithmPreset(1450, 2.9, 1.25, 0.11, 0.14, 1.55, 116),
    ),
    AlgorithmSpec(
        "attention_surface",
        "Attention Surface",
        "Softmax geometry over query-key interactions.",
        "Attn(Q,K,V) = softmax(QK^T/√d)V",
        4,
        "advanced_theory",
        AlgorithmPreset(1500, 3.05, 1.2, 0.12, 0.15, 1.7, 120),
    ),
    AlgorithmSpec(
        "scaling_laws",
        "Scaling Laws",
        "Performance trends across model/data scale.",
        "L(N,D) ≈ A N^-α + B D^-β + C",
        5,
        "research_frontiers",
        AlgorithmPreset(1550, 3.2, 1.3, 0.15, 0.18, 1.9, 124),
    ),
    AlgorithmSpec(
        "grokking",
        "Grokking Transition",
        "Delayed generalization phase shift dynamics.",
        "gen_gap(t) = L_test(t) - L_train(t)",
        5,
        "research_frontiers",
        AlgorithmPreset(1620, 3.35, 1.28, 0.16, 0.20, 2.05, 128),
    ),
    AlgorithmSpec(
        "information_bound",
        "Information Bound",
        "Information transfer upper bounds under entropy constraints.",
        "I(X;Y) ≤ min(H(X), H(Y))",
        6,
        "theoretical_limits",
        AlgorithmPreset(1700, 3.6, 1.35, 0.17, 0.20, 2.2, 132),
    ),
    AlgorithmSpec(
        "thermo_learning",
        "Thermodynamics of Learning",
        "Energy/work dynamics across optimization states.",
        "ΔE = W - Q",
        6,
        "theoretical_limits",
        AlgorithmPreset(1760, 3.75, 1.32, 0.18, 0.22, 2.35, 136),
    ),
    AlgorithmSpec(
        "laplacian_diffusion",
        "Laplacian Diffusion",
        "Spatial smoothing and denoising through Laplacian-like flow.",
        "X_{t+1} = X_t + αΔX_t",
        3,
        "field_dynamics",
        AlgorithmPreset(1220, 2.15, 1.02, 0.10, 0.07, 1.00, 96),
    ),
    AlgorithmSpec(
        "advection_transport",
        "Advection Transport",
        "Directional tensor transport under a velocity-like drift field.",
        "X_{t+1} = X_t - dt(v · ∇X_t)",
        3,
        "field_dynamics",
        AlgorithmPreset(1260, 2.25, 1.08, 0.11, 0.08, 1.08, 98),
    ),
    AlgorithmSpec(
        "reaction_diffusion",
        "Reaction Diffusion",
        "Coupled field reaction and diffusion for pattern emergence.",
        "∂u/∂t = D∇²u + R(u,v)",
        4,
        "field_dynamics",
        AlgorithmPreset(1320, 2.35, 1.16, 0.11, 0.09, 1.20, 104),
    ),
    AlgorithmSpec(
        "spectral_filter",
        "Spectral Filter",
        "Frequency-selective mutation over tensor energy bands.",
        "X' = F^{-1}(H(ω) ⊙ F(X))",
        4,
        "spectral_methods",
        AlgorithmPreset(1340, 2.45, 1.05, 0.10, 0.08, 1.28, 106),
    ),
    AlgorithmSpec(
        "wave_propagation",
        "Wave Propagation",
        "Second-order wave-like propagation through the tensor field.",
        "∂²X/∂t² = c²∇²X",
        4,
        "field_dynamics",
        AlgorithmPreset(1380, 2.55, 1.14, 0.10, 0.10, 1.36, 110),
    ),
    AlgorithmSpec(
        "hamiltonian_step",
        "Hamiltonian Step",
        "Energy-conserving-style state updates over position and momentum.",
        "q_{t+1}=q_t+∂H/∂p,  p_{t+1}=p_t-∂H/∂q",
        5,
        "geometric_learning",
        AlgorithmPreset(1440, 2.70, 1.20, 0.09, 0.09, 1.52, 114),
    ),
    AlgorithmSpec(
        "attention_message_passing",
        "Attention Message Passing",
        "Nonlocal coupling via affinity-driven message aggregation.",
        "M = softmax(QK^T)V",
        5,
        "graph_dynamics",
        AlgorithmPreset(1480, 2.80, 1.18, 0.10, 0.11, 1.66, 118),
    ),
    AlgorithmSpec(
        "tensor_decomposition",
        "Tensor Decomposition",
        "Factorized tensor reconstruction and rank-structured mutation.",
        "X ≈ Σ_r a_r ⊗ b_r ⊗ c_r",
        5,
        "geometric_learning",
        AlgorithmPreset(1520, 2.92, 1.22, 0.11, 0.11, 1.78, 122),
    ),
    AlgorithmSpec(
        "topology_regularization",
        "Topology Regularization",
        "Shape-aware smoothing that penalizes unstable curvature.",
        "L = L_task + λL_topo",
        6,
        "constraint_systems",
        AlgorithmPreset(1580, 3.05, 1.24, 0.12, 0.12, 1.92, 126),
    ),
    AlgorithmSpec(
        "entropy_flow",
        "Entropy Flow",
        "Entropy-constrained dynamics balancing collapse and exploration.",
        "X_{t+1} = X_t - η∇(E - τH)",
        6,
        "stochastic_dynamics",
        AlgorithmPreset(1620, 3.18, 1.26, 0.13, 0.14, 2.06, 130),
    ),
    AlgorithmSpec(
        "stochastic_process",
        "Stochastic Process",
        "Noise-driven updates with drift and mean reversion behavior.",
        "dX_t = θ(μ-X_t)dt + σdW_t",
        6,
        "stochastic_dynamics",
        AlgorithmPreset(1660, 3.28, 1.20, 0.14, 0.16, 2.18, 132),
    ),
    AlgorithmSpec(
        "constraint_projection",
        "Constraint Projection",
        "Projection-like correction onto structured admissible states.",
        "X_{t+1} = Π_C(X_t)",
        6,
        "constraint_systems",
        AlgorithmPreset(1700, 3.38, 1.18, 0.14, 0.15, 2.30, 134),
    ),
)


ALGO_MAP = {spec.key: spec for spec in ALGORITHM_SPECS}
DEFAULT_ALGO_KEYS: tuple[str, ...] = ("tensor_ops", "chain_rule", "gradient_descent")


def list_algorithm_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in ALGORITHM_SPECS)


def resolve_algorithm_keys(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_ALGO_KEYS
    s = raw.strip().lower()
    if not s or s == "default":
        return DEFAULT_ALGO_KEYS
    if s == "all":
        return list_algorithm_keys()

    out: list[str] = []
    for part in s.split(","):
        key = part.strip()
        if not key:
            continue
        if key not in ALGO_MAP:
            allowed = ", ".join(list_algorithm_keys())
            raise ValueError(f"Unknown algo '{key}'. Available: {allowed}")
        if key not in out:
            out.append(key)
    if not out:
        return DEFAULT_ALGO_KEYS
    return tuple(out)


def specs_for_keys(keys: tuple[str, ...]) -> tuple[AlgorithmSpec, ...]:
    return tuple(ALGO_MAP[k] for k in keys)


def build_tui_algorithms(keys: tuple[str, ...] | None = None) -> tuple[dict[str, object], ...]:
    selected = specs_for_keys(keys if keys is not None else DEFAULT_ALGO_KEYS)
    return tuple(
        {
            "key": spec.key,
            "title": spec.title,
            "description": spec.description,
            "formula": spec.formula,
            "complexity": spec.complexity,
            "source_module": spec.source_module,
            "preset": {
                "samples": spec.preset.samples,
                "freq": spec.preset.freq,
                "amplitude": spec.preset.amplitude,
                "damping": spec.preset.damping,
                "noise": spec.preset.noise,
                "phase": spec.preset.phase,
                "grid": spec.preset.grid,
            },
        }
        for spec in selected
    )


def build_tui_profiles(keys: tuple[str, ...] | None = None) -> tuple[dict[str, object], ...]:
    algos = build_tui_algorithms(keys)
    return tuple(
        {
            "id": algo["key"],
            "title": algo["title"],
            "complexity": algo["complexity"],
            "algorithms": (algo,),
        }
        for algo in algos
    )
