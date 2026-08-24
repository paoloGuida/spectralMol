from __future__ import annotations
import numpy as np
from .config import CFG

def build_fourier_basis(L: int, K: int) -> np.ndarray:
    # Defensive casting: some configs/CLI args may come through as floats (e.g. from YAML/JSON).
    # NumPy expects integer shapes and Python expects integer ranges.
    L = int(L)
    K = int(K)
    if L <= 0:
        raise ValueError(f"L must be a positive integer, got {L!r}")
    if K < 0:
        raise ValueError(f"K must be a non-negative integer, got {K!r}")

    i = np.arange(L, dtype=np.float64)[:, None]
    Phi = np.ones((L, 1 + 2 * K), dtype=np.float64)
    for k in range(1, K + 1):
        Phi[:, k] = np.cos(2.0 * np.pi * k * i[:, 0] / L)
        Phi[:, K + k] = np.sin(2.0 * np.pi * k * i[:, 0] / L)
    return Phi

def theta_to_Z(Theta: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    return Phi @ Theta

def mutate_Theta(Theta: np.ndarray, gen: int, generations: int, rng: np.random.RandomState) -> np.ndarray:
    Y = Theta.copy()
    M, Dloc = Y.shape

    anneal = 1.0 - 0.20 * (gen / max(1, generations - 1))
    sigma = CFG.GAUSS_STD_THETA * anneal

    mask = (rng.rand(M, Dloc) < CFG.P_PARAM_NOISE)
    if np.any(mask):
        Y[mask] += rng.normal(0.0, sigma, size=int(mask.sum()))

    row_reset = (rng.rand(M) < CFG.P_ROW_RESET)
    if np.any(row_reset):
        Y[row_reset, :] = rng.normal(0.0, CFG.THETA_INIT_STD, size=(int(row_reset.sum()), Dloc))

    if CFG.CLIP_THETA_NORM and CFG.CLIP_THETA_NORM > 0:
        norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, CFG.CLIP_THETA_NORM / norms)
        Y = Y * scale

    return Y

def make_initial_population(pop_size: int, M: int, D: int, rng: np.random.RandomState) -> list[np.ndarray]:
    pop: list[np.ndarray] = []
    for _ in range(pop_size):
        Theta = rng.normal(0.0, CFG.THETA_INIT_STD, size=(M, D)).astype(np.float64)
        Theta = mutate_Theta(Theta, gen=0, generations=CFG.GENERATIONS, rng=rng)
        pop.append(Theta)
    return pop
