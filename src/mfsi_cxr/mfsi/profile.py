"""Profile and margin (Phase 3, steps 7-8), with a per-finding offset b (Zeinab 2026-10-15; plan: b = 0).

p_+ = exp(-d_+/tau) / (exp(-d_+/tau) + exp(-(d_- - b)/tau)) = sigmoid((d_- - d_+ - b)/tau)   (lse stable)
r   = |p_+ - p_-| = |2 p_+ - 1| = tanh(|d_- - d_+ - b| / (2 tau))
tau and b are scalars or one value per finding (shape (L,), broadcast over the last axis of (N, L) distances).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp


def profile_pos(d_pos: np.ndarray, d_neg: np.ndarray, tau: float | np.ndarray,
                b: float | np.ndarray = 0.0) -> np.ndarray:
    """p_+ via sigmoid of the shifted, scaled distance gap."""
    return expit((d_neg - d_pos - b) / tau)


def profile_pos_lse(d_pos: np.ndarray, d_neg: np.ndarray, tau: float | np.ndarray,
                    b: float | np.ndarray = 0.0) -> np.ndarray:
    """p_+ via the softmax definition with log-sum-exp (reference form; equals profile_pos)."""
    logits = np.stack(np.broadcast_arrays(-d_pos / tau, -(d_neg - b) / tau))
    return np.exp(logits[0] - logsumexp(logits, axis=0))


def margin(d_pos: np.ndarray, d_neg: np.ndarray, tau: float | np.ndarray,
           b: float | np.ndarray = 0.0) -> np.ndarray:
    """r = tanh(|d_- - d_+ - b| / (2 tau))."""
    return np.tanh(np.abs(d_neg - d_pos - b) / (2.0 * tau))
