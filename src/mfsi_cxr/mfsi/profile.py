"""Profile and margin (Phase 3, steps 7-8).

p_+ = exp(-d_+/tau) / (exp(-d_+/tau) + exp(-d_-/tau)) = sigmoid((d_- - d_+)/tau)   (log-sum-exp stable)
r   = |p_+ - p_-| = |2 p_+ - 1| = tanh(|d_- - d_+| / (2 tau))
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp


def profile_pos(d_pos: np.ndarray, d_neg: np.ndarray, tau: float) -> np.ndarray:
    """p_+ via sigmoid of the scaled distance gap."""
    return expit((d_neg - d_pos) / tau)


def profile_pos_lse(d_pos: np.ndarray, d_neg: np.ndarray, tau: float) -> np.ndarray:
    """p_+ via the softmax definition with log-sum-exp (reference form; equals profile_pos)."""
    logits = np.stack([-d_pos / tau, -d_neg / tau])
    return np.exp(logits[0] - logsumexp(logits, axis=0))


def margin(d_pos: np.ndarray, d_neg: np.ndarray, tau: float) -> np.ndarray:
    """r = tanh(|d_- - d_+| / (2 tau))."""
    return np.tanh(np.abs(d_neg - d_pos) / (2.0 * tau))
