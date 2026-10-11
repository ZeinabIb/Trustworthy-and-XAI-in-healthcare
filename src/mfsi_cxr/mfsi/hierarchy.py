"""Coherence flags (Phase 3, step 12) from the approved label tree (constants.PARENT, ancestors()).

flag_l = [y_hat_l = 1] and, for some ANCESTOR a, [y_hat_a = 0] and [R_a >= theta_a]: the classifier calls the
child while confidently denying an ancestor (the head is flat, so this can happen). Ancestors, not only the
direct parent (DECISIONS 2026-10-11). No Finding rule (approved): No Finding positive together with a
pathology positive is a conflict; the label whose counterpart is reliable gets flagged.
"""

from __future__ import annotations

import numpy as np

from mfsi_cxr.constants import FINDINGS, NO_FINDING, PATHOLOGIES, ancestors


def coherence_flags(y_hat: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """y_hat (N, L) bool, r (N, L) reliability, theta (L,) per-finding 'reliable' threshold -> (N, L)."""
    y_hat = y_hat.astype(bool)
    reliable = r >= theta[None, :]
    flags = np.zeros_like(y_hat)
    for j, f in enumerate(FINDINGS):
        for a in ancestors(f):
            k = FINDINGS.index(a)
            flags[:, j] |= y_hat[:, j] & ~y_hat[:, k] & reliable[:, k]
    nf = FINDINGS.index(NO_FINDING)
    path = [FINDINGS.index(f) for f in PATHOLOGIES]
    for j in path:
        conflict = y_hat[:, j] & y_hat[:, nf]
        flags[:, j] |= conflict & reliable[:, nf]
        flags[:, nf] |= conflict & reliable[:, j]
    return flags
