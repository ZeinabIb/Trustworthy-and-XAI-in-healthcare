"""Baselines for RQ1 (Phase 4), all from the same frozen exports; every score is a TRUST score (N, L),
higher = more reliable.

- msp: max(p, 1 - p) of the classifier probability.
- ensemble_var: minus the variance of p across seeds (needs >= 2 seeds; absent otherwise).
- Distance baselines run in MFSI's train-PCA space (Zeinab 2026-10-15), per modality and on the concatenation
  [P_img z_img ; P_txt z_txt]; clouds and prototypes from TRAIN, as in MFSI:
  - tied_maha_*: Lee et al. 2018, one covariance shared by all class-centred train vectors (pooled over the
    28 finding x class clouds), class means per finding; trust = -min_c d_c.
  - euclid_*: trust = -min_c ||z - mu_c||.
  - lof_*: LocalOutlierFactor(novelty=True) fit on train; trust = score_samples (higher = more inlying),
    one value per sample, repeated over findings.
- mfsi_img / mfsi_txt: single-modality MFSI, R = r^img / r^txt (the decisive comparison); mfsi: R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi.clouds import cloud_masks, prototype
from mfsi_cxr.mfsi.covariance import make_pd, pooled_covariance
from mfsi_cxr.mfsi.distance import Gaussian
from mfsi_cxr.mfsi.fit import MODALITIES
from mfsi_cxr.mfsi.reduce import fit_projection

MFSI_SCORES = {"mfsi": "R", "mfsi_img": "r_img", "mfsi_txt": "r_txt"}


def msp(probs: np.ndarray) -> np.ndarray:
    return np.maximum(probs, 1.0 - probs)


def ensemble_var(probs_by_seed: list[np.ndarray]) -> np.ndarray | None:
    """-var over seeds of p, or None with fewer than two seeds (rows must be aligned across seeds)."""
    if len(probs_by_seed) < 2:
        return None
    return -np.var(np.stack(probs_by_seed), axis=0)


def project(train: dict[str, np.ndarray], data: dict[str, np.ndarray], cfg: dict[str, Any],
            params: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(train, data) embeddings in the MFSI projection per space: img, txt and cat (concatenated)."""
    spaces = {}
    for name, key in MODALITIES:
        proj = fit_projection(train[key], cfg["mfsi"]["reduce"], name)
        if proj.k != params[f"reduce_{name}"]["k"]:
            raise ValueError(f"{name}: projection rebuilt with k={proj.k}, params say "
                             f"{params[f'reduce_{name}']['k']}")
        spaces[name] = (proj(train[key]), proj(data[key]))
    spaces["cat"] = tuple(np.hstack([spaces["img"][i], spaces["txt"][i]]) for i in (0, 1))
    return spaces


def _class_means(z: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for j in range(len(FINDINGS)):
        mu_p = prototype(z[pos[:, j]]) if pos[:, j].any() else prototype(z)
        out.append((mu_p, prototype(z[neg[:, j]])))
    return out


def tied_mahalanobis(z_train: np.ndarray, y_train: np.ndarray, z: np.ndarray, cfg: dict[str, Any],
                     name: str) -> np.ndarray:
    m = cfg["mfsi"]
    pos, neg = cloud_masks(y_train, m["uncertain_policy"])
    shared = pooled_covariance([z_train[pos[:, j]] for j in range(len(FINDINGS))]
                               + [z_train[neg[:, j]] for j in range(len(FINDINGS))])
    shared, _ = make_pd(shared, m["ridge_factor"], f"tied:{name}")
    cols = []
    for mu_p, mu_n in _class_means(z_train, pos, neg):
        cols.append(-np.minimum(Gaussian(mu_p, shared).distance(z), Gaussian(mu_n, shared).distance(z)))
    return np.stack(cols, 1)


def euclidean(z_train: np.ndarray, y_train: np.ndarray, z: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    pos, neg = cloud_masks(y_train, cfg["mfsi"]["uncertain_policy"])
    z = z.astype(np.float64)
    return np.stack([-np.minimum(np.linalg.norm(z - mu_p, axis=1), np.linalg.norm(z - mu_n, axis=1))
                     for mu_p, mu_n in _class_means(z_train, pos, neg)], 1)


def lof(z_train: np.ndarray, z: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor(n_neighbors=cfg["eval"]["lof_neighbors"], novelty=True)
    model.fit(z_train)
    return np.repeat(model.score_samples(z)[:, None], len(FINDINGS), 1)


def mfsi_scores(table: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    """(N, L) arrays of R, r_img, r_txt from the long MFSI table (rows sample-major, FINDINGS order)."""
    if list(table["finding"].iloc[: len(FINDINGS)]) != list(FINDINGS) or len(table) != n * len(FINDINGS):
        raise ValueError("MFSI table is not sample-major in FINDINGS order")
    return {name: table[col].to_numpy().reshape(n, len(FINDINGS)) for name, col in MFSI_SCORES.items()}


def all_scores(train: dict[str, np.ndarray], data: dict[str, np.ndarray], table: pd.DataFrame,
               cfg: dict[str, Any], params: dict[str, Any],
               probs_by_seed: list[np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Every trust score for one split and seed, keyed by name. ensemble_var only with >= 2 seeds."""
    n = len(data["study_id"])
    scores = mfsi_scores(table, n)
    scores["msp"] = msp(data["probs"])
    ens = ensemble_var(probs_by_seed or [data["probs"]])
    if ens is not None:
        scores["ensemble_var"] = ens
    for space, (zt, z) in project(train, data, cfg, params).items():
        scores[f"tied_maha_{space}"] = tied_mahalanobis(zt, train["y"], z, cfg, space)
        scores[f"euclid_{space}"] = euclidean(zt, train["y"], z, cfg)
        scores[f"lof_{space}"] = lof(zt, z, cfg)
    return scores
