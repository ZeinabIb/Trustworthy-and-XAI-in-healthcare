"""MFSI fit (train), tuning (val) and scoring (any split) from the frozen Phase 2 exports.

- Clouds, prototypes, covariances: TRAIN only. alpha: K-fold CV over train positives (mean held-out Gaussian
  log-likelihood). tau (per modality, global), y_hat thresholds, theta_R per stratum: VAL only.
- Scoring a split needs only the fitted parameters and the train exports; its labels are copied into the
  output for later analysis but never used. The test split is scored, never looked at.
- Each modality is first projected onto its train-PCA subspace (mfsi.reduce; reduce.py), then everything
  below runs in k dims. k and the variance kept are saved; scoring rebuilds the projection and checks k.
- mfsi_params.json keeps alpha / tau / thresholds; covariances are rebuilt from train when scoring
  (deterministic, seconds; avoids ~470 MB of matrices per seed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi.clouds import cloud_masks, prototype
from mfsi_cxr.mfsi.covariance import fit_covariances, make_pd, raw_covariance, shrink
from mfsi_cxr.mfsi.distance import Gaussian
from mfsi_cxr.mfsi.escalation import escalate, stratum_thresholds, yhat_threshold
from mfsi_cxr.mfsi.hierarchy import coherence_flags
from mfsi_cxr.mfsi.profile import margin, profile_pos
from mfsi_cxr.mfsi.reduce import Projection, fit_projection
from mfsi_cxr.mfsi.reliability import ONE_WITNESS, TWO_WITNESS, agreement, reliability
from mfsi_cxr.utils.logging import get_logger

log = get_logger("mfsi")
MODALITIES = (("img", "z_img"), ("txt", "z_txt"))
# Output columns of mfsi_{split}_seed{k}.parquet, in the order of plan 3 "Outputs".
COLUMNS = ("patient_id", "study_id", "finding", "y", "y_hat", "prob", "p_img_pos", "p_txt_pos", "d_img_pos",
           "d_img_neg", "d_txt_pos", "d_txt_neg", "r_img", "r_txt", "A_cos", "A_js", "R", "R_mean", "stratum",
           "flag", "escalate", "n_studies", "n_images", "text_len")


class SplitError(ValueError):
    """An MFSI fit function was given an export of the wrong split (leakage guard)."""


def load_split(cfg: dict[str, Any], split: str, seed: int) -> dict[str, np.ndarray]:
    """The full-model export of a split, after checking its sidecar really says that split."""
    from mfsi_cxr.models.extract import export_paths, load_export
    from mfsi_cxr.utils.io import read_meta

    path = export_paths(cfg, cfg["mfsi"]["embeddings"], split, seed)[0]
    data = load_export(path, cfg)
    if read_meta(path).get("split") != split:
        raise SplitError(f"{path.name}: sidecar split {read_meta(path).get('split')!r} != {split!r}")
    data["_split"] = np.array(split)
    return data


def _require(data: dict[str, np.ndarray], split: str) -> None:
    if str(data.get("_split")) != split:
        raise SplitError(f"expected the {split} split, got {data.get('_split')!r}")


def choose_alpha_cv(grid: list[float], folds: int, seed: int, ridge_factor: float):
    """alpha chooser: K-fold CV over the finding's train positives, mean held-out Gaussian log-likelihood."""
    def choose(finding: str, zp: np.ndarray, target: np.ndarray) -> float:
        if len(zp) < folds:
            log.info("alpha %s: %d positives < %d folds -> alpha = 0", finding, len(zp), folds)
            return 0.0
        rng = np.random.default_rng(seed)
        fold = rng.permutation(len(zp)) % folds
        scores = []
        for alpha in grid:
            ll = []
            for k in range(folds):
                tr, te = zp[fold != k], zp[fold == k]
                cov, _ = make_pd(shrink(raw_covariance(tr), target, alpha), ridge_factor, f"cv:{finding}")
                ll.append(Gaussian(prototype(tr), cov).log_likelihood(te).mean())
            scores.append(float(np.mean(ll)))
        return float(grid[int(np.argmax(scores))])
    return choose


def ece(prob: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - prob[sel].mean())
    return float(total)


def tau_targets(y: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Class label for calibrating the profile: 1 positive, 0 negative or blank; uncertain excluded (or
    positive under the 'ones' ablation). Returns (label, used) of shape (N, L)."""
    pos, neg = cloud_masks(y, policy)
    return pos.astype(float), pos | neg


@dataclass
class Modality:
    gauss_pos: dict[str, Gaussian]
    gauss_neg: dict[str, Gaussian]
    info: dict[str, dict]
    proj: Projection


def build_modality(train: dict[str, np.ndarray], key: str, alphas, cfg: dict[str, Any],
                   name: str) -> Modality:
    _require(train, "train")
    m = cfg["mfsi"]
    proj = fit_projection(train[key], m["reduce"], name)
    z = proj(train[key])
    pos, neg = cloud_masks(train["y"], m["uncertain_policy"])
    covs = fit_covariances(z, pos, neg, alphas, m["ridge_factor"], m["cond_max"], name)
    gp, gn = {}, {}
    for j, f in enumerate(FINDINGS):
        zp = z[pos[:, j]] if pos[:, j].any() else z  # a finding without positives: fall back to all (logged)
        if not pos[:, j].any():
            log.warning("%s:%s: no train positives; positive prototype = global mean", name, f)
        gp[f] = Gaussian(prototype(zp), covs.pos[f])
        gn[f] = Gaussian(prototype(z[neg[:, j]]), covs.neg[f])
    return Modality(gp, gn, covs.info, proj)


def distances(mod: Modality, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = mod.proj(z)
    d_pos = np.stack([mod.gauss_pos[f].distance(z) for f in FINDINGS], 1)
    d_neg = np.stack([mod.gauss_neg[f].distance(z) for f in FINDINGS], 1)
    return d_pos, d_neg


def fit_tau(d_pos: np.ndarray, d_neg: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Global tau per modality minimising ECE of p+ against the class label on VAL."""
    m = cfg["mfsi"]
    g = m["tau_grid"]
    grid = np.geomspace(g["min"], g["max"], g["n"])
    label, used = tau_targets(y, m["uncertain_policy"])
    gap = (d_neg - d_pos)[used]
    target = label[used]
    errs = [ece(expit(gap / t), target, m["ece_bins"]) for t in grid]
    best = int(np.argmin(errs))
    return {"tau": float(grid[best]), "ece_after": float(errs[best]),
            "ece_before_tau1": ece(expit(gap), target, m["ece_bins"]),
            "at_grid_boundary": best in (0, len(grid) - 1)}


def fit_mfsi(train: dict[str, np.ndarray], val: dict[str, np.ndarray], cfg: dict[str, Any],
             text_informative: dict[str, bool], seed: int) -> dict[str, Any]:
    """All MFSI parameters: alpha (train CV), tau, y_hat thresholds, theta_R per stratum (val)."""
    _require(train, "train")
    _require(val, "val")
    m = cfg["mfsi"]
    params: dict[str, Any] = {"seed": seed, "n_train": int(len(train["study_id"])),
                              "n_val": int(len(val["study_id"])), "config": m,
                              "text_informative": text_informative}
    chooser = choose_alpha_cv(m["alpha_grid"], m["alpha_folds"], m["alpha_seed"], m["ridge_factor"])
    mods = {}
    for name, key in MODALITIES:
        mods[name] = build_modality(train, key, chooser, cfg, name)
        d_pos, d_neg = distances(mods[name], val[key])
        params[f"tau_{name}"] = fit_tau(d_pos, d_neg, val["y"], cfg)
        params[f"cov_{name}"] = mods[name].info
        params[f"reduce_{name}"] = mods[name].proj.info()
        params[f"alpha_{name}"] = {f: mods[name].info[f]["alpha"] for f in FINDINGS}
        if params[f"tau_{name}"]["at_grid_boundary"]:
            log.warning("tau_%s hit the grid boundary (%.3g): STOP-AND-ASK", name,
                        params[f"tau_{name}"]["tau"])
    scored = score_arrays(mods, val, params, cfg, thresholds=None)
    label, used = tau_targets(val["y"], m["uncertain_policy"])
    params["yhat_threshold"] = {f: yhat_threshold(val["probs"][used[:, j], j], label[used[:, j], j],
                                                  m["yhat_rule"]) for j, f in enumerate(FINDINGS)}
    strata = np.array([stratum_of(f, text_informative) for f in FINDINGS])
    params["strata"] = dict(zip(FINDINGS, strata.tolist(), strict=True))
    params["theta_R"] = stratum_thresholds(scored["R"], strata, m["escalate_fractions"])
    return params


def stratum_of(finding: str, text_informative: dict[str, bool]) -> str:
    return TWO_WITNESS if text_informative.get(finding, False) else ONE_WITNESS


def score_arrays(mods: dict[str, Modality], data: dict[str, np.ndarray], params: dict[str, Any],
                 cfg: dict[str, Any], thresholds: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """Per (sample, finding) arrays (N, L). Labels of `data` are not used."""
    m = cfg["mfsi"]
    out: dict[str, np.ndarray] = {}
    for name, key in MODALITIES:
        tau = params[f"tau_{name}"]["tau"]
        d_pos, d_neg = distances(mods[name], data[key])
        out[f"d_{name}_pos"], out[f"d_{name}_neg"] = d_pos, d_neg
        out[f"p_{name}_pos"] = profile_pos(d_pos, d_neg, tau)
        out[f"r_{name}"] = margin(d_pos, d_neg, tau)
    out["A_cos"] = agreement(out["p_img_pos"], out["p_txt_pos"], "cos")
    out["A_js"] = agreement(out["p_img_pos"], out["p_txt_pos"], "js")
    a = agreement(out["p_img_pos"], out["p_txt_pos"], m["agreement"])
    two = reliability(a, out["r_img"], out["r_txt"], m["combine"])
    two_mean = reliability(a, out["r_img"], out["r_txt"], "mean")
    one_witness = np.array([stratum_of(f, params["text_informative"]) == ONE_WITNESS for f in FINDINGS])
    out["R"] = np.where(one_witness[None, :], out["r_img"], two)        # R_one = r_img for one-witness
    out["R_mean"] = np.where(one_witness[None, :], out["r_img"], two_mean)
    if thresholds is not None:
        thr = np.array([thresholds["yhat_threshold"][f] for f in FINDINGS])
        out["y_hat"] = data["probs"] >= thr[None, :]
        op = str(m["operating_fraction"])
        theta_r = np.array([thresholds["theta_R"][thresholds["strata"][f]][op] for f in FINDINGS])
        out["flag"] = coherence_flags(out["y_hat"], out["R"], theta_r)
        out["escalate"] = escalate(out["R"], theta_r, out["flag"])
    return out


def score_split(train: dict[str, np.ndarray], data: dict[str, np.ndarray], params: dict[str, Any],
                cfg: dict[str, Any]) -> pd.DataFrame:
    """Long table (one row per sample x finding) with the columns of plan 3 'Outputs'."""
    alphas = {name: params[f"alpha_{name}"] for name, _ in MODALITIES}
    mods = {name: build_modality(train, key, alphas[name], cfg, name) for name, key in MODALITIES}
    for name, _ in MODALITIES:
        if mods[name].proj.k != params[f"reduce_{name}"]["k"]:
            raise ValueError(f"{name}: projection rebuilt with k={mods[name].proj.k}, params say "
                             f"{params[f'reduce_{name}']['k']} (different train export or config?)")
    s = score_arrays(mods, data, params, cfg, thresholds=params)
    n, n_findings = data["probs"].shape
    rows = {
        "patient_id": np.repeat(data["patient_id"], n_findings),
        "study_id": np.repeat(data["study_id"], n_findings),
        "finding": np.tile(np.array(FINDINGS), n), "y": data["y"].ravel(), "prob": data["probs"].ravel(),
        "stratum": np.tile(np.array([params["strata"][f] for f in FINDINGS]), n),
    }
    for k in ("y_hat", "p_img_pos", "p_txt_pos", "d_img_pos", "d_img_neg", "d_txt_pos", "d_txt_neg",
              "r_img", "r_txt", "A_cos", "A_js", "R", "R_mean", "flag", "escalate"):
        rows[k] = s[k].ravel()
    for k in ("n_studies", "n_images", "text_len"):
        rows[k] = np.repeat(data[k], n_findings)
    return pd.DataFrame(rows)[list(COLUMNS)]


def write_params(params: dict[str, Any], path: Path, cfg: dict[str, Any], seed: int) -> None:
    from mfsi_cxr.utils.io import write_meta

    path.write_text(json.dumps(params, indent=2, default=float), encoding="utf-8")
    write_meta(path, cfg, seed)

