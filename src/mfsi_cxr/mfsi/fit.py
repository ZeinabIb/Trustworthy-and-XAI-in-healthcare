"""MFSI fit (train), tuning (val) and scoring (any split) from the frozen Phase 2 exports.

- Clouds, prototypes, covariances: TRAIN only. alpha: K-fold CV over train positives (mean held-out Gaussian
  log-likelihood). tau, y_hat thresholds, theta_R (per finding; theta_scope): VAL only. tau is per finding
  and modality (mfsi.tau_scope: finding; Zeinab 2026-10-15), falling back to the modality's global tau when a
  finding has fewer than tau_min_class_count val positives or negatives; global: the plan's start.
  With mfsi.offset each also gets an offset b, p+ = sigmoid((d_- - d_+ - b)/tau), fitted by log-loss (Platt;
  Zeinab 2026-10-15): under ECE a constant p+ = prevalence would win.
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
from scipy.optimize import minimize
from scipy.special import expit

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.mfsi import yhat
from mfsi_cxr.mfsi.clouds import cloud_masks, prototype
from mfsi_cxr.mfsi.covariance import fit_covariances, make_pd, raw_covariance, shrink
from mfsi_cxr.mfsi.distance import Gaussian
from mfsi_cxr.mfsi.escalation import escalate, finding_thresholds, stratum_thresholds
from mfsi_cxr.mfsi.hierarchy import coherence_flags
from mfsi_cxr.mfsi.profile import margin, profile_pos
from mfsi_cxr.mfsi.reduce import Projection, fit_projection
from mfsi_cxr.mfsi.reliability import ONE_WITNESS, TWO_WITNESS, agreement, reliability
from mfsi_cxr.utils.logging import get_logger

log = get_logger("mfsi")
MODALITIES = (("img", "z_img"), ("txt", "z_txt"))
THETA_SCOPES = ("finding", "stratum")
# Output columns of mfsi_{split}_seed{k}.parquet, in the order of plan 3 "Outputs", plus prob_cal (y_hat v2:
# the decision-centred probability q of the y_hat rule; MSP = max(q, 1 - q)).
COLUMNS = ("patient_id", "study_id", "finding", "y", "y_hat", "prob", "prob_cal", "p_img_pos", "p_txt_pos",
           "d_img_pos", "d_img_neg", "d_txt_pos", "d_txt_neg", "r_img", "r_txt", "A_cos", "A_js", "R",
           "R_mean", "stratum", "flag", "escalate", "n_studies", "n_images", "text_len")


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


def log_loss(logit: np.ndarray, target: np.ndarray) -> float:
    """Mean binary log-loss of sigmoid(logit) against 0/1 targets (stable form)."""
    return float(np.mean(np.logaddexp(0.0, logit) - target * logit))


def _evaluate(gap: np.ndarray, target: np.ndarray, tau: float, b: float, bins: int) -> dict[str, Any]:
    """ECE and log-loss of p+ before (tau = 1, b = 0) and after calibration."""
    if not len(gap):
        return dict.fromkeys(("ece_before_tau1", "ece_after", "logloss_before_tau1", "logloss_after"))
    after = (gap - b) / tau
    return {"ece_before_tau1": ece(expit(gap), target, bins), "ece_after": ece(expit(after), target, bins),
            "logloss_before_tau1": log_loss(gap, target), "logloss_after": log_loss(after, target)}


def _tau_search(gap: np.ndarray, target: np.ndarray, m: dict[str, Any]) -> dict[str, Any]:
    """ECE objective (plan 3 as written): tau on the log grid, no offset."""
    g = m["tau_grid"]
    grid = np.geomspace(g["min"], g["max"], g["n"])
    best = int(np.argmin([ece(expit(gap / t), target, m["ece_bins"]) for t in grid]))
    tau = float(grid[best])
    return {"tau": tau, "b": 0.0, "at_grid_boundary": best in (0, len(grid) - 1),
            **_evaluate(gap, target, tau, 0.0, m["ece_bins"])}


def _platt(gap: np.ndarray, target: np.ndarray, m: dict[str, Any]) -> dict[str, Any]:
    """Log-loss objective (Zeinab 2026-10-15): p+ = sigmoid((gap - b)/tau), fitted as the convex logistic
    regression sigmoid(a gap + c) with a = 1/tau bounded to the tau_grid range (tau > 0, so the direction of
    the gap is never flipped) and c = -b/tau free (fixed at 0 without offset). L-BFGS-B, analytic gradient."""
    g = m["tau_grid"]
    a_lo, a_hi = 1.0 / g["max"], 1.0 / g["min"]
    offset = bool(m["offset"])

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        s = w[0] * gap + w[1]
        resid = expit(s) - target
        return log_loss(s, target), np.array([np.mean(resid * gap), np.mean(resid) if offset else 0.0])

    prev = float(np.clip(target.mean(), 1e-6, 1 - 1e-6))
    x0 = np.array([float(np.clip(1.0, a_lo, a_hi)), float(np.log(prev / (1 - prev))) if offset else 0.0])
    res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                   bounds=[(a_lo, a_hi), (None, None) if offset else (0.0, 0.0)])
    a, c = float(res.x[0]), float(res.x[1])
    tau, b = 1.0 / a, -c / a
    edge = bool(np.isclose(a, a_lo, rtol=1e-6) or np.isclose(a, a_hi, rtol=1e-6))
    return {"tau": tau, "b": b, "at_grid_boundary": edge, "converged": bool(res.success),
            **_evaluate(gap, target, tau, b, m["ece_bins"])}


def fit_tau(d_pos: np.ndarray, d_neg: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Profile calibration per modality on VAL, p+ = sigmoid((d_- - d_+ - b)/tau) against the class label:
    the global (tau, b) (all findings pooled) and, under tau_scope 'finding', one (tau, b) per finding (global
    fallback when a class has fewer than tau_min_class_count val cases). Objective 'logloss' (Platt; b fitted
    when offset is true) or 'ece' (plan as written; b = 0). `per_finding[f]` holds what scoring uses."""
    m = cfg["mfsi"]
    if m["tau_scope"] not in ("finding", "global"):
        raise ValueError(f"mfsi.tau_scope must be finding or global, got {m['tau_scope']!r}")
    if m["tau_objective"] not in ("logloss", "ece"):
        raise ValueError(f"mfsi.tau_objective must be logloss or ece, got {m['tau_objective']!r}")
    if m["tau_objective"] == "ece" and m["offset"]:
        raise ValueError("mfsi.offset needs tau_objective logloss: under ECE a constant p+ = prevalence wins")
    fit = _platt if m["tau_objective"] == "logloss" else _tau_search
    label, used = tau_targets(y, m["uncertain_policy"])
    gap = d_neg - d_pos
    out = fit(gap[used], label[used], m)
    per: dict[str, dict[str, Any]] = {}
    for j, f in enumerate(FINDINGS):
        g_j, lab = gap[used[:, j], j], label[used[:, j], j]
        n_pos, n_neg = int(lab.sum()), int(len(lab) - lab.sum())
        rec: dict[str, Any] = {"n_pos": n_pos, "n_neg": n_neg}
        if m["tau_scope"] == "finding" and min(n_pos, n_neg) >= m["tau_min_class_count"]:
            rec.update(fit(g_j, lab, m), source="finding")
        else:
            if m["tau_scope"] == "finding":
                log.info("tau %s: %d val positives / %d negatives < %d -> global tau", f, n_pos, n_neg,
                         m["tau_min_class_count"])
            rec.update(tau=out["tau"], b=out["b"], source="global", at_grid_boundary=False,
                       **_evaluate(g_j, lab, out["tau"], out["b"], m["ece_bins"]))
        per[f] = rec
    out["scope"], out["objective"], out["per_finding"] = m["tau_scope"], m["tau_objective"], per
    return out


def tau_vector(tau_params: dict[str, Any], key: str = "tau") -> np.ndarray:
    """The (L,) tau (or offset, key='b') used for scoring, in FINDINGS order."""
    return np.array([tau_params["per_finding"][f][key] for f in FINDINGS])


def fit_mfsi(train: dict[str, np.ndarray], val: dict[str, np.ndarray], cfg: dict[str, Any],
             text_informative: dict[str, bool], seed: int) -> dict[str, Any]:
    """All MFSI parameters: alpha (train CV), tau, y_hat thresholds, theta_R per finding and stratum (val)."""
    _require(train, "train")
    _require(val, "val")
    m = cfg["mfsi"]
    if m["theta_scope"] not in THETA_SCOPES:
        raise ValueError(f"mfsi.theta_scope must be one of {THETA_SCOPES}, got {m['theta_scope']!r}")
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
        edge = [f for f, r in params[f"tau_{name}"]["per_finding"].items() if r["at_grid_boundary"]]
        if edge:
            log.warning("tau_%s at the grid boundary for %s: STOP-AND-ASK", name, edge)
    scored = score_arrays(mods, val, params, cfg, thresholds=None)
    label, used = tau_targets(val["y"], m["uncertain_policy"])
    # y_hat v2 (amendment 1): fitted on ALL val scored rows, frozen here, applied unchanged to test.
    params["yhat"] = {"rule": m["yhat_rule"], "version": yhat.VERSION, "eps": m["platt_eps"],
                      "per_finding": {f: yhat.fit_rule(m["yhat_rule"], val["probs"][used[:, j], j],
                                                       label[used[:, j], j], m["platt_eps"])
                                      for j, f in enumerate(FINDINGS)}}
    strata = np.array([stratum_of(f, text_informative) for f in FINDINGS])
    params["strata"] = dict(zip(FINDINGS, strata.tolist(), strict=True))
    params["theta_R"] = stratum_thresholds(scored["R"], strata, m["escalate_fractions"])
    params["theta_R_finding"] = finding_thresholds(scored["R"], FINDINGS, m["escalate_fractions"])
    params["theta_scope"] = m["theta_scope"]
    return params


def theta_vector(params: dict[str, Any], fraction: float) -> np.ndarray:
    """The (L,) theta_R in use: each finding's own (theta_scope 'finding') or its stratum's."""
    op = str(fraction)
    if params["theta_scope"] == "finding":
        return np.array([params["theta_R_finding"][f][op] for f in FINDINGS])
    return np.array([params["theta_R"][params["strata"][f]][op] for f in FINDINGS])


def stratum_of(finding: str, text_informative: dict[str, bool]) -> str:
    return TWO_WITNESS if text_informative.get(finding, False) else ONE_WITNESS


def score_arrays(mods: dict[str, Modality], data: dict[str, np.ndarray], params: dict[str, Any],
                 cfg: dict[str, Any], thresholds: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """Per (sample, finding) arrays (N, L). Labels of `data` are not used."""
    m = cfg["mfsi"]
    out: dict[str, np.ndarray] = {}
    for name, key in MODALITIES:
        tau, b = tau_vector(params[f"tau_{name}"]), tau_vector(params[f"tau_{name}"], "b")
        d_pos, d_neg = distances(mods[name], data[key])
        out[f"d_{name}_pos"], out[f"d_{name}_neg"] = d_pos, d_neg
        out[f"p_{name}_pos"] = profile_pos(d_pos, d_neg, tau, b)
        out[f"r_{name}"] = margin(d_pos, d_neg, tau, b)
    out["A_cos"] = agreement(out["p_img_pos"], out["p_txt_pos"], "cos")
    out["A_js"] = agreement(out["p_img_pos"], out["p_txt_pos"], "js")
    a = agreement(out["p_img_pos"], out["p_txt_pos"], m["agreement"])
    two = reliability(a, out["r_img"], out["r_txt"], m["combine"])
    two_mean = reliability(a, out["r_img"], out["r_txt"], "mean")
    one_witness = np.array([stratum_of(f, params["text_informative"]) == ONE_WITNESS for f in FINDINGS])
    out["R"] = np.where(one_witness[None, :], out["r_img"], two)        # R_one = r_img for one-witness
    out["R_mean"] = np.where(one_witness[None, :], out["r_img"], two_mean)
    if thresholds is not None:
        y = thresholds["yhat"]
        out["prob_cal"], out["y_hat"], out["msp"] = yhat.apply_all(
            y["rule"], [y["per_finding"][f] for f in FINDINGS], data["probs"], y["eps"])
        theta_r = theta_vector(thresholds, m["operating_fraction"])
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
    for k in ("y_hat", "prob_cal", "p_img_pos", "p_txt_pos", "d_img_pos", "d_img_neg", "d_txt_pos",
              "d_txt_neg",
              "r_img", "r_txt", "A_cos", "A_js", "R", "R_mean", "flag", "escalate"):
        rows[k] = s[k].ravel()
    for k in ("n_studies", "n_images", "text_len"):
        rows[k] = np.repeat(data[k], n_findings)
    return pd.DataFrame(rows)[list(COLUMNS)]


def write_params(params: dict[str, Any], path: Path, cfg: dict[str, Any], seed: int) -> None:
    from mfsi_cxr.utils.io import write_meta

    path.write_text(json.dumps(params, indent=2, default=float), encoding="utf-8")
    write_meta(path, cfg, seed)

