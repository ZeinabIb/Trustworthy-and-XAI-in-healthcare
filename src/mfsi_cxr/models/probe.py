"""Text-informativeness probe (Phase 2.5): which findings are two-witness (text informative) vs one-witness.

Per finding: logistic regression on z_txt alone (standardised), fit on train, AUROC on val with a bootstrap CI
over PATIENTS (a patient contributes several samples). text_informative = lower CI bound > probe.text_cutoff.
The same probe on z_img is saved as a reference. Val only; the test split is never read here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.utils.logging import get_logger

log = get_logger("probe")


def bootstrap_auroc(y: np.ndarray, score: np.ndarray, groups: np.ndarray, n_boot: int, level: float,
                    seed: int) -> tuple[float, float, float]:
    """AUROC with a percentile CI from resampling groups (patients) with replacement."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            stats.append(roc_auc_score(y[idx], score[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    alpha = (1 - level) / 2
    return point, float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def probe_table(train: dict[str, np.ndarray], val: dict[str, np.ndarray], key: str,
                cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Per finding: val AUROC (+ CI) of a logistic probe on `key` (z_txt or z_img)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pcfg = cfg["probe"]
    rows = []
    for j, finding in enumerate(FINDINGS):
        tr = train["label_mask"][:, j] > 0
        va = val["label_mask"][:, j] > 0
        y_tr, y_va = train["y"][tr, j] == 1, val["y"][va, j] == 1
        row = {"finding": finding, "embedding": key, "n_train": int(tr.sum()), "n_val": int(va.sum()),
               "n_val_pos": int(y_va.sum()), "auroc": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        if len(np.unique(y_tr)) == 2 and len(np.unique(y_va)) == 2:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=pcfg["C"], max_iter=pcfg["max_iter"]))
            clf.fit(train[key][tr], y_tr)
            score = clf.predict_proba(val[key][va])[:, 1]
            row["auroc"], row["ci_low"], row["ci_high"] = bootstrap_auroc(
                y_va.astype(int), score, val["patient_id"][va], cfg["eval"]["bootstrap_B"],
                pcfg["ci_level"], seed)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["informative"] = table["ci_low"] > pcfg["text_cutoff"]  # NaN compares False
    return table


def text_informative_mask(table: pd.DataFrame) -> dict[str, Any]:
    """The mask Phase 3 reads, plus a loud flag when z_txt carries nothing for any finding."""
    txt = table[table["embedding"] == "z_txt"].set_index("finding")
    mask = {f: bool(txt.loc[f, "informative"]) for f in FINDINGS}
    none = not any(mask.values())
    if none:
        log.error("z_txt is not informative for ANY finding: the MFSI agreement term would be meaningless")
    return {"text_informative": mask, "no_finding_text_informative": none}
