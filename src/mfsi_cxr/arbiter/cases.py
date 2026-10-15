"""Escalated cases -> PacketInputs (Phase 5), from artifacts only.

Sources: the MFSI table of the split (escalate flag, classifier and MFSI outputs; the label column `y` is
never copied), text_inputs.parquet (the leakage-safe index pre-read text and the ordered prior study ids),
radgraph_facts.parquet (facts of PRIOR studies; a fact is used for finding f if f is in its `findings`),
and mfsi_params (y_hat rule, theta_R). Prior rank 1 = the most recent prior. Cases are a seeded sample of
the escalated (sample, finding) rows, ordered by sample then label-tree order (roots first), so a child's
hierarchy check can use its ancestors' final decisions when they were arbitrated first.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from mfsi_cxr.arbiter.packet import Modality, PacketInputs
from mfsi_cxr.arbiter.rules import Context
from mfsi_cxr.constants import FINDINGS, PARENT
from mfsi_cxr.mfsi.covariance import topological_order

USED_COLUMNS = ("patient_id", "study_id", "finding", "y_hat", "prob_cal", "p_img_pos", "p_txt_pos", "r_img",
                "r_txt", "R", "stratum", "escalate")
CHILDREN = {p: [c for c, q in PARENT.items() if q == p] for p in set(PARENT.values())}


def threshold_of(params: dict[str, Any]) -> float:
    """Decision threshold on the calibrated probability shown to the arbiter (0.5 for the y_hat v2 rules)."""
    return 0.5 if params["yhat"]["rule"] in ("platt", "youden_coherent") else float("nan")


def select_cases(table: pd.DataFrame, n_cases: int, seed: int) -> pd.DataFrame:
    esc = table.loc[table["escalate"].astype(bool), list(USED_COLUMNS)].reset_index(drop=True)
    if n_cases < len(esc):
        esc = esc.iloc[np.sort(np.random.default_rng(seed).choice(len(esc), n_cases, replace=False))]
    order = {f: k for k, f in enumerate(topological_order())}
    return esc.assign(_o=esc["finding"].map(order)).sort_values(["study_id", "_o"]).drop(columns="_o")


def select_cases_by_patient(table: pd.DataFrame, n_min: int,
                            seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pre-stated Phase 6 sample (DECISIONS 2026-10-15): patients with escalated cases in a seeded random
    order, each with ALL its escalated findings, until >= n_min cases. Returns (cases, sampling summary)."""
    esc = table.loc[table["escalate"].astype(bool), list(USED_COLUMNS)].reset_index(drop=True)
    patients = np.array(sorted(esc["patient_id"].unique()))
    order = patients[np.random.default_rng(seed).permutation(len(patients))]
    counts = esc.groupby("patient_id").size()
    taken = np.cumsum(counts.loc[order].to_numpy())
    k = int(np.searchsorted(taken, n_min) + 1) if len(order) else 0
    chosen = set(order[:k])
    sample = esc[esc["patient_id"].isin(chosen)]
    rank = {f: j for j, f in enumerate(topological_order())}
    sample = sample.assign(_o=sample["finding"].map(rank)).sort_values(["study_id", "_o"]).drop(columns="_o")
    info = {"n_cases": len(sample), "n_escalated": len(esc), "case_fraction": len(sample) / max(len(esc), 1),
            "n_patients": len(chosen), "n_patients_escalated": len(patients),
            "patient_fraction": len(chosen) / max(len(patients), 1), "seed": seed, "n_min": n_min}
    return sample, info


def derangement(n: int, rng: np.random.Generator) -> np.ndarray | None:
    """A permutation with no fixed point (None for n < 2), by seeded rejection sampling."""
    if n < 2:
        return None
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def _modality(row: pd.Series, finding: str, informative: dict[str, bool]) -> Modality:
    return Modality(float(row["p_img_pos"]), float(row["p_txt_pos"]), float(row["r_img"]),
                    float(row["r_txt"]), float(row["R"]), str(row["stratum"]),
                    bool(informative.get(finding, False)))


def _stable_int(*parts: str) -> int:
    """Process-independent integer from strings (Python's hash() is randomised per process)."""
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


class CaseBuilder:
    def __init__(self, table: pd.DataFrame, texts: pd.DataFrame, facts: pd.DataFrame, params: dict[str, Any],
                 informative: dict[str, bool], theta: np.ndarray) -> None:
        self.rows = table[list(USED_COLUMNS)].set_index(["study_id", "finding"])
        self.texts = texts.set_index("study_id")
        self.facts = facts
        self.informative = informative
        self.threshold = threshold_of(params)
        self.theta = dict(zip(FINDINGS, theta, strict=True))
        self.by_finding = {f: g.reset_index() for f, g in table[list(USED_COLUMNS)].groupby("finding")}

    def prior_facts(self, study_id: str, finding: str) -> tuple[dict[str, Any], ...]:
        priors = list(self.texts.loc[study_id, "prior_study_ids"])
        if not priors:
            return ()
        rank = {s: k + 1 for k, s in enumerate(priors)}
        f = self.facts[self.facts["study_id"].isin(rank)]
        f = f[f["findings"].map(lambda xs: finding in list(xs))]
        return tuple({"rank": rank[r.study_id], "entity": r.entity_text, "certainty": r.certainty,
                      "anatomy": r.anatomy if isinstance(r.anatomy, str) else ""} for r in f.itertuples())

    def inputs(self, study_id: str, finding: str) -> PacketInputs:
        row = self.rows.loc[(study_id, finding)]
        parent = None
        if finding in PARENT:
            p = self.rows.loc[(study_id, PARENT[finding])]
            parent = (PARENT[finding], bool(p["y_hat"]), float(p["prob_cal"]))
        children = tuple((c, bool(self.rows.loc[(study_id, c), "y_hat"])) for c in CHILDREN.get(finding, []))
        return PacketInputs(case_id=f"{study_id}|{finding}", finding=finding,
                            index_text=str(self.texts.loc[study_id, "index_text"]),
                            prior_facts=self.prior_facts(study_id, finding), prob_cal=float(row["prob_cal"]),
                            threshold=self.threshold, y_hat=bool(row["y_hat"]),
                            modality=_modality(row, finding, self.informative),
                            parent=parent, children=children)

    def donor(self, study_id: str, finding: str, patient_id: str, seed: int) -> Modality:
        """Modality info of another patient's sample, same finding (shuffled_across_patients control)."""
        g = self.by_finding[finding]
        g = g[g["patient_id"] != patient_id]
        pick = np.random.default_rng([seed, _stable_int(study_id, finding)]).integers(len(g))
        return _modality(g.iloc[int(pick)], finding, self.informative)

    def context(self, study_id: str, final: dict[str, bool]) -> Context:
        """Hierarchy context: classifier calls, overridden by final decisions already made for this sample."""
        classifier = {f: bool(self.rows.loc[(study_id, f), "y_hat"]) for f in FINDINGS}
        reliable = {f: float(self.rows.loc[(study_id, f), "R"]) >= self.theta[f] for f in FINDINGS}
        return Context({**classifier, **final}, classifier, reliable)
