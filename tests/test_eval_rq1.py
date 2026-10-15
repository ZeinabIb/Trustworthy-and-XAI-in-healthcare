"""RQ1 engine and baselines end to end on synthetic exports (no real data)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from tests.test_mfsi_fit import INFORMATIVE, _export

from mfsi_cxr.constants import FINDINGS
from mfsi_cxr.eval.baselines import all_scores, ensemble_var, euclidean, lof, msp, tied_mahalanobis
from mfsi_cxr.eval.rq1 import Inputs, evaluate
from mfsi_cxr.eval.selective import correctness
from mfsi_cxr.mfsi.fit import fit_mfsi, score_split


@pytest.fixture
def cfg(smoke_cfg: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(smoke_cfg)
    c["eval"]["bootstrap_B"] = 30
    return c


@pytest.fixture
def setup(cfg: dict[str, Any]):
    train, val = _export(300, "train", 0), _export(150, "val", 1)
    params = fit_mfsi(train, val, cfg, INFORMATIVE, seed=0)
    table = score_split(train, val, params, cfg)
    n = len(val["study_id"])
    q = table["prob_cal"].to_numpy().reshape(n, len(FINDINGS))
    scores = all_scores(train, val, table, cfg, params, q, [q])
    y_hat = table["y_hat"].to_numpy().reshape(n, len(FINDINGS))
    return train, val, params, table, scores, y_hat


def test_all_scores_shapes_and_names(setup) -> None:
    _, val, _, _, scores, _ = setup
    families = ("tied_maha", "euclid", "lof")
    spaces = ("img", "txt", "cat")
    expected = {"mfsi", "mfsi_img", "mfsi_txt", "msp"} | {f"{b}_{s}" for b in families for s in spaces}
    assert set(scores) == expected  # one seed: no ensemble_var
    for k, v in scores.items():
        assert v.shape == (len(val["study_id"]), len(FINDINGS)) and np.isfinite(v).all(), k


def test_baseline_orientation() -> None:
    """Higher trust = more reliable for every wrapper."""
    assert msp(np.array([0.99]))[0] > msp(np.array([0.5]))[0]
    rng = np.random.default_rng(0)
    zt = rng.normal(size=(200, 3))
    y = np.zeros((200, len(FINDINGS)))
    y[:100] = 1.0
    zt[:100] += 4
    near, far = np.array([[4.0, 0, 0], [0, 0, 0]]), np.array([[30.0, 30, 30]])
    z = np.vstack([near, far])
    cfg = {"mfsi": {"uncertain_policy": "ignore", "ridge_factor": 1e-3}, "eval": {"lof_neighbors": 20}}
    for t in (tied_mahalanobis(zt, y, z, cfg, "t"), euclidean(zt, y, z, cfg), lof(zt, z, cfg)):
        assert (t[:2, 0] > t[2, 0]).all()
    assert ensemble_var([np.zeros((2, 3))]) is None
    v = ensemble_var([np.array([[0.5]]), np.array([[0.5]]), np.array([[0.9]])])
    assert v[0, 0] < 0 and ensemble_var([np.array([[0.5]])] * 2)[0, 0] == 0


def test_evaluate_oracle_is_best_and_reference_diff_is_zero(setup, cfg: dict[str, Any]) -> None:
    _, val, _, _, scores, y_hat = setup
    correct, _ = correctness(val["y"], y_hat, cfg["eval"]["blank_policy"])
    scores = dict(scores, oracle=correct + 1e-3 * np.random.default_rng(1).normal(size=correct.shape))
    out = evaluate(Inputs([scores], val["y"], [y_hat], val["patient_id"], "negative"), cfg)
    pf = out["per_finding"]
    for f, g in pf.groupby("finding"):
        assert g.loc[g.score == "oracle", "aurc"].iloc[0] <= g["aurc"].min() + 1e-12, f
        oracle_auroc = g.loc[g.score == "oracle", "auroc"].iloc[0]
        assert oracle_auroc == pytest.approx(1.0) or g["error_rate"].iloc[0] == 0
    ref = out["main"].set_index("score").loc["mfsi"]
    assert ref["aurc_minus_ref"] == 0 and ref["aurc_minus_ref_lo"] == 0 and ref["aurc_minus_ref_hi"] == 0
    m = out["main"].set_index("score")
    assert (m["aurc_lo"] <= m["aurc"] + 1e-12).all() and (m["aurc"] <= m["aurc_hi"] + 1e-12).all()
    assert set(out["delong"]["score"]) == set(scores) - {"mfsi"}


def test_cross_modal_rule_cannot_hold_for_one_witness(setup, cfg: dict[str, Any]) -> None:
    _, val, params, _, scores, y_hat = setup
    out = evaluate(Inputs([scores], val["y"], [y_hat], val["patient_id"], "negative"), cfg)
    cm = out["cross_modal"].set_index("finding")
    for f in FINDINGS:
        if params["strata"][f] == "one_witness":
            assert cm.loc[f, "diff"] <= 1e-12 and not cm.loc[f, "claim_holds"]  # R = r_img: never better


def test_seed_mean_and_range(setup, cfg: dict[str, Any]) -> None:
    _, val, _, _, scores, y_hat = setup
    shuffled = {k: np.random.default_rng(2).permutation(v) for k, v in scores.items()}
    out = evaluate(Inputs([scores, shuffled], val["y"], [y_hat, y_hat], val["patient_id"], "negative"), cfg)
    one = evaluate(Inputs([scores], val["y"], [y_hat], val["patient_id"], "negative"), cfg)
    two = evaluate(Inputs([shuffled], val["y"], [y_hat], val["patient_id"], "negative"), cfg)
    pf, a, b = (x["per_finding"].set_index(["score", "finding"]) for x in (out, one, two))
    assert np.allclose(pf["aurc"], (a["aurc"] + b["aurc"]) / 2, equal_nan=True)
    assert np.allclose(pf["aurc_seed_min"], np.minimum(a["aurc"], b["aurc"]), equal_nan=True)


def test_sanity_anchors_random_and_oracle(setup, cfg: dict[str, Any]) -> None:
    from mfsi_cxr.eval.rq1 import oracle_aurc
    from mfsi_cxr.eval.selective import aurc

    _, val, _, _, scores, y_hat = setup
    correct, used = correctness(val["y"], y_hat, "negative")
    pf = evaluate(Inputs([scores], val["y"], [y_hat], val["patient_id"], "negative"), cfg)["per_finding"]
    for j, f in enumerate(FINDINGS):
        c = correct[used[:, j], j]
        rows = pf[pf.finding == f]
        assert np.allclose(rows["aurc_random"], 1 - c.mean())
        assert np.allclose(rows["aurc_oracle"], aurc(c.astype(float), c))  # trust = correctness itself
        assert (rows["aurc_oracle"] <= rows["aurc"] + 1e-12).all()
        assert np.array_equal(rows["below_random"], rows["aurc"] > rows["aurc_random"])
    assert oracle_aurc(4, 2) == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)


def test_finding_mask_makes_other_findings_na(setup, cfg: dict[str, Any]) -> None:
    _, val, _, _, scores, y_hat = setup
    mask = np.zeros(len(FINDINGS), bool)
    mask[[FINDINGS.index("Edema"), FINDINGS.index("Fracture")]] = True
    out = evaluate(Inputs([scores], val["y"], [y_hat], val["patient_id"], "exclude", mask), cfg)
    pf = out["per_finding"]
    assert pf.loc[~pf.finding.isin(["Edema", "Fracture"]), "aurc"].isna().all()
    m = out["main"].set_index("score")
    assert (m["n_findings"] == 2).all()
    e = pf[pf.finding.isin(["Edema", "Fracture"])].groupby("score")["aurc"].mean()
    assert np.allclose(m.loc[e.index, "aurc"], e)


def test_exclude_eligibility_counts(cfg: dict[str, Any]) -> None:
    from mfsi_cxr.eval.pipeline import exclude_eligibility

    y = np.full((40, len(FINDINGS)), np.nan)
    y[:10, 0], y[10:20, 0] = 1.0, 0.0       # 10 / 10 -> eligible
    y[:10, 1], y[10:19, 1] = 1.0, 0.0       # 10 / 9  -> not
    y[:5, 2] = -1.0
    t = exclude_eligibility(cfg, y).set_index("finding")
    assert t.loc[FINDINGS[0], "eligible"] and not t.loc[FINDINGS[1], "eligible"]
    assert t.loc[FINDINGS[2], "n_uncertain"] == 5 and t.loc[FINDINGS[2], "n_blank"] == 35
