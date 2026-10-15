"""RQ2 evaluation (Phase 6) from the experiment records. Labels enter only here (evaluation side).

A  Does it help? On SHOWN cases and on all sampled cases (human review = no decision): arbiter vs classifier
   accuracy, fixed / broken / net, exact McNemar, patient-bootstrap CI of the net change; system coverage and
   risk (arbiter decides SHOWN cases, defers the rest) vs the classifier alone at the same coverage (MSP
   order); arbiter calls per case and per study.
B  Does it listen? Flip rate of the post-vote label vs the main run, per condition; pre-stated primary test:
   flip(shuffle) - flip(rerun) with a paired patient-bootstrap CI (H2a supported iff its lower end > 0).
   Modality swap reported by direction; agreement with the classifier, revealed (main) vs blinded;
   cannot_decide rate by decile of the R shown (main and shuffled).
C  Does it hold up? Verified-claim share (overall, per finding), claims per case, cases with >= 1 / 0
   verified claims; unanimous / 2-1 / no-majority; cannot_decide; label-tree rejections; human review by
   cause; AUROC of the verified fraction vs arbiter correctness; label-tree violations of the classifier alone
   vs the full system.
Correctness uses eval.blank_policy (blank = negative by default); uncertain labels are excluded.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mfsi_cxr.arbiter.experiments import CONDITIONS
from mfsi_cxr.arbiter.rules import violations
from mfsi_cxr.eval.stats import bootstrap, percentile_ci

SHOWN = "SHOWN"


def _label_frame(records: list[dict[str, Any]], labels: dict[tuple[str, str], float],
                 y_hat: dict[tuple[str, str], bool], msp: dict[tuple[str, str], float],
                 blank_policy: str) -> pd.DataFrame:
    rows = []
    for r in records:
        key = (r["study_id"], r["finding"])
        y = labels[key]
        scored = (y in (0.0, 1.0)) or (np.isnan(y) and blank_policy == "negative")
        truth = bool(y == 1.0)
        m = r["main"]
        decided = m["status"] == SHOWN and m["decision"] in ("present", "absent")
        rows.append({"case_id": r["case_id"], "patient_id": r["patient_id"], "study_id": r["study_id"],
                     "finding": r["finding"], "scored": scored, "truth": truth, "classifier": y_hat[key],
                     "msp": msp[key], "status": m["status"], "decided": decided,
                     "arbiter": (m["decision"] == "present") if decided else None,
                     "n_claims": len(m["claims"]),
                     "n_verified": sum(c["tag"] == "VERIFIED" for c in m["claims"]),
                     "n_calls_main": m["n_calls"], "n_calls_total": m["n_calls"] + sum(r["calls"].values()),
                     "main_label": r["main_label"], "run_decisions": m["run_decisions"],
                     "R_main": r["R_shown"]["main"], "R_shuffle": r["R_shown"]["shuffle"],
                     "two_witness": r["two_witness"], "image_more_reliable": r["image_more_reliable"],
                     **{f"label_{c}": r["labels"].get(c) for c in CONDITIONS}})
    return pd.DataFrame(rows)


def _ci(values: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    return percentile_ci(values, cfg["eval"]["alpha"])


def _boot(df: pd.DataFrame, stat, cfg: dict[str, Any]) -> np.ndarray:
    e = cfg["eval"]
    return bootstrap(df["patient_id"].to_numpy(), lambda rows: stat(df.iloc[rows]), e["bootstrap_B"],
                     e["bootstrap_seed"])


def experiment_a(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, sub in (("shown", df[df["decided"] & df["scored"]]), ("all_sampled", df[df["scored"]])):
        clf_ok = sub["classifier"] == sub["truth"]
        arb_ok = sub["decided"] & (sub["arbiter"] == sub["truth"])
        fixed = int((~clf_ok & arb_ok).sum())
        broken = int((clf_ok & ~arb_ok & sub["decided"]).sum())

        def net(s: pd.DataFrame) -> float:
            c = s["classifier"] == s["truth"]
            a = s["decided"] & (s["arbiter"] == s["truth"])
            return float(((~c & a).sum() - (c & ~a & s["decided"]).sum()) / max(len(s), 1))

        lo, hi = _ci(_boot(sub, net, cfg), cfg) if len(sub) else (np.nan, np.nan)
        p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else np.nan
        rows.append({"population": name, "n": len(sub), "n_decided": int(sub["decided"].sum()),
                     "classifier_accuracy": float(clf_ok.mean()) if len(sub) else np.nan,
                     "arbiter_accuracy_decided": float(arb_ok[sub["decided"]].mean())
                     if sub["decided"].any() else np.nan, "fixed": fixed, "broken": broken,
                     "net_per_case": net(sub) if len(sub) else np.nan,
                     "net_lo": lo, "net_hi": hi, "mcnemar_exact_p": p})
    # system coverage / risk on the sampled escalated cases vs the classifier alone at the same coverage
    s = df[df["scored"]]
    cov = float(s["decided"].mean()) if len(s) else np.nan
    d = s[s["decided"]]
    sys_risk = float((d["arbiter"] != d["truth"]).mean()) if len(d) else np.nan
    k = int(round(cov * len(s))) if len(s) else 0
    top = s.sort_values("msp", ascending=False).head(k)
    clf_risk = float((top["classifier"] != top["truth"]).mean()) if k else np.nan
    rows.append({"population": "coverage_at_system_coverage", "n": len(s), "coverage": cov,
                 "system_risk": sys_risk, "classifier_risk_same_coverage": clf_risk,
                 "calls_per_case": float(df["n_calls_main"].mean()),
                 "calls_per_study": float(df.groupby("study_id")["n_calls_main"].sum().mean())})
    return pd.DataFrame(rows)


def _flip(df: pd.DataFrame, cond: str) -> pd.Series:
    sub = df[df[f"label_{cond}"].notna()]
    return sub[f"label_{cond}"] != sub["main_label"]


def experiment_b(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for c in CONDITIONS:
        sub = df[df[f"label_{c}"].notna()]
        if not len(sub):
            rows.append({"condition": c, "n": 0})
            continue
        lo, hi = _ci(_boot(sub, lambda s, c=c: float((s[f"label_{c}"] != s["main_label"]).mean()), cfg), cfg)
        rows.append({"condition": c, "n": len(sub), "flip_rate": float(_flip(df, c).mean()), "lo": lo,
                     "hi": hi})
    for direction, mask in (("image_was_more_reliable", df["image_more_reliable"]),
                            ("text_was_more_reliable", ~df["image_more_reliable"])):
        sub = df[mask & df["label_swap_modalities"].notna()]
        rows.append({"condition": f"swap_modalities[{direction}]", "n": len(sub),
                     "flip_rate": float((sub["label_swap_modalities"] != sub["main_label"]).mean())
                     if len(sub) else np.nan})
    flips = pd.DataFrame(rows)
    both = df[df["label_shuffle"].notna()]

    def diff(s: pd.DataFrame) -> float:
        shuffle = (s["label_shuffle"] != s["main_label"]).mean()
        return float(shuffle - (s["label_rerun"] != s["main_label"]).mean())

    if len(both):
        lo, hi = _ci(_boot(both, diff, cfg), cfg)
        point = diff(both)
    else:
        point, lo, hi = np.nan, np.nan, np.nan
    clf_lab = df["classifier"].map({True: "present", False: "absent"})
    extra = pd.DataFrame([{
        "primary": "flip(shuffle) - flip(rerun)", "n": len(both), "diff": point, "lo": lo, "hi": hi,
        "H2a_supported": bool(lo > 0) if np.isfinite(lo) else False,
        "agree_classifier_revealed": float((df["main_label"] == clf_lab).mean()),
        "agree_classifier_blinded": float((df["label_blinded"] == clf_lab).mean())}])
    return flips, extra


def cannot_decide_by_r(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    for which, rcol, lcol in (("main", "R_main", "main_label"), ("shuffle", "R_shuffle", "label_shuffle")):
        sub = df[df[rcol].notna() & df[lcol].notna()]
        if len(sub) < 2:
            continue
        bins = pd.qcut(sub[rcol].rank(method="first"), min(n_bins, len(sub)), labels=False)
        for b, g in sub.groupby(bins):
            rows.append({"packet": which, "R_decile": int(b), "n": len(g), "R_min": float(g[rcol].min()),
                         "R_max": float(g[rcol].max()),
                         "cannot_decide_rate": float((g[lcol] == "cannot_decide").mean())})
    return pd.DataFrame(rows)


def experiment_c(df: pd.DataFrame, ctx_fn, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import roc_auc_score

    shown = df[df["decided"]]
    agreement = Counter()
    for runs in df["run_decisions"]:
        valid = [x for x in runs if x is not None]
        top = Counter(valid).most_common(1)[0][1] if valid else 0
        agreement["unanimous" if top == len(runs) else "two_one" if top == 2 else "no_majority"] += 1
    frac = shown["n_verified"] / shown["n_claims"].where(shown["n_claims"] > 0)
    correct = shown["scored"] & (shown["arbiter"] == shown["truth"])
    ok = frac.notna() & shown["scored"]
    auc = (roc_auc_score(correct[ok], frac[ok]) if ok.any() and correct[ok].nunique() == 2 else np.nan)
    finals: dict[str, dict[str, bool]] = {}
    for r in shown.itertuples():
        finals.setdefault(r.study_id, {})[r.finding] = bool(r.arbiter)
    clf_viol = sys_viol = 0
    for r in df.itertuples():
        clf = "present" if r.classifier else "absent"
        clf_viol += bool(violations(r.finding, clf, ctx_fn(r.study_id, {}), True, False))
        final = ("present" if r.arbiter else "absent") if r.decided else clf
        sys_viol += bool(violations(r.finding, final, ctx_fn(r.study_id, finals.get(r.study_id, {})), True,
                                    False))
    n = len(df)
    overall = pd.DataFrame([{
        "n_cases": n, "n_shown": len(shown),
        "claims_per_shown_case": float(shown["n_claims"].mean()) if len(shown) else np.nan,
        "verified_claim_share": float(shown["n_verified"].sum() / max(shown["n_claims"].sum(), 1)),
        "cases_ge1_verified": float((shown["n_verified"] >= 1).mean()) if len(shown) else np.nan,
        "cases_0_verified": float((shown["n_verified"] == 0).mean()) if len(shown) else np.nan,
        **{f"runs_{k}": v / n for k, v in agreement.items()},
        "cannot_decide_rate": float((df["main_label"] == "cannot_decide").mean()),
        **{f"status_{k}": float(v / n) for k, v in df["status"].value_counts().items()},
        "auroc_verified_fraction_vs_correct": float(auc),
        "tree_violation_rate_classifier": clf_viol / max(n, 1),
        "tree_violation_rate_system": sys_viol / max(n, 1)}])
    per_finding = (shown.groupby("finding").agg(n=("case_id", "size"), claims=("n_claims", "sum"),
                                                verified=("n_verified", "sum")).reset_index()
                   .assign(verified_share=lambda t: t["verified"] / t["claims"].where(t["claims"] > 0)))
    return overall, per_finding


def evaluate(records: list[dict[str, Any]], labels, y_hat, msp, ctx_fn,
             cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    df = _label_frame(records, labels, y_hat, msp, cfg["eval"]["blank_policy"])
    flips, primary = experiment_b(df, cfg)
    c_all, c_finding = experiment_c(df, ctx_fn, cfg)
    return {"rq2_help": experiment_a(df, cfg), "rq2_listen": flips, "rq2_listen_primary": primary,
            "rq2_cannot_decide_vs_R": cannot_decide_by_r(df, cfg["eval"]["r_deciles"]),
            "rq2_holdup": c_all, "rq2_holdup_per_finding": c_finding}
