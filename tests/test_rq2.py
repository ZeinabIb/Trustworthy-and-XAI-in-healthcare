"""Phase 6: derangement shuffle, patient sampling, modality swap, flip rates, noise-floor seeds."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.test_arbiter import _inputs

from mfsi_cxr.arbiter import llm_client as lc
from mfsi_cxr.arbiter.cases import derangement, select_cases_by_patient
from mfsi_cxr.arbiter.experiments import CONDITIONS
from mfsi_cxr.arbiter.packet import Modality, NotApplicableError, build_packet
from mfsi_cxr.arbiter.run import voted
from mfsi_cxr.eval.arbiter_eval import experiment_b


def test_derangement_is_seeded_and_fixed_point_free() -> None:
    for n in range(2, 12):
        p = derangement(n, np.random.default_rng(n))
        assert sorted(p) == list(range(n)) and not np.any(p == np.arange(n))
        assert np.array_equal(p, derangement(n, np.random.default_rng(n)))
    assert derangement(1, np.random.default_rng(0)) is None


def test_shuffle_preserves_the_marginal_of_r_per_finding() -> None:
    from mfsi_cxr.arbiter.experiments import shuffle_donors

    class FakeBuilder:
        def inputs(self, study, finding):
            return _inputs(modality=Modality(0.5, 0.5, 0.5, 0.5, float(study[1:]) / 10, "two_witness", True))

    cases = pd.DataFrame({"study_id": [f"s{k}" for k in range(6)] + ["s9"],
                          "finding": ["Edema"] * 6 + ["Fracture"]})
    donors = shuffle_donors(cases, FakeBuilder(), seed=0)
    own = {f"s{k}|Edema": k / 10 for k in range(6)}
    got = [donors[c].R for c in own]
    assert sorted(got) == sorted(own.values())                      # same multiset of R
    assert all(r != donors[c].R for c, r in own.items())             # nobody keeps their own
    assert donors["s9|Fracture"] is None                             # single case: not deranged
    assert [d.R for d in shuffle_donors(cases, FakeBuilder(), seed=0).values() if d] == \
        [d.R for d in donors.values() if d]                          # seeded


def test_patient_sampling_takes_whole_patients() -> None:
    rng = np.random.default_rng(1)
    rows = [{"patient_id": f"p{k % 30}", "study_id": f"p{k % 30}/s1", "finding": f, "y_hat": True,
             "prob_cal": 0.5, "p_img_pos": 0.5, "p_txt_pos": 0.5, "r_img": 0.5, "r_txt": 0.5, "R": 0.5,
             "stratum": "one_witness", "escalate": bool(rng.random() < 0.5)}
            for k in range(30) for f in ("Edema", "Fracture", "Lung Opacity", "Pneumonia")]
    table = pd.DataFrame(rows)
    sample, info = select_cases_by_patient(table, 20, seed=0)
    assert info["n_cases"] >= 20 and len(sample) == info["n_cases"]
    esc = table[table["escalate"]]
    for pid in sample["patient_id"].unique():  # every escalated case of a chosen patient
        assert len(sample[sample["patient_id"] == pid]) == len(esc[esc["patient_id"] == pid])
    again, _ = select_cases_by_patient(table, 20, seed=0)
    assert again.equals(sample)


def test_swap_modalities(smoke_cfg: dict[str, Any]) -> None:
    p = build_packet(_inputs(), smoke_cfg, reliability="swap_modalities")
    mod = next(t for i, t in p.items if p.roles[i] == "modality")
    assert "image profile P(present) = 0.30" in mod and "r_img = 0.40" in mod and "r_txt = 0.60" in mod
    one = Modality(0.8, 0.5, 0.6, 0.0, 0.6, "one_witness", False)
    with pytest.raises(NotApplicableError):
        build_packet(_inputs(modality=one), smoke_cfg, reliability="swap_modalities")


def test_flip_rates_on_scripted_decisions(smoke_cfg: dict[str, Any]) -> None:
    cfg = copy.deepcopy(smoke_cfg)
    cfg["eval"]["bootstrap_B"] = 50
    n = 40
    main = ["present"] * n
    df = pd.DataFrame({"patient_id": [f"p{k}" for k in range(n)], "main_label": main,
                       "image_more_reliable": [k % 2 == 0 for k in range(n)],
                       **{f"label_{c}": main for c in CONDITIONS}})
    df["label_rerun"] = ["absent" if k < 4 else "present" for k in range(n)]       # 10% noise
    df["label_shuffle"] = ["absent" if k < 20 else "present" for k in range(n)]    # 50% flips
    df["label_swap_modalities"] = [None] * n
    flips, primary = experiment_b(df.assign(classifier=True), cfg)
    f = flips.set_index("condition")
    assert f.loc["rerun", "flip_rate"] == pytest.approx(0.1)
    assert f.loc["shuffle", "flip_rate"] == pytest.approx(0.5)
    assert f.loc["swap_modalities", "n"] == 0
    assert primary["diff"].iloc[0] == pytest.approx(0.4) and primary["H2a_supported"].iloc[0]


def test_noise_floor_uses_different_seeds(smoke_cfg: dict[str, Any], tmp_path) -> None:
    seen: list[int] = []
    client = lc.CachedClient(lc.MockBackend(lambda m, s: seen.append(s) or json.dumps(
        {"decision": "absent", "claims": []})), tmp_path, 0.7, 0.95, 100)
    cfg = copy.deepcopy(smoke_cfg)
    p = build_packet(_inputs(), cfg)
    voted(client, cfg, p)
    noise = copy.deepcopy(cfg)
    noise["arbiter"]["seeds"] = list(cfg["arbiter"]["experiments"]["noise_seeds"])
    voted(client, noise, p)
    assert seen[:3] == cfg["arbiter"]["seeds"] and seen[3:] == noise["arbiter"]["seeds"]
    assert not set(cfg["arbiter"]["seeds"]) & set(noise["arbiter"]["seeds"])
