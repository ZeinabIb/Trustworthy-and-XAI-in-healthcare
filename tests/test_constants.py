"""The approved CheXpert label tree (docs/DECISIONS.md, 2026-10-11)."""

from __future__ import annotations

from mfsi_cxr.constants import FINDINGS, NO_FINDING, PARENT, PATHOLOGIES, ancestors

# Edges of the published figure (Pham et al., Fig. 1), including the drawn Lung Opacity -> Pneumonia.
FIGURE_EDGES = {
    ("Enlarged Cardiomediastinum", "Cardiomegaly"),
    ("Lung Opacity", "Lung Lesion"),
    ("Lung Opacity", "Edema"),
    ("Lung Opacity", "Consolidation"),
    ("Lung Opacity", "Atelectasis"),
    ("Lung Opacity", "Pneumonia"),
    ("Consolidation", "Pneumonia"),
}


def test_tree_uses_known_findings_and_has_no_cycle() -> None:
    assert set(PARENT) | set(PARENT.values()) <= set(FINDINGS)
    for f in FINDINGS:
        assert f not in ancestors(f)


def test_every_figure_edge_is_an_ancestor_relation() -> None:
    for parent, child in FIGURE_EDGES:
        assert parent in ancestors(child), (parent, child)


def test_no_ancestor_relation_beyond_the_figure() -> None:
    implied = {(a, f) for f in FINDINGS for a in ancestors(f)}
    assert implied == FIGURE_EDGES


def test_pneumonia_ancestors_nearest_first() -> None:
    assert ancestors("Pneumonia") == ("Consolidation", "Lung Opacity")
    assert ancestors("Pleural Effusion") == ()


def test_pathologies_exclude_no_finding_and_support_devices() -> None:
    assert NO_FINDING not in PATHOLOGIES and "Support Devices" not in PATHOLOGIES
    assert len(PATHOLOGIES) == 12  # CheXpert: 12 pathologies + No Finding + Support Devices
