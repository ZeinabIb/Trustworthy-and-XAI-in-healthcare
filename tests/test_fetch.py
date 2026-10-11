from __future__ import annotations

from pathlib import Path

import pytest

from mfsi_cxr.data.fetch import (
    UnsafeDestinationError,
    check_destination,
    check_refs,
    merge_parquet_parts,
    png_table_path,
)
from mfsi_cxr.utils.config import REPO_ROOT


def test_destination_inside_repo_is_refused() -> None:
    with pytest.raises(UnsafeDestinationError, match="repo"):
        check_destination(REPO_ROOT / "data")
    with pytest.raises(UnsafeDestinationError):
        check_destination(REPO_ROOT)


@pytest.mark.parametrize("folder", ["OneDrive/chexpert", "Dropbox/x", "Google Drive/x"])
def test_synced_destination_is_refused(tmp_path: Path, folder: str) -> None:
    with pytest.raises(UnsafeDestinationError, match="synced"):
        check_destination(tmp_path / folder, repo_root=tmp_path / "repo")


def test_missing_drive_is_refused() -> None:
    letter = next(c for c in "QRSTUVWXYZ" if not Path(f"{c}:/").exists())
    with pytest.raises(UnsafeDestinationError, match="drive"):
        check_destination(Path(f"{letter}:/chexpert_plus"))


def test_labels_ref_as_metadata_ref_is_refused() -> None:
    ref = "aimi.chexpert_plus:5yyj:v1_0.chexpert_labels:pmec"
    with pytest.raises(UnsafeDestinationError, match="labels table"):
        check_refs(ref, ref)
    with pytest.raises(UnsafeDestinationError, match="labels table"):
        check_refs(ref, ref + "/findings_fixed.json")
    check_refs(ref, None)
    check_refs(ref, "aimi.chexpert_plus:5yyj:v1_0.df_chexpert_plus_240401:bavj")


def test_png_table_path() -> None:
    assert png_table_path("train/patient00001/study2/view1_frontal.jpg") == (
        "train", "patient00001/study2/view1_frontal.png")
    assert png_table_path("valid\\patient64541\\study1\\view1_frontal.jpg")[0] == "valid"
    with pytest.raises(ValueError):
        png_table_path("patient00001/study2/view1_frontal.jpg")


def test_local_image_path_maps_jpg_to_png(tmp_path: Path) -> None:
    from mfsi_cxr.data.images import local_image_path

    p = local_image_path("train/patient00001/study2/view1_frontal.jpg", tmp_path, "png")
    assert p == tmp_path / "train" / "patient00001" / "study2" / "view1_frontal.png"
    with pytest.raises(NotImplementedError):
        local_image_path("train/p/s/v.jpg", tmp_path, "dicom")


def test_merge_parquet_parts(tmp_path: Path) -> None:
    import pandas as pd

    parts = []
    for i in range(3):
        p = tmp_path / f"{i:06d}.parquet"
        pd.DataFrame({"path_to_image": [f"img{i}a", f"img{i}b"], "age": [i, i]}).to_parquet(p)
        parts.append(p)
    out = merge_parquet_parts(list(reversed(parts)), tmp_path / "meta.parquet")
    df = pd.read_parquet(out)
    assert len(df) == 6 and df["path_to_image"].tolist()[:2] == ["img0a", "img0b"]
    assert not (tmp_path / "meta.parquet.tmp").exists()


def test_plain_local_folder_is_accepted(tmp_path: Path) -> None:
    dest = tmp_path / "chexpert_plus"
    assert check_destination(dest, repo_root=tmp_path / "repo") == dest.resolve()


class _FakeFile:
    def __init__(self, name: str, fail: bool) -> None:
        self.name, self.fail = name, fail

    def download(self, folder: str) -> None:
        (Path(folder) / self.name).write_text("partial" if self.fail else "complete", encoding="utf-8")
        if self.fail:
            raise ConnectionError("dropped")


def _fake_redivis(monkeypatch: pytest.MonkeyPatch, fail: set[str], calls: list[str]) -> None:
    import sys
    import types

    class Table:
        def __init__(self, ref: str) -> None:
            pass

        def file(self, name: str) -> _FakeFile:
            calls.append(name)
            return _FakeFile(name, name in fail)

    monkeypatch.setitem(sys.modules, "redivis", types.SimpleNamespace(table=Table))


def test_interrupted_file_download_leaves_no_final_file(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from mfsi_cxr.data.fetch import fetch_table_files

    calls: list[str] = []
    _fake_redivis(monkeypatch, {"b.json"}, calls)
    with pytest.raises(ConnectionError):
        fetch_table_files("ref", ["a.json", "b.json"], tmp_path / "out")
    assert (tmp_path / "out" / "a.json").read_text(encoding="utf-8") == "complete"
    assert not (tmp_path / "out" / "b.json").exists()

    _fake_redivis(monkeypatch, set(), calls)
    calls.clear()
    out = fetch_table_files("ref", ["a.json", "b.json"], tmp_path / "out")
    assert calls == ["b.json"]  # a.json skipped, b.json retried
    assert [p.read_text(encoding="utf-8") for p in out] == ["complete", "complete"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.json", "b.json"]
