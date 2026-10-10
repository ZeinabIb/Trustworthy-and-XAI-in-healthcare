from __future__ import annotations

from pathlib import Path

import pytest

from mfsi_cxr.data.fetch import UnsafeDestinationError, check_destination, check_refs
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
    check_refs(ref, None)


def test_plain_local_folder_is_accepted(tmp_path: Path) -> None:
    dest = tmp_path / "chexpert_plus"
    assert check_destination(dest, repo_root=tmp_path / "repo") == dest.resolve()
