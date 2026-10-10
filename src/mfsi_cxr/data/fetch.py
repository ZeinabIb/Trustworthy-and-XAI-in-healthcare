"""Download the small raw tables (labels files, metadata table) from Redivis into the data root.

Run by Zeinab only, after confirming the data use agreement allows the destination. Never downloads
images. Authentication is the Redivis client's own browser login (credentials cached by the client in
the user's home folder, outside the repo); no credential is ever read from or written to the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mfsi_cxr.utils.config import REPO_ROOT

# Path fragments that mark a cloud-synced folder; patient data must never land there.
SYNCED_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud")


class UnsafeDestinationError(ValueError):
    """The destination is inside the repo or in a cloud-synced folder."""


def check_destination(dest: Path, repo_root: Path = REPO_ROOT) -> Path:
    if not Path(dest.anchor or ".").exists():
        raise UnsafeDestinationError(
            f"drive {dest.anchor} does not exist; choose a folder on an existing drive"
        )
    dest = dest.resolve()
    repo = repo_root.resolve()
    if dest == repo or repo in dest.parents:
        raise UnsafeDestinationError(f"{dest} is inside the git repo; use a folder outside it")
    lowered = str(dest).lower()
    for marker in SYNCED_MARKERS:
        if marker in lowered:
            raise UnsafeDestinationError(
                f"{dest} looks cloud-synced ({marker}); use a local encrypted folder"
            )
    return dest


def fetch_label_files(table_ref: str, files: list[str], dest: Path) -> list[Path]:
    import redivis

    table = redivis.table(table_ref)
    out = []
    for name in files:
        target = dest / name
        if target.exists():
            out.append(target)
            continue
        table.file(name).download(str(dest))
        out.append(target)
    return out


def fetch_metadata_table(table_ref: str, dest_file: Path) -> Path:
    import pyarrow.parquet as pq
    import redivis

    if dest_file.exists():
        return dest_file
    if dest_file.suffix != ".parquet":
        raise ValueError(f"metadata is saved as parquet; got {dest_file.name}")
    table = redivis.table(table_ref).to_arrow_table()
    tmp = dest_file.with_name(dest_file.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(dest_file)  # a killed download never leaves a partial file under the final name
    return dest_file


def check_refs(labels_ref: str, metadata_ref: str | None) -> None:
    if metadata_ref and metadata_ref == labels_ref:
        raise UnsafeDestinationError(
            "--metadata-table is the labels table; copy the ref from the df_chexpert_plus_240401 table page"
        )


def fetch_all(cfg: dict[str, Any], dest: Path, labels_ref: str, metadata_ref: str | None) -> list[Path]:
    check_refs(labels_ref, metadata_ref)
    dest = check_destination(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fetch = cfg["fetch"]
    written = fetch_label_files(labels_ref, list(fetch["label_files"]), dest)
    if metadata_ref:
        written.append(fetch_metadata_table(metadata_ref, dest / cfg["data"]["metadata_file"]))
    return written
