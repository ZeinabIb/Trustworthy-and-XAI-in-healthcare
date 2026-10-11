"""Download raw data from Redivis into the data root: label files, the metadata table, and (separately,
on request) the PNG images of the selected patients only.

Run by Zeinab only, after confirming the data use agreement allows the destination. Authentication is
the Redivis client's own browser login (credentials cached by the client in the user's home folder,
outside the repo); no credential is ever read from or written to the repo.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from mfsi_cxr.utils.config import REPO_ROOT
from mfsi_cxr.utils.logging import get_logger

log = get_logger("fetch")

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


def fetch_table_files(table_ref: str, files: list[str], dest: Path) -> list[Path]:
    """Download named files of a Redivis file table into dest; skips files already present.

    Each file is downloaded into a staging folder and moved into place when complete, so an interrupted
    run never leaves a partial file under the final name (which a restart would then skip).
    """
    import shutil

    import redivis

    table = redivis.table(table_ref)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in files:
        target = dest / name
        if not target.exists():
            staging = dest / f"_download_{Path(name).stem}"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            table.file(name).download(str(staging))
            (staging / name).replace(target)
            shutil.rmtree(staging)
        out.append(target)
    return out


def fetch_metadata_table(table_ref: str, dest_file: Path) -> Path:
    """Server-side parquet export, then file download (byte-range retries), then merge to one file.

    A streamed read (`to_arrow_table`) expires after ~30 min, too short on a slow connection.
    """
    import shutil

    import redivis

    if dest_file.exists():
        return dest_file
    if dest_file.suffix != ".parquet":
        raise ValueError(f"metadata is saved as parquet; got {dest_file.name}")
    staging = dest_file.parent / f"_export_{dest_file.stem}"
    if staging.exists():
        shutil.rmtree(staging)  # an export is only reusable within one run
    staging.mkdir(parents=True)
    parts = redivis.table(table_ref).download(str(staging), format="parquet")
    merge_parquet_parts([Path(p) for p in parts], dest_file)
    shutil.rmtree(staging)
    return dest_file


def merge_parquet_parts(parts: list[Path], dest_file: Path) -> Path:
    """Merge export parts into `dest_file` atomically (temp file, then rename)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not parts:
        raise ValueError("export produced no files")
    table = pa.concat_tables([pq.read_table(p) for p in sorted(parts)], promote_options="default")
    tmp = dest_file.with_name(dest_file.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(dest_file)  # a killed run never leaves a partial file under the final name
    return dest_file


def check_refs(labels_ref: str, metadata_ref: str | None) -> None:
    if not metadata_ref:
        return
    labels_table = labels_ref.split("/")[0]
    if metadata_ref.split("/")[0] == labels_table or "/" in metadata_ref:
        raise UnsafeDestinationError(
            "--metadata-table must be the df_chexpert_plus_240401 table itself, not the labels table "
            "or a file; leave it out to use fetch.metadata_table from configs/data.yaml"
        )


def fetch_all(cfg: dict[str, Any], dest: Path, labels_ref: str, metadata_ref: str | None) -> list[Path]:
    check_refs(labels_ref, metadata_ref)
    dest = check_destination(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fetch = cfg["fetch"]
    written = fetch_table_files(labels_ref, list(fetch["label_files"]), dest)
    if metadata_ref:
        written.append(fetch_metadata_table(metadata_ref, dest / cfg["data"]["metadata_file"]))
    return written


def fetch_radgraph(cfg: dict[str, Any], dest: Path) -> list[Path]:
    """RadGraph XL annotation files into dest/<data.radgraph_dir>/."""
    dest = check_destination(dest)
    return fetch_table_files(cfg["fetch"]["radgraph_table"], list(cfg["radgraph"]["files"].values()),
                             dest / cfg["data"]["radgraph_dir"])


def png_table_path(image_path: str) -> tuple[str, str]:
    """'train/patient1/study2/view1_frontal.jpg' -> ('train', 'patient1/study2/view1_frontal.png').

    The metadata names images by their original .jpg path with a split prefix; the PNG tables hold
    the same files as patient/study/view .png without the prefix.
    """
    parts = PurePosixPath(image_path.replace("\\", "/")).with_suffix(".png").parts
    if len(parts) != 4 or parts[0] not in ("train", "valid"):
        raise ValueError(f"unexpected image path {image_path!r}")
    return parts[0], "/".join(parts[1:])


def fetch_images(cfg: dict[str, Any], dest: Path, image_paths: list[str]) -> tuple[int, int]:
    """Download the given images (as PNG) under dest/<split>/...; skips files already present.

    Each file goes to a .part name first and is renamed when complete, so an interrupted run can simply
    be restarted. Returns (downloaded, skipped).
    """
    import redivis

    dest = check_destination(dest)
    tables = {"train": redivis.table(cfg["fetch"]["png_train_table"]),
              "valid": redivis.table(cfg["fetch"]["png_valid_table"])}
    downloaded = skipped = 0
    for i, rel in enumerate(image_paths, start=1):
        split, table_path = png_table_path(rel)
        target = dest / split / table_path
        if target.exists():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        if part.exists():
            part.unlink()
        tables[split].file(table_path).download(str(part), progress=False)
        part.replace(target)
        downloaded += 1
        if i % 50 == 0 or i == len(image_paths):
            log.info("images %d/%d (downloaded %d, skipped %d)", i, len(image_paths), downloaded, skipped)
    return downloaded, skipped
