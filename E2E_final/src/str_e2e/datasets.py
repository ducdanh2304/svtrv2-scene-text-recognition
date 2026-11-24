"""Input sources: CSV manifest, image directory and standard OpenOCR LMDB."""
from __future__ import annotations
import csv
from pathlib import Path
from collections.abc import Iterator
from .types import Sample

_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def from_csv(csv_file: str | Path, image_root: str | Path | None = None) -> Iterator[Sample]:
    csv_file = Path(csv_file).resolve()
    base = Path(image_root).resolve() if image_root else csv_file.parent
    seen: set[str] = set()
    with csv_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("Manifest must have image_path column")
        for i, row in enumerate(reader, 1):
            sample_id = row.get("id") or str(i)
            if sample_id in seen:
                raise ValueError(f"Duplicate ID in manifest: {sample_id}")
            seen.add(sample_id)
            rel = Path(row["image_path"])
            path = rel if rel.is_absolute() else base / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            yield Sample(sample_id=str(sample_id), image_path=str(path), gt=row.get("gt"))


def from_directory(directory: str | Path) -> Iterator[Sample]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUFFIXES:
            yield Sample(sample_id=path.relative_to(root).as_posix(), image_path=str(path))


def from_lmdb(directory: str | Path, *, indices: list[int] | None = None) -> Iterator[Sample]:
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("LMDB input requires: pip install lmdb") from exc
    source = Path(directory)
    if source.is_file() and source.name == "data.mdb":
        source = source.parent
    if not (source / "data.mdb").is_file():
        raise FileNotFoundError(source / "data.mdb")
    env = lmdb.open(str(source), subdir=True, readonly=True, lock=False, readahead=False,
                    max_readers=1)
    try:
        with env.begin(write=False) as txn:
            total = int((txn.get(b"num-samples") or b"0").decode())
            if total <= 0:
                raise ValueError("LMDB lacks a valid num-samples key")
            requested = indices if indices is not None else range(1, total + 1)
            for idx in requested:
                if not (1 <= idx <= total):
                    raise ValueError(f"LMDB index outside 1..{total}: {idx}")
                raw = txn.get(f"image-{idx:09d}".encode())
                label = txn.get(f"label-{idx:09d}".encode())
                if raw is None or label is None:
                    raise ValueError(f"LMDB record {idx} has a missing image or label")
                yield Sample(sample_id=f"{idx:09d}", image_bytes=raw,
                             gt=label.decode("utf-8"))
    finally:
        env.close()
