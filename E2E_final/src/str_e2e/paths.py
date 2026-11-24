"""Paths that remain stable after an editable pip install from the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "research_upstream"


def require_file(path: str | Path | None, name: str) -> Path:
    if not path:
        raise ValueError(f"Missing required asset: {name}. Set assets.{name} in config.")
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"{name}: {p}")
    return p.resolve()


def require_dir(path: str | Path | None, name: str) -> Path:
    if not path:
        raise ValueError(f"Missing required asset: {name}. Set assets.{name} in config.")
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_dir():
        raise FileNotFoundError(f"{name}: {p}")
    return p.resolve()
