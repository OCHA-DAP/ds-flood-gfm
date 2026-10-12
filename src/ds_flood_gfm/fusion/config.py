"""Load the fusion YAML config and resolve local/blob paths.

The config is the single source of every threshold, window and path the
pipeline uses. Nothing downstream should hard-code a number that lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ds_flood_gfm.constants import PROJECT_PREFIX

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fusion.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def local_root(self) -> Path:
        return REPO_ROOT / self.raw["paths"]["local_root"]

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / self.raw["paths"]["results_dir"]

    def local(self, *parts: str) -> Path:
        """Path under the gitignored local cache; parent dirs are created."""
        p = self.local_root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def blob_path(self, kind: str, *parts: str) -> str:
        """Blob path following the KB convention
        ``{PROJECT_PREFIX}/{raw|processed}/{datasource}/{filename}``."""
        if kind not in ("raw", "processed"):
            raise ValueError(f"kind must be 'raw' or 'processed', got {kind!r}")
        base = self.raw["paths"]["blob"][kind]
        return "/".join([PROJECT_PREFIX, base, *parts])


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"fusion config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=p)
