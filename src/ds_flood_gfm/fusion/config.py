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


def pull_raw(cfg: Config, local: Path, datasource: str) -> bool:
    """Fetch `local`'s file from blob raw/{datasource}/{name} if it exists there.

    Returns False only when the blob does not exist (absence); other errors raise.
    Raw inputs are pushed there by scripts/fusion/phase6_package.py.
    """
    import ocha_stratus as stratus
    from azure.core.exceptions import ResourceNotFoundError

    b = cfg.raw["paths"]["blob"]
    cc = stratus.get_container_client(container_name=b["container"], stage=b["stage"])
    try:
        data = cc.get_blob_client(cfg.blob_path("raw", datasource, local.name)).download_blob().readall()
    except ResourceNotFoundError:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return True


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"fusion config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=p)
