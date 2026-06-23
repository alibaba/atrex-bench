"""Shared utilities for Atrex-Bench."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def save_json(data: dict, path: Path) -> None:
    """Save data to a JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_timestamp(timestamp: str | None = None) -> str:
    """Return the second-level timestamp used for timestamped artifacts."""
    if timestamp is not None:
        return timestamp
    return datetime.now().strftime("%Y%m%d-%H%M%S")
