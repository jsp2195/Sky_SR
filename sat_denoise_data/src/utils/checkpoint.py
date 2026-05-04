"""Checkpoint save/load helpers."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch


def save_checkpoint(path: str, state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def is_better(metric_value: float, best_value: Optional[float], mode: str = "min") -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return metric_value < best_value
    return metric_value > best_value
