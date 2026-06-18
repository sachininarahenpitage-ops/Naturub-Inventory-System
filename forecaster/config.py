"""
config.py
=========
Single place for the few external numbers the tool needs. Edit config.yaml; this
module loads it with safe defaults so the package still runs out of the box.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_DEFAULTS = {
    "service_level_z": 1.65,        # 95% service level for safety stock
    "default_horizon_days": 30,
    "lead_time_days": {"_default": 45},
    "stock_source": {"mode": "stub", "file_path": None, "assumed_stock": None},
}


@lru_cache(maxsize=1)
def get_config() -> dict:
    cfg = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    return cfg


def lead_time_for(material: str) -> int:
    """Lead time (days) for a raw material; falls back to the configured default.

    NOTE: lead times are PLACEHOLDERS until the company supplies the real
    per-material values (see config.yaml).
    """
    table = get_config().get("lead_time_days", {})
    if material in table:
        return int(table[material])
    return int(table.get("_default", 45))


def service_level_z() -> float:
    return float(get_config().get("service_level_z", 1.65))


def reorder_basis_horizon(lead_time_days: int) -> int:
    """Pick the model horizon whose window best matches the lead time.

    The reorder point needs 'expected demand over the lead time'. With lead times
    of 60-100 days, the 90-day model's demand rate is a better basis than the
    30-day rate (which would understate demand when extrapolated). Rule: use the
    90-day model when lead time >= 60 days, else the 30-day model.
    """
    return 90 if lead_time_days >= 60 else 30
