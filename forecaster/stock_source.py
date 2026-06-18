"""
stock_source.py
===============
Where "current stock on hand" comes from.

>>> FUTURE DEV NOTE <<<
Current stock will eventually be read from the company database via an API. That
integration is intentionally NOT built yet. Everything else in this package calls
`get_current_stock(item)` and does not care where the number comes from, so wiring
the real API later means implementing ONE function (`_from_api`) and setting
`stock_source.mode: api` in config.yaml. No other code changes.

Modes (config.yaml -> stock_source.mode):
  stub : return assumed_stock, or a rough heuristic from recent demand history
  file : read a CSV (item,current_stock) the company drops in
  api  : call the DB API   [NOT IMPLEMENTED - raises NotImplementedError]
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import get_config


@lru_cache(maxsize=1)
def _stock_file(path: str) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["item"], df["current_stock"]))


def _from_file(item: str) -> float:
    cfg = get_config()["stock_source"]
    path = cfg.get("file_path")
    if not path or not Path(path).exists():
        raise FileNotFoundError(
            f"stock_source.mode='file' but file_path is missing/not found: {path!r}"
        )
    table = _stock_file(path)
    if item not in table:
        raise KeyError(f"item '{item}' not present in stock file {path!r}")
    return float(table[item])


def _from_api(item: str) -> float:
    # ---------------------------------------------------------------------
    # TODO (future): call the company DB API and return current stock on hand.
    # Example shape:
    #     resp = requests.get(f"{BASE_URL}/stock/{item}", headers=auth, timeout=10)
    #     return float(resp.json()["on_hand"])
    # Until then this path is deliberately disabled.
    # ---------------------------------------------------------------------
    raise NotImplementedError(
        "DB API stock source is not implemented yet. Use mode 'stub' or 'file', "
        "or pass current_stock explicitly to the prediction call."
    )


def _stub(item: str) -> float:
    """Placeholder stock until the API is wired.

    If config sets a fixed assumed_stock, use it. Otherwise return a clearly
    flagged heuristic (~30 days of average recent demand) so the tool is runnable
    for demos. NOT a real inventory figure.
    """
    cfg = get_config()["stock_source"]
    if cfg.get("assumed_stock") is not None:
        return float(cfg["assumed_stock"])
    # heuristic from bundled history, purely so demos have a number
    from . import artifacts
    try:
        attrs = artifacts.item_attributes(item)
        # ~30 days of demand as a stand-in "you currently hold about a month"
        return round(float(attrs.get("total_issued", 0.0)) / max(1, 1558) * 30, 1)
    except Exception:
        return 0.0


def get_current_stock(item: str, override: float | None = None) -> tuple[float, str]:
    """Return (current_stock, source_label).

    `override` (an explicit value passed by the caller) always wins - this is how
    the company's own system can feed real stock in today, before the API exists.
    """
    if override is not None:
        return float(override), "override"
    mode = get_config()["stock_source"].get("mode", "stub")
    if mode == "file":
        return _from_file(item), "file"
    if mode == "api":
        return _from_api(item), "api"
    return _stub(item), "stub (PLACEHOLDER - not real stock)"
