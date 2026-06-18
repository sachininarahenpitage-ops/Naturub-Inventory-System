"""
artifacts.py
============
Load the bundled, frozen model artifacts. Models are used AS-IS (no retraining):
  - four LightGBM boosters (daily 30/90-day, weekly 4/13-week)
  - item_master.csv      static per-item attributes + each item's best baseline
  - category_levels.json exact training category levels, per grain
  - ship_table.csv       per item x grain x horizon: did ML beat the baseline?

Everything is loaded once and cached.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# (grain, horizon_periods) -> booster filename
MODEL_FILES = {
    ("daily", 30): "lgbm_daily_h30.txt",
    ("daily", 90): "lgbm_daily_h90.txt",
    ("weekly", 4): "lgbm_weekly_h4.txt",     # 4 weeks  ~ 30 days
    ("weekly", 13): "lgbm_weekly_h13.txt",   # 13 weeks ~ 90 days
}

# map a calendar horizon in days to the (grain, model-units) key
HORIZON_TO_MODEL = {
    ("daily", 30): ("daily", 30),
    ("daily", 90): ("daily", 90),
    ("weekly", 30): ("weekly", 4),
    ("weekly", 90): ("weekly", 13),
}


@lru_cache(maxsize=None)
def load_model(grain: str, horizon_periods: int) -> lgb.Booster:
    fname = MODEL_FILES[(grain, horizon_periods)]
    return lgb.Booster(model_file=str(ARTIFACT_DIR / fname))


@lru_cache(maxsize=1)
def load_category_levels() -> dict:
    with open(ARTIFACT_DIR / "category_levels.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_item_master() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "item_master.csv")


@lru_cache(maxsize=1)
def load_ship_table() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "ship_table.csv")


def known_items() -> list[str]:
    return load_item_master()["item"].tolist()


def item_attributes(item: str) -> dict:
    m = load_item_master()
    row = m[m["item"] == item]
    if row.empty:
        raise KeyError(f"Unknown item '{item}'. Known items: {len(m)} (see item_master.csv).")
    return row.iloc[0].to_dict()


def ml_wins(item: str, grain: str, horizon_days: int) -> bool:
    """Did ML beat this item's baseline by >=10% for this grain+horizon?"""
    t = load_ship_table()
    hit = t[(t["item"] == item) & (t["grain"] == grain) & (t["horizon"] == horizon_days)]
    if hit.empty:
        return False
    return bool(hit.iloc[0]["ml_wins"])
