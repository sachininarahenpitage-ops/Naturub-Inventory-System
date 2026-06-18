"""
forecast.py
===========
Forecast total demand over the next 30 or 90 days for one item.

Routing ("baselines are the floor"):
  - Use the trained ML booster only where it beat the item's baseline by >=10%
    on the Phase-4 held-out test (ship_table.csv).
  - Otherwise fall back to that item's committed baseline (MA-7/30/90 or Croston).

Also returns a demand-uncertainty estimate (sigma over the horizon), used for
safety stock and the forecast band. sigma_horizon = std(daily demand) * sqrt(H),
the standard demand-over-lead-time spread.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import artifacts, baselines, features


@dataclass
class DemandForecast:
    item: str
    grain: str
    horizon_days: int
    forecast: float          # expected total demand over the horizon
    method: str              # 'ml' or the baseline name (ma30, croston, ...)
    ml_forecast: float
    baseline_forecast: float
    ml_used: bool
    sigma_horizon: float     # std of total demand over the horizon
    n_history: int

    def as_dict(self) -> dict:
        return asdict(self)


def _prepare_history(item: str, history: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Attach static attributes from the item master if the caller didn't supply
    them, and resample to weekly when needed."""
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    attrs = artifacts.item_attributes(item)
    for a in features.STATIC_ATTRS:
        if a not in h.columns:
            h[a] = attrs[a]
    if grain == "weekly":
        h = features.to_weekly(h)
    return h.sort_values("date").reset_index(drop=True)


def _sigma_horizon(history: pd.DataFrame, horizon_days: int) -> float:
    h = history.sort_values("date").copy()
    h["date"] = pd.to_datetime(h["date"])
    daily = h.set_index("date")["y"].asfreq("D").fillna(0.0)
    daily_std = float(daily.tail(180).std()) if len(daily) else 0.0
    if np.isnan(daily_std):
        daily_std = 0.0
    return daily_std * np.sqrt(horizon_days)


def forecast_demand(item: str, history: pd.DataFrame, horizon_days: int,
                    grain: str = "daily") -> DemandForecast:
    """Forecast total demand over `horizon_days` (30 or 90) for `item`.

    `history` needs columns [date, y]; static attributes are looked up if absent.
    """
    if horizon_days not in (30, 90):
        raise ValueError("horizon_days must be 30 or 90")
    if grain not in ("daily", "weekly"):
        raise ValueError("grain must be 'daily' or 'weekly'")

    raw = history.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if len(raw) < 2:
        raise ValueError(
            f"history for '{item}' has {len(raw)} row(s); need at least 2 to build "
            "lag/rolling features. Provide the item's demand history."
        )
    series = _prepare_history(item, raw, grain)

    levels = artifacts.load_category_levels()[grain]
    _, model_units = artifacts.HORIZON_TO_MODEL[(grain, horizon_days)]
    model = artifacts.load_model(grain, model_units)

    origin = features.make_origin_row(series, levels)
    ml_pred = max(0.0, float(model.predict(origin)[0]))   # demand is non-negative

    attrs = artifacts.item_attributes(item)
    base_pred = baselines.baseline_forecast(raw, attrs.get("best_baseline", "ma30"), horizon_days)

    use_ml = artifacts.ml_wins(item, grain, horizon_days)
    chosen = ml_pred if use_ml else base_pred
    method = "ml" if use_ml else str(attrs.get("best_baseline", "ma30")).lower()

    return DemandForecast(
        item=item, grain=grain, horizon_days=horizon_days,
        forecast=round(chosen, 3), method=method,
        ml_forecast=round(ml_pred, 3), baseline_forecast=round(base_pred, 3),
        ml_used=use_ml, sigma_horizon=round(_sigma_horizon(raw, horizon_days), 3),
        n_history=int(len(raw)),
    )
