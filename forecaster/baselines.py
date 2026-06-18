"""
baselines.py
============
The simple baselines that are "the floor". Used wherever the trained ML model did
NOT beat the baseline by >=10% (per ship_table.csv). No training required - these
are computed directly from the item's own recent history.

All baselines forecast TOTAL demand over the next `horizon_days` (matching the ML
target, which is a forward sum of demand).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _daily_series(history: pd.DataFrame) -> pd.Series:
    h = history.sort_values("date").copy()
    h["date"] = pd.to_datetime(h["date"])
    return h.set_index("date")["y"].asfreq("D").fillna(0.0)


def moving_average(history: pd.DataFrame, window_days: int, horizon_days: int) -> float:
    """Average daily demand over the last `window_days`, scaled to the horizon."""
    y = _daily_series(history)
    recent = y.tail(window_days)
    rate = float(recent.mean()) if len(recent) else 0.0
    return max(0.0, rate * horizon_days)


def croston(history: pd.DataFrame, horizon_days: int, alpha: float = 0.1) -> float:
    """Croston's method for intermittent demand -> daily rate, scaled to horizon."""
    y = _daily_series(history).to_numpy()
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return 0.0
    z = y[nz[0]]          # demand-size estimate
    p = 1.0               # interval estimate (periods between demands)
    last = nz[0]
    for i in nz[1:]:
        gap = i - last
        z = z + alpha * (y[i] - z)
        p = p + alpha * (gap - p)
        last = i
    rate = (z / p) if p > 0 else 0.0
    return max(0.0, float(rate) * horizon_days)


def baseline_forecast(history: pd.DataFrame, best_baseline: str, horizon_days: int) -> float:
    """Dispatch to the item's committed best baseline (from baseline_to_beat)."""
    name = (best_baseline or "ma30").lower()
    if name == "ma7":
        return moving_average(history, 7, horizon_days)
    if name == "ma30":
        return moving_average(history, 30, horizon_days)
    if name == "ma90":
        return moving_average(history, 90, horizon_days)
    if name == "croston":
        return croston(history, horizon_days)
    return moving_average(history, 30, horizon_days)
