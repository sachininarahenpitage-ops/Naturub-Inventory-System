"""
features.py
===========
Rebuild the 22 model features from raw demand history.

Every formula here was reverse-engineered from the training parquet files and
verified to match the stored features *exactly* (floating-point precision) for
all 23 items and both grains. Do not change a formula without re-running
tests/test_smoke.py, which checks that predictions still reproduce phase4_eval.

Input  : a per-item history with columns [date, y] (y = units issued that day)
         plus the static attributes [family, material, stock_type, reorder_qty].
Output : a single-row DataFrame holding the 22 features in MODEL order, with the
         categorical columns cast to the exact training levels.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Feature order expected by every booster (verified against booster.feature_name()).
FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_30", "lag_60", "lag_90",
    "rmean_7", "rstd_7", "rmean_30", "rstd_30", "rmean_90", "rstd_90",
    "dow", "month", "is_month_start", "is_month_end", "days_since_issue",
    "family", "material", "stock_type", "reorder_qty", "reorder_ratio",
]
CATEGORICAL = ["family", "material", "stock_type", "dow", "month"]
STATIC_ATTRS = ["family", "material", "stock_type", "reorder_qty"]

LAGS = [1, 7, 14, 30, 60, 90]
ROLL_WINDOWS = [7, 30, 90]


def to_weekly(history: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily (date, y) history to weekly (week ending Sunday) by summing.

    Verified: weekly training `y` == daily `y` resampled with rule 'W-SUN'.
    Static attributes are carried through unchanged.
    """
    h = history.sort_values("date").copy()
    h["date"] = pd.to_datetime(h["date"])
    weekly_y = h.set_index("date")["y"].resample("W-SUN").sum()
    out = weekly_y.reset_index()
    for a in STATIC_ATTRS:
        if a in h.columns:
            out[a] = h[a].iloc[-1]
    return out


def _days_since_issue(y: pd.Series) -> np.ndarray:
    """Periods since the last non-zero demand. Resets to 0 on every issue.

    On a daily series this is 'days since last issue'; on a weekly series it is
    'weeks since last issue' (verified against both parquet files).
    """
    yv = y.to_numpy()
    out = np.zeros(len(yv), dtype=int)
    for i in range(len(yv)):
        if yv[i] > 0:
            out[i] = 0
        else:
            out[i] = out[i - 1] + 1 if i > 0 else 0
    return out


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Compute the full feature table for one item's series (no category casting).

    `history` must be time-sorted and at the desired grain already (use
    to_weekly() first for weekly models). Returns the history with feature
    columns appended; the last row is the live forecast origin.
    """
    h = history.sort_values("date").reset_index(drop=True).copy()
    h["date"] = pd.to_datetime(h["date"])
    y = h["y"]

    for k in LAGS:
        h[f"lag_{k}"] = y.shift(k)
    for w in ROLL_WINDOWS:
        shifted = y.shift(1)                       # shift(1) prevents target leakage
        h[f"rmean_{w}"] = shifted.rolling(w).mean()
        h[f"rstd_{w}"] = shifted.rolling(w).std()  # ddof=1 (pandas default)

    h["dow"] = h["date"].dt.dayofweek
    h["month"] = h["date"].dt.month
    h["is_month_start"] = h["date"].dt.is_month_start.astype(int)
    h["is_month_end"] = h["date"].dt.is_month_end.astype(int)
    h["days_since_issue"] = _days_since_issue(y)

    if "reorder_qty" in h.columns:
        rq = h["reorder_qty"].replace(0, np.nan)
        h["reorder_ratio"] = h["rmean_30"] / rq    # verified: rmean_30 / reorder_qty
    else:
        h["reorder_ratio"] = np.nan
    return h


def _apply_levels(row: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """Cast categoricals to the EXACT training levels.

    Correctness-critical: LightGBM stores categorical splits as integer codes, so
    the category ordering at inference must match training (e.g. weekly dow==[6]).
    """
    row = row.copy()
    for c in CATEGORICAL:
        cats = levels[c]
        val = row[c].iloc[0]
        # ints arrive as numpy types; normalise so they match the stored levels
        if c in ("dow", "month"):
            val = int(val)
        else:
            val = str(val)
        row[c] = pd.Categorical([val], categories=cats)
    return row


def make_origin_row(history: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """Build the single feature row for the most recent date (the forecast origin)."""
    feat = build_feature_frame(history)
    origin = feat.iloc[[-1]][FEATURES].copy()
    return _apply_levels(origin, levels)
