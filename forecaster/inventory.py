"""
inventory.py
============
The inventory decisions described in the requirements transcript:

  1. Reorder point (MODEL-BASED) = expected demand over the lead time
                                   + safety stock (z * demand spread over LT).
  2. RED / GREEN status          = is current stock below the reorder point?
  3. Order feasibility           = can current stock cover an incoming order?
  4. Replenishment deadline      = by when must the shortfall be ordered/arrive,
                                   given the delivery date and the material lead time?
  5. Forward projection          = using the demand forecast, when does stock fall
                                   below the reorder point / hit zero?

The demand forecast (forecast.py) provides the expected demand rate; lead times
come from config.yaml (per raw material).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import config
from .forecast import DemandForecast


def _daily_rate(fc: DemandForecast) -> float:
    return fc.forecast / fc.horizon_days if fc.horizon_days else 0.0


@dataclass
class ReorderPoint:
    reorder_point: float
    lead_time_days: int
    expected_demand_over_lead_time: float
    safety_stock: float
    service_level_z: float
    daily_demand_rate: float
    basis: str          # 'model' or baseline name behind the forecast

    def as_dict(self): return asdict(self)


def compute_reorder_point(fc: DemandForecast, material: str) -> ReorderPoint:
    """ROP = (rate * lead_time) + z * (daily_std * sqrt(lead_time))."""
    lt = config.lead_time_for(material)
    z = config.service_level_z()
    rate = _daily_rate(fc)
    daily_std = fc.sigma_horizon / np.sqrt(fc.horizon_days) if fc.horizon_days else 0.0
    expected_lt = rate * lt
    safety = z * daily_std * np.sqrt(lt)
    return ReorderPoint(
        reorder_point=round(expected_lt + safety, 2),
        lead_time_days=lt,
        expected_demand_over_lead_time=round(expected_lt, 2),
        safety_stock=round(safety, 2),
        service_level_z=z,
        daily_demand_rate=round(rate, 4),
        basis=fc.method,
    )


def stock_status(current_stock: float, reorder_point: float) -> dict:
    """RED = understocked (at/below reorder point), GREEN otherwise."""
    below = current_stock <= reorder_point
    return {
        "status": "RED" if below else "GREEN",
        "understocked": bool(below),
        "current_stock": round(float(current_stock), 2),
        "reorder_point": round(float(reorder_point), 2),
        "margin_to_reorder": round(float(current_stock - reorder_point), 2),
    }


def order_feasibility(current_stock: float, order_qty: float) -> dict:
    shortfall = max(0.0, order_qty - current_stock)
    return {
        "order_qty": round(float(order_qty), 2),
        "current_stock": round(float(current_stock), 2),
        "can_fulfill_now": bool(current_stock >= order_qty),
        "shortfall": round(float(shortfall), 2),
    }


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.to_datetime(d).date()


def replenishment_plan(current_stock: float, order_qty: float, delivery_date,
                       material: str, today=None) -> dict:
    """When must the shortfall be ordered so the order ships on time?

    must_arrive_by = delivery_date            (stock has to be in before shipping)
    must_order_by  = delivery_date - lead_time (so it arrives in time)
    """
    today = _as_date(today) if today is not None else date.today()
    delivery = _as_date(delivery_date)
    lt = config.lead_time_for(material)
    shortfall = max(0.0, order_qty - current_stock)

    if shortfall <= 0:
        return {
            "shortfall": 0.0, "lead_time_days": lt,
            "message": "Current stock already covers the order; no replenishment needed.",
            "must_arrive_by": delivery.isoformat(), "must_order_by": None,
            "days_until_order_deadline": None, "already_late": False,
        }

    must_arrive_by = delivery
    must_order_by = delivery - timedelta(days=lt)
    days_until = (must_order_by - today).days
    return {
        "shortfall": round(float(shortfall), 2),
        "lead_time_days": lt,
        "must_arrive_by": must_arrive_by.isoformat(),
        "must_order_by": must_order_by.isoformat(),
        "days_until_order_deadline": days_until,
        "already_late": days_until < 0,
        "message": (
            f"Order {shortfall:.0f} more units by {must_order_by.isoformat()} "
            f"(lead time {lt}d) so they arrive before {must_arrive_by.isoformat()}."
            + (" WARNING: order deadline is already in the past." if days_until < 0 else "")
        ),
    }


def project_depletion(current_stock: float, fc: DemandForecast, reorder_point: float,
                      today=None) -> dict:
    """Use the forecast demand rate to project when stock crosses key levels."""
    today = _as_date(today) if today is not None else date.today()
    rate = _daily_rate(fc)
    if rate <= 0:
        return {"daily_demand_rate": 0.0, "days_to_reorder_point": None,
                "days_to_stockout": None, "reorder_point_date": None,
                "stockout_date": None,
                "crosses_reorder_within_horizon": False}
    days_to_rop = max(0.0, (current_stock - reorder_point) / rate)
    days_to_zero = max(0.0, current_stock / rate)
    return {
        "daily_demand_rate": round(rate, 4),
        "days_to_reorder_point": round(days_to_rop, 1),
        "days_to_stockout": round(days_to_zero, 1),
        "reorder_point_date": (today + timedelta(days=int(days_to_rop))).isoformat(),
        "stockout_date": (today + timedelta(days=int(days_to_zero))).isoformat(),
        "crosses_reorder_within_horizon": days_to_rop <= fc.horizon_days,
    }


def historical_breaches(history: pd.DataFrame, reorder_point: float) -> list[str]:
    """Months (YYYY-MM) where end-of-month *cumulative-demand pace* implies stock
    would have sat below the reorder point. We don't have true on-hand history, so
    this flags months whose 30-day trailing demand exceeded the reorder point - a
    proxy for 'demand pressure' until real stock-balance history is available.
    """
    h = history.sort_values("date").copy()
    h["date"] = pd.to_datetime(h["date"])
    daily = h.set_index("date")["y"].asfreq("D").fillna(0.0)
    trailing30 = daily.rolling(30).sum()
    monthly = trailing30.resample("ME").max()
    flagged = monthly[monthly > reorder_point]
    return [d.strftime("%Y-%m") for d in flagged.index]
