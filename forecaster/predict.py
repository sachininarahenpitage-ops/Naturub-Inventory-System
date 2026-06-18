"""
predict.py
==========
The single high-level entry point. Given an item and its demand history (and,
optionally, an incoming order), it returns one self-describing decision bundle:

    forecast        - expected demand over the horizon (ML where it wins, else baseline)
    reorder_point   - model-based, with safety stock
    status          - RED / GREEN vs the reorder point
    projection      - when stock crosses the reorder point / runs out
    order           - feasibility + shortfall            (if order_qty given)
    replenishment   - order-by / arrive-by deadline      (if delivery_date given)

Stock-on-hand comes from stock_source (a stub today; DB API later) unless the
caller passes `current_stock` explicitly.
"""
from __future__ import annotations
from datetime import date

import pandas as pd

from . import artifacts, config, inventory, stock_source
from .forecast import forecast_demand


def predict(item: str,
            history: pd.DataFrame,
            horizon_days: int = 30,
            grain: str = "daily",
            order_qty: float | None = None,
            delivery_date=None,
            current_stock: float | None = None,
            today=None) -> dict:
    """Run the full inventory decision for one item.

    Parameters
    ----------
    item          : item code (must exist in item_master.csv)
    history       : DataFrame with at least [date, y]; static attrs looked up if absent
    horizon_days  : 30 or 90
    grain         : 'daily' or 'weekly'
    order_qty     : optional incoming order size -> feasibility + replenishment
    delivery_date : optional required delivery date -> replenishment deadline
    current_stock : optional explicit stock on hand (else taken from stock_source)
    today         : optional 'as-of' date (defaults to system date)
    """
    if item not in artifacts.known_items():
        raise KeyError(f"Unknown item '{item}'. {len(artifacts.known_items())} known items.")

    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    attrs = artifacts.item_attributes(item)
    material = attrs["material"]

    fc = forecast_demand(item, hist, horizon_days, grain)

    # Reorder point uses the model horizon that best matches the lead time
    # (long lead times -> 90-day demand basis). May differ from the display horizon.
    lt = config.lead_time_for(material)
    rop_horizon = config.reorder_basis_horizon(lt)
    rop_fc = fc if rop_horizon == horizon_days else forecast_demand(item, hist, rop_horizon, grain)
    rop = inventory.compute_reorder_point(rop_fc, material)
    stock_val, stock_src = stock_source.get_current_stock(item, override=current_stock)

    bundle = {
        "item": item,
        "material": material,
        "family": attrs["family"],
        "as_of": (inventory._as_date(today).isoformat() if today is not None
                  else date.today().isoformat()),
        "current_stock": round(float(stock_val), 2),
        "current_stock_source": stock_src,
        "forecast": fc.as_dict(),
        "reorder_point": rop.as_dict(),
        "status": inventory.stock_status(stock_val, rop.reorder_point),
        "projection": inventory.project_depletion(stock_val, fc, rop.reorder_point, today),
        "historical_demand_pressure_months": inventory.historical_breaches(hist, rop.reorder_point),
    }

    if order_qty is not None:
        bundle["order"] = inventory.order_feasibility(stock_val, order_qty)
        if delivery_date is not None:
            bundle["replenishment"] = inventory.replenishment_plan(
                stock_val, order_qty, delivery_date, material, today)

    return bundle
