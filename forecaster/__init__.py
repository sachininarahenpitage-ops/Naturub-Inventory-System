"""
yarn_inventory.forecaster
=========================
Demand forecasting + inventory decisions for the dense Tier-A yarn items.

Public API:
    from forecaster import predict, forecast_demand, known_items

    bundle = predict("YPOL0004", history_df, horizon_days=30,
                     order_qty=1000, delivery_date="2026-07-25")
"""
from .predict import predict
from .forecast import forecast_demand, DemandForecast
from .artifacts import known_items, item_attributes

__all__ = ["predict", "forecast_demand", "DemandForecast",
           "known_items", "item_attributes"]
__version__ = "0.1.0"
