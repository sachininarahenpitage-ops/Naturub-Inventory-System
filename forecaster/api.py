"""
api.py
======
FastAPI backend. This is the "connectivity" seam: today it runs locally (offline,
on the company's own machine); later the same surface can be deployed remotely or
have its current-stock source pointed at the company DB (see stock_source.py).

Endpoints
---------
  GET  /                serve the dashboard (frontend/index.html)
  GET  /api/data        precomputed dashboard payload (items + monthly history)
  GET  /api/items       list of known item codes
  POST /api/predict     run the live engine for one item

Run:
  pip install -r requirements.txt
  uvicorn forecaster.api:app --reload
  # open http://127.0.0.1:8000
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import artifacts
from .predict import predict

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "index.html"
UI_DATA = ROOT / "artifacts" / "ui_data.json"

app = FastAPI(title="REEL — yarn inventory & reorder planner", version="0.1.0")


def _clean(obj):
    """Replace NaN/Inf with None so the response is always valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


class HistoryPoint(BaseModel):
    date: str
    y: float


class PredictRequest(BaseModel):
    item: str
    history: list[HistoryPoint]
    horizon_days: int = 30
    grain: str = "daily"
    order_qty: float | None = None
    delivery_date: str | None = None
    current_stock: float | None = None
    today: str | None = None


@app.get("/")
def index():
    if not FRONTEND.exists():
        raise HTTPException(500, "frontend/index.html not found")
    return FileResponse(FRONTEND)


@app.get("/api/data")
def dashboard_data():
    if UI_DATA.exists():
        return JSONResponse(json.loads(UI_DATA.read_text()))
    raise HTTPException(404, "ui_data.json not bundled")


@app.get("/api/items")
def items():
    return {"items": artifacts.known_items()}


@app.post("/api/predict")
def run_predict(req: PredictRequest):
    if req.item not in artifacts.known_items():
        raise HTTPException(404, f"unknown item '{req.item}'")
    hist = pd.DataFrame([h.model_dump() for h in req.history])
    if hist.empty:
        raise HTTPException(400, "history is empty")
    try:
        bundle = predict(
            req.item, hist, horizon_days=req.horizon_days, grain=req.grain,
            order_qty=req.order_qty, delivery_date=req.delivery_date,
            current_stock=req.current_stock, today=req.today,
        )
        return _clean(bundle)
    except (ValueError, KeyError) as e:   # bad input -> clean 400, not a 500
        raise HTTPException(400, str(e))
