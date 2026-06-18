"""
cli.py
======
Command-line interface. Reads a history file (CSV or Parquet) with at least
columns [item, date, y], runs the full inventory decision for one item, and
prints the bundle as JSON (or a short human summary).

Examples
--------
  python -m forecaster.cli --list-items

  python -m forecaster.cli \
      --history examples/sample_history.csv --item YPOL0004 \
      --horizon 30 --order-qty 40000 --delivery-date 2026-07-25 \
      --current-stock 20000 --today 2026-04-09

  python -m forecaster.cli --history examples/sample_history.csv \
      --item YPOL0004 --summary
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import artifacts
from .predict import predict


def _load_history(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"history file not found: {path}")
    df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p)
    missing = {"item", "date", "y"} - set(df.columns)
    if missing:
        sys.exit(f"history file missing columns: {sorted(missing)}")
    return df


def _summary(b: dict) -> str:
    s = b["status"]
    fc = b["forecast"]
    rp = b["reorder_point"]
    lines = [
        f"Item {b['item']}  ({b['family']} / {b['material']})   as of {b['as_of']}",
        f"  Current stock : {b['current_stock']:>12,.1f}   [{b['current_stock_source']}]",
        f"  Reorder point : {rp['reorder_point']:>12,.1f}   "
        f"(lead {rp['lead_time_days']}d, safety {rp['safety_stock']:,.0f})",
        f"  STATUS        : {s['status']}  (margin {s['margin_to_reorder']:+,.1f})",
        f"  Demand next {fc['horizon_days']}d : {fc['forecast']:,.1f}  via {fc['method']}",
        f"  Stockout est. : {b['projection']['stockout_date']} "
        f"({b['projection']['days_to_stockout']} days)",
    ]
    if "order" in b:
        o = b["order"]
        lines.append(f"  Order {o['order_qty']:,.0f}: "
                     f"{'OK' if o['can_fulfill_now'] else f'short {o['shortfall']:,.0f}'}")
    if "replenishment" in b and b["replenishment"].get("must_order_by"):
        r = b["replenishment"]
        lines.append(f"  Replenish     : order by {r['must_order_by']} "
                     f"-> arrive by {r['must_arrive_by']}"
                     + ("  [ALREADY LATE]" if r["already_late"] else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forecaster", description="Yarn inventory decision tool")
    ap.add_argument("--list-items", action="store_true", help="print known item codes and exit")
    ap.add_argument("--history", help="CSV/Parquet with columns item,date,y")
    ap.add_argument("--item", help="item code to evaluate")
    ap.add_argument("--horizon", type=int, default=30, choices=[30, 90])
    ap.add_argument("--grain", default="daily", choices=["daily", "weekly"])
    ap.add_argument("--order-qty", type=float, default=None)
    ap.add_argument("--delivery-date", default=None)
    ap.add_argument("--current-stock", type=float, default=None)
    ap.add_argument("--today", default=None, help="as-of date (YYYY-MM-DD)")
    ap.add_argument("--summary", action="store_true", help="human summary instead of JSON")
    args = ap.parse_args(argv)

    if args.list_items:
        print("\n".join(artifacts.known_items()))
        return
    if not args.history or not args.item:
        ap.error("--history and --item are required (unless --list-items)")

    df = _load_history(args.history)
    hist = df[df["item"] == args.item][["date", "y"]]
    if hist.empty:
        sys.exit(f"no history rows for item '{args.item}' in {args.history}")

    bundle = predict(args.item, hist, horizon_days=args.horizon, grain=args.grain,
                     order_qty=args.order_qty, delivery_date=args.delivery_date,
                     current_stock=args.current_stock, today=args.today)
    print(_summary(bundle) if args.summary else json.dumps(bundle, indent=2, default=str))


if __name__ == "__main__":
    main()
