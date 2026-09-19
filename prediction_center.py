"""
Prediction Center (BETA) - trend-based forecasting, NOT machine learning.

Reads the real, already-accumulating security_state_snapshots table and
projects each CyberState dimension forward using simple linear trend
extrapolation - the same disclosed, non-ML approach validated in the
original CyberState prototype (TrendForecaster).

This module is intentionally honest about its limits:
- With fewer than MIN_SNAPSHOTS_FOR_FORECAST days of history, it returns
  an explicit "insufficient history" result rather than a fake number.
- Every response is labeled as a statistical trend forecast, not an AI
  prediction, so nothing here ever overclaims what it is.
- "Confidence" is derived from how much real history exists, not from
  any trained model - more days of data = more confidence, in a
  transparent, explainable way.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

DIMENSIONS = ["overall_score", "identity_security", "endpoint_security", "behavioral_stability"]
MIN_SNAPSHOTS_FOR_FORECAST = 5
LOOKBACK_DAYS = 30
HORIZONS = (7, 30, 90)


def fetch_recent_snapshots(engine, company_id: int, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Reads real rows from security_state_snapshots for one company.
    Table columns (confirmed from the live schema Replit created):
    id, company_id, date, overall_score, identity_security,
    endpoint_security, behavioral_stability, top_factors, created_at
    """
    query = """
        select date, overall_score, identity_security,
               endpoint_security, behavioral_stability
        from security_state_snapshots
        where company_id = %(company_id)s
        order by date asc
    """
    df = pd.read_sql(query, engine, params={"company_id": company_id})
    if not df.empty:
        df = df.tail(lookback_days).reset_index(drop=True)
    return df


def _forecast_series(series: pd.Series, horizons=HORIZONS) -> dict:
    """Linear trend extrapolation - identical logic to the validated prototype."""
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series.values, 1)
    last_x = len(series) - 1

    forecasts = {
        h: round(float(np.clip(slope * (last_x + h) + intercept, 0, 100)), 1)
        for h in horizons
    }
    direction = "improving" if slope > 0.05 else "declining" if slope < -0.05 else "stable"
    return {
        "slope_per_day": round(float(slope), 3),
        "direction": direction,
        "forecasts": forecasts,
    }


def _confidence_label(n_days: int) -> dict:
    """
    Transparent, non-ML confidence signal based purely on how much real
    history exists. Thresholds are a starting hypothesis, easy to tune.
    """
    if n_days < 10:
        level, note = "low", f"Based on only {n_days} days of history - treat as directional, not precise."
    elif n_days < 21:
        level, note = "moderate", f"Based on {n_days} days of history."
    else:
        level, note = "higher", f"Based on {n_days} days of history - trend is more established."
    return {"level": level, "days_of_history": n_days, "note": note}


def generate_prediction(engine, company_id: int) -> dict:
    """Main entry point: real snapshot history -> forecast response."""
    df = fetch_recent_snapshots(engine, company_id)

    if len(df) < MIN_SNAPSHOTS_FOR_FORECAST:
        return {
            "status": "insufficient_history",
            "days_of_history": len(df),
            "minimum_required": MIN_SNAPSHOTS_FOR_FORECAST,
            "message": (
                f"Not enough CyberState history yet to forecast "
                f"({len(df)}/{MIN_SNAPSHOTS_FOR_FORECAST} days). "
                f"Forecasts will appear automatically once enough daily "
                f"snapshots have accumulated."
            ),
        }

    result = {
        "status": "ok",
        "method": "linear_trend_forecast",
        "disclaimer": "Statistical trend forecast, not an AI/ML prediction.",
        "confidence": _confidence_label(len(df)),
        "dimensions": {},
    }
    for dim in DIMENSIONS:
        result["dimensions"][dim] = _forecast_series(df[dim])

    return result
