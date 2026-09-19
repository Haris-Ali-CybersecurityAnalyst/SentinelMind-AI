"""
CyberState Engine Prototype (v1 — statistical, not ML)
=======================================================
This is a working proof-of-concept for SentinelMind's CyberState engine.
It deliberately does NOT use trained machine learning. Every number here
is produced by transparent statistics (rolling baselines, z-scores, weighted
formulas, linear trend extrapolation) so that every output is explainable and
defensible in an investor demo.

Pipeline stages, matching the architecture we designed:

    Telemetry data
        -> FeatureExtractor      (raw events -> daily numeric features)
        -> BaselineComparator    (per-org rolling mean/std -> z-scores)
        -> StateScorer           (z-scores -> explainable 0-100 scores)
        -> StateHistory          (snapshots stored over time)
        -> TrendForecaster       (linear trend, NOT a trained model)
        -> narrate_state()       (template explanation - stand-in for Alfred)

Everything below is organization-specific: baselines are computed only
from that org's own history, which is the "one organization != another
organization" principle from the master context doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. SYNTHETIC TELEMETRY (stands in for real Windows Agent / log ingestion)
# ---------------------------------------------------------------------------

def generate_synthetic_telemetry(org_id: str = "acme-corp", days: int = 120,
                                  seed: int = 7) -> pd.DataFrame:
    """
    Simulates daily aggregated security telemetry for one organization.
    In production this comes from the existing alert/incident/log tables
    (4624/4625/4672/4688 etc.), aggregated per day per org.

    We inject a realistic "creeping incident" around day 95-104: an
    attacker slowly escalating privileges and moving laterally, which is
    exactly the kind of gradual state change CyberState should catch
    before a rule-based system fires a single high-confidence alert.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)

    df = pd.DataFrame({"date": dates, "org_id": org_id})

    # Baseline "normal" organizational behavior (mirrors real event IDs
    # your Windows agent already collects: 4625, 4624, 4672/4728/4732/4720,
    # 4740, 4104, 4688, 1102)
    df["failed_logins"] = rng.poisson(5, days)               # 4625
    df["offhours_logins"] = rng.poisson(1, days)              # 4624, off-hours
    df["privilege_events"] = rng.poisson(0.3, days)           # 4672/4728/4732/4720
    df["account_lockouts"] = rng.poisson(0.05, days)          # 4740
    df["powershell_events"] = rng.poisson(2, days)            # 4104
    df["new_processes"] = rng.poisson(20, days)                # 4688
    df["log_tampering_events"] = np.zeros(days, dtype=int)     # 1102, normally never fires
    df["new_devices"] = rng.poisson(0.4, days)

    # Weekly seasonality: quieter on weekends (real orgs behave this way)
    weekday = dates.weekday
    weekend_mask = weekday >= 5
    df.loc[weekend_mask, "failed_logins"] = rng.poisson(2, weekend_mask.sum())
    df.loc[weekend_mask, "new_processes"] = rng.poisson(6, weekend_mask.sum())

    # Injected creeping incident: days 95-104, escalating identity abuse
    # that culminates in the attacker clearing the audit log (1102) to
    # cover their tracks - the single highest-signal event in the set.
    incident_start, incident_end = 95, 104
    for i in range(incident_start, min(incident_end + 1, days)):
        severity = (i - incident_start + 1) / (incident_end - incident_start + 1)
        df.loc[i, "failed_logins"] += int(rng.poisson(6 * severity))
        df.loc[i, "offhours_logins"] += int(rng.poisson(3 * severity))
        df.loc[i, "privilege_events"] += int(rng.poisson(2 * severity))
        if severity > 0.6:
            df.loc[i, "powershell_events"] += int(rng.poisson(4 * severity))
        if i == incident_end - 1:  # attacker covers tracks near the end
            df.loc[i, "log_tampering_events"] = 1

    return df


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTOR
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "failed_logins", "offhours_logins", "privilege_events",
    "account_lockouts", "powershell_events", "new_processes",
    "log_tampering_events", "new_devices",
]

# Maps each feature to the real Windows event IDs in your log_entries
# table (event_id column), per the agent's confirmed event coverage.
EVENT_ID_MAP = {
    "failed_logins": ["4625"],
    "offhours_logins": ["4624"],          # filtered by hour, see aggregate below
    "privilege_events": ["4672", "4728", "4732", "4720"],
    "account_lockouts": ["4740"],
    "powershell_events": ["4104"],
    "new_processes": ["4688"],
    "log_tampering_events": ["1102"],      # audit log cleared - treated as critical, see StateScorer
}


class FeatureExtractor:
    """
    In production: queries the alerts/incidents/log_entries tables,
    grouped by org_id and day, and returns the same shape of DataFrame
    this synthetic generator produces. The rest of the pipeline is
    identical whether the data is real or synthetic.
    """

    @staticmethod
    def extract(telemetry: pd.DataFrame) -> pd.DataFrame:
        return telemetry[["date", "org_id", *FEATURE_COLUMNS]].copy()


# ---------------------------------------------------------------------------
# 3. BASELINE COMPARATOR (organization-specific, rolling statistics)
# ---------------------------------------------------------------------------

class BaselineComparator:
    """
    Computes a trailing rolling mean/std per feature, per organization,
    and converts each day's value into a z-score against that org's own
    recent history. No cross-tenant data is ever used.
    """

    def __init__(self, window: int = 21, min_periods: int = 7):
        self.window = window
        self.min_periods = min_periods

    def compute_zscores(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        for col in FEATURE_COLUMNS:
            roll_mean = (
                out[col].shift(1)
                .rolling(self.window, min_periods=self.min_periods)
                .mean()
            )
            roll_std = (
                out[col].shift(1)
                .rolling(self.window, min_periods=self.min_periods)
                .std()
                .clip(lower=0.5)  # floor so quiet features don't explode z-scores
            )
            out[f"{col}_z"] = (out[col] - roll_mean) / roll_std
        return out


# ---------------------------------------------------------------------------
# 4. STATE SCORER (explainable, weighted, 0-100 per dimension)
# ---------------------------------------------------------------------------

# Each state dimension is built from a weighted combination of feature
# z-scores. Weights are a starting hypothesis, not a trained parameter -
# they should be tuned with a security analyst's input over time.
DIMENSION_WEIGHTS = {
    "identity_security": {
        "failed_logins_z": 0.30,
        "offhours_logins_z": 0.20,
        "privilege_events_z": 0.30,
        "account_lockouts_z": 0.20,
    },
    "endpoint_security": {
        "new_processes_z": 0.40,
        "powershell_events_z": 0.35,
        "account_lockouts_z": 0.25,
    },
    "behavioral_stability": {
        "new_devices_z": 0.5,
        "offhours_logins_z": 0.5,
    },
}

OVERALL_WEIGHTS = {
    "identity_security": 0.45,
    "endpoint_security": 0.35,
    "behavioral_stability": 0.20,
}

PENALTY_SCALE = 9.0  # how many points a z=1.0 deviation costs

# Some events are not "statistically unusual" - they are simply always
# bad, regardless of the org's baseline (a baseline of ~0 occurrences
# means even one event produces a huge z-score anyway, but we make the
# rule explicit rather than relying on statistics alone). This is the
# "rules + behavioral ML" hybrid the master doc calls for: CyberState
# does not replace rule-based detection, it adds organizational context
# around it.
HARD_PENALTY_RULES = {
    # feature -> (dimensions penalized, points per event, reason)
    "log_tampering_events": (
        ["identity_security", "endpoint_security"], 40,
        "audit log cleared - possible evidence tampering",
    ),
}


class StateScorer:
    def score(self, zscore_row: pd.Series) -> dict:
        dimension_scores = {}
        contributions = []  # for explainability: which features hurt the score
        hard_penalties = {d: 0.0 for d in DIMENSION_WEIGHTS}

        for dim, weights in DIMENSION_WEIGHTS.items():
            penalty = 0.0
            for feat, w in weights.items():
                z = zscore_row.get(feat, 0.0)
                z = 0.0 if pd.isna(z) else z
                positive_z = max(z, 0.0)  # only "worse than normal" hurts score
                contrib = positive_z * w * PENALTY_SCALE
                penalty += contrib
                if contrib > 0.5:
                    contributions.append((dim, feat.replace("_z", ""), round(contrib, 1)))
            dimension_scores[dim] = penalty  # raw penalty, hard rules added below

        # Apply hard penalty rules (rule-based layer on top of the stats)
        for feat, (dims, points, reason) in HARD_PENALTY_RULES.items():
            count = zscore_row.get(feat, 0)
            count = 0 if pd.isna(count) else count
            if count > 0:
                for dim in dims:
                    hard_penalties[dim] += points * count
                    contributions.append((dim, feat, round(points * count, 1)))

        for dim in dimension_scores:
            total_penalty = dimension_scores[dim] + hard_penalties[dim]
            dimension_scores[dim] = round(float(np.clip(100 - total_penalty, 0, 100)), 1)

        overall = round(
            sum(dimension_scores[d] * w for d, w in OVERALL_WEIGHTS.items()), 1
        )
        contributions.sort(key=lambda x: -x[2])

        return {
            "overall_score": overall,
            **dimension_scores,
            "top_factors": contributions[:3],
        }


# ---------------------------------------------------------------------------
# 5. STATE HISTORY (the snapshot store)
# ---------------------------------------------------------------------------

@dataclass
class StateHistory:
    snapshots: list = field(default_factory=list)

    def add(self, date, org_id, score_dict):
        self.snapshots.append({"date": date, "org_id": org_id, **score_dict})

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshots)


def run_state_engine(telemetry: pd.DataFrame) -> pd.DataFrame:
    features = FeatureExtractor.extract(telemetry)
    with_z = BaselineComparator().compute_zscores(features)
    scorer = StateScorer()
    history = StateHistory()

    for _, row in with_z.iterrows():
        result = scorer.score(row)
        history.add(row["date"], row["org_id"], result)

    return history.as_dataframe()


# ---------------------------------------------------------------------------
# 6. TREND FORECASTER (linear extrapolation - explicitly NOT ML)
# ---------------------------------------------------------------------------

class TrendForecaster:
    """
    Projects a dimension score forward using simple linear regression on
    the most recent N snapshots. This is intentionally not a trained
    model: it is disclosed to investors/users as a trend-based forecast,
    not an AI prediction, until real incident-outcome data exists to
    train and validate something more sophisticated.
    """

    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def forecast(self, series: pd.Series, horizons=(7, 30, 90)) -> dict:
        recent = series.tail(self.lookback).reset_index(drop=True)
        if len(recent) < 5:
            return {"error": "insufficient history"}

        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent.values, 1)

        last_x = len(recent) - 1
        forecasts = {
            h: round(float(np.clip(slope * (last_x + h) + intercept, 0, 100)), 1)
            for h in horizons
        }
        return {
            "slope_per_day": round(float(slope), 3),
            "direction": "improving" if slope > 0.05 else "declining" if slope < -0.05 else "stable",
            "forecasts": forecasts,
        }


# ---------------------------------------------------------------------------
# 7. NARRATION (template stand-in for Alfred's LLM explanation layer)
# ---------------------------------------------------------------------------

def narrate_state(latest_row: pd.Series, forecast: dict) -> str:
    factors = latest_row.get("top_factors", [])
    if factors:
        dim, feat, contrib = factors[0]
        factor_txt = f"driven mainly by {feat.replace('_', ' ')} ({contrib} pt impact on {dim.replace('_', ' ')})"
    else:
        factor_txt = "with no significant contributing anomalies"

    direction = forecast.get("direction", "stable")
    f30 = forecast.get("forecasts", {}).get(30, "n/a")

    return (
        f"Current overall CyberState: {latest_row['overall_score']}/100, "
        f"{factor_txt}. Trend is {direction}; at the current trajectory, "
        f"the trend-based forecast (not an AI prediction) projects "
        f"{f30}/100 in 30 days."
    )