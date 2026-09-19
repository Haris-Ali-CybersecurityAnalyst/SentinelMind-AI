"""
Organization Behavior Model - a persisted, incrementally-learning
statistical model per (organization, feature, day-type).

This is deliberately NOT a neural network or trained ML model. It's an
exponentially-weighted online mean/variance estimator - a well-known,
fully explainable statistical technique. Per the master context doc's
own principle: "Do not force a neural network merely because 'AI'
sounds more advanced" - this is the right-sized tool for the data
volume any single organization will realistically produce.

What makes this a genuine "learning model" rather than a live query:
- It is PERSISTED per organization (not recomputed from a raw log scan
  each time)
- It UPDATES incrementally with each new observation (true online
  learning, in the literal statistical sense)
- It ADAPTS to legitimate drift over time via exponential weighting,
  rather than being stuck on a fixed historical window forever
- It separates weekday/weekend behavior, so "normal" is contextual,
  not a single blended number
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date


ALPHA_DEFAULT = 2 / (21 + 1)   # ~21-observation effective memory, matches
                                 # the window already used in CyberState v1
MIN_OBSERVATIONS_FOR_CONFIDENCE = 10
VARIANCE_FLOOR = 0.25           # avoid divide-by-near-zero on quiet features


@dataclass
class FeatureBaselineState:
    """
    Persisted per (company_id, feature, day_type). This is the actual
    'model' - small, cheap to store, cheap to update, fully inspectable.
    """
    n: int = 0
    mean: float = 0.0
    var: float = 0.0
    last_updated: date | None = None

    def confidence(self) -> str:
        if self.n < MIN_OBSERVATIONS_FOR_CONFIDENCE:
            return "warming_up"
        elif self.n < MIN_OBSERVATIONS_FOR_CONFIDENCE * 3:
            return "moderate"
        else:
            return "established"


def update_baseline(state: FeatureBaselineState, x: float, obs_date: date,
                     alpha: float = ALPHA_DEFAULT) -> FeatureBaselineState:
    """
    Online exponentially-weighted mean/variance update. This is the core
    'learning' step - called once per organization per feature per day,
    with that day's observed count. No raw history needs to be re-read;
    the model IS the state.
    """
    if state.n == 0:
        return FeatureBaselineState(n=1, mean=x, var=0.0, last_updated=obs_date)

    delta = x - state.mean
    new_mean = state.mean + alpha * delta
    new_var = (1 - alpha) * (state.var + alpha * delta * delta)
    return FeatureBaselineState(
        n=state.n + 1, mean=new_mean, var=new_var, last_updated=obs_date,
    )


def score_deviation(state: FeatureBaselineState, x: float) -> dict:
    """
    Given today's observation and the organization's learned baseline,
    return a z-score and confidence label - the same explainable
    contract CyberState already uses downstream.
    """
    if state.n == 0:
        return {"z": 0.0, "confidence": "no_data", "note": "no prior observations"}

    std = math.sqrt(max(state.var, VARIANCE_FLOOR))
    z = (x - state.mean) / std
    return {
        "z": round(z, 2),
        "learned_mean": round(state.mean, 2),
        "confidence": state.confidence(),
        "observations": state.n,
    }


class OrganizationBehaviorModel:
    """
    One of these per organization. Holds a FeatureBaselineState for
    every (feature, day_type) pair - a genuinely learned, organization-
    specific model, small enough to persist as a handful of rows.
    """

    def __init__(self, company_id: int):
        self.company_id = company_id
        self.states: dict[tuple[str, str], FeatureBaselineState] = {}

    def _day_type(self, obs_date: date) -> str:
        return "weekend" if obs_date.weekday() >= 5 else "weekday"

    def observe(self, feature: str, x: float, obs_date: date, alpha: float = ALPHA_DEFAULT) -> dict:
        day_type = self._day_type(obs_date)
        key = (feature, day_type)
        prior_state = self.states.get(key, FeatureBaselineState())

        # Score against the PRIOR state before updating - never let
        # today's observation contaminate today's own anomaly score.
        result = score_deviation(prior_state, x)
        result["day_type"] = day_type

        self.states[key] = update_baseline(prior_state, x, obs_date, alpha)
        return result

    def snapshot(self) -> dict:
        """Human-readable dump of everything this org's model has learned - for explainability/debugging."""
        return {
            f"{feat}_{day_type}": {
                "mean": round(s.mean, 2),
                "std": round(math.sqrt(max(s.var, VARIANCE_FLOOR)), 2),
                "observations": s.n,
                "confidence": s.confidence(),
            }
            for (feat, day_type), s in self.states.items()
        }
