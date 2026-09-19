"""
Bridge: takes one day's raw feature counts for one company, runs them
through that company's learned OrganizationBehaviorModel, and produces
a row in EXACTLY the shape cyberstate_engine.StateScorer already expects
(raw counts + "_z" suffixed z-scores in one dict/Series).

This is the ONLY new integration point. StateScorer itself, the hard
penalty rules, snapshot storage, Prediction Center, and Alfred are all
completely unchanged - they don't know or care that the z-scores now
come from a persisted learning model instead of a live rolling-window
SQL query.
"""

from __future__ import annotations
from datetime import date
from org_behavior_model import OrganizationBehaviorModel
from cyberstate_engine import FEATURE_COLUMNS


def compute_scored_row(model: OrganizationBehaviorModel, daily_counts: dict, obs_date: date) -> dict:
    """
    daily_counts: {"failed_logins": 3, "offhours_logins": 1, ...} - one
    day's raw counts for one company, same shape FeatureExtractor
    already produces.

    Returns a dict with both raw counts AND "<feature>_z" scores,
    exactly matching what BaselineComparator.compute_zscores() used to
    produce - so StateScorer.score() needs zero changes.

    IMPORTANT: this both scores AND updates the model in one call. The
    model's own observe() method guarantees today's count is scored
    against YESTERDAY's baseline, then folded into today's update - so
    call this exactly once per company per day.
    """
    row = dict(daily_counts)  # raw counts pass through unchanged
    for feature in FEATURE_COLUMNS:
        count = daily_counts.get(feature, 0)
        result = model.observe(feature, count, obs_date)
        row[f"{feature}_z"] = result["z"]
    return row
