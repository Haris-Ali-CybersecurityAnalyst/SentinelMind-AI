"""
Persistence layer for OrganizationBehaviorModel.

The model itself (org_behavior_model.py) is pure Python with no DB
dependency, by design - easy to test in isolation. This module is the
thin adapter that loads a company's learned state from the database at
the start of a run and saves it back at the end, so the model survives
restarts and doesn't need to replay history.

Proposed table name: behavior_baselines - this matches a table name the
master context doc already anticipated in its future-models list
(Section 25), so this aligns with planned architecture rather than
introducing a new naming convention.

Schema:
    behavior_baselines(
        id            serial primary key,
        company_id    integer not null references companies(id),
        feature       varchar not null,
        day_type      varchar not null,      -- 'weekday' or 'weekend'
        n             integer not null default 0,
        mean          double precision not null default 0,
        var           double precision not null default 0,
        last_updated  date,
        updated_at    timestamp with time zone default now(),
        unique (company_id, feature, day_type)
    )
"""

from __future__ import annotations
from datetime import date
from org_behavior_model import OrganizationBehaviorModel, FeatureBaselineState


def load_org_model(engine, company_id: int) -> OrganizationBehaviorModel:
    """Reconstructs a company's learned model from persisted rows."""
    import pandas as pd
    query = """
        select feature, day_type, n, mean, var, last_updated
        from behavior_baselines
        where company_id = %(company_id)s
    """
    df = pd.read_sql(query, engine, params={"company_id": company_id})

    model = OrganizationBehaviorModel(company_id=company_id)
    for _, row in df.iterrows():
        key = (row["feature"], row["day_type"])
        model.states[key] = FeatureBaselineState(
            n=int(row["n"]),
            mean=float(row["mean"]),
            var=float(row["var"]),
            last_updated=row["last_updated"],
        )
    return model


def save_org_model(engine, model: OrganizationBehaviorModel) -> None:
    """
    Upserts every (feature, day_type) state for this company. Only
    called once per batch run per company - cheap, small number of rows
    (features x 2 day-types, currently at most ~16 rows per company).
    """
    from sqlalchemy import text
    upsert_sql = text("""
        insert into behavior_baselines (company_id, feature, day_type, n, mean, var, last_updated, updated_at)
        values (:company_id, :feature, :day_type, :n, :mean, :var, :last_updated, now())
        on conflict (company_id, feature, day_type)
        do update set n = :n, mean = :mean, var = :var,
                      last_updated = :last_updated, updated_at = now()
    """)
    with engine.begin() as conn:
        for (feature, day_type), state in model.states.items():
            conn.execute(upsert_sql, {
                "company_id": model.company_id,
                "feature": feature,
                "day_type": day_type,
                "n": state.n,
                "mean": state.mean,
                "var": state.var,
                "last_updated": state.last_updated,
            })
