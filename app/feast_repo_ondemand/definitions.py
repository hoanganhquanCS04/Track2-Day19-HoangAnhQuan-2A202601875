"""On-demand feature transformation: the request-time ratio (NB8).

Deck reference: "On-Demand Feature: Tinh Tai Thoi Diem Request".

The strongest fraud feature is `amount / avg_amount_7d`. It cannot be
materialised: `amount` does not exist until the transaction arrives. An
on-demand feature view joins a STORED feature with REQUEST data and applies one
formula -- the same formula for training and for serving, which is the whole
point of a feature store.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, RequestSource, ValueType
from feast.types import Float64, Int64

# GOTCHA (feast 0.65): `from feast import on_demand_feature_view` gives you the
# MODULE, not the decorator -- feast/__init__.py exports only the class
# `OnDemandFeatureView`. The docs still show the short import; it raises
# "TypeError: 'module' object is not callable" at apply time. Import the
# decorator from its module explicitly:
from feast.on_demand_feature_view import on_demand_feature_view

_HERE = Path(__file__).resolve().parent

# value_type is NOT optional here. pandas 2.x + pyarrow 25 write string
# columns as arrow `large_string`; with no declared type feast infers the join
# key as JSON and materialize dies with
#   ValueError: Invalid JSON string for JSON type
# ...raised deep inside type_map, naming neither the column nor the entity.
# app/feast_repo/feature_views.py declares STRING on both entities, which is
# exactly why that repo materializes and this one did not.
user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.STRING)

user_spend_source = FileSource(
    name="user_spend_source",
    path=str(_HERE / "data" / "user_spend.parquet"),
    timestamp_field="event_timestamp",
)

# Stored, materialisable: a 7-day rolling average computed in batch.
user_spend_stats = FeatureView(
    name="user_spend_stats",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="avg_amount_7d", dtype=Float64),
        Field(name="txn_count_7d", dtype=Int64),
    ],
    source=user_spend_source,
    online=True,
)

# Known only at request time.
txn_request = RequestSource(
    name="txn",
    schema=[Field(name="amount", dtype=Float64)],
)


@on_demand_feature_view(
    sources=[user_spend_stats, txn_request],
    schema=[
        Field(name="amount_vs_avg", dtype=Float64),
        Field(name="is_spike", dtype=Int64),
    ],
    mode="python",
)
def amount_vs_avg(inputs: dict) -> dict:
    """Ratio of this transaction to the user's own recent baseline.

    Absolute amount is a weak fraud signal (a large purchase is normal for a
    large spender). The ratio against THIS user's baseline is what carries
    signal -- and it can only be computed once `amount` is known.
    """
    ratios = [
        (a / m) if m else 0.0
        for a, m in zip(inputs["amount"], inputs["avg_amount_7d"])
    ]
    return {
        "amount_vs_avg": ratios,
        "is_spike": [int(r > 3.0) for r in ratios],
    }
