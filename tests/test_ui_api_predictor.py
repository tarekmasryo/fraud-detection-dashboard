from __future__ import annotations

import pandas as pd

from fraud_dashboard.ui import predictors
from fraud_dashboard.ui.api_client import ApiConfig
from fraud_dashboard.ui.predictors import ApiPredictor


def test_api_predictor_chunks_large_batches_and_forwards_policy(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_fetch_metadata(_cfg):
        return {"limits": {"max_batch_records": 2}}

    def fake_predict_batch(*, cfg, model, records, threshold=None, policy=None):
        calls.append(
            {
                "cfg": cfg,
                "model": model,
                "records": records,
                "threshold": threshold,
                "policy": policy,
            }
        )
        return {"results": [{"proba_fraud": 0.25} for _ in records]}

    monkeypatch.setattr(predictors, "fetch_metadata", fake_fetch_metadata)
    monkeypatch.setattr(predictors, "predict_batch", fake_predict_batch)

    predictor = ApiPredictor(ApiConfig(base_url="http://api:8000"))
    df = pd.DataFrame({"Amount": [1, 2, 3, 4, 5]})

    probs, _elapsed = predictor.predict_proba_batch(df, model_key="rf", policy="balanced")

    assert probs.tolist() == [0.25] * 5
    assert [len(call["records"]) for call in calls] == [2, 2, 1]
    assert {call["policy"] for call in calls} == {"balanced"}
    assert {call["threshold"] for call in calls} == {None}
    assert {call["model"] for call in calls} == {"rf"}


def test_api_predictor_forwards_manual_threshold(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(
        predictors,
        "fetch_metadata",
        lambda _cfg: {"limits": {"max_batch_records": 1000}},
    )

    def fake_predict_batch(*, cfg, model, records, threshold=None, policy=None):
        calls.append({"threshold": threshold, "policy": policy, "records": records})
        return {"results": [{"proba_fraud": 0.7} for _ in records]}

    monkeypatch.setattr(predictors, "predict_batch", fake_predict_batch)

    predictor = ApiPredictor(ApiConfig(base_url="http://api:8000"))
    df = pd.DataFrame({"Amount": [10, 20]})

    probs, _elapsed = predictor.predict_proba_batch(df, model_key="xgb", threshold=0.42)

    assert probs.tolist() == [0.7, 0.7]
    assert calls == [
        {
            "threshold": 0.42,
            "policy": None,
            "records": [{"Amount": 10}, {"Amount": 20}],
        }
    ]
