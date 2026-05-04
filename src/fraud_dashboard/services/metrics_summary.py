from __future__ import annotations

from typing import Any

from fraud_dashboard.platform.store import OpsStore


def build_metrics_summary(store: OpsStore) -> dict[str, Any]:
    """Return dashboard-friendly operational metrics from persisted state."""

    return store.metrics_summary()
