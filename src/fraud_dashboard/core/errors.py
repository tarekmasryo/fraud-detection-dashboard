from __future__ import annotations


class FraudDashboardError(RuntimeError):
    """Base error for controlled runtime failures."""


class ArtifactContractError(FraudDashboardError):
    """Raised when artifact metadata, policy, or runtime compatibility is invalid."""


class PredictionRuntimeError(FraudDashboardError):
    """Raised when a model cannot score the supplied input."""
