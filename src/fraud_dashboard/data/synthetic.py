from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticCreditcardSpec:
    n_rows: int = 5000
    fraud_rate: float = 0.002
    random_state: int = 42
    include_label: bool = True


def generate_synthetic_creditcard(spec: SyntheticCreditcardSpec = SyntheticCreditcardSpec()) -> pd.DataFrame:
    """Generate a small, self-contained demo dataset similar to the classic credit-card fraud schema.

    Columns:
      - Time (float): seconds since the first transaction
      - V1..V28 (float): anonymized features (approx N(0, 1))
      - Amount (float): transaction amount
      - Class (int, optional): 0/1 fraud label with strong class imbalance
    """

    rng = np.random.default_rng(spec.random_state)

    n = int(spec.n_rows)
    if n <= 0:
        return pd.DataFrame()

    time = rng.uniform(0.0, 172800.0, size=n)  # up to ~2 days
    v = rng.normal(loc=0.0, scale=1.0, size=(n, 28))
    amount = rng.lognormal(mean=3.0, sigma=1.0, size=n)

    df = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount

    if spec.include_label:
        # Create a weak-but-realistic signal: fraud tends to correlate with some latent features and higher amounts.
        z = (
            0.8 * (df["V10"].to_numpy())
            - 0.6 * (df["V14"].to_numpy())
            + 0.25 * np.log1p(df["Amount"].to_numpy())
            + rng.normal(0.0, 0.35, size=n)
        )
        # Calibrate baseline rate to match the target fraud_rate.
        # We shift z so that mean(sigmoid(z)) ~= fraud_rate.
        target = float(np.clip(spec.fraud_rate, 1e-6, 0.2))
        shift = np.log(target / (1.0 - target))
        p = 1.0 / (1.0 + np.exp(-(z + shift)))
        y = (rng.uniform(0.0, 1.0, size=n) < p).astype(int)
        df["Class"] = y

    return df
