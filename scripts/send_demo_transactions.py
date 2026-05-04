import argparse
import os
from pathlib import Path

import httpx
import pandas as pd

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
LABEL_COL = "Class"


def find_csv(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return p

    # Try common locations
    candidates = [
        Path("data/demo_creditcard.csv"),
        Path("creditcard.csv"),
        Path("data/creditcard.csv"),
        Path("datasets/creditcard.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    hits = list(Path(".").rglob("creditcard*.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError("Could not find a creditcard CSV (try --csv path/to/creditcard.csv)")


def build_record(row: pd.Series) -> dict:
    return {k: float(row[k]) for k in FEATURES}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay a few labeled sample transactions against the FastAPI /v1/predictions endpoint."
    )
    ap.add_argument("--csv", default=None, help="Path to creditcard.csv (or a compatible CSV)")
    ap.add_argument("--model", default="rf", choices=["rf", "xgb"])
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=3, help="How many fraud samples to send")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--api-key",
        default=os.getenv("FRAUD_API_KEY", ""),
        help="Optional API key sent as X-API-Key for protected local runs.",
    )
    ap.add_argument(
        "--bearer-token",
        default=os.getenv("FRAUD_BEARER_TOKEN", ""),
        help="Optional bearer token for protected local runs.",
    )
    args = ap.parse_args()

    csv_path = find_csv(args.csv)
    df = pd.read_csv(csv_path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV must include '{LABEL_COL}' column to select fraud samples.")

    fraud_df = df[df[LABEL_COL] == 1]
    if fraud_df.empty:
        raise ValueError("No fraud samples found (Class=1).")

    fraud = fraud_df.sample(n=min(args.n, len(fraud_df)), random_state=args.seed)
    legit = df[df[LABEL_COL] == 0].sample(n=1, random_state=args.seed)

    url = f"{args.host.rstrip('/')}/v1/predictions"

    print(f"CSV: {csv_path}")
    print(f"POST: {url}  model={args.model}\n")

    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    if args.bearer_token:
        token = args.bearer_token.removeprefix("Bearer ").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=30.0, headers=headers) as c:
        rec_legit = build_record(legit.iloc[0])
        r = c.post(url, json={"record": rec_legit, "model": args.model})
        r.raise_for_status()
        print("LEGIT (Class=0):", r.status_code, r.json())

        print("\nFRAUD samples (Class=1):")
        for i in range(len(fraud)):
            rec = build_record(fraud.iloc[i])
            r = c.post(url, json={"record": rec, "model": args.model})
            r.raise_for_status()
            print(f"  #{i + 1}:", r.status_code, r.json())


if __name__ == "__main__":
    main()
