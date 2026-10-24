"""Authorized local IEEE-CIS ingestion and explicitly synthetic fixtures."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_transactions(n_rows=3000, seed=42):
    rng = np.random.default_rng(seed)
    card = rng.integers(1000, 1100, n_rows)
    amount = rng.lognormal(3.5, 1, n_rows).round(2)
    device = rng.choice(["desktop", "mobile", "unknown"], n_rows, p=[0.5, 0.45, 0.05])
    probability = 1 / (1 + np.exp(-(-4 + 0.006 * amount + 0.9 * (device == "unknown") + 1.1 * (card % 13 == 0))))
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n_rows + 1),
            "TransactionDT": np.arange(n_rows) * 180,
            "TransactionAmt": amount,
            "card1": card,
            "addr1": rng.integers(1, 15, n_rows),
            "DeviceInfo": device,
            "ProductCD": rng.choice(["W", "C", "R"], n_rows),
            "P_emaildomain": rng.choice(["mail.example", "other.example", None], n_rows),
            "isFraud": rng.binomial(1, probability),
        }
    )


def load_transactions(data_dir=None, n_rows=3000):
    if data_dir is None:
        return synthetic_transactions(n_rows), "SYNTHETIC DEMONSTRATION"
    root = Path(data_dir)
    tx = pd.read_csv(root / "train_transaction.csv")
    identity = pd.read_csv(root / "train_identity.csv")
    if tx.TransactionID.duplicated().any() or identity.TransactionID.duplicated().any():
        raise ValueError("TransactionID must be unique in both tables")
    tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    validate_transactions(tx)
    return tx.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True), "IEEE-CIS local authorized data"


def validate_transactions(df):
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    if not required.issubset(df):
        raise ValueError(f"Missing columns: {required - set(df)}")
    if df.TransactionID.duplicated().any() or df[list(required)].isna().any().any():
        raise ValueError("Duplicate IDs or missing required values")
    if not df.isFraud.isin([0, 1]).all() or (df.TransactionAmt < 0).any():
        raise ValueError("Invalid label or amount")


def fingerprint(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def chronological_splits(df):
    times = np.sort(df.TransactionDT.unique())
    if len(times) < 20:
        raise ValueError("At least 20 distinct timestamps required")
    cuts = [times[int(len(times) * q)] for q in [0.55, 0.70, 0.85]]
    t = df.TransactionDT
    masks = [t < cuts[0], (t >= cuts[0]) & (t < cuts[1]), (t >= cuts[1]) & (t < cuts[2]), t >= cuts[2]]
    return {name: np.flatnonzero(mask) for name, mask in zip(["train", "validation", "calibration", "test"], masks)}


def reserve_governance(df, indices):
    """Separate calibration and governance without dividing tied timestamps."""
    times = np.sort(df.iloc[indices].TransactionDT.unique())
    if len(times) < 2:
        raise ValueError("Calibration/governance need distinct relative times")
    boundary = times[len(times) // 2]
    values = df.iloc[indices].TransactionDT.to_numpy()
    return indices[values < boundary], indices[values >= boundary]
