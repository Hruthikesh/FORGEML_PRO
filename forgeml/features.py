"""Current fields and strictly earlier unlabeled observations. No target encoding."""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "ProductCD",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "DeviceType",
]
GROUPS = {"tabular", "temporal", "graph", "all"}


def canonical(value):
    if pd.isna(value):
        return "__MISSING__"
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def engineer(df, groups="all", history=None):
    if groups not in GROUPS:
        raise ValueError("Unknown feature group")
    original = df.copy().reset_index(drop=True)
    original["_request_order"] = np.arange(len(original))
    if history is not None and len(history):
        past = history.copy()
        past["_request_order"] = -1
        past = past[past.TransactionDT < original.TransactionDT.max()]
        if "TransactionID" in past and "TransactionID" in original:
            past = past[~past.TransactionID.isin(original.TransactionID)]
        frame = pd.concat([past, original], ignore_index=True)
    else:
        frame = original.copy()
    frame = frame.sort_values(["TransactionDT"], kind="stable")
    records = []
    counts = defaultdict(int)
    totals = defaultdict(float)
    last = {}
    windows = defaultdict(deque)
    parent = {}
    sizes = {}
    degree = defaultdict(int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def entities(row):
        result = []
        for key in ["card1", "addr1", "DeviceInfo"]:
            value = row.get(key)
            if pd.notna(value) and str(value).lower() not in ["unknown", "nan", "none", ""]:
                result.append(f"{key}:{canonical(value)}")
        return result

    for timestamp, batch in frame.groupby("TransactionDT", sort=True):
        pending = []
        for _, row in batch.iterrows():
            amount = float(row.get("TransactionAmt", 0))
            rec = {
                "_request_order": row["_request_order"],
                "amount": amount,
                "log_amount": np.log1p(max(0, amount)),
                "relative_hour": int(timestamp // 3600) % 24,
                "relative_day": int(timestamp // 86400) % 7,
            }
            for column in RAW_COLUMNS[2:]:
                value = row.get(column)
                rec[column] = canonical(value)
            rec["missing_fields"] = sum(pd.isna(row.get(c)) for c in RAW_COLUMNS)
            ent = entities(row)
            key = next((e for e in ent if e.startswith("card1:")), None)
            if groups in ["temporal", "all"]:
                if key:
                    while windows[key] and windows[key][0] <= timestamp - 86400:
                        windows[key].popleft()
                rec.update(
                    {
                        "prior_card_count": counts[key] if key else 0,
                        "prior_card_mean": totals[key] / counts[key] if key and counts[key] else 0,
                        "seconds_since_previous": float(timestamp - last[key]) if key in last else -1,
                        "card_24h_count": len(windows[key]) if key else 0,
                    }
                )
            if groups in ["graph", "all"]:
                roots = {find(e) for e in ent if e in parent}
                rec.update(
                    {
                        "prior_entity_degree": sum(degree[e] for e in ent),
                        "prior_component_size": max([sizes[r] for r in roots], default=0),
                        "prior_components_linked": len(roots),
                        "observed_entity_count": len(ent),
                    }
                )
            records.append(rec)
            pending.append((key, amount, ent))
        for key, amount, ent in pending:
            if key:
                counts[key] += 1
                totals[key] += amount
                last[key] = timestamp
                windows[key].append(timestamp)
            for e in ent:
                if e not in parent:
                    parent[e] = e
                    sizes[e] = 1
                degree[e] += 1
            if ent:
                root = find(ent[0])
                for e in ent[1:]:
                    other = find(e)
                    if root != other:
                        parent[other] = root
                        sizes[root] += sizes[other]
    result = pd.DataFrame(records)
    result = result[result._request_order >= 0].sort_values("_request_order").drop(columns="_request_order")
    result.index = original.index
    return result
