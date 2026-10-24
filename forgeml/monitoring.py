"""Windowed drift and delayed-label historical replay."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def psi(reference, current):
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(r, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return float(abs(np.mean(c) - np.mean(r)) > 1e-8)
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(r, bins=edges)[0] / max(1, len(r))
    b = np.histogram(c, bins=edges)[0] / max(1, len(c))
    a, b = np.maximum(a, 1e-6), np.maximum(b, 1e-6)
    return float(np.sum((b - a) * np.log(b / a)))


def drift_report(reference_features, current_features, reference_scores, current_scores):
    features = []
    for column in reference_features.select_dtypes(include="number").columns:
        if column in current_features:
            value = psi(reference_features[column], current_features[column])
            features.append(
                {
                    "feature": column,
                    "psi": value,
                    "warning": value > 0.2,
                    "reference_missing": float(reference_features[column].isna().mean()),
                    "current_missing": float(current_features[column].isna().mean()),
                }
            )
    return {
        "features": features,
        "prediction_psi": psi(reference_scores, current_scores),
        "warning": any(f["warning"] for f in features),
        "interpretation": "Covariate drift is a warning, not proof of predictive degradation.",
    }


def replay(bundle, raw_df, batch_size=100, label_delay=2, induced_shift=False):
    from .features import engineer
    from .training import predict_bundle

    if label_delay < 0 or batch_size < 1:
        raise ValueError("Invalid replay configuration")
    frame = raw_df.sort_values("TransactionDT").copy()
    windows = []
    pending = []
    replay_bundle = dict(bundle)
    replay_history = bundle.get("history", pd.DataFrame()).copy()
    for index, start in enumerate(range(0, len(frame), batch_size)):
        batch = frame.iloc[start : start + batch_size].copy()
        if induced_shift:
            batch["TransactionAmt"] *= 2
        begin = time.perf_counter()
        replay_bundle["history"] = replay_history
        probabilities = predict_bundle(replay_bundle, batch)
        elapsed = time.perf_counter() - begin
        features = engineer(batch, bundle.get("group", "all"), history=replay_history)
        report = drift_report(bundle["reference_features"], features, bundle["reference_scores"], probabilities)
        replay_history = pd.concat(
            [replay_history, batch.drop(columns=["isFraud"], errors="ignore")], ignore_index=True
        ).drop_duplicates("TransactionID", keep="last")
        pending.append((index, batch.isFraud.to_numpy() if "isFraud" in batch else None, probabilities))
        eligible = [item for item in pending if item[0] <= index - label_delay and item[1] is not None]
        quality = None
        if eligible:
            labels = np.concatenate([p[1] for p in eligible])
            scores = np.concatenate([p[2] for p in eligible])
            quality = {
                "labeled_count": len(labels),
                "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
                "brier": float(brier_score_loss(labels, scores)),
            }
        windows.append(
            {
                "window": index,
                "transactions": len(batch),
                "relative_start": int(batch.TransactionDT.min()),
                "relative_end": int(batch.TransactionDT.max()),
                "latency_ms": 1000 * elapsed / len(batch),
                "alert_rate": float(np.mean(probabilities >= bundle["threshold"])),
                "mean_score": float(np.mean(probabilities)),
                "drift": report,
                "delayed_quality": quality,
            }
        )
    return {
        "mode": bundle["mode"],
        "windows": windows,
        "label_delay_batches": label_delay,
        "induced_shift": induced_shift,
        "note": "Replay labels become visible only after the configured batch delay. Last delayed batches remain pending.",
    }


def evidently_report(reference, current, output_path):
    """Optional Evidently >=0.7 report, independently labelled from native PSI.

    API source: https://github.com/evidentlyai/evidently (Report/DataDriftPreset example).
    Native drift/replay still works when this optional integration is unavailable.
    """
    import importlib.util
    from pathlib import Path

    if importlib.util.find_spec("evidently") is None:
        return {"status": "unavailable", "reason": "Evidently package unavailable"}
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset(method="psi")])
        snapshot = report.run(current, reference)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))
        output_path.with_suffix(".json").write_text(snapshot.json(), encoding="utf-8")
        return {"status": "verified", "html": str(output_path), "json": str(output_path.with_suffix(".json"))}
    except Exception as exc:  # noqa: BLE001 -- optional dependency/backend failure must not interrupt native monitoring
        return {"status": "failed", "reason": str(exc)}
