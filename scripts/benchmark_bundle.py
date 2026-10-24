"""Measure direct Python bundle latency, excluding HTTP transport."""

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forgeml.training import predict_bundle

root = Path(__file__).resolve().parents[1]
bundle = joblib.load(root / "artifacts/bundle.joblib")
sample = pd.DataFrame(json.loads((root / "artifacts/examples.json").read_text())[:1])
predict_bundle(bundle, sample)
latencies = []
for _ in range(12):
    start = time.perf_counter()
    predict_bundle(bundle, sample)
    latencies.append((time.perf_counter() - start) * 1000)
result = {
    "mode": bundle["mode"],
    "model_version": bundle["version"],
    "transport": "DIRECT PYTHON BUNDLE; no HTTP, no concurrency, no network overhead",
    "requests": len(latencies),
    "concurrency": 1,
    "warmup_requests": 1,
    "history_rows": len(bundle["history"]),
    "p50_ms": float(np.quantile(latencies, 0.5)),
    "p95_ms": float(np.quantile(latencies, 0.95)),
    "throughput_rps": 1000 * len(latencies) / sum(latencies),
    "latencies_ms": latencies,
    "hardware": {"platform": platform.platform(), "processor": platform.processor()},
    "limitation": "Reconstructs past features per call; not optimized for production latency.",
}
(root / "artifacts/bundle-benchmark.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "latencies_ms"}, indent=2))
