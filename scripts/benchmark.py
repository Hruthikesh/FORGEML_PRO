"""Benchmark the actual HTTP API under documented bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np


def benchmark(base_url: str, requests: int, concurrency: int):
    with httpx.Client(base_url=base_url, timeout=60) as client:
        client.get("/ready").raise_for_status()
        payload = client.get("/api/examples").json()
        examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
        sample = examples[0]
        if "transaction" in sample:
            sample = sample["transaction"]
        allowed = {
            "TransactionID",
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
            "DeviceInfo",
            "DeviceType",
            "ProductCD",
            "P_emaildomain",
            "R_emaildomain",
        }
        sample = {key: value for key, value in sample.items() if key in allowed}
        for _ in range(5):
            client.post("/predict", json=sample).raise_for_status()

        def issue(_):
            started = time.perf_counter()
            response = client.post("/predict", json=sample)
            response.raise_for_status()
            return (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            latency = list(pool.map(issue, range(requests)))
        duration = time.perf_counter() - started
        return {
            "endpoint": "/predict",
            "transport": "actual HTTP; loopback unless URL changed",
            "requests": requests,
            "concurrency": concurrency,
            "warmup_requests": 5,
            "errors": 0,
            "duration_seconds": duration,
            "throughput_rps": requests / duration,
            "p50_ms": float(np.quantile(latency, 0.5)),
            "p95_ms": float(np.quantile(latency, 0.95)),
            "mean_ms": statistics.mean(latency),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": sys.version.split()[0],
            },
            "latencies_ms": latency,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    if not 1 <= args.requests <= 100000 or not 1 <= args.concurrency <= 128:
        parser.error("requests must be 1..100000; concurrency 1..128")
    result = benchmark(args.url, args.requests, args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "latencies_ms"}, indent=2))


if __name__ == "__main__":
    main()
