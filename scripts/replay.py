"""Execute historical replay against the trained inference bundle."""

import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--label-delay", type=int, default=2, help="Label delay in replay batches")
    parser.add_argument("--induce-shift", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay.json"))
    args = parser.parse_args()
    response = httpx.post(
        args.url + "/api/replay",
        json={"batch_size": args.batch_size, "label_delay": args.label_delay, "induce_shift": args.induce_shift},
        timeout=180,
    )
    response.raise_for_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    print(f"Replay saved to {args.output}")


if __name__ == "__main__":
    main()
