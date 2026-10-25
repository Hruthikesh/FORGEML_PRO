"""Local demonstration registry with auditable, quality-gated transitions."""

import copy
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

_LOCK = RLock()
CRITERIA = {"min_ap_gain": 0.005, "max_brier_regression": 0.02, "max_latency_ms": 250}


def _save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return state


def _event(action, **kwargs):
    return {"timestamp": datetime.now(UTC).isoformat(), "action": action, **kwargs}


def load_registry(path):
    with _LOCK:
        return json.loads(Path(path).read_text(encoding="utf-8"))


def initialize_registry(path, version, metrics, reset=False):
    with _LOCK:
        if Path(path).exists() and not reset:
            return load_registry(path)
        if Path(path).exists() and reset:
            archive = Path(path).with_name(
                "registry-archive-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + ".json"
            )
            archive.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        return _save(
            path,
            {
                "champion": version,
                "previous": None,
                "challenger": None,
                "versions": {version: metrics},
                "history": [_event("registered", version=version)],
                "criteria": CRITERIA,
            },
        )


def evaluate_candidate(path, challenger_version, champion_metrics, challenger_metrics, eligible=True):
    with _LOCK:
        state = load_registry(path)
        if challenger_version == state["champion"]:
            raise ValueError("Challenger must differ from champion")
        state["challenger"] = challenger_version
        state["versions"][challenger_version] = challenger_metrics
        reasons = []
        for label, measured in [("champion", champion_metrics), ("challenger", challenger_metrics)]:
            for metric in ("average_precision", "brier", "latency_ms"):
                value = measured.get(metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    reasons.append(f"Missing or non-finite {label} {metric}")
                elif value < 0 or (metric != "latency_ms" and value > 1):
                    reasons.append(f"Invalid {label} {metric}")
        # Invalid measurements must fail closed, including missing latency.
        if reasons:
            state["versions"][challenger_version] = {
                key: value for key, value in challenger_metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(value)
            }
            state["history"].append(_event("rejected", version=challenger_version, reasons=reasons))
            return _save(path, state)
        if not eligible:
            reasons.append("Ineligible or unavailable delayed labels")
        if (
            challenger_metrics.get("average_precision", 0)
            < champion_metrics.get("average_precision", 0) + CRITERIA["min_ap_gain"]
        ):
            reasons.append("Insufficient average-precision improvement")
        if challenger_metrics.get("brier", 1) > champion_metrics.get("brier", 1) + CRITERIA["max_brier_regression"]:
            reasons.append("Calibration regression")
        if challenger_metrics.get("latency_ms", 0) > CRITERIA["max_latency_ms"]:
            reasons.append("Latency budget exceeded")
        if not reasons:
            state["previous"] = state["champion"]
            state["champion"] = challenger_version
        state["history"].append(
            _event(
                "rejected" if reasons else "promoted",
                version=challenger_version,
                reasons=reasons,
                champion_metrics=champion_metrics,
                challenger_metrics=challenger_metrics,
            )
        )
        return _save(path, state)


def rollback(path):
    with _LOCK:
        state = load_registry(path)
        if not state.get("previous"):
            raise ValueError("No previous champion available")
        old = state["champion"]
        state["champion"], state["previous"] = state["previous"], old
        state["history"].append(_event("rollback", version=state["champion"], from_version=old))
        return _save(path, state)


def governance_action(output_dir, action, version=None):
    """Use trained challenger and reserved forward window; never final holdout."""
    import shutil

    out = Path(output_dir)
    path = out / "registry.json"
    with _LOCK:
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        comparison = report["governance_comparison"]
        state = load_registry(path)
        previous_state = copy.deepcopy(state)
        if action in ["evaluate", "promote"]:
            candidate = comparison["challenger_version"]
            if version and version != candidate:
                raise ValueError("No trained candidate for requested version")
            if state["champion"] != comparison["champion_version"]:
                raise ValueError("Comparison stale for current champion; retrain offline")
            if not (out / "challenger.joblib").is_file():
                raise FileNotFoundError("Trained challenger bundle is missing")
            state = evaluate_candidate(
                path, candidate, comparison["champion_metrics"], comparison["challenger_metrics"]
            )
            if state["champion"] == candidate:
                try:
                    temporary = out / "bundle.next.joblib"
                    shutil.copyfile(out / "challenger.joblib", temporary)
                    os.replace(temporary, out / "bundle.joblib")
                except OSError:
                    _save(path, previous_state)
                    raise
        elif action == "reject":
            state["history"].append(
                _event("rejected", version=comparison["challenger_version"], reasons=["Explicit operator rejection"])
            )
            state = _save(path, state)
        elif action == "rollback":
            expected_version = state.get("previous")
            source = (
                "champion-original.joblib"
                if expected_version == comparison["champion_version"]
                else "challenger.joblib"
            )
            if not expected_version:
                raise ValueError("No previous champion available")
            if not (out / source).is_file():
                raise FileNotFoundError("Previous champion bundle is missing")
            state = rollback(path)
            try:
                temporary = out / "bundle.next.joblib"
                shutil.copyfile(out / source, temporary)
                os.replace(temporary, out / "bundle.joblib")
            except OSError:
                _save(path, previous_state)
                raise
        else:
            raise ValueError("Unknown governance action")
        return state
