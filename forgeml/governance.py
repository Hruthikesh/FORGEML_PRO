"""Local demonstration registry with auditable, quality-gated transitions."""

import json
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
        if action in ["evaluate", "promote"]:
            candidate = comparison["challenger_version"]
            if version and version != candidate:
                raise ValueError("No trained candidate for requested version")
            if state["champion"] != comparison["champion_version"]:
                raise ValueError("Comparison stale for current champion; retrain offline")
            state = evaluate_candidate(
                path, candidate, comparison["champion_metrics"], comparison["challenger_metrics"]
            )
            if state["champion"] == candidate:
                shutil.copy2(out / "challenger.joblib", out / "bundle.joblib")
        elif action == "reject":
            state["history"].append(
                _event("rejected", version=comparison["challenger_version"], reasons=["Explicit operator rejection"])
            )
            state = _save(path, state)
        elif action == "rollback":
            state = rollback(path)
            source = (
                "champion-original.joblib"
                if state["champion"] == comparison["champion_version"]
                else "challenger.joblib"
            )
            shutil.copy2(out / source, out / "bundle.joblib")
        else:
            raise ValueError("Unknown governance action")
        return state
