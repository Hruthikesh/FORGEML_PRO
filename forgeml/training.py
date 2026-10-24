"""Reproducible time-ordered model experiments and versioned serving bundles."""

import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import chronological_splits, fingerprint, load_transactions, validate_transactions
from .features import engineer


def metrics(y, p, threshold=0.5):
    y = np.asarray(y)
    pred = p >= threshold
    k = max(1, int(0.05 * len(y)))
    top = np.argsort(-p)[:k]
    false_positive = int(np.sum(pred & (y == 0)))
    fpr, tpr, _ = roc_curve(y, p) if len(np.unique(y)) == 2 else ([0], [0], [0])
    return {
        "average_precision": float(average_precision_score(y, p)) if y.sum() else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "precision_at_k": float(y[top].mean()),
        "recall_at_budget": float(y[top].sum() / max(1, y.sum())),
        "review_budget": k,
        "recall_at_fpr_01": float(max([t for f, t in zip(fpr, tpr) if f <= 0.01], default=0)),
        "false_positives_per_1000": 1000 * false_positive / len(y),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "assumed_cost": float(5 * false_positive + 100 * np.sum((~pred) & (y == 1))),
        "threshold": float(threshold),
        "count": len(y),
    }


def make_pipeline(x, estimator):
    cats = x.select_dtypes(include=["object", "string"]).columns.tolist()
    nums = [c for c in x if c not in cats]
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _calibrated(bundle, p):
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return calibrator.predict_proba(logits)[:, 1]
    return p


def predict_bundle(bundle, raw_df):
    features = engineer(raw_df, bundle.get("group", "all"), history=bundle.get("history"))
    p = bundle["model"].predict_proba(features[bundle["feature_columns"]])[:, 1]
    return _calibrated(bundle, p)


def _curves(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    fraction, mean = calibration_curve(y, p, n_bins=8, strategy="uniform")
    # Downsample dense curves only for display, retaining measured metrics separately.
    indices = np.unique(np.linspace(0, len(precision) - 1, min(150, len(precision))).astype(int))
    return {
        "precision": precision[indices].tolist(),
        "recall": recall[indices].tolist(),
        "calibration_predicted": mean.tolist(),
        "calibration_observed": fraction.tolist(),
    }


def _json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v), allow_nan=False),
        encoding="utf-8",
    )


def run_training(output_dir=Path("artifacts"), data_dir=None, n_rows=3000, seed=42, optuna_trials=3):
    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, mode = load_transactions(data_dir, n_rows)
    validate_transactions(df)
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    splits = chronological_splits(df)
    # Reserve half the original calibration interval for a forward governance comparison.
    cal = splits["calibration"]
    from .data import reserve_governance

    splits["calibration"], splits["governance"] = reserve_governance(df, cal)
    y = df.isFraud.to_numpy()
    if any(len(np.unique(y[ix])) < 2 for ix in splits.values()):
        raise ValueError("Each chronological interval needs both classes; provide more rows")
    features = {g: engineer(df, g) for g in ["tabular", "temporal", "graph", "all"]}
    estimators = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=700, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=2
        ),
        "RF Unweighted": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=4, random_state=seed, n_jobs=2
        ),
    }
    unavailable = []
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        estimators["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.06, random_state=seed, n_jobs=2, eval_metric="logloss"
        )
    else:
        unavailable.append("XGBoost: package unavailable")
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier

        estimators["LightGBM"] = LGBMClassifier(
            n_estimators=100, num_leaves=15, verbosity=-1, random_state=seed, n_jobs=2
        )
    else:
        unavailable.append("LightGBM: package unavailable")
    models = []
    fitted = {}
    train, validation = splits["train"], splits["validation"]
    for name, estimator in estimators.items():
        x = features["all"]
        model = make_pipeline(x, estimator)
        begin = time.perf_counter()
        model.fit(x.iloc[train], y[train])
        elapsed = time.perf_counter() - begin
        p = model.predict_proba(x.iloc[validation])[:, 1]
        version = f"{name.lower().replace(' ', '-')}-all"
        models.append(
            {
                "name": name,
                "version": version,
                "group": "all",
                "training_seconds": elapsed,
                "metrics": metrics(y[validation], p),
                "curves": _curves(y[validation], p),
            }
        )
        fitted[version] = (model, "all")
    for group in ["tabular", "temporal", "graph"]:
        x = features[group]
        model = make_pipeline(
            x,
            RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=2
            ),
        )
        begin = time.perf_counter()
        model.fit(x.iloc[train], y[train])
        elapsed = time.perf_counter() - begin
        p = model.predict_proba(x.iloc[validation])[:, 1]
        version = f"random-forest-{group}"
        models.append(
            {
                "name": f"RF / {group}",
                "version": version,
                "group": group,
                "training_seconds": elapsed,
                "metrics": metrics(y[validation], p),
                "curves": _curves(y[validation], p),
            }
        )
        fitted[version] = (model, group)
    tuning = []
    if importlib.util.find_spec("optuna"):
        import optuna

        for model_name in [n for n in ["XGBoost", "LightGBM"] if n in estimators]:

            def objective(trial, base_estimator=estimators[model_name]):
                candidate = clone(base_estimator)
                candidate.set_params(
                    n_estimators=trial.suggest_int("n_estimators", 50, 150),
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                )
                pipeline = make_pipeline(features["all"], candidate)
                pipeline.fit(features["all"].iloc[train], y[train])
                return average_precision_score(
                    y[validation], pipeline.predict_proba(features["all"].iloc[validation])[:, 1]
                )

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=optuna_trials)
            tuned_estimator = clone(estimators[model_name]).set_params(**study.best_params)
            tuned_model = make_pipeline(features["all"], tuned_estimator)
            begin = time.perf_counter()
            tuned_model.fit(features["all"].iloc[train], y[train])
            tuned_duration = time.perf_counter() - begin
            tuned_scores = tuned_model.predict_proba(features["all"].iloc[validation])[:, 1]
            tuned_version = model_name.lower() + "-tuned-all"
            models.append(
                {
                    "name": model_name + " / Optuna",
                    "version": tuned_version,
                    "group": "all",
                    "training_seconds": tuned_duration,
                    "metrics": metrics(y[validation], tuned_scores),
                    "curves": _curves(y[validation], tuned_scores),
                }
            )
            fitted[tuned_version] = (tuned_model, "all")
            tuning.append(
                {
                    "model": model_name,
                    "best_parameters": study.best_params,
                    "best_average_precision": study.best_value,
                    "trials": [{"number": t.number, "params": t.params, "value": t.value} for t in study.trials],
                }
            )
    else:
        unavailable.append("Optuna: package unavailable")
    selected = max(models, key=lambda record: record["metrics"]["average_precision"])
    model, group = fitted[selected["version"]]
    x = features[group]
    calibration = splits["calibration"]
    raw = model.predict_proba(x.iloc[calibration])[:, 1]
    calibrator = LogisticRegression(C=1, random_state=seed).fit(
        np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6))).reshape(-1, 1), y[calibration]
    )
    bundle = {
        "model": model,
        "group": group,
        "feature_columns": list(x.columns),
        "threshold": 5 / 105,
        "policy_version": "cost-fp5-fn100-v1",
        "version": selected["version"] + "-" + fingerprint(df)[:8],
        "mode": mode,
        "calibrator": calibrator,
        "history": df.iloc[: splits["test"][0]].drop(columns="isFraud"),
        "reference_features": x.iloc[train],
        "reference_scores": model.predict_proba(x.iloc[train])[:, 1],
        "schema_version": 1,
        "feature_contract": "current fields plus strictly earlier unlabeled transactions; no future labels",
        "dataset_sha256": fingerprint(df),
    }
    bundle["reference_scores"] = _calibrated(bundle, bundle["reference_scores"])
    # Freeze selection, calibration and assumed-cost policy before one final test evaluation.
    test = splits["test"]
    p = _calibrated(bundle, model.predict_proba(x.iloc[test])[:, 1])
    holdout = metrics(y[test], p, bundle["threshold"])
    raw_holdout = model.predict_proba(x.iloc[test])[:, 1]
    holdout["raw_brier"] = float(brier_score_loss(y[test], raw_holdout))
    curves = _curves(y[test], p)
    raw_curves = _curves(y[test], raw_holdout)
    governance = splits["governance"]
    champion_scores = _calibrated(bundle, model.predict_proba(x.iloc[governance])[:, 1])
    champion_metrics = metrics(y[governance], champion_scores, bundle["threshold"])
    # Challenger expands only into earlier, eligible development observations.
    challenger = make_pipeline(
        x,
        RandomForestClassifier(
            n_estimators=120, max_depth=9, min_samples_leaf=4, class_weight="balanced", random_state=seed + 1, n_jobs=2
        ),
    )
    expanded = np.concatenate([train, validation])
    challenger.fit(x.iloc[expanded], y[expanded])
    challenger_bundle = {
        **bundle,
        "model": challenger,
        "calibrator": None,
        "version": "challenger-rf-" + fingerprint(df)[:8],
    }
    challenger_metrics = metrics(y[governance], challenger.predict_proba(x.iloc[governance])[:, 1], bundle["threshold"])
    for candidate_bundle, candidate_metrics in [(bundle, champion_metrics), (challenger_bundle, challenger_metrics)]:
        begin = time.perf_counter()
        predict_bundle(candidate_bundle, df.iloc[governance].drop(columns="isFraud"))
        candidate_metrics["latency_ms"] = 1000 * (time.perf_counter() - begin) / len(governance)
        candidate_metrics["latency_measurement"] = (
            "mean per-row wall time in reserved-window batch, includes feature reconstruction"
        )
    joblib.dump(bundle, out / "bundle.joblib")
    joblib.dump(bundle, out / "champion-original.joblib")
    joblib.dump(challenger_bundle, out / "challenger.joblib")
    # Retain named fit candidates for transparent offline comparison.
    for version, (candidate, candidate_group) in fitted.items():
        joblib.dump(
            {
                **bundle,
                "model": candidate,
                "calibrator": None,
                "group": candidate_group,
                "feature_columns": list(features[candidate_group].columns),
                "version": version,
            },
            out / (version + ".joblib"),
        )
    df.iloc[governance].to_csv(out / "replay-data.csv", index=False)
    examples = json.loads(df.iloc[governance[:8]].drop(columns="isFraud").to_json(orient="records"))
    _json(out / "examples.json", examples)
    from .monitoring import evidently_report, replay

    monitoring = replay(bundle, df.iloc[governance].head(300), batch_size=100, label_delay=2)
    _json(out / "replay.json", monitoring)
    nodes = []
    edges = []
    seen = set()
    for _, row in df.iloc[test[:18]].iterrows():
        tid = f"T{int(row.TransactionID)}"
        nodes.append({"id": tid, "type": "transaction", "label": tid})
        for key in ["card1", "addr1", "DeviceInfo"]:
            value = row.get(key)
            if pd.notna(value):
                eid = f"{key}:{value}"
                if eid not in seen:
                    nodes.append({"id": eid, "type": "entity", "label": eid})
                    seen.add(eid)
                edges.append({"source": tid, "target": eid})
    graph = {
        "nodes": nodes,
        "edges": edges,
        "note": "Bounded historical sample. Attributes are weak anonymized relationship hints, not confirmed customer identities.",
    }
    explanation = {"status": "unavailable", "reason": "SHAP package unavailable", "features": []}
    if importlib.util.find_spec("shap"):
        try:
            import shap

            # Explain the selected estimator before its separate probability calibration.
            selected_estimator = model.named_steps["model"]
            transformed = model.named_steps["preprocess"].transform(x.iloc[governance[:64]])
            background = model.named_steps["preprocess"].transform(x.iloc[train[:100]])
            if isinstance(selected_estimator, LogisticRegression):
                explainer = shap.LinearExplainer(selected_estimator, background)
                method = "SHAP LinearExplainer"
            else:
                explainer = shap.TreeExplainer(selected_estimator)
                method = "SHAP TreeExplainer"
            values = explainer(transformed)
            contributions = np.asarray(values.values)
            if contributions.ndim == 3:
                contributions = contributions[:, :, 1]
            names = model.named_steps["preprocess"].get_feature_names_out()
            order = np.argsort(-np.abs(contributions[0]))[:12]
            importance = np.mean(np.abs(contributions), axis=0)
            global_order = np.argsort(-importance)[:20]
            explanation = {
                "status": "verified",
                "method": method,
                "model": bundle["version"],
                "transaction_id": int(df.iloc[governance[0]].TransactionID),
                "raw_probability": float(model.predict_proba(x.iloc[governance[:1]])[0, 1]),
                "calibrated_probability": float(champion_scores[0]),
                "note": "Contributions explain the selected raw estimator, not the nonlinear calibration mapping. They are model explanations, not causal effects.",
                "features": [
                    {
                        "feature": str(names[i]),
                        "contribution": float(contributions[0, i]),
                        "value": float(transformed[0, i]),
                    }
                    for i in order
                ],
                "global_features": [
                    {"feature": str(names[i]), "mean_absolute_contribution": float(importance[i])} for i in global_order
                ],
                "global_sample_size": len(transformed),
            }
        except Exception as exc:  # noqa: BLE001 -- optional explainer errors must be reported without losing trained models
            explanation["reason"] = str(exc)
    _json(out / "explanations.json", explanation)
    from .governance import initialize_registry

    registry = initialize_registry(out / "registry.json", bundle["version"], champion_metrics, reset=True)
    report = {
        "mode": mode,
        "dataset_sha256": fingerprint(df),
        "rows": len(df),
        "fraud_prevalence": float(y.mean()),
        "seed": seed,
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "splits": {
            name: {
                "rows": len(ix),
                "start": int(df.iloc[ix].TransactionDT.min()),
                "end": int(df.iloc[ix].TransactionDT.max()),
            }
            for name, ix in splits.items()
        },
        "models": models,
        "selected_model": selected["name"],
        "selected_version": bundle["version"],
        "holdout": holdout,
        "curves": curves,
        "raw_curves": raw_curves,
        "threshold": bundle["threshold"],
        "policy": {
            "false_positive_cost": 5,
            "false_negative_cost": 100,
            "note": "Assumed abstract cost units, not recovered money",
        },
        "graph": graph,
        "monitoring": monitoring,
        "evidently": evidently_report(x.iloc[train], x.iloc[governance], out / "evidently.html"),
        "governance": registry,
        "governance_comparison": {
            "champion_version": bundle["version"],
            "challenger_version": challenger_bundle["version"],
            "champion_metrics": champion_metrics,
            "challenger_metrics": challenger_metrics,
            "evaluation_window": "governance; disjoint from final test",
        },
        "tuning": tuning,
        "explanation": explanation,
        "unavailable": unavailable,
        "training_seconds": time.perf_counter() - started,
        "tracking": {"status": "unavailable", "reason": "MLflow package unavailable"},
        "limitations": [
            "Synthetic demonstration results are not IEEE-CIS evidence.",
            "Identifiers are weak relationship hints.",
            "No statistical-significance claim.",
            "Serving uses an immutable history snapshot; replay advances an in-memory history independently of serving.",
            "Governance actions are local simulation, not financial production authorization.",
        ],
    }
    if importlib.util.find_spec("mlflow"):
        try:
            import mlflow

            mlflow.set_tracking_uri((out / "mlruns").resolve().as_uri())
            mlflow.set_experiment("ForgeML Pro")
            revision = (
                subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
                or "uncommitted-workspace"
            )
            runs = []
            for entry in models:
                with mlflow.start_run(run_name=entry["version"]) as run:
                    mlflow.log_params(
                        {
                            "seed": seed,
                            "dataset_sha256": fingerprint(df),
                            "group": entry["group"],
                            "mode": mode,
                            "code_revision": revision,
                        }
                    )
                    candidate = fitted[entry["version"]][0]
                    estimator_params = candidate.named_steps["model"].get_params(deep=False)
                    mlflow.log_params({"estimator_" + key: str(value)[:500] for key, value in estimator_params.items()})
                    mlflow.log_metrics({k: v for k, v in entry["metrics"].items() if isinstance(v, (int, float))})
                    mlflow.log_metric("training_seconds", entry["training_seconds"])
                    mlflow.log_dict(entry["curves"], "evaluation/validation-curves.json")
                    mlflow.log_dict(report["splits"], "data/splits.json")
                    mlflow.log_dict(
                        {
                            "group": entry["group"],
                            "features": list(features[entry["group"]].columns),
                            "contract": bundle["feature_contract"],
                        },
                        "features/contract.json",
                    )
                    mlflow.log_artifact(str(out / (entry["version"] + ".joblib")), artifact_path="bundles")
                    runs.append({"version": entry["version"], "run_id": run.info.run_id})
            report["tracking"] = {
                "status": "verified",
                "uri": str(out / "mlruns"),
                "runs": runs,
                "code_revision": revision,
            }
        except Exception as exc:  # noqa: BLE001 -- tracking backend failure is recorded while preserving local experiment evidence
            report["tracking"] = {"status": "failed", "reason": str(exc)}
    _json(out / "report.json", report)
    return report
