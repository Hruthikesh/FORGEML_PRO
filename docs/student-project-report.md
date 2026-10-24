# Student project report: ForgeML Pro

## Problem

Fraud detection is an imbalanced classification problem: a model can achieve high accuracy by predicting that every transaction is legitimate. A useful experiment therefore needs metrics that describe alert quality and missed fraud, plus an evaluation design that respects time.

## Objective

Develop a reproducible local ML project that compares transaction, temporal and graph-derived features; calibrates predicted probabilities; exposes inference through a service implementation; and makes experimental evidence understandable through a dashboard.

The intended real dataset is IEEE-CIS. The executed demonstration currently uses 3,000 synthetic transactions. Real-data access and additional runtime verification remain pending. This report describes the existing implementation, not a completed real-world fraud system.

## Approach

1. Validate transaction identifiers and the one-to-one transaction/identity join.
2. Order observations by relative TransactionDT and reserve separate train, validation, calibration, governance and final evaluation periods.
3. Create current-transaction features and historical features using only strictly earlier observations. Keep tied timestamps together at period boundaries.
4. Compare dummy, logistic regression, random forest and XGBoost configurations. Use controlled random-forest ablations to investigate feature groups.
5. Select by validation average precision and fit probability calibration on its separate period.
6. Save preprocessing, model, calibration, feature history and decision policy in one bundle.
7. Replay later observations, withhold outcomes until a simulated delay, and compare a trained challenger against the champion on a reserved governance period.

## Results from the executed synthetic experiment

| Measure | Result |
|---|---:|
| Candidate configurations | 8 |
| Selected model | Unweighted random forest |
| Final evaluation rows | 450 |
| Average precision | 0.3323 |
| ROC-AUC | 0.6457 |
| Calibrated Brier score | 0.0185 |
| Automated tests | 24 passed; 10 transport tests skipped |

All values come from artifacts/report.json and test evidence. The saved model reproduces the recorded evaluation metrics. These values do not establish performance on IEEE-CIS or real fraud.

The graph-only augmentation scored lower than the tabular RF baseline on the validation fixture (AP 0.0877 versus 0.1269, rounded); temporal augmentation scored 0.1370. The combined weighted RF scored 0.1574. This is a useful negative finding: graph relationships should be evaluated rather than assumed to help. The final unweighted RF winner is a separate class-weighting comparison.

The trained challenger was rejected because its average-precision gain was too small and its calibration was worse. Promotion and rollback mechanics are tested separately with fixtures; no successful real challenger promotion is claimed.

## Engineering evidence

The repository includes source, trained synthetic bundles, experiment artifacts, a six-view dashboard, service validation, replay monitoring, a local registry, Docker/CI definitions and seven generated research figures. Direct singleton Python inference measured approximately 724 ms p50 and 785 ms p95 over 12 calls, including feature-history reconstruction. This is a limitation to improve, not a low-latency production claim.

## Limitations

Authorized real data is still needed. Local network restrictions prevented HTTP/browser checks and genuine application screenshots. Docker was unavailable. Optional LightGBM, Optuna, MLflow, SHAP and Evidently paths have not run because their dependencies were unavailable. There is no production deployment, financial integration, formal uncertainty analysis or distributed streaming feature store.

## Conclusion

The executed demonstration shows a reproducible way to test fraud-model ideas while keeping temporal evaluation and serving artifacts consistent. The next scientific step is a new frozen IEEE-CIS experiment. The next engineering step is live transport/browser validation and efficient stateful historical features. Full completion is tracked in VERIFICATION.md.
