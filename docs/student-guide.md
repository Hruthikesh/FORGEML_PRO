# Understanding and presenting ForgeML

Use this guide to understand the implementation before presenting it. Explain only work and results you can reproduce; adapt the wording to reflect your own contributions.

## Read the code in this order

| File | Question it answers |
|---|---|
| forgeml/data.py | Where do rows come from, and how are joins and time periods validated? |
| forgeml/features.py | What was known before a transaction, and how is future information excluded? |
| forgeml/training.py | How are models compared, calibrated and saved? |
| forgeml/policies.py | How does a probability become a review decision? |
| forgeml/monitoring.py | What changes during replay, and when are labels allowed to appear? |
| forgeml/governance.py | What evidence is needed to replace a model? |
| forgeml/api.py | How does validated input reach the actual saved bundle? |
| frontend/app.js | How do the six views consume backend artifacts? |
| tests/test_ml.py | Which leakage and reproducibility failures are prevented? |

## A short project introduction

“ForgeML is a local fraud-detection ML project. It compares transaction, temporal and graph-derived features using chronological evaluation. It also includes calibrated scoring, a dashboard, delayed-label monitoring and model promotion checks. The current executed results use synthetic data; real IEEE-CIS experiments and some runtime checks remain pending.”

## Questions to be ready for

**Why average precision rather than accuracy?**
Fraud is rare. A trivial all-legitimate model can score highly on accuracy. Average precision summarizes precision–recall behavior across ranking thresholds. Also discuss review budgets and false positives rather than quoting one metric alone.

**Why chronological splits?**
The model must predict later observations using earlier information. Random splitting can make future patterns or shared entities appear in both development and evaluation data in unrealistic ways.

**How are graph features prevented from leaking?**
Features are read before relationships for the current timestamp are inserted. Same-time transactions cannot see each other. Outcome labels are not graph inputs. Inspect the corresponding tests and be able to demonstrate a future-row change leaving an earlier feature unchanged.

**Did graph features improve the model?**
Not consistently. In the synthetic validation run, the graph augmentation alone performed worse than the tabular RF baseline. Report the controlled ablation and its limits rather than presenting graph ML as an automatic improvement.

**Why use calibration?**
A useful ranking does not guarantee useful probabilities. Calibration learns a mapping on a separate eligible development period; Brier score and reliability plots assess probability quality. Calibration is not an uncertainty guarantee.

**Why was the challenger rejected?**
Its AP gain did not meet the configured minimum and its Brier score regressed. The gates preserve this outcome. Fixture tests separately exercise successful promotion and rollback mechanics.

**What is the largest engineering limitation?**
Each singleton call reconstructs past features from the saved history, making direct inference slow. A durable incremental feature state with parity tests would be the next implementation step.

**What has actually been verified?**
Read VERIFICATION.md and the saved test logs. Separate direct model/service tests from HTTP, browser and container checks. Do not call skipped transport tests passing endpoint tests.

## Suggested walkthrough

1. Explain the problem and clearly state synthetic-data mode.
2. Show the model comparison and describe one negative finding.
3. Trace a transaction through features, the saved estimator, calibration and policy.
4. Show a test that catches future-data leakage.
5. Explain delayed labels and the actual rejected challenger.
6. Close with one concrete next experiment and one engineering improvement.

For the live dashboard steps, use docs/demo-script.md after the server can run. The offline snapshot is a read-only artifact viewer; its controls cannot perform inference or governance actions.
