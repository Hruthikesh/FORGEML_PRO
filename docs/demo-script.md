# Reproducible demonstration

## Install and train

From repository root with Python 3.11+ and network access:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,research]"
python -m scripts.train --rows 3000 --trials 3
python -m scripts.plot_results
python -m scripts.build_snapshot
python -m uvicorn forgeml.api:app --host 127.0.0.1 --port 8000
```

Visit http://127.0.0.1:8000. All bundled experiment results are synthetic until retrained with authorized data. `frontend/snapshot.html` is a separate offline artifact viewer with mutations disabled and a permanent provenance banner.

## Walkthrough

1. Overview: inspect mode, labeled prevalence, selected model and holdout period.
2. Laboratory: compare real validation metrics and RF feature-group ablations. Separate validation/model-selection scores from champion holdout scores.
3. Prediction: choose a synthetic example, submit it, inspect actual probability/version/policy. The explanation panel shows a recorded explanation artifact, not a per-request SHAP claim. Without SHAP it clearly reports unavailability.
4. Graph: inspect bounded transaction–attribute relationships; attributes are weak hints, not confirmed customers.
5. Monitoring: run replay with delayed labels. Optionally induce an explicitly labeled amount shift; compare PSI warnings with quality on eligible labeled windows.
6. Governance: evaluate the trained challenger. The current fixture can reject it. Do not invent a promotion; successful promotion/rejection/rollback are tested with separate fixtures. Rollback works only after a genuine prior promotion exists.
7. API: inspect /health, /ready, /model-info and /metrics. With FastAPI installed, inspect /docs.
8. Benchmarks: `python -m scripts.benchmark --requests 100 --concurrency 4`. This is an actual HTTP benchmark and requires the running service. `python -m scripts.benchmark_bundle` is explicitly direct Python only.

## Tests and screenshots

```powershell
python -m pytest -q --basetemp .test-tmp
python -m ruff check .
node scripts/check_frontend.mjs
python -m scripts.capture_screenshots
```

The screenshot helper prints the exact URL and expected image names for manual browser captures; it does not synthesize screenshots. Capture each live view only after the app visibly shows real backend artifacts. Browser screenshots were blocked in the original execution environment and are not included as fabricated assets.

## Real data

Accept the competition's access requirements in your own Kaggle account, download the authorized files, and place `train_transaction.csv` and `train_identity.csv` under `data/raw/`. Then run:

```powershell
python -m scripts.train --data-dir data/raw --trials 3
python -m scripts.plot_results
python -m scripts.build_snapshot
```

This uses the whole supplied data; full-scale resource use is unverified. Record a new experiment and do not compare synthetic results as if they were IEEE-CIS scores. Keep the final holdout untouched during design and promotion.
