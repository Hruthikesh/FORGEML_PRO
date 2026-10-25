# Demo walkthrough

## Start the app

Follow the setup commands in the [README](../README.md). Train a model, start the API and open http://127.0.0.1:8000.
The included demo uses synthetic transactions.

## Explore the project

1. **Overview:** check the selected model, class balance and final test metrics.
2. **Model comparison:** compare validation scores. The feature-group chart compares random forests using the same model settings.
3. **Try a prediction:** load an example, change its amount and submit it. Inspect the probability, threshold and model version. Recorded SHAP explanations, when available, belong to the training run.
4. **Transaction graph:** explore transactions that share card, address or device attributes.
5. **Replay & drift:** replay batches with delayed labels. Try the amount-shift option and compare distribution changes with later model quality.
6. **Model versions:** evaluate the trained candidate. Promotion requires better average precision, acceptable calibration and latency. Rollback becomes available after a successful promotion.

API documentation is at `/docs`. Health, readiness, model details and telemetry are at `/health`, `/ready`, `/model-info` and `/metrics`.

## Run checks and capture screenshots

```powershell
python -m pytest -q
python -m ruff check .
node scripts/check_frontend.mjs
python -m pip install -e ".[browser]"
python -m playwright install chromium
python scripts/check_browser.py
```

The browser check needs a running API. It tests all six views and a prediction, then saves screenshots in `browser-results/`.
For an HTTP benchmark, run `python -m scripts.benchmark --requests 100 --concurrency 4` while the API is running.

## Try real data

Put authorized IEEE-CIS CSV files in `data/raw/`, then run:

```powershell
python -m scripts.train --data-dir data/raw --output-dir runs/ieee
$env:FORGEML_ARTIFACT_DIR = "runs/ieee"
python -m uvicorn forgeml.api:app --host 127.0.0.1 --port 8000
```

Keep these outputs outside the committed demo artifacts. Full-dataset runtime and performance still need evaluation.
