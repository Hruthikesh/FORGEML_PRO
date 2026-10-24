"""Print a screenshot checklist; never fabricate browser screenshots."""

from pathlib import Path

assets = Path(__file__).resolve().parents[1] / "docs/assets"
for view, filename in [
    ("overview", "dashboard-overview.png"),
    ("laboratory", "model-laboratory.png"),
    ("prediction", "fraud-prediction.png"),
    ("graph", "graph-intelligence.png"),
    ("monitoring", "drift-monitoring.png"),
    ("governance", "model-governance.png"),
]:
    print(f"Open http://127.0.0.1:8000/#{view}; verify live data; save browser screenshot to {assets / filename}")
print("shap-explanation.png: capture only after actual SHAP artifact generation; no substitute image.")
