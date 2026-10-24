"""Generate reproducible figures from recorded experiments, never invented metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
OUT.mkdir(parents=True, exist_ok=True)
report = json.loads((ROOT / "artifacts/report.json").read_text())
plt.rcParams.update(
    {
        "figure.facecolor": "#0b1120",
        "axes.facecolor": "#111b2e",
        "savefig.facecolor": "#0b1120",
        "text.color": "#e6edf7",
        "axes.labelcolor": "#aab9d2",
        "xtick.color": "#aab9d2",
        "ytick.color": "#aab9d2",
        "axes.edgecolor": "#33425d",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
    }
)
COLORS = ["#42e5b4", "#6a9fff", "#b999ff", "#ffce75", "#fd8e9e", "#67d4ed", "#b5de79", "#a3a9c2"]


def save(fig, name):
    fig.text(
        0.02,
        0.015,
        "ForgeML Pro  |  SYNTHETIC DEMONSTRATION · executed experiment artifacts",
        color="#9aacc9",
        fontsize=9,
    )
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


models = report["models"]
fig, ax = plt.subplots(figsize=(10, 5))
names = [m["name"] for m in models]
values = [m["metrics"]["average_precision"] for m in models]
ax.barh(names, values, color=COLORS[: len(models)])
ax.invert_yaxis()
ax.set(
    xlabel="Average precision on chronological validation period",
    title="Model laboratory · measured validation performance",
)
for i, v in enumerate(values):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, max(values) * 1.22)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "model-comparison.png")

fig, ax = plt.subplots(figsize=(9, 5))
for m, c in zip(models, COLORS):
    ax.plot(m["curves"]["recall"], m["curves"]["precision"], label=m["name"], color=c, linewidth=1.6)
ax.set(
    xlabel="Recall", ylabel="Precision", title="Precision–recall · chronological validation", xlim=(0, 1), ylim=(0, 1)
)
ax.legend(fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "precision-recall.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([0, 1], [0, 1], color="#61718f", linestyle="--", label="Ideal calibration")
curves = report["curves"]
ax.plot(
    curves["calibration_predicted"],
    curves["calibration_observed"],
    "o-",
    color=COLORS[0],
    label="Frozen champion · final holdout",
)
ax.set(
    xlabel="Mean predicted probability",
    ylabel="Observed fraction fraud",
    title="Reliability · calibrated holdout predictions",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.legend()
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "calibration.png")

fig, ax = plt.subplots(figsize=(11, 6))
ax.set(xlim=(0, 12), ylim=(0, 7))
ax.axis("off")
boxes = [
    (0.2, 5.4, "Authorized IEEE-CIS\nor synthetic fixture"),
    (3.3, 5.4, "Strict temporal &\ngraph features"),
    (6.4, 5.4, "Train / validate /\ncalibrate / final test"),
    (9.5, 5.4, "Versioned bundle\n+ JSON evidence"),
    (0.2, 2.9, "MLflow tracking*\nOptuna / SHAP*"),
    (3.3, 2.9, "Model registry\nquality gates"),
    (6.4, 2.9, "FastAPI preferred*\nStarlette fallback"),
    (9.5, 2.9, "Six-view dashboard\n+ artifact snapshot"),
    (3.3, 0.5, "Challenger training\n& guarded rollback"),
    (6.4, 0.5, "Replay + delayed labels\nPSI / metrics / Evidently*"),
]
for i, (x, y, label) in enumerate(boxes):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            2.3,
            1.15,
            boxstyle="round,pad=0.12",
            facecolor="#15283a",
            edgecolor=COLORS[i % len(COLORS)],
            linewidth=1.5,
        )
    )
    ax.text(x + 1.15, y + 0.57, label, ha="center", va="center", fontsize=9)
for a, b in [(0, 1), (1, 2), (2, 3), (3, 6), (6, 7), (6, 9), (9, 8), (8, 5), (5, 6), (2, 4)]:
    x, y, _ = boxes[a]
    u, v, _ = boxes[b]
    ax.annotate(
        "",
        xy=(u + 1.1, v + 0.55),
        xytext=(x + 1.15, y + 0.55),
        arrowprops={"arrowstyle": "->", "color": "#627993", "connectionstyle": "arc3,rad=.1"},
        zorder=0,
    )
ax.set_title("ForgeML Pro · offline research → local serving → governance", fontsize=16)
fig.text(
    0.03,
    0.065,
    "* Optional or environment-blocked integrations are identified in VERIFICATION.md; this is an architecture diagram, not deployment evidence.",
    fontsize=8,
    color="#aab9d2",
)
save(fig, "system-architecture.png")

windows = report["monitoring"]["windows"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot([w["window"] for w in windows], [w["mean_score"] for w in windows], "o-", color=COLORS[0])
axes[0].set(title="Replay prediction scores", xlabel="Chronological batch", ylabel="Mean calibrated risk")
axes[1].bar([w["window"] for w in windows], [w["drift"]["prediction_psi"] for w in windows], color=COLORS[1])
axes[1].set(
    title="Prediction distribution shift", xlabel="Chronological batch", ylabel="PSI · warning, not degradation proof"
)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "replay-monitoring.png")
print("Generated five figures from actual report.json")

# Measured direct-bundle latency (not an HTTP benchmark).
benchmark_path = ROOT / "artifacts/bundle-benchmark.json"
if benchmark_path.exists():
    benchmark = json.loads(benchmark_path.read_text())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["p50", "p95"], [benchmark["p50_ms"], benchmark["p95_ms"]], color=COLORS[:2])
    ax.set(ylabel="Wall-clock milliseconds", title="Direct Python inference · concurrency 1 · 12 measured calls")
    for i, value in enumerate([benchmark["p50_ms"], benchmark["p95_ms"]]):
        ax.text(i, value + 5, f"{value:.1f} ms", ha="center")
    fig.text(
        0.13,
        0.085,
        "No HTTP transport · immutable history reconstructed per call · synthetic bundle",
        fontsize=9,
        color="#aab9d2",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "inference-benchmark.png")

# Reproducible visualization of actual sampled relationships (not a UI screenshot).
graph = report["graph"]
transactions = [n for n in graph["nodes"] if n["type"] == "transaction"]
entities = [n for n in graph["nodes"] if n["type"] == "entity"]
positions = {}
for i, node in enumerate(transactions):
    positions[node["id"]] = (0, i)
for i, node in enumerate(entities):
    positions[node["id"]] = (1 + 0.15 * (i % 2), i * max(1, len(transactions) - 1) / max(1, len(entities) - 1))
fig, ax = plt.subplots(figsize=(10, 8))
for edge in graph["edges"]:
    x, y = positions[edge["source"]]
    u, v = positions[edge["target"]]
    ax.plot([x, u], [y, v], color="#3a526a", linewidth=0.6, alpha=0.7)
for node in graph["nodes"]:
    x, y = positions[node["id"]]
    ax.scatter(x, y, s=28, c=COLORS[0] if node["type"] == "transaction" else COLORS[1], zorder=3)
    ax.text(
        x - 0.035 if x == 0 else x + 0.035, y, node["label"], ha="right" if x == 0 else "left", va="center", fontsize=6
    )
ax.axis("off")
ax.set_title("Observed transaction–attribute relationships · bounded holdout sample")
fig.text(
    0.08,
    0.06,
    "Weak anonymized attributes are not verified customer identities. This is a generated research figure.",
    fontsize=9,
    color="#aab9d2",
)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "graph-relationships.png")
