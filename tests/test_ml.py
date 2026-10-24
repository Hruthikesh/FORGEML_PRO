import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forgeml.data import chronological_splits, load_transactions, synthetic_transactions
from forgeml.features import engineer
from forgeml.governance import evaluate_candidate, initialize_registry, rollback
from forgeml.monitoring import psi


def test_temporal_features_ignore_future_rows_and_labels():
    df = synthetic_transactions(100)
    df["card1"] = 123
    full = engineer(df)
    prefix = engineer(df.iloc[:40])
    pd.testing.assert_frame_equal(prefix, full.iloc[:40])
    changed = df.copy()
    changed["isFraud"] = 1 - changed.isFraud
    pd.testing.assert_frame_equal(full, engineer(changed))
    assert full.iloc[0].prior_card_count == 0
    assert full.iloc[1].prior_card_count == 1


def test_same_timestamp_does_not_see_peers():
    df = synthetic_transactions(20)
    df["card1"] = 10
    df.loc[:1, "TransactionDT"] = 0
    features = engineer(df)
    assert features.loc[0, "prior_card_count"] == features.loc[1, "prior_card_count"] == 0
    assert features.loc[2, "prior_card_count"] == 2


def test_serving_history_parity_and_numeric_identity():
    df = synthetic_transactions(100)
    df["card1"] = 123
    history = df.iloc[:80].copy()
    history["card1"] = history.card1.astype(float)
    batched = engineer(df.iloc[80:], history=history)
    whole = engineer(df).iloc[80:].reset_index(drop=True)
    pd.testing.assert_frame_equal(batched, whole)
    single = engineer(df.iloc[[80]], history=history)
    pd.testing.assert_frame_equal(single, whole.iloc[:1])


def test_splits_are_chronological_and_disjoint():
    df = synthetic_transactions(100)
    splits = chronological_splits(df)
    for a, b in zip(list(splits.values())[:-1], list(splits.values())[1:]):
        assert set(a).isdisjoint(b)
        assert df.iloc[a].TransactionDT.max() < df.iloc[b].TransactionDT.min()


def test_join_cardinality_rejects_duplicates(tmp_path):
    df = synthetic_transactions(100)
    df.to_csv(tmp_path / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [1, 1], "DeviceType": ["mobile", "mobile"]}).to_csv(
        tmp_path / "train_identity.csv", index=False
    )
    with pytest.raises(ValueError, match="unique"):
        load_transactions(tmp_path)


def test_governance_promote_reject_and_rollback(tmp_path):
    p = tmp_path / "registry.json"
    champion = {"average_precision": 0.3, "brier": 0.1}
    initialize_registry(p, "a", champion)
    state = evaluate_candidate(p, "bad", champion, {"average_precision": 0.2, "brier": 0.09})
    assert state["champion"] == "a"
    assert state["history"][-1]["action"] == "rejected"
    state = evaluate_candidate(p, "good", champion, {"average_precision": 0.4, "brier": 0.09})
    assert state["champion"] == "good"
    assert rollback(p)["champion"] == "a"


def test_ineligible_labels_block_promotion(tmp_path):
    p = tmp_path / "registry.json"
    metrics = {"average_precision": 0.3, "brier": 0.1}
    initialize_registry(p, "a", metrics)
    state = evaluate_candidate(p, "b", metrics, {"average_precision": 0.9, "brier": 0.01}, eligible=False)
    assert state["champion"] == "a"


def test_drift_identity_and_shift():
    reference = np.linspace(0, 100, 1000)
    assert psi(reference, reference) == pytest.approx(0)
    assert psi(reference, reference + 100) > 0.2


def test_missing_entities_do_not_create_shared_identity():
    frame = synthetic_transactions(20)
    frame["card1"] = None
    frame["addr1"] = None
    frame["DeviceInfo"] = None
    result = engineer(frame)
    assert (result.prior_card_count == 0).all()
    assert (result.prior_entity_degree == 0).all()


def test_future_history_excluded_from_replayed_instant():
    frame = synthetic_transactions(100)
    frame["card1"] = 123
    result = engineer(frame.iloc[[20]], history=frame.iloc[21:])
    assert result.iloc[0].prior_card_count == 0


def test_probability_bundle_roundtrip(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression

    from forgeml.training import make_pipeline, predict_bundle

    frame = synthetic_transactions(200)
    features = engineer(frame)
    model = make_pipeline(features, LogisticRegression(max_iter=500))
    model.fit(features.iloc[:150], frame.isFraud.iloc[:150])
    bundle = {
        "model": model,
        "feature_columns": list(features.columns),
        "group": "all",
        "history": frame.iloc[:150].drop(columns="isFraud"),
        "calibrator": None,
    }
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    before = predict_bundle(bundle, frame.iloc[150:160])
    after = predict_bundle(joblib.load(path), frame.iloc[150:160])
    np.testing.assert_allclose(before, after)
    assert ((after >= 0) & (after <= 1)).all()


def test_delayed_quality_hides_unavailable_labels():
    import joblib

    from forgeml.monitoring import replay

    if not Path("artifacts/bundle.joblib").exists():
        pytest.skip("Training bundle not generated")
    bundle = joblib.load("artifacts/bundle.joblib")
    frame = pd.read_csv("artifacts/replay-data.csv").head(90)
    report = replay(bundle, frame, batch_size=30, label_delay=2)
    assert report["windows"][0]["delayed_quality"] is None
    assert report["windows"][1]["delayed_quality"] is None
    assert report["windows"][2]["delayed_quality"]["labeled_count"] == 30


def test_replayed_batch_deduplicates_snapshot_overlap():
    frame = synthetic_transactions(100)
    expected = engineer(frame).iloc[50:60].reset_index(drop=True)
    actual = engineer(frame.iloc[50:60], history=frame.iloc[:80])
    pd.testing.assert_frame_equal(expected, actual)


def test_serialized_champion_matches_reported_holdout():
    import joblib

    from forgeml.data import fingerprint
    from forgeml.training import metrics, predict_bundle

    report_path = Path("artifacts/report.json")
    if not report_path.exists():
        pytest.skip("Training evidence not generated")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["mode"] != "SYNTHETIC DEMONSTRATION":
        pytest.skip("Real restricted dataset intentionally not bundled")
    frame = synthetic_transactions(report["rows"], report["seed"])
    assert fingerprint(frame) == report["dataset_sha256"]
    test = frame[frame.TransactionDT >= report["splits"]["test"]["start"]]
    bundle = joblib.load("artifacts/champion-original.joblib")
    scores = predict_bundle(bundle, test.drop(columns="isFraud"))
    observed = metrics(test.isFraud, scores, bundle["threshold"])
    for metric in ["average_precision", "roc_auc", "brier", "precision", "recall"]:
        assert observed[metric] == pytest.approx(report["holdout"][metric], abs=1e-12)


def test_calibration_governance_boundary_keeps_timestamps_disjoint():
    from forgeml.data import reserve_governance

    frame = pd.DataFrame({"TransactionDT": [0, 1, 1, 1, 2, 3]})
    calibration, governance = reserve_governance(frame, np.arange(len(frame)))
    assert frame.iloc[calibration].TransactionDT.max() < frame.iloc[governance].TransactionDT.min()
