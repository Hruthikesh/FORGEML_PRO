# ForgeML Pro model card

Purpose: local experiments in fraud detection using historical transactions.

Data: current trained bundles use 3,000 explicitly synthetic observations. The fixture has a known synthetic signal involving amount/device/card patterns, so performance does not establish real fraud-detection usefulness. IEEE-CIS ingestion is implemented but real-data execution awaits authorized local CSVs.

Selection: chronological validation average precision, followed by separate sigmoid/logistic probability calibration. All final metrics, model version, seed, dataset SHA-256 and boundaries live in artifacts/report.json. Graph ablations use the same RF family/seed; cross-family winners are not controlled graph ablations.

Decision policy: approval/review demonstration at assumed cost threshold 5/(5+100); no real transaction is blocked. Alternative precision-constrained and top-K functions are in forgeml/policies.py. Probability calibration does not supply epistemic uncertainty or a confidence guarantee. Confidence intervals and subgroup fairness analysis are not implemented.

Governance: a challenger trains offline on earlier labeled development windows and is compared on a reserved forward governance period. Promotion requires AP gain >=0.005, Brier regression <=0.02 and mean per-row batch latency <=250ms. Current champion/challenger comparison and actual decision are recorded. Tiny windows and few positive examples make the decision uncertain. The local JSON registry is protected by a process lock, not a distributed transactional registry. No model promotion uses the final test period.

Use limitations: no authentication, authorization, rate limiting, adversarial security validation, durable streaming state, financial integration, production deployment, subgroup evaluation, or formal statistical significance claims. Bind the demo to localhost. Load only trusted joblib bundles because deserialization can execute code. Raw Kaggle data is excluded from Git.
