# Methodology and feature availability

`TransactionDT` is a relative offset, never a real calendar date. Transaction/identity joins are one-to-one on `TransactionID`; duplicate identifiers fail ingestion. Missing identity matches stay missing.

| Data | Available at scoring? | Treatment |
|---|---|---|
| Amount, product, card/address/device attributes | Assumed yes | Current transaction features; identity attributes must actually arrive before scoring |
| Earlier transaction observations | Yes in this simulation | Strictly smaller TransactionDT; same-timestamp rows cannot see one another |
| Fraud outcome | No | Excluded from features and API schema |
| Delayed labels | Only after simulated delay | Used for eligible replay quality windows |
| Final holdout labels | Evaluation only | Never candidate promotion evidence |

Training uses 55% of unique times, validation 15%, calibration 7.5%, governance 7.5%, final holdout 15%. Counts can vary with tied timestamps. The calibration-to-governance boundary uses unique relative times, keeping tied timestamps together. Preprocessing fits only on training. Validation selects the model; a separate later calibration period fits logistic probability calibration. The fixed decision threshold uses assumed FP=5/FN=100 cost units. Selection freezes before final evaluation. Preliminary engineering reruns used deterministic synthetic fixtures; no claim of a pristine scientific final result across iterative development is made. A new IEEE-CIS evaluation must pre-register the frozen configuration and reserve its final period.

Weak card/address/device values are relationship hints. Missing attributes never create one universal customer node. Union-find graph features are computed before adding the current timestamp's observations. No fraud-rate encoding is used, avoiding assumptions about historical fraud-label availability. Included features: degree, component size, linked prior components, entity count; temporal card counts, historical amount mean, time since previous transaction, and 24-hour activity. No GNN is claimed.

Serving reconstructs features from an immutable unlabeled history snapshot in the bundle. Replay advances a separate in-memory history. This is reproducible but slow and is not a durable streaming feature store. Earlier overlapping transaction IDs are removed from snapshot inputs to avoid self history. Each batch itself supplies earlier observations in that batch; future observations never contribute to earlier rows.

Full-data scaling is unverified: row iteration and dense one-hot encoding can be expensive on IEEE-CIS. Bounded graph display differs from complete feature computation. Identifiers are not verified people. Costs are abstract assumptions, not recovered money. SHAP values, when optional SHAP is available, are model contributions rather than causal effects.
