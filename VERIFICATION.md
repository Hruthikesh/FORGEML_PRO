# Verification matrix

Recorded 24 October 2026. The full user definition of done is **not yet met**. Source code is distinguished from runtime evidence.

| Requirement | Status | Evidence / remaining condition |
|---|---|---|
| Source repository initialized | VERIFIED | Local Git main branch; no external remote/publication |
| Authorized IEEE-CIS acquisition | BLOCKED | User will provide CSVs; no Kaggle credential shared |
| Real-data ingestion execution | BLOCKED | CSV join implementation exists; fixture cardinality tests passed |
| Synthetic training execution | VERIFIED | artifacts/report.json; 3,000 rows, seed 42, fingerprint recorded |
| Chronological split correctness | VERIFIED | 15 ML tests include period disjointness and strict same-time handling and distinct calibration/governance timestamps |
| Tabular/logistic/dummy/RF/XGBoost | VERIFIED | Eight executed candidate configurations and serialized bundles |
| LightGBM | IMPLEMENTED BUT UNVERIFIED | Optional dependency unavailable |
| Past-only temporal/graph features | VERIFIED | Feature leakage tests; actual controlled RF ablations |
| Optuna optimization | IMPLEMENTED BUT UNVERIFIED | Conditional tuned-model path; package unavailable |
| Final synthetic holdout | VERIFIED | 450 rows; serialized champion reproduces recorded AP/ROC/Brier |
| Final IEEE-CIS holdout | BLOCKED | No real data; no real fraud results claimed |
| Calibration and cost threshold | VERIFIED | Separate calibration period; raw/calibrated Brier and curves |
| Alternative precision/top-K policies | VERIFIED | Actual policy tests; deployment policy remains fixed assumed cost |
| Epistemic uncertainty / confidence intervals | NOT IMPLEMENTED | Calibration is not a confidence bound |
| SHAP global/local | IMPLEMENTED BUT UNVERIFIED | Conditional selected-tree artifacts; package unavailable |
| MLflow experiment records | IMPLEMENTED BUT UNVERIFIED | Conditional params/revision/artifact logging; package unavailable |
| JSON experiment evidence | VERIFIED | Actual metrics, splits, seed, data fingerprint, fit durations |
| Bundle serialization and parity | VERIFIED | Roundtrip and report-to-bundle equality tests |
| Pydantic validation / direct service logic | VERIFIED | Seven service tests; actual trained-bundle prediction |
| FastAPI execution | IMPLEMENTED BUT UNVERIFIED | FastAPI missing; explicit Starlette fallback imported |
| Actual HTTP endpoints / metrics transport | BLOCKED | Windows socketpair hangs; 10 transport tests explicitly skipped |
| Six-view dashboard implementation | VERIFIED | JS syntax + VM data/render checks; all six data-driven views |
| Live dashboard/backend interaction | BLOCKED | Local server cannot initialize network sockets |
| Offline artifact viewer generation | VERIFIED | Embedded JSON equality to recorded artifacts; mutations disabled |
| Browser visual QA / screenshots | BLOCKED | Loopback refused; file URL blocked by browser policy; no workaround attempted |
| Historical replay / delayed-label quality | VERIFIED | Actual replay.json and withholding/history tests |
| Drift / missingness diagnostics | VERIFIED | Actual PSI and score reports; warnings do not prove degradation |
| Evidently report execution | IMPLEMENTED BUT UNVERIFIED | Optional adapter, package unavailable |
| Offline challenger training and comparison | VERIFIED | Real separate forward governance window |
| Actual candidate decision | VERIFIED | Rejected by gates; no forced promotion |
| Promotion/rejection/rollback mechanics | VERIFIED | Test fixtures exercise all paths; not claimed as empirical challenger success |
| Durable distributed governance | NOT IMPLEMENTED | Local JSON/process-lock demonstration only |
| Docker configuration / CI definition | IMPLEMENTED BUT UNVERIFIED | Dockerfile, Compose, CI workflow and smoke script |
| Container smoke run | BLOCKED | Docker executable absent |
| Direct Python latency benchmark | VERIFIED | bundle-benchmark.json: 12 measured calls, one warmup, concurrency 1 |
| HTTP load benchmark | BLOCKED | Script exists, local transport unavailable |
| Research charts | VERIFIED | Seven real figures generated from artifacts; inspected architecture PNG |
| README image existence | VERIFIED | All linked local visual assets exist; no fabricated screenshots |
| Unit/integration suite | VERIFIED | 24 passed, 10 explicit transport skips; two dependency deprecation warnings |
| Lint / frontend checks | VERIFIED | Ruff and Node checks passed; final evidence in artifacts |
| Target Downloads folder write | BLOCKED | Read succeeds; write probe returned Access denied despite user's workspace update |
| GitHub publication | NOT IMPLEMENTED | User prohibited publication without authorization |

## Environment and decisions

Network package installation failed DNS. An escalation request was rejected by the current granular approval policy. Reused existing local packages/cached wheels inside a workspace venv; runtime versions are recorded separately. No credentials were read or committed.

Local workspace relocation and GitHub publication are tracked separately from technical verification. Temporary execution logs, recovery planning documents and trained binary artifacts are excluded from the source repository. Reproduce bundles with the documented training command.
