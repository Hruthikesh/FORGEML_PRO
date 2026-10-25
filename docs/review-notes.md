# ForgeML Pro review

Reviewed 25 October 2026.

The project has a useful student-portfolio scope: model comparison, time-aware features, an API, a dashboard and repeatable checks. Its strongest part is testing for future-data leakage rather than relying on a single accuracy score.

## Changes made

- Fixed the Docker model-loading failure by including PyArrow and boosting libraries.
- Fixed the model-versions page when no challenger has been registered.
- Rejected non-finite values, invalid identifiers and invalid transaction amounts during training.
- Made model promotion reject missing or invalid quality and latency measurements.
- Restored the previous registry state if saving a replacement model fails.
- Kept request count and latency totals accurate after the rolling history fills up.
- Kept real-data outputs separate from the committed synthetic demo, and limited serving examples to supported input fields.
- Preserved optional fields when loading prediction examples and kept blank values as missing values.
- Fixed the standalone viewer, simplified project documentation and improved chart and feature-label readability.
- Removed a duplicate navigation render that could reset the prediction form. The browser test checks the actual submitted transaction and missing optional field.
- Added real browser captures and regression checks.

## Verification

GitHub Actions passes 47 Python tests, lint, frontend checks, Docker build, container prediction and browser checks. The browser test covers six dashboard pages, a live prediction, a missing optional field and mobile horizontal overflow. See [verification notes](../VERIFICATION.md) for the recorded run.

## Remaining work

The main next step is evaluation with authorized IEEE-CIS data. Current scores describe a small synthetic experiment. Full-dataset efficiency, repeated forward validation and multi-user access control remain outside this local demo's verified scope.
