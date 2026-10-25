# Verification notes

Updated 25 October 2026.

[Successful build and browser run](https://github.com/Hruthikesh/FORGEML_PRO/actions/runs/40171802950) for code commit `382e52a`.

| Check | Result |
| --- | --- |
| Synthetic training | Passed with 3,000 transactions |
| Python tests | 47 passed, no skips; one dependency deprecation warning |
| Ruff | Passed |
| JavaScript syntax and six-view rendering | Passed |
| Docker image build | Passed |
| Container readiness and prediction | Passed |
| Browser navigation | All six views passed without JavaScript errors |
| Browser prediction | Selected example and missing optional field verified in the submitted request |
| Mobile overview | No horizontal page overflow at 390px width |
| README screenshots | Captured from the running app in Chromium and visually reviewed |
| README links and images | Local targets checked |

The screenshots come from the linked CI run, which includes the optional research packages. The committed reference report and plots describe the earlier eight-model run. They are separate runs and can select different models.

The local environment passed the non-HTTP checks; its socket restrictions prevented local HTTP testing. HTTP, Docker and browser results above were verified in GitHub Actions.

## Remaining scope

Current results use synthetic data. Full IEEE-CIS training and performance remain unverified until authorized CSV files are available.
The model registry uses local files and a process lock. It is not a distributed service, and the direct Python benchmark is not an HTTP load test.

See the [review notes](docs/review-notes.md) for the issues fixed in this pass.
