# ThreatGuard ML audit — 14 October 2026

## Verdict

The application runs, but current models are not reliable enough to claim correct outputs on arbitrary unseen real-world inputs. API integration is working; major problems are URL generalization, evaluation leakage, brittle normalization, and SMS false positives/false negatives. No application or model files were changed.

## Application verification

- Started Vite at http://localhost:5173 and verified the existing backend at http://localhost:8000 through the dashboard and live API.
- PostgreSQL container healthy; initialized missing tables without deleting history.
- Frontend production build and ESLint passed. Build reports a 926.68 kB JavaScript chunk (276.25 kB gzip).
- All 13 existing API tests passed against a separate SQLite database at /tmp/threatguard_audit_20261014.sqlite. These are integration checks, not proof of ML quality. PostgreSQL history was preserved.
- Four live API calls matched direct model predictions exactly for prediction, score, confidence and reasons. Five audit analyses including the browser test were added to application history.
- Camera/QR hardware and full production deployment were outside this ML-focused verification.

## Recomputed results

| Evaluation | Rows | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Actual URL UCI holdout reconstructed from V5 training split | 47,219 | 99.655% | 99.990% | 99.199% | 99.593% |
| Above holdout excluding training hostnames | 42,596 | 99.786% | 99.987% | 99.445% | 99.716% |
| SMS original test set | 8,201 | 97.830% | 96.888% | 97.210% | 97.049% |
| SMS excluding normalized overlaps with train and validation | 8,186 | 97.826% | 96.873% | 97.196% | 97.034% |

URL production model, vectorizer and scaler SHA-256 hashes match the V5 training artifacts. Training rows were reconstructed with the same datasets, deduplication, ordering, stratification and random seed as train_model_v5.py. Historical training provenance is not independently recorded, so the reconstruction assumes these source datasets remain the ones used for training. Hostname exclusion is not registrable-domain exclusion: related subdomains can still span splits. These are same-source historical holdouts, not a new time-separated external benchmark.

## Findings

### 1. The published unseen-domain evaluation includes training examples

ml/url_detection/evaluate_v5.py loads the existing V5 model, then independently shuffles UCI hostnames into a new test split. V5 was trained using a different random row split over the combined UCI and augmentation data. The evaluator does not retrain on its new training domains.

Of the 47,005 claimed unseen test URLs, 37,548 (79.88%) match reconstructed training URLs exactly; 38,452 have a hostname present in training. The existing 99.887% unseen-domain accuracy is therefore not a valid unseen-domain result. The separately reconstructed genuine holdout above remains strong, but does not establish performance on new sources or realistic out-of-distribution inputs.

### 2. URL model fails on realistic benign paths and equivalent spellings

Live API outputs:

| Input | Prediction | Risk score /100 |
|---|---|---:|
| https://github.com | legitimate | 2.18 |
| https://github.com/ | phishing | 97.59 |
| https://docs.python.org/3/library/urllib.parse.html | phishing | 100.00 |
| https://accounts.google.com/ | phishing | 99.78 |
| https://en.wikipedia.org/wiki/Logistic_regression | phishing | 97.92 |

The Python documentation result was also reproduced in the live dashboard. Other listed fresh URL predictions were generated directly using the deployed inference engine, except the GitHub pair which was additionally checked through the live API.

The original UCI CSV contains 134,850 benign rows, all with empty/root paths. build_benign_deep_paths.py adds only /docs/, /wiki/, /en-US/, and /research/. These synthetic paths do not cover realistic nested paths, filenames, queries, identifiers, or login flows. Strong dependence on path structure is observed directly: for the Python documentation URL, standardized path_length contributes +15.23 to the model logit and num_dots contributes +15.24 (individual correlated-feature contributions are not causal estimates).

Inference: dataset structure and narrow augmentation encourage shortcuts that explain the gap between strong same-source scores and poor fresh examples. Threshold adjustment alone cannot resolve extremely confident false positives.

### 3. Fresh SMS probes expose generalization failures

24 author-written messages, absent after lowercase/whitespace normalization from available train/validation/test data: 15 correct, 6 benign messages flagged as scam, 3 intended scam scenarios missed.

Examples:

- Routine electricity bill reminder directing use of the official app: scam, 94.28 risk.
- Routine parcel arrival notice: scam, 76.14 risk.
- Safety warning that staff will never ask for a password or OTP: scam, 56.47 risk.
- New-number family impersonation asking for an urgent transfer and no call: legitimate, 34.22 risk.
- Hindi message asking for OTP and PIN to prevent account closure: legitimate, 4.30 risk.
- Hindi advance-fee prize message: legitimate, 48.88 risk.

These are small, deliberately challenging diagnostic cases with author-assigned scenario labels, not an unbiased accuracy benchmark or externally verified messages. They demonstrate failure modes, not an estimate of overall real-world accuracy.

The original SMS test set has only 11 Hindi scam examples; it detects 7 and misses 4 (63.64% recall). Its 99.13% Hindi accuracy is dominated by 449 benign examples and must not be used to imply reliable Hindi scam detection.

### 4. SMS split contamination is small but measurable

After lowercase/whitespace normalization, 13 test messages also occur in training and 2 in validation. Removing both leaves 8,186 test messages. Metrics barely change, so this contamination does not explain all SMS failures. Near-duplicate/template leakage was not exhaustively measured.

### 5. Invalid inputs receive confident legitimacy labels

The live API accepts 'not a url' with HTTP 200 and returns legitimate at 0.01 risk. Direct inference also accepts 'https://', 'javascript:alert(1)', and 'hello world'. URLAnalysisRequest checks string length only; URLRiskEngine.normalize_url only strips whitespace and prepends a scheme. These should receive input-validation errors rather than threat assessments.

### 6. Confidence and explanations can overstate evidence

The API's confidence is max(p, 1-p), not a measured per-example probability of correctness. The dashboard displayed 'Confidence: 100.00%' for a false-positive documentation URL. Reasons are keyword/structural heuristics, not contributions from the fitted classifier; the same false positive displayed 'No major heuristic signals detected.' Model-based explanations and calibrated uncertainty should be distinguished from heuristic warnings.

## Fresh URL probe interpretation

18 fresh URL strings were absent from the combined available URL corpus: eight expected benign service URLs and ten synthetic phishing-style structures using reserved .example domains. All eight expected benign URLs were flagged as phishing, while all ten suspicious structures were flagged. The synthetic URLs are not verified malicious sites and were never visited. Benign-service paths were selected as diagnostic examples, not exhaustively live-verified resources. Do not present the resulting 10/18 score as real-world phishing accuracy.

## Recommended order of repair

1. Persist actual training/validation/test membership and artifact hashes. Split before fitting by registrable domain and preferably time/source; retain a truly untouched external test set. Replace the current misleading evaluation.
2. Collect representative labeled benign URLs with real paths, queries, login routes and varied hosts; include modern phishing sources. Keep all variants/templates from the same group in one split. Do not blanket-allowlist trusted hosting domains.
3. Validate URL syntax and HTTP(S) schemes, normalize semantically equivalent root URLs consistently during both training and inference, and add metamorphic regression checks such as the GitHub slash case.
4. Expand SMS benign transaction/security/recruitment notices and Hindi/Hinglish scams, group near-duplicate templates before splitting, and report per-language/per-scam-type recall and false-positive rates.
5. Fit calibration using validation data and revise certainty wording. Evaluate changes on held-out data rather than tuning on these audit cases. Treat these probes as regression cases after repair, not future independent evaluation data.

## Reproduction and artifacts

Run from the repository in WSL:

```sh
.venv/bin/python audit/verify_ml.py
.venv/bin/python audit/diagnose_paths.py
.venv/bin/python audit/check_live_api.py
```

The last script requires a running backend and adds four diagnostic rows to history. The other scripts do not modify models or application data.

- ml_audit_results.json: metrics, overlap counts, fresh failures, malformed URL outputs.
- fresh_probes.csv: all 42 fresh cases, intended labels and outputs.
- url_holdout_predictions.csv and sms_holdout_predictions.csv: per-row benchmark outputs.
- path_diagnostics.json: structural sensitivity and feature contribution diagnostics.
- live_api_checks.json: live responses and equality with direct inference.

No retraining or production fixes were made during this audit. Existing untracked data/processed/sms/ and run.sh were preserved.
