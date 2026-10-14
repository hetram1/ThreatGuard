# ThreatGuard ML repair report — 14 October 2026

## Outcome

Implemented a reproducible, evaluated research-demo release (`reliable-v1`) in the existing application. Invalid input handling, canonical root URLs, split provenance, model integrity, calibration evaluation and score explanations are repaired. Both models were retrained and a separate external URL benchmark was evaluated once after development was frozen.

**The models are not production-ready and do not reliably classify every unseen input.** URL generalization improves on the external benchmark but remains weak; there is a substantial recall tradeoff. Hindi scam recall and transactional-message false positives remain unresolved. These limitations are demonstrated, not hidden.

## Changes and reasons

| Area | Implemented change | Reason |
|---|---|---|
| URL validation | HTTP(S) only; valid domain/IP, port, escaping and authority checks; 422 errors on dedicated and unified routes | Invalid text must not be assessed as legitimate |
| Canonicalization | Shared host/scheme/IDNA/default-port normalization and equivalent empty/root paths | Eliminate representation-only score changes |
| URL model | Retrained hostname character TF-IDF + logistic regression | Remove path/source shortcuts unsupported by representative benign path data |
| SMS model | Retrained word/character TF-IDF + logistic regression, with held-out sigmoid calibration | Improve reproducibility and probability reliability without fabricated labels |
| Data splits | Registrable-domain URL groups; normalized SMS templates and MinHash/Jaccard connected components | Keep related examples together |
| Provenance | SHA-256 source/model/code hashes; persisted memberships; immutable release directories | Make data and deployed artifacts auditable |
| Evaluation | Retired the contaminated V5 evaluator; isolated internal and external workflows | Prevent inflated unseen-domain claims |
| UI | Model estimate instead of confidence; scope warnings; separate learned contributions and heuristic indicators | Explain what the score does and does not establish |
| API tests | Disposable in-memory SQLite via dependency overrides | The former fixture deleted configured history; tests now cannot do that |
| Storage | Existing schema and old rows preserved; new model versions stored normally | Maintain compatibility and history |

The URL model now ignores path/query features in its probability calculation. It still validates and preserves them in normalized input and can display structural heuristic indicators. This is a deliberate scope reduction, not a full solution for phishing hosted on compromised legitimate domains. No allowlist, famous-domain exception or audit-case lookup was added.

## Data inventory and leakage controls

Original datasets were inspected before use. Existing untracked `data/processed/sms/` and `run.sh` were preserved. Original model binaries remain intact. Historical memberships and hashes are under `data/reliable/v2/`; the compact manifest is also saved as `models/releases/reliable-v1/dataset_manifest.json`.

Historical URL membership is reconstructed from the V5 code and the currently available source files. The absence of an original run manifest means historical execution cannot be conclusively attested. SMS original train/validation/test file membership is preserved verbatim. New memberships are recorded as the preparation runs.

- URL: 234,894 normalized, non-conflicting rows; 197,708 registrable-domain groups. One conflicting normalized URL was quarantined. The synthetic Majestic path/root augmentations were inventoried for legacy reproduction but excluded from new training.
- SMS: 76,685 retained rows; 74,731 templates connected into 72,698 groups. 2,033 near-duplicate component unions. One invalid/empty row quarantined; 8,917 rows from explicitly synthetic, benchmark, and spam-only source categories excluded.
- SMS source labels are inherited, not independently re-annotated. The original dataset card advertises mixed annotation provenance. Excluding named synthetic categories does not prove all remaining upstream data are human-authored or correctly labeled.
- SMS grouping masks changing numbers, email addresses and URLs, then uses 128-permutation MinHash candidates and exact word-bigram Jaccard >=0.85. This is approximate candidate retrieval, not an exhaustive proof against paraphrase leakage.
- Zero overlap across the saved split groups and normalized row hashes for both tasks. URL grouping uses the pinned offline Public Suffix List and includes private suffixes.

| Task | Train | Selection | Calibration | Internal test |
|---|---:|---:|---:|---:|
| URL | 164,446 | 23,528 | 23,712 | 23,208 |
| SMS | 53,612 | 7,743 | 7,785 | 7,545 |

Groups are assigned deterministically by a salted SHA-256 bucket. TF-IDF and classifier parameters are fitted only on training rows. Two predefined candidates are selected by selection F1, with Brier as a tie-breaker. Threshold is fixed at 0.5. Sigmoid calibration is fitted on a separate calibration partition and retained only if it improves selection Brier. Test metrics do not choose a model or threshold.

## Internal held-out results

These historical source-based holdouts are separated by group, but are not a new external source.

| Task | Rows | Accuracy | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| URL | 23,208 | 84.50% | 87.99% | 73.48% | 80.08% | 7.38% |
| SMS | 7,545 | 97.60% | 97.34% | 96.28% | 96.81% | 1.60% |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

- URL: `[[12380, 987], [2610, 7231]]`.
- SMS: `[[4619, 75], [106, 2745]]`.

For a before/after comparison, restrict further to common test rows whose retained groups were not seen by the legacy training/validation data. This yields 3,842 URL rows and 703 SMS rows. Legacy membership is reconstructed; semantic/template exclusion is not exhaustive.

| Common held-out comparison | Before | After |
|---|---:|---:|
| URL accuracy | 99.82% | 86.05% |
| URL F1 | 99.71% | 76.14% |
| URL false-positive rate | 0.00% | 7.09% |
| SMS accuracy | 97.16% | 97.44% |
| SMS F1 | 95.88% | 96.28% |
| SMS false-positive rate | 1.96% | 1.53% |

The URL decline on the old source distribution is real and is not omitted. Removing its shortcut improves behavior on the separate external source but sacrifices same-source performance and phishing recall. Do not compare the old contaminated 99.887% number directly to the new grouped result as if they used the same test design.

## External URL evaluation

Source: [Phishing Website Dataset, Zenodo 8041387](https://doi.org/10.5281/zenodo.8041387), published 2 July 2023, CC BY 4.0. Publisher labels distinguish phishing and non-phishing website snapshots. Source metadata and checksums are saved locally. Only the URL and date columns are used for evaluation; no pages or suspicious target URLs are visited.

The source has 10,395 rows. Four URLs fail strict canonicalization and four duplicate normalized URLs are removed, leaving 10,387 unique valid rows. There are 1,335 exact development URL overlaps; domain exclusion removes 3,660 rows in total. The retained set has **6,727 rows, 6,341 registrable domains, 4,270 benign labels and 2,457 phishing labels**, with zero remaining exact URL or registrable-domain overlap against available URL development data and audit probes.

The retained scan dates range from 8 December 2018 to 8 May 2023. This is a historical, source-disjoint, overlap-filtered benchmark; it is not a 2026 threat feed or proof of all upstream independence. Its collection and label noise can differ from the deployment population.

| Metric on exactly the same external rows | Legacy | Repaired |
|---|---:|---:|
| Accuracy | 42.49% | 65.08% |
| Precision | 38.78% | 51.74% |
| Recall | 99.27% | 65.53% |
| F1 | 55.77% | 57.82% |
| False-positive rate | 90.19% | 35.18% |
| Brier score (lower is better) | 0.5161 | 0.2439 |
| ROC-AUC | 0.6537 | 0.7195 |

- Legacy confusion matrix: `[[419, 3851], [18, 2439]]`.
- Repaired confusion matrix: `[[2768, 1502], [847, 1610]]`.

The replacement reduces the extreme false-alarm problem but misses more phishing cases. **35.18% false positives and 65.53% recall are insufficient for a trusted production detector.** The release is a defensible baseline for an internship demonstration, with explicit limitations.

The external data were downloaded and checksum-verified without using their examples in training, calibration or candidate selection. An opening receipt binds evaluation to the frozen release manifest. The evaluator refuses repeat use. Now that results are published, this dataset must not be called untouched in future tuning; acquire a new final benchmark for a future model-development cycle.

No independent SMS benchmark was acquired. The [Mendeley SMS phishing dataset](https://data.mendeley.com/datasets/f45bkkt8pr/1) was examined as a candidate, but its lineage may overlap the existing `Smishing_Dataset`/UCI-derived corpus; independence was not assumed. The [ISCX URL dataset](https://www.unb.ca/cic/datasets/url-2016.html) offers more representative URL paths, but its download endpoint requested registration details. No personal information was submitted.

## Calibration and language performance

SMS selection Brier improved from 0.017009 to 0.016542 with held-out sigmoid calibration; log loss improved from 0.066081 to 0.061928. Calibration was retained. Its final test Brier is 0.018875.

URL sigmoid calibration worsened selection Brier from 0.108460 to 0.108831 and was rejected. The UI explicitly says uncalibrated. No claim that these scores estimate real-world probability of correctness is made. Test calibration-bin summaries are in `evaluation.json`.

| SMS language | Test rows | Scam labels | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| English | 6,974 | 2,747 | 97.24% | 96.29% | 96.76% | 1.77% |
| Hindi | 456 | 6 | 100.00% | 33.33% | 50.00% | 0.00% |
| Hinglish | 115 | 98 | 100.00% | 100.00% | 100.00% | 0.00% |

Hindi detects only 2 of 6 scams. Hinglish's result is on a small, source-dependent sample and does not establish perfect general detection. Per-language and per-source confusion matrices are saved in the evaluation JSON. Representative independently labeled Hindi/Hinglish and transactional-message data remain necessary.

## Audit regressions

All 42 author-written audit probes remain diagnostics, not training data or independent final evidence.

- URL cases: 10/18 to 15/18 correct. All ten synthetic phishing-style structures remain flagged. Five of eight expected benign URLs now pass; GitHub, Google Accounts and Microsoft's security page remain false positives.
- SMS cases: 15/24 to 16/24 correct. Six benign transactional/security messages and two intended scam scenarios still fail.
- `github.com`, `github.com/`, and default-port/case variants now return exactly the same score. They still get a wrong phishing prediction (88.61); consistency is fixed, classification is not.
- Python's `urllib.parse` documentation URL changes from 100.00 phishing risk to 33.05, classified legitimate. This does not establish page safety.
- Invalid `not a url`, missing hosts, unsupported schemes, invalid ports and malformed bracketed authorities now return structured 422 errors and create no history rows.

Eleven unresolved classification cases are XFAIL tests with explicit reasons; they actually run inference. Do not count them as passes. The complete before/after probe outputs are in `audit_regressions.csv`.

## Verification

- **134 passed, 11 XFAIL**, with one existing Starlette/httpx deprecation warning.
- Frontend production build and ESLint pass. The existing large-bundle warning remains (about 927 kB uncompressed JavaScript).
- A second independent training run produced byte-identical URL and SMS `.joblib` SHA-256 hashes. It used the same fixed candidates, seeds and prepared data; no tuning followed external results.
- Direct inference matches live API predictions, probabilities, feature contributions and artifact hashes.
- Live invalid-input errors verified; updated UI verified in the browser.
- All **41 pre-existing PostgreSQL records** verified unchanged by an ordered-row SHA-256 snapshot. New diagnostic checks append ordinary history entries; no old records were deleted or updated.
- SQLite tests verify history insert/count/limit and response compatibility. PostgreSQL live checks verify the existing schema. No migration required.
- Camera hardware was not exercised; the existing QR feature was preserved.

## Artifacts and commands

- `models/active_release.json`: active evaluated release pointer.
- `models/releases/reliable-v1/manifest.json`: parameters, model and preprocessing hashes, validation/calibration evidence.
- `models/releases/reliable-v1/dataset_manifest.json`: compact source inventory, exclusion counts and split checks.
- `models/releases/reliable-v1/evaluation.json`: internal and common before/after metrics.
- `models/releases/reliable-v1/external_evaluation.json`: one-shot external comparison.
- `models/releases/reliable-v1/audit_regressions.csv`: all audit cases and outcomes.
- `audit/repair_live_verification.json`: API/artifact identity and history preservation.
- `audit/reproducibility_verification.json`: identical repeat-training artifact hashes.
- `requirements-reliable.lock`: exact verified Python environment.
- `README.md`: reproducible setup, training, evaluation and promotion commands.
- `docs/INTERVIEW_GUIDE.md`: beginner-friendly architecture and algorithms, two-minute explanation and follow-up questions.

Original legacy model files and original metrics remain for comparison. The active pointer and release manifest are authoritative; old top-level model manifests describe archived models. No destructive cleanup of original or untracked data occurred. Failed initial preparation output is preserved separately under `data/reliable/v1`; the complete authoritative preparation is `v2`.

## Remaining work that data limits prevent completing honestly

1. Full-URL generalization with representative, independently labeled development URLs containing real paths, login routes, filenames and queries. No synthetic URL mutation was silently labeled benign. The new external corpus stays evaluation-only as requested.
2. Better Hindi scam coverage and resolution of transactional-message false positives using verified labels and fresh source/time-separated data.
3. Independently verified SMS external testing, stronger paraphrase/source-lineage checks, uncertainty intervals and a more representative deployment prevalence assessment.
4. Production requirements such as access control, rate limits, operations monitoring and live reputation/content analysis are outside this repair and must not be implied by the demo.

These are genuine unresolved limitations. The repository now makes them measurable and explainable instead of masking them with inflated accuracy.

Final handoff verification on 15 October 2026: backend and frontend remained running; malformed bracketed hosts, empty ports and repeated raw @ characters returned HTTP 422 through the live API. The original 41-row history snapshot still matched. The final dashboard screenshot shows the documentation URL assessment, uncalibrated-score wording, scope limits and model evidence. No model tuning or repeat external scoring occurred during handoff.
