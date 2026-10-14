# Understanding ThreatGuard and explaining it in an interview

## The complete journey

1. A person enters a URL or message, or extracts a URL from a QR code in React.
2. The browser sends JSON to FastAPI. Pydantic validates it. Unsupported schemes, malformed hostnames, whitespace inside URLs and invalid ports are rejected before model inference or history insertion.
3. Shared preprocessing runs in both training and prediction. URL schemes/hosts are normalized, internationalized domains become IDNA ASCII, default ports disappear, and an empty root path and `/` share one representation. Meaningful path/query characters are preserved in the normalized input. SMS uses Unicode NFKC and whitespace normalization while keeping Hindi characters.
4. The URL classifier uses the hostname; the SMS classifier uses message text. TF-IDF turns character fragments (3–5 characters) and, for SMS, words/word pairs into sparse numbers. Frequent fragments across many documents receive less distinguishing weight. The vocabulary and weights are fitted only on training rows.
5. Logistic regression learns a weighted sum plus an intercept. A sigmoid converts that score into a model probability. This is a transparent, efficient baseline for sparse text features; it is not a large language model.
6. The selected SMS model has a second sigmoid fitted on a separate calibration partition. Calibration tries to make scores better match observed frequencies. It cannot correct all label errors or distribution shifts. URL calibration was rejected because it worsened validation Brier score.
7. A fixed 0.5 threshold determines the stored class. The dashboard shows the score, scope limits, raw-model feature contributions and separate heuristic indicators. It does not call max(p, 1-p) “confidence.”
8. SQLAlchemy writes the analysis to PostgreSQL. Existing schemas and historical records are preserved. Historical scores are not silently recomputed using new models.

## What “unseen” means here

A new row is not necessarily new information. Two URLs on the same site, or the same SMS with a changed OTP, can leak across an ordinary random split. URL groups use the registrable domain from a pinned offline Public Suffix List, including private hosting suffixes. SMS groups mask changing numbers, links and email addresses, then connect highly similar word-bigram templates using MinHash candidate search and exact Jaccard checks. Whole connected groups get one split.

The split roles are training (about 70%), model selection (10%), calibration (10%) and internal test (10%). Percentages apply to hashed groups, so exact row counts vary. Two predetermined logistic-regression candidates are compared on selection F1, with Brier score breaking ties. The internal test never chooses hyperparameters or thresholds.

The external URL dataset comes from a different published research source and is filtered to remove URLs and registrable domains found anywhere in development, including audit examples. Its labels describe historical observations, not today's web. No external SMS performance claim is made.

## Reading the metrics

- Precision: of the items flagged as threats, how many truly had threat labels?
- Recall: of the labeled threats, how many did we detect?
- F1: harmonic mean balancing precision and recall.
- False-positive rate: false alarms divided by all labeled benign examples.
- Confusion matrix: `[[true negatives, false positives], [false negatives, true positives]]`.
- Brier score: average squared difference between predicted probabilities and binary labels; lower is better.
- Calibration bins: among items near a particular predicted probability, what proportion actually had the threat label?
- Accuracy alone can hide a weak minority class. For example, many benign Hindi messages can make accuracy look good even when most Hindi scams are missed.

## Why the repaired URL model is narrower

The original benign URL data had only root URLs. Its artificial path augmentation used four short paths. Features like path length and dot count became collection-pattern shortcuts. They gave almost perfect same-source scores but classified ordinary documentation paths as phishing. Using only hostname character features removes that particular shortcut without inventing labels or allowlisting famous sites. The cost is real: compromised pages and malicious paths on otherwise benign hosts cannot be detected by this model. Some well-known hosts still trigger false positives. Better, independently labeled full URLs are needed before claiming full-URL robustness.

## Reproducibility and integrity

A preparation manifest contains input hashes and row/group/split membership. A release manifest contains parameters, library versions, calibration choice, model hashes and preprocessing-code hashes. The active pointer identifies one evaluated release. Loading fails if the expected manifest, model or inference-code hashes differ. Hashes detect accidental changes; they are not cryptographic signatures proving that an untrusted model is safe. Only load trusted local artifacts.

The earlier URL split is reconstructed from recorded code and current hashed inputs; its historical execution cannot be retroactively proven. The new pipeline saves membership at the time it runs.

## Two-minute explanation

“ThreatGuard is my end-to-end machine-learning cybersecurity demonstration. It analyzes URLs and SMS messages through a React dashboard and FastAPI backend, and saves analysis history in PostgreSQL.

The most important part of the project was learning that high accuracy can be misleading. My original URL model scored extremely well, but an audit found training examples inside the supposed unseen-domain test. It also classified a Python documentation page as phishing, and adding a slash to GitHub changed the prediction.

I repaired the evaluation first. I now save dataset hashes and actual split membership. URLs are grouped by registrable domain, and SMS messages by normalized and near-duplicate templates. Training, selection, calibration and test data have separate roles. I also preserve an external URL benchmark exclusively for final evaluation and remove development-domain overlap before scoring it.

The models use TF-IDF and logistic regression. I kept them deliberately understandable. For URLs, the available labels had a strong path-collection bias, so the new model is explicitly hostname-only. For SMS, I removed clearly synthetic, benchmark and spam-only sources, retrained the text classifier, and selected sigmoid calibration using validation data. I added strict URL validation and show model feature contributions separately from heuristic warnings.

The application verifies model hashes, keeps history intact, and has automated software and regression tests. I report both improvements and unresolved failures. Hindi scam recall and some false positives remain weaknesses, and I would need better labeled data before claiming production-level reliability. The main result is a reproducible project whose limitations and evaluation I can defend.”

## Likely follow-up questions

**Why logistic regression instead of deep learning?** Sparse TF-IDF features work well with a linear classifier, train quickly, require modest hardware and expose feature contributions. A transformer is a future comparison, not a substitute for trustworthy data splits.

**What caused leakage?** The old evaluator loaded a model already trained on a random row split, then invented a separate domain split. Many supposed test rows had already been used in training.

**Why not just allowlist GitHub or Google?** That hides a symptom and can miss compromised content or hosted phishing. A legitimate parent platform does not establish the safety of every page.

**Why not train on the audit failures?** They are developer diagnostics already used to understand failure modes. Adding their labels would make that set unsuitable as independent evidence. This repair does not use them to fit the models.

**How do you avoid preprocessing mismatch?** Both paths import the same preprocessing functions, model bundles contain fitted TF-IDF objects, and release hashes cover the inference code.

**Does calibration improve accuracy?** Not necessarily. It targets probability reliability. I compare Brier/log loss and classification metrics separately, using validation for selection.

**Is a 90% score a 90% chance of being malicious?** Not reliably for arbitrary real-world inputs. It is a model estimate tied to the evaluated distribution; shifts in language, source or prevalence can invalidate that interpretation.

**Why are some tests expected to fail?** They record unresolved ML classification limitations. Software contract tests pass separately. The report counts these failures explicitly rather than silently removing hard examples.

**What would you do next?** Collect properly licensed, independently labeled real URL paths and more Hindi/Hinglish transactional/scam messages; audit label quality; use source/time-disjoint evaluation; improve duplicate grouping; compare models on validation data; then test once on a fresh external collection.
