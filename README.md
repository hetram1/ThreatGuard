# ThreatGuard: reproducible ML demonstration

ThreatGuard detects lexical phishing-domain and scam-message patterns with a React dashboard, FastAPI service and PostgreSQL history. It is a research demonstration, not a guarantee that a website or message is safe.

The 14 October 2026 repair replaces the contaminated URL evaluation with saved domain/template-grouped splits, verified model releases and explicit limitations. Read [the repair report](docs/ML_REPAIR_REPORT.md) and [the interview guide](docs/INTERVIEW_GUIDE.md).

## Run the application

Clone https://github.com/hetram1/ThreatGuard first, then use WSL/Linux from the repository root. The active model bundles and checksum manifests are included; source datasets are needed only for retraining and full split-membership verification. The verified environment is Python 3.14.4 and Node 20.20.2. Exact Python package versions are in `requirements-reliable.lock`; frontend versions are in `frontend/package-lock.json`.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-reliable.lock
docker compose up -d postgres
.venv/bin/python -m backend.app.database.init_db
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```sh
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Dashboard: http://localhost:5173 ; API documentation: http://localhost:8000/docs . Set `DATABASE_URL` and `VITE_API_BASE_URL` when needed. No database migration is required; historical rows retain their original model version and scores.

## Reproduce training and evaluation

The two compressed verified model bundles (5.3 MB total) are included in Git under models/releases/reliable-v1, so cloning supplies the exact inference artifacts. Raw datasets and historical checkpoints remain local and are not published. Full training requires the original source datasets. On another machine obtain the cited source datasets, retain their original labels, and verify their SHA-256 hashes against the saved preparation manifest before claiming an exact reproduction.

```sh
# A new output directory is mandatory: existing preparations/releases are never overwritten.
.venv/bin/python -m ml.reliable.prepare --output data/reliable/my-run
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m ml.reliable.train --data data/reliable/my-run --output models/releases/my-run
.venv/bin/python -m ml.reliable.evaluate --release models/releases/my-run
.venv/bin/python -m ml.reliable.promote --release models/releases/my-run
```

Restart the backend after promotion. Preparation saves original membership, source hashes, cleaned rows, groups, split membership and excluded-row counts. Training uses train/selection/calibration partitions; evaluation uses the separate internal test partition. The old V5 unseen-domain evaluator is retired and exits with an explanation.

External URL data is strictly evaluation-only. The existing snapshot has already been opened for the final evaluation of `reliable-v1`; it is now a published benchmark, not an untouched test for future model development. Do not use it to select improvements while continuing to call it independent. Acquire a new approved source/version for future final testing. The original one-time workflow is:

```sh
.venv/bin/python -m ml.reliable.fetch_external
# Only after freezing all development and model selection:
.venv/bin/python -m ml.reliable.evaluate_external --release models/releases/reliable-v1
```

No separately verified independent SMS test set was acquired. Domain exclusion and template clustering reduce leakage; they cannot prove that every semantic paraphrase or shared upstream source has been removed.

## Tests

```sh
.venv/bin/python -m pytest -q
cd frontend
npm run build
npm run lint
```

API tests always use isolated in-memory SQLite and do not delete PostgreSQL history. Known classification failures are explicit XFAIL cases, distinct from passing software tests. All audit probes are regression cases, never training labels or independent final-test evidence.

## Important limits

- The current URL model examines the hostname only. Paths, page content, redirects and live reputation are not scored. This removes a demonstrated path/source shortcut but sacrifices path-specific detection. It is not a domain allowlist.
- SMS uses supplied source labels after excluding explicitly synthetic, benchmark and spam-only source categories. Upstream label quality is not independently guaranteed.
- Hindi scam coverage is inadequate. Some everyday transactional messages still cause false alarms.
- Risk estimates are not certainty. Feature contributions describe the classifier; heuristic indicators are shown separately.

See the report for both improvements and regressions, dataset sizes, confusion matrices, calibration and external results.

## Cleanup and historical reproduction

The frozen model.py and preprocessing.py files retain their exact original bytes. Shared heuristic signals are separated from historical model loading; analysis routes share one persistence/error handler. QR scanning and charts load as separate frontend chunks. Superseded experiments were removed after reference review and backup; their original source remains in Git history. Historical V5 URL and original SMS training scripts, required V2 URL features, dataset acquisition/builders, baseline model artifacts, audit metadata and all current regression cases are retained. An archived upstream README is preserved in docs/history/README-before-repair.md.

On a fresh clone, inference and regression tests run using the included models. The one full source-dataset membership test skips explicitly until the local data/reliable/v2 files are restored or regenerated from the original source snapshots. To compare a new evaluation against historical models, pass --compare-legacy; default evaluation does not require those optional local binaries. Follow dataset SHA-256 values in models/releases/reliable-v1/dataset_manifest.json rather than assuming a current upstream download is byte-identical. The original UCI acquisition entry point is ml/url_detection/download_dataset.py; the SMS source is https://huggingface.co/datasets/sidzzz07/scamshield-dataset and Majestic data came from https://majestic.com/reports/majestic-million. The original SMS revision was not captured, so exact source recovery cannot be guaranteed from a new download alone.
