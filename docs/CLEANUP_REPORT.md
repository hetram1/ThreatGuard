# Cleanup report — 15 October 2026

Allocated file bytes before: 4,652,339,200; after verification: 4,465,872,896; saved: 186,466,304 bytes.
Source lines excluding generated files: 16,300 to 7,762.

Removed targets and exact sizes: audit/cleanup_deleted.json. Removed failed duplicate preparation, repeated training output, obsolete experiments/intermediate feature CSVs, old prediction exports, unused assets and one-time patch utilities. Exact duplicate historical vectorizers now use hash-preserving relative symlinks. Original raw/SMS data, active model bundles, baseline checkpoints, split metadata, audit evidence, all regression cases and application history are preserved.

Refactoring consolidated API inference/persistence/error handling and heuristic signals, removed unused imports/debug demonstrations, and extracted lazy QR/charts. The initial JavaScript bundle fell from 927.52 kB to 239.25 kB; chart and QR chunks are 316.52 and 372.11 kB and load separately. Frontend build/lint pass; backend/ML tests: 134 passed, 11 documented expected classification failures.

The two exact compressed model bundles total 5,321,583 bytes and are distributed in Git with verified hashes. Full split-data verification requires local datasets and skips explicitly on a fresh clone. Source acquisition and snapshot limitations are documented in README.md. Model/preprocessing bytes and reported benchmarks are unchanged.

Rollback branch: codex/pre-cleanup-20261015. Recovery bundle/source archive: /tmp/threatguard-cleanup-20261015 (about 99 MB outside the project). No history rewrite, force push, pruning, database deletion or dataset deletion occurred. Environments remain installed; Git object repacking and dependency removal were deferred to meet the requested quick finish. The original upstream README is archived in docs/history/README-before-repair.md.
