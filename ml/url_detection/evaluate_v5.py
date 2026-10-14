"""The old V5 evaluator leaked training rows into a newly invented test split.

Use the versioned pipeline. Archived V5 metrics are historical and invalid for
unseen-domain claims. This entry point intentionally cannot regenerate them.
"""
raise SystemExit('Retired: V5 unseen-domain evaluation was contaminated. Run python -m ml.reliable.evaluate --release models/releases/<release> after grouped training.')
