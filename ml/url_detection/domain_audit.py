from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.model_selection import GroupShuffleSplit

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = (
    ROOT
    / "data"
    / "raw"
    / "phiusiil.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "url_xgboost_v2.joblib"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 60)
print("ThreatGuard — Domain Generalization Audit")
print("=" * 60)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n===== LOADING DATA =====")

df = pd.read_csv(RAW_DATA)

print(
    "Original rows:",
    len(df),
)

print(
    "Unique URLs:",
    df["URL"].nunique(),
)


# ------------------------------------------------------------
# NORMALIZE DOMAIN GROUPS
# ------------------------------------------------------------

def normalize_domain(value: str) -> str:

    value = str(value).strip().lower()

    # Remove markdown-like wrappers that may occur
    # in dataset representations.
    value = re.sub(
        r"^\[",
        "",
        value,
    )

    value = re.sub(
        r"\]\(.*$",
        "",
        value,
    )

    value = re.sub(
        r"^https?://",
        "",
        value,
    )

    value = value.split("/")[0]

    value = value.split(":")[0]

    value = value.strip()

    if value.startswith("www."):
        value = value[4:]

    return value


df["domain_group"] = (
    df["Domain"]
    .fillna("")
    .astype(str)
    .apply(normalize_domain)
)

# Fall back to URL-derived grouping when Domain is empty.
empty_domain = (
    df["domain_group"]
    .str.len()
    .eq(0)
)

if empty_domain.any():

    fallback = (
        df.loc[
            empty_domain,
            "URL"
        ]
        .astype(str)
        .str.lower()
        .str.replace(
            r"^https?://",
            "",
            regex=True,
        )
        .str.split("/")
        .str[0]
        .str.split(":")
        .str[0]
    )

    df.loc[
        empty_domain,
        "domain_group"
    ] = fallback


print(
    "Unique domain groups:",
    df["domain_group"].nunique(),
)


# ------------------------------------------------------------
# REMOVE EXACT DUPLICATE URLS
# ------------------------------------------------------------

before = len(df)

df = (
    df
    .drop_duplicates(
        subset=["URL"],
        keep="first",
    )
    .reset_index(drop=True)
)

print(
    "Removed duplicate URLs:",
    before - len(df),
)


# ------------------------------------------------------------
# BUILD FEATURES
# ------------------------------------------------------------

print("\n===== BUILDING FEATURES =====")

X = extract_features_from_series(
    df["URL"]
)

X = X[FEATURE_NAMES]

y = (
    df["label"] == 0
).astype(int)

groups = df["domain_group"]


# ------------------------------------------------------------
# CHECK FEATURE SIGNATURE DUPLICATION
# ------------------------------------------------------------

print(
    "\n===== FEATURE SIGNATURE AUDIT ====="
)

feature_signature = (
    X.astype(str)
    .agg("|".join, axis=1)
)

signature_counts = (
    feature_signature
    .value_counts()
)

duplicate_signature_rows = int(
    (
        signature_counts
        .loc[
            signature_counts > 1
        ]
        .sum()
        - signature_counts[
            signature_counts > 1
        ].count()
    )
)

print(
    "Rows sharing duplicated feature signatures:",
    duplicate_signature_rows,
)

print(
    "Unique feature signatures:",
    feature_signature.nunique(),
)

print(
    "Total feature rows:",
    len(feature_signature),
)


# ------------------------------------------------------------
# GROUP SPLIT
# ------------------------------------------------------------

print("\n===== DOMAIN-GROUPED SPLIT =====")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

test_domains = groups.iloc[test_idx]

print(
    "Training samples:",
    len(train_idx),
)

print(
    "Unseen-domain test samples:",
    len(test_idx),
)

print(
    "Training domains:",
    groups.iloc[train_idx].nunique(),
)

print(
    "Test domains:",
    test_domains.nunique(),
)

overlap = (
    set(
        groups.iloc[train_idx]
    )
    &
    set(test_domains)
)

print(
    "Domain overlap:",
    len(overlap),
)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n===== LOADING V2 MODEL =====")

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded:",
    MODEL_PATH,
)


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# THRESHOLD AUDIT
# ------------------------------------------------------------

print(
    "\n===== UNSEEN-DOMAIN PERFORMANCE ====="
)

threshold_results = []

for threshold in [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    result = {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    threshold_results.append(
        result
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accuracy : {result['accuracy']:.4f}"
    )

    print(
        f"Precision: {result['precision']:.4f}"
    )

    print(
        f"Recall   : {result['recall']:.4f}"
    )

    print(
        f"F1       : {result['f1']:.4f}"
    )

    print(
        "Confusion:",
        [[tn, fp], [fn, tp]],
    )


# ------------------------------------------------------------
# AUC
# ------------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)

print(
    "\nROC-AUC:",
    f"{roc_auc:.4f}",
)

print(
    "PR-AUC:",
    f"{pr_auc:.4f}",
)


# ------------------------------------------------------------
# MANUAL SANITY TEST
# ------------------------------------------------------------

print(
    "\n===== MANUAL SANITY TEST ====="
)

manual_urls = [
    "https://www.google.com",
    "https://www.amazon.com",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.wikipedia.org",
    "https://example.com",
    "https://paypal.com/login",
    "https://www.paypal.com/login",
    "http://192.168.1.10/login",
    "http://secure-account-verification.example.com/login",
    "http://paypal-login-security.example.com",
    "http://account-verification-payment.example.com/login",
    "http://free-gift-card-claim.example.com",
]

manual_X = extract_features_from_series(
    pd.Series(manual_urls)
)

manual_X = manual_X[
    FEATURE_NAMES
]

manual_probabilities = (
    model.predict_proba(
        manual_X
    )[:, 1]
)

manual = pd.DataFrame(
    {
        "url": manual_urls,
        "phishing_probability":
            manual_probabilities,
        "risk_score":
            manual_probabilities * 100,
    }
)

manual["classification"] = (
    manual[
        "phishing_probability"
    ]
    >= 0.50
).map(
    {
        True: "PHISHING",
        False: "LEGITIMATE",
    }
)

print(
    manual.to_string(
        index=False
    )
)

manual.to_csv(
    OUTPUT_DIR
    / "domain_audit_manual_tests.csv",
    index=False,
)


# ------------------------------------------------------------
# DOMAIN-LEVEL ERROR ANALYSIS
# ------------------------------------------------------------

predictions = (
    probabilities >= 0.50
).astype(int)

errors = df.iloc[
    test_idx
].copy()

errors[
    "phishing_probability"
] = probabilities

errors[
    "prediction"
] = predictions

errors[
    "correct"
] = (
    predictions == y_test.to_numpy()
)

false_positives = errors[
    (errors["label"] == 1)
    &
    (errors["prediction"] == 1)
]

false_negatives = errors[
    (errors["label"] == 0)
    &
    (errors["prediction"] == 0)
]

false_positives[
    [
        "URL",
        "Domain",
        "label",
        "phishing_probability",
    ]
].to_csv(
    OUTPUT_DIR
    / "domain_false_positives.csv",
    index=False,
)

false_negatives[
    [
        "URL",
        "Domain",
        "label",
        "phishing_probability",
    ]
].to_csv(
    OUTPUT_DIR
    / "domain_false_negatives.csv",
    index=False,
)

print(
    "\nFalse positives:",
    len(false_positives),
)

print(
    "False negatives:",
    len(false_negatives),
)


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary = {
    "audit": "domain_grouped_generalization",
    "dataset_rows_after_url_dedup": int(len(df)),
    "unique_domains": int(
        df["domain_group"].nunique()
    ),
    "training_samples": int(
        len(train_idx)
    ),
    "test_samples": int(
        len(test_idx)
    ),
    "training_domains": int(
        groups.iloc[train_idx].nunique()
    ),
    "test_domains": int(
        test_domains.nunique()
    ),
    "domain_overlap": int(
        len(overlap)
    ),
    "duplicated_feature_signature_rows": int(
        duplicate_signature_rows
    ),
    "unique_feature_signatures": int(
        feature_signature.nunique()
    ),
    "roc_auc": float(
        roc_auc
    ),
    "pr_auc": float(
        pr_auc
    ),
    "threshold_results":
        threshold_results,
    "model":
        str(MODEL_PATH),
}

with open(
    OUTPUT_DIR
    / "domain_audit_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\n===== SAVED ====="
)

print(
    OUTPUT_DIR
    / "domain_audit_summary.json"
)

print(
    OUTPUT_DIR
    / "domain_audit_manual_tests.csv"
)

print(
    OUTPUT_DIR
    / "domain_false_positives.csv"
)

print(
    OUTPUT_DIR
    / "domain_false_negatives.csv"
)

print(
    "\n=========================================="
)

print(
    "STEP 9 AUDIT COMPLETE"
)

print(
    "=========================================="
)
