from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"
AUDIT_DIR = MODEL_DIR / "audit"


# ============================================================
# CURATED SANITY DATASET
# ============================================================

# IMPORTANT:
# This is NOT used for training.
# It is strictly a developer sanity benchmark.

cases = [

    # --------------------------------------------------------
    # BENIGN
    # --------------------------------------------------------

    ("https://www.google.com", 0, "benign"),
    ("https://www.amazon.com", 0, "benign"),
    ("https://github.com", 0, "benign"),
    ("https://www.microsoft.com", 0, "benign"),
    ("https://www.apple.com", 0, "benign"),
    ("https://www.wikipedia.org", 0, "benign"),
    ("https://example.com", 0, "benign"),
    ("https://www.python.org", 0, "benign"),
    ("https://www.mozilla.org", 0, "benign"),
    ("https://www.amazon.in", 0, "benign"),
    ("https://www.flipkart.com", 0, "benign"),
    ("https://www.gov.in", 0, "benign"),
    ("https://uidai.gov.in", 0, "benign"),
    ("https://www.rbi.org.in", 0, "benign"),
    ("https://www.nasa.gov", 0, "benign"),
    ("https://www.github.com", 0, "benign"),
    ("https://stackoverflow.com", 0, "benign"),
    ("https://docs.python.org/3/", 0, "benign"),
    ("https://developer.mozilla.org/en-US/", 0, "benign"),
    ("https://www.wikipedia.org/wiki/India", 0, "benign"),

    # --------------------------------------------------------
    # SUSPICIOUS
    # --------------------------------------------------------

    ("http://192.168.1.10/login", 1, "suspicious"),
    ("http://192.168.0.20/verify/account", 1, "suspicious"),
    ("http://secure-account-verification.example.com/login", 1, "suspicious"),
    ("http://paypal-login-security.example.com", 1, "suspicious"),
    ("http://account-verification-payment.example.com/login", 1, "suspicious"),
    ("http://free-gift-card-claim.example.com", 1, "suspicious"),
    ("http://google-security-login.example.com", 1, "suspicious"),
    ("http://microsoft-account-verify.example.com", 1, "suspicious"),
    ("http://paypal-security-check.example.com", 1, "suspicious"),
    ("http://secure-login-update.example.com", 1, "suspicious"),

    ("http://paypa1-secure-login.example.com", 1, "suspicious"),
    ("http://paypal.verify-account.example.com/login", 1, "suspicious"),
    ("http://google.account-security.example.com/login", 1, "suspicious"),
    ("http://microsoft.verify-user.example.com/auth", 1, "suspicious"),
    ("http://bank-account-confirm.example.com/update", 1, "suspicious"),
    ("http://wallet-security-check.example.com", 1, "suspicious"),
    ("http://login-session-expired.example.com/verify", 1, "suspicious"),
    ("http://secure-payment-confirmation.example.com", 1, "suspicious"),
    ("http://account-locked-verification.example.com", 1, "suspicious"),
    ("http://urgent-security-alert.example.com/login", 1, "suspicious"),
]


df = pd.DataFrame(
    cases,
    columns=[
        "url",
        "expected",
        "category",
    ],
)


# ============================================================
# MODEL 1 — XGBOOST V2
# ============================================================

print("\n===== XGBOOST V2 =====")

xgb_model = joblib.load(
    MODEL_DIR / "url_xgboost_v2.joblib"
)

X_xgb = extract_features_from_series(
    df["url"]
)

X_xgb = X_xgb[
    FEATURE_NAMES
]

xgb_prob = (
    xgb_model
    .predict_proba(X_xgb)[:, 1]
)

xgb_pred = (
    xgb_prob >= 0.50
).astype(int)


# ============================================================
# MODEL 2 — CHARACTER TF-IDF
# ============================================================

print("\n===== CHARACTER TF-IDF =====")

char_model = joblib.load(
    MODEL_DIR / "url_char_logreg.joblib"
)

vectorizer = joblib.load(
    MODEL_DIR
    / "url_char_vectorizer.joblib"
)

scaler = joblib.load(
    MODEL_DIR
    / "url_char_scaler.joblib"
)

X_char = vectorizer.transform(
    df["url"]
)

X_engineered = extract_features_from_series(
    df["url"]
)

X_engineered = X_engineered[
    FEATURE_NAMES
]

X_scaled = scaler.transform(
    X_engineered
)

from scipy.sparse import (
    hstack,
    csr_matrix,
)

X_char_combined = hstack(
    [
        X_char,
        csr_matrix(X_scaled),
    ],
    format="csr",
)

char_prob = (
    char_model
    .predict_proba(
        X_char_combined
    )[:, 1]
)

char_pred = (
    char_prob >= 0.50
).astype(int)


# ============================================================
# RESULT TABLE
# ============================================================

result = df.copy()

result["xgb_probability"] = xgb_prob
result["xgb_prediction"] = xgb_pred
result["xgb_correct"] = (
    xgb_pred == df["expected"].to_numpy()
)

result["char_probability"] = char_prob
result["char_prediction"] = char_pred
result["char_correct"] = (
    char_pred == df["expected"].to_numpy()
)


# ============================================================
# PRINT ALL
# ============================================================

pd.set_option(
    "display.max_rows",
    100,
)

print(
    "\n===== COMPLETE SANITY RESULTS =====\n"
)

print(
    result[
        [
            "url",
            "category",
            "expected",
            "xgb_probability",
            "xgb_prediction",
            "xgb_correct",
            "char_probability",
            "char_prediction",
            "char_correct",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true,
    pred,
):
    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "confusion_matrix":
            cm.tolist(),
    }


y_true = df["expected"].to_numpy()

xgb_metrics = metrics(
    y_true,
    xgb_pred,
)

char_metrics = metrics(
    y_true,
    char_pred,
)


print(
    "\n===== XGBOOST SANITY METRICS ====="
)

print(
    json.dumps(
        xgb_metrics,
        indent=2,
    )
)

print(
    "\n===== CHARACTER MODEL SANITY METRICS ====="
)

print(
    json.dumps(
        char_metrics,
        indent=2,
    )
)


# ============================================================
# CATEGORY METRICS
# ============================================================

print(
    "\n===== CATEGORY BREAKDOWN ====="
)

for category in [
    "benign",
    "suspicious",
]:

    subset = (
        result["category"]
        == category
    )

    xgb_correct = int(
        result.loc[
            subset,
            "xgb_correct"
        ].sum()
    )

    char_correct = int(
        result.loc[
            subset,
            "char_correct"
        ].sum()
    )

    total = int(
        subset.sum()
    )

    print(
        f"\n{category.upper()}:"
    )

    print(
        f"XGBoost  : {xgb_correct}/{total}"
    )

    print(
        f"Char TFIDF: {char_correct}/{total}"
    )


# ============================================================
# SAVE
# ============================================================

output_csv = (
    AUDIT_DIR
    / "model_selection_sanity.csv"
)

result.to_csv(
    output_csv,
    index=False,
)


summary = {
    "benchmark": (
        "curated developer sanity set"
    ),
    "total_cases": int(len(df)),
    "benign_cases": int(
        (df["category"] == "benign").sum()
    ),
    "suspicious_cases": int(
        (df["category"] == "suspicious").sum()
    ),
    "xgboost": xgb_metrics,
    "character_tfidf": char_metrics,
}


with open(
    AUDIT_DIR
    / "model_selection_summary.json",
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
    output_csv
)

print(
    AUDIT_DIR
    / "model_selection_summary.json"
)

print(
    "\n=========================================="
)

print(
    "STEP 9.3 COMPLETE"
)

print(
    "=========================================="
)
