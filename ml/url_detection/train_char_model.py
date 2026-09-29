from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = ROOT / "data" / "raw" / "phiusiil.csv"

MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "url_char_logreg.joblib"

VECTORIZER_PATH = MODEL_DIR / "url_char_vectorizer.joblib"

SCALER_PATH = MODEL_DIR / "url_char_scaler.joblib"

METRICS_PATH = MODEL_DIR / "url_char_model_metrics.json"


print("=" * 60)
print("ThreatGuard — Character-Level URL Model")
print("=" * 60)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n===== LOADING DATA =====")

df = pd.read_csv(RAW_DATA)

df = (
    df
    .drop_duplicates(
        subset=["URL"],
        keep="first",
    )
    .reset_index(drop=True)
)

urls = (
    df["URL"]
    .fillna("")
    .astype(str)
)

# UCI label:
# 0 = phishing
# 1 = legitimate
#
# ThreatGuard convention:
# 0 = legitimate
# 1 = phishing

y = (
    df["label"] == 0
).astype(int)


print(
    "Rows:",
    len(df),
)

print(
    "Phishing:",
    int(y.sum()),
)

print(
    "Legitimate:",
    int((y == 0).sum()),
)


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


train_urls = urls.iloc[train_idx]

test_urls = urls.iloc[test_idx]

y_train = y.iloc[train_idx]

y_test = y.iloc[test_idx]


print(
    "\nTraining:",
    len(train_idx),
)

print(
    "Test:",
    len(test_idx),
)


# ------------------------------------------------------------
# CHARACTER TF-IDF
# ------------------------------------------------------------

print(
    "\n===== CHARACTER TF-IDF ====="
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=120000,
    sublinear_tf=True,
    lowercase=True,
)

X_train_char = vectorizer.fit_transform(
    train_urls
)

X_test_char = vectorizer.transform(
    test_urls
)

print(
    "Character vocabulary:",
    len(vectorizer.vocabulary_),
)

print(
    "Train character matrix:",
    X_train_char.shape,
)

print(
    "Test character matrix:",
    X_test_char.shape,
)


# ------------------------------------------------------------
# ENGINEERED FEATURES
# ------------------------------------------------------------

print(
    "\n===== ENGINEERED FEATURES ====="
)

X_all_engineered = extract_features_from_series(
    urls
)

X_all_engineered = X_all_engineered[
    FEATURE_NAMES
]

X_train_engineered = (
    X_all_engineered.iloc[train_idx]
)

X_test_engineered = (
    X_all_engineered.iloc[test_idx]
)


# ------------------------------------------------------------
# SCALE ENGINEERED FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_engineered
)

X_test_scaled = scaler.transform(
    X_test_engineered
)

X_train_scaled = csr_matrix(
    X_train_scaled
)

X_test_scaled = csr_matrix(
    X_test_scaled
)


# ------------------------------------------------------------
# COMBINE FEATURES
# ------------------------------------------------------------

print(
    "\n===== COMBINING FEATURES ====="
)

X_train = hstack(
    [
        X_train_char,
        X_train_scaled,
    ],
    format="csr",
)

X_test = hstack(
    [
        X_test_char,
        X_test_scaled,
    ],
    format="csr",
)

print(
    "Combined train matrix:",
    X_train.shape,
)

print(
    "Combined test matrix:",
    X_test.shape,
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print(
    "\n===== TRAINING LOGISTIC REGRESSION ====="
)

model = LogisticRegression(
    C=4.0,
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)

cm = confusion_matrix(
    y_test,
    predictions,
)


print(
    "\n===== TEST RESULTS ====="
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)


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
    "https://www.python.org",
    "https://www.mozilla.org",
    "https://www.amazon.in",
    "https://www.flipkart.com",
    "https://www.gov.in",
    "https://uidai.gov.in",
    "https://www.rbi.org.in",

    "https://paypal.com/login",
    "https://www.paypal.com/login",
    "http://192.168.1.10/login",
    "http://secure-account-verification.example.com/login",
    "http://paypal-login-security.example.com",
    "http://account-verification-payment.example.com/login",
    "http://free-gift-card-claim.example.com",
    "http://google-security-login.example.com",
    "http://microsoft-account-verify.example.com",
    "http://paypal-security-check.example.com",
    "http://secure-login-update.example.com",
    "http://192.168.0.20/verify/account",
]


manual_urls_series = pd.Series(
    manual_urls
)

manual_char = vectorizer.transform(
    manual_urls_series
)

manual_engineered = extract_features_from_series(
    manual_urls_series
)

manual_engineered = manual_engineered[
    FEATURE_NAMES
]

manual_scaled = scaler.transform(
    manual_engineered
)

manual_scaled = csr_matrix(
    manual_scaled
)

manual_X = hstack(
    [
        manual_char,
        manual_scaled,
    ],
    format="csr",
)

manual_probabilities = (
    model.predict_proba(
        manual_X
    )[:, 1]
)

manual_predictions = (
    manual_probabilities >= 0.50
)

manual_result = pd.DataFrame(
    {
        "url": manual_urls,
        "phishing_probability":
            manual_probabilities,
        "risk_score":
            manual_probabilities * 100,
        "classification": np.where(
            manual_predictions,
            "PHISHING",
            "LEGITIMATE",
        ),
    }
)

print(
    manual_result.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SAVE MANUAL TESTS
# ------------------------------------------------------------

manual_result.to_csv(
    MODEL_DIR
    / "audit"
    / "char_model_manual_tests.csv",
    index=False,
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH,
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH,
)

joblib.dump(
    scaler,
    SCALER_PATH,
)


# ------------------------------------------------------------
# SAVE METRICS
# ------------------------------------------------------------

metrics = {
    "model": "character_tfidf_logistic_regression",
    "dataset_rows": int(len(df)),
    "train_samples": int(len(train_idx)),
    "test_samples": int(len(test_idx)),
    "character_ngram_range": [3, 5],
    "character_max_features": 120000,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "confusion_matrix": cm.tolist(),
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


print(
    "\n===== SAVED ====="
)

print(
    MODEL_PATH
)

print(
    VECTORIZER_PATH
)

print(
    SCALER_PATH
)

print(
    METRICS_PATH
)

print(
    "\n=========================================="
)

print(
    "STEP 9.2 COMPLETE"
)

print(
    "=========================================="
)
