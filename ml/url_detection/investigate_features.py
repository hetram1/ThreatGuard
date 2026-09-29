from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "url_xgboost_v2.joblib"

OUTPUT = ROOT / "models" / "audit" / "sanity_feature_analysis.csv"


# ------------------------------------------------------------
# MANUAL URL SET
# ------------------------------------------------------------

urls = [
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

    # Additional realistic benign examples
    "https://www.python.org",
    "https://www.mozilla.org",
    "https://www.amazon.in",
    "https://www.flipkart.com",
    "https://www.gov.in",
    "https://uidai.gov.in",
    "https://www.rbi.org.in",

    # Additional suspicious patterns
    "http://google-security-login.example.com",
    "http://microsoft-account-verify.example.com",
    "http://paypal-security-check.example.com",
    "http://secure-login-update.example.com",
    "http://192.168.0.20/verify/account",
]


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

X = extract_features_from_series(
    pd.Series(urls)
)

X = X[FEATURE_NAMES]


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = joblib.load(MODEL_PATH)

probabilities = model.predict_proba(X)[:, 1]

result = X.copy()

result.insert(
    0,
    "url",
    urls,
)

result["phishing_probability"] = probabilities

result["risk_score"] = probabilities * 100

result["classification"] = np.where(
    probabilities >= 0.50,
    "PHISHING",
    "LEGITIMATE",
)


# ------------------------------------------------------------
# PRINT PREDICTIONS
# ------------------------------------------------------------

print("\n===== PREDICTIONS =====")

print(
    result[
        [
            "url",
            "phishing_probability",
            "risk_score",
            "classification",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# PRINT IMPORTANT FEATURES
# ------------------------------------------------------------

important_features = [
    "url_length",
    "domain_length",
    "path_length",
    "num_digits",
    "num_special_chars",
    "num_dots",
    "num_hyphens",
    "num_slashes",
    "num_question_marks",
    "num_ampersands",
    "num_equals",
    "subdomain_count",
    "host_token_count",
    "path_token_count",
    "has_ip_address",
    "has_port",
    "has_userinfo",
    "encoded_char_count",
    "has_hex_encoding",
    "url_entropy",
    "domain_entropy",
    "suspicious_keyword_count",
    "has_suspicious_keyword",
    "repeated_separator_count",
    "has_double_slash_path",
    "hyphenated_host",
    "numeric_host",
    "very_long_host",
    "very_long_url",
    "excessive_subdomains",
    "suspicious_tld",
    "longest_host_token",
    "longest_path_token",
]


print("\n===== IMPORTANT FEATURE VALUES =====")

display_columns = [
    "url"
] + [
    f
    for f in important_features
    if f in result.columns
]

print(
    result[
        display_columns
    ].to_string(index=False)
)


# ------------------------------------------------------------
# MODEL FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n===== MODEL FEATURE IMPORTANCE =====")

importance = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    importance.head(20).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# EXPLANATION:
# COMPARE BENIGN VS SUSPICIOUS
# ------------------------------------------------------------

print(
    "\n===== BENIGN / SUSPICIOUS FEATURE COMPARISON ====="
)

benign = result[
    result["classification"]
    == "LEGITIMATE"
]

suspicious = result[
    result["classification"]
    == "PHISHING"
]

comparison_rows = []

for feature in important_features:

    if feature not in result.columns:
        continue

    comparison_rows.append(
        {
            "feature": feature,
            "benign_mean": benign[feature].mean()
            if len(benign)
            else np.nan,
            "suspicious_mean": suspicious[feature].mean()
            if len(suspicious)
            else np.nan,
        }
    )

comparison = pd.DataFrame(
    comparison_rows
)

print(
    comparison.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

result.to_csv(
    OUTPUT,
    index=False,
)

print(
    "\nSaved:",
    OUTPUT,
)

print(
    "\n=========================================="
)

print(
    "STEP 9.1 COMPLETE"
)

print(
    "=========================================="
)
