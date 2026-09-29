from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from url_features_v3 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models"
    / "url_xgboost_v3.joblib"
)

OUTPUT_PATH = (
    ROOT
    / "models"
    / "audit"
    / "v3_sanity_results.csv"
)


cases = [

    # BENIGN
    ("https://www.google.com", 0),
    ("https://www.amazon.com", 0),
    ("https://github.com", 0),
    ("https://www.microsoft.com", 0),
    ("https://www.apple.com", 0),
    ("https://www.wikipedia.org", 0),
    ("https://example.com", 0),
    ("https://www.python.org", 0),
    ("https://www.mozilla.org", 0),
    ("https://www.amazon.in", 0),
    ("https://www.flipkart.com", 0),
    ("https://www.gov.in", 0),
    ("https://uidai.gov.in", 0),
    ("https://www.rbi.org.in", 0),
    ("https://www.nasa.gov", 0),
    ("https://www.github.com", 0),
    ("https://stackoverflow.com", 0),
    ("https://docs.python.org/3/", 0),
    ("https://developer.mozilla.org/en-US/", 0),
    ("https://www.wikipedia.org/wiki/India", 0),

    # SUSPICIOUS
    ("http://192.168.1.10/login", 1),
    ("http://192.168.0.20/verify/account", 1),
    ("http://secure-account-verification.example.com/login", 1),
    ("http://paypal-login-security.example.com", 1),
    ("http://account-verification-payment.example.com/login", 1),
    ("http://free-gift-card-claim.example.com", 1),
    ("http://google-security-login.example.com", 1),
    ("http://microsoft-account-verify.example.com", 1),
    ("http://paypal-security-check.example.com", 1),
    ("http://secure-login-update.example.com", 1),
    ("http://paypa1-secure-login.example.com", 1),
    ("http://paypal.verify-account.example.com/login", 1),
    ("http://google.account-security.example.com/login", 1),
    ("http://microsoft.verify-user.example.com/auth", 1),
    ("http://bank-account-confirm.example.com/update", 1),
    ("http://wallet-security-check.example.com", 1),
    ("http://login-session-expired.example.com/verify", 1),
    ("http://secure-payment-confirmation.example.com", 1),
    ("http://account-locked-verification.example.com", 1),
    ("http://urgent-security-alert.example.com/login", 1),
]


df = pd.DataFrame(
    cases,
    columns=[
        "url",
        "expected",
    ],
)


print(
    "=========================================="
)

print(
    "ThreatGuard — V3 Sanity Test"
)

print(
    "=========================================="
)


model = joblib.load(
    MODEL_PATH
)

X = extract_features_from_series(
    df["url"]
)

X = X[
    FEATURE_NAMES
]

probabilities = (
    model
    .predict_proba(X)[:, 1]
)

predictions = (
    probabilities >= 0.50
).astype(int)

df["probability"] = probabilities

df["prediction"] = predictions

df["correct"] = (
    predictions
    == df["expected"].to_numpy()
)

df["classification"] = np.where(
    predictions == 1,
    "PHISHING",
    "LEGITIMATE",
)


print(
    "\n===== V3 SANITY RESULTS ====="
)

print(
    df.to_string(
        index=False
    )
)


print(
    "\n===== SUMMARY ====="
)

correct = int(
    df["correct"].sum()
)

total = len(df)

print(
    "Correct:",
    correct,
    "/",
    total,
)

print(
    "Accuracy:",
    f"{correct / total:.4f}",
)


print(
    "\n===== MISCLASSIFICATIONS ====="
)

misclassified = df[
    ~df["correct"]
]

if len(misclassified) == 0:

    print(
        "NONE"
    )

else:

    print(
        misclassified.to_string(
            index=False
        )
    )


df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    "\nSaved:",
    OUTPUT_PATH
)

print(
    "\n=========================================="
)

print(
    "V3 SANITY TEST COMPLETE"
)

print(
    "=========================================="
)
