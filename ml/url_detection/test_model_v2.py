from pathlib import Path
import joblib
import pandas as pd

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

model = joblib.load(
    ROOT / "models" / "url_xgboost_v2.joblib"
)

urls = [
    "https://www.google.com",
    "https://www.amazon.com",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://example.com",
    "https://paypal.com/login",
    "http://192.168.1.10/login",
    "http://secure-account-verification.example.com/login",
    "http://verify-account-security.example.com/update",
    "http://paypal-login-security.example.com",
    "http://account-verification-payment.example.com/login",
]

X = extract_features_from_series(
    pd.Series(urls)
)

X = X[FEATURE_NAMES]

probabilities = model.predict_proba(
    X
)[:, 1]

result = pd.DataFrame(
    {
        "url": urls,
        "phishing_probability": probabilities,
        "risk_score": probabilities * 100,
    }
)

result["classification"] = result[
    "phishing_probability"
].apply(
    lambda x:
        "PHISHING"
        if x >= 0.50
        else "LEGITIMATE"
)

print(
    result.to_string(
        index=False
    )
)

print("\n===== EXPECTED BEHAVIOR =====")
print(
    "Benign well-known/example URLs should NOT"
    " all receive extreme phishing scores."
)
print(
    "Clearly suspicious constructions should"
    " generally receive elevated scores."
)
