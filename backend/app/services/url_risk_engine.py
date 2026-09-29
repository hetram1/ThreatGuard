from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd
from scipy.sparse import hstack

from ml.url_detection.url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "models"
    / "url_detector.joblib"
)

VECTORIZER_PATH = (
    ROOT
    / "models"
    / "url_detector_vectorizer.joblib"
)

SCALER_PATH = (
    ROOT
    / "models"
    / "url_detector_scaler.joblib"
)

THRESHOLD = 0.50


class URLRiskEngine:
    """
    ThreatGuard production URL inference engine.

    Production model:
        Character TF-IDF
        +
        45 engineered URL features
        +
        Logistic Regression

    The inference path intentionally mirrors the V5
    training/evaluation pipeline.
    """

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        expected_features = len(
            self.vectorizer.vocabulary_
        ) + len(FEATURE_NAMES)

        actual_features = self.model.coef_.shape[1]

        if actual_features != expected_features:
            raise RuntimeError(
                "URL model feature contract mismatch: "
                f"model expects {actual_features}, "
                f"pipeline produces {expected_features}."
            )

        if self.scaler.n_features_in_ != len(
            FEATURE_NAMES
        ):
            raise RuntimeError(
                "URL scaler feature contract mismatch: "
                f"scaler expects {self.scaler.n_features_in_}, "
                f"features provide {len(FEATURE_NAMES)}."
            )

    @staticmethod
    def normalize_url(url: str) -> str:
        url = str(url).strip()

        if not url:
            raise ValueError(
                "URL cannot be empty."
            )

        if "://" not in url:
            url = "https://" + url

        return url

    def _predict_probability(
        self,
        url: str,
    ) -> float:
        urls = pd.Series(
            [url],
            dtype="string",
        )

        char_features = (
            self.vectorizer.transform(urls)
        )

        engineered = (
            extract_features_from_series(urls)
        )

        engineered = engineered[
            FEATURE_NAMES
        ]

        engineered_scaled = (
            self.scaler.transform(
                engineered
            )
        )

        combined = hstack(
            [
                char_features,
                engineered_scaled,
            ],
            format="csr",
        )

        probability = (
            self.model
            .predict_proba(combined)[:, 1][0]
        )

        return float(probability)

    @staticmethod
    def _risk_band(
        probability: float,
    ) -> str:
        if probability < 0.20:
            return "LOW"

        if probability < 0.50:
            return "MEDIUM"

        if probability < 0.80:
            return "HIGH"

        return "CRITICAL"

    @staticmethod
    def _structural_signals(
        url: str,
    ) -> dict[str, Any]:
        features = (
            extract_features_from_series(
                pd.Series(
                    [url],
                    dtype="string",
                )
            )
            .iloc[0]
            .to_dict()
        )

        reasons: list[str] = []

        if features["has_ip_address"]:
            reasons.append(
                "URL uses an IP address instead of a hostname."
            )

        if features["has_port"]:
            reasons.append(
                "URL specifies a non-default port."
            )

        if features["has_userinfo"]:
            reasons.append(
                "URL contains embedded user information."
            )

        if features["encoded_char_count"] > 3:
            reasons.append(
                "URL contains multiple percent-encoded characters."
            )

        if features["excessive_subdomains"]:
            reasons.append(
                "URL contains an unusually deep subdomain structure."
            )

        if features["hyphenated_host"]:
            reasons.append(
                "Hostname contains multiple hyphen separators."
            )

        if features["very_long_host"]:
            reasons.append(
                "Hostname is unusually long."
            )

        if features["very_long_url"]:
            reasons.append(
                "URL is unusually long."
            )

        if features["suspicious_keyword_count"] > 0:
            reasons.append(
                "URL contains security-sensitive keywords."
            )

        if features["suspicious_tld"]:
            reasons.append(
                "Top-level domain is present in the model's higher-risk TLD set."
            )

        if features["has_hex_encoding"]:
            reasons.append(
                "URL contains hexadecimal or encoded character patterns."
            )

        if features["has_double_slash_path"]:
            reasons.append(
                "URL path contains repeated slash separators."
            )

        if features["numeric_host"]:
            reasons.append(
                "Hostname contains a numeric-only host component."
            )

        return {
            "features": features,
            "reasons": reasons,
        }

    def analyze(
        self,
        url: str,
    ) -> dict[str, Any]:
        normalized_url = self.normalize_url(
            url
        )

        probability = (
            self._predict_probability(
                normalized_url
            )
        )

        signals = (
            self._structural_signals(
                normalized_url
            )
        )

        risk_band = self._risk_band(
            probability
        )

        prediction = (
            "phishing"
            if probability >= THRESHOLD
            else "legitimate"
        )

        return {
            "url": normalized_url,
            "prediction": prediction,
            "risk_score": round(
                probability * 100,
                2,
            ),
            "phishing_probability": round(
                probability,
                6,
            ),
            "confidence": round(
                max(
                    probability,
                    1.0 - probability,
                ),
                6,
            ),
            "risk_band": risk_band,
            "threshold": THRESHOLD,
            "model": {
                "name": "ThreatGuard URL Detector",
                "version": "v5",
                "type": (
                    "Character TF-IDF + "
                    "engineered URL features + "
                    "Logistic Regression"
                ),
            },
            "reasons": signals["reasons"],
            "features": signals["features"],
        }


if __name__ == "__main__":
    engine = URLRiskEngine()

    test_urls = [
        "https://www.google.com",
        "https://github.com",
        "https://example.com",
        "https://docs.python.org/3/",
        "https://www.wikipedia.org/",
        "https://www.mozilla.org/",
        "https://amazon.in",
        "https://docs.github.com/en",
        "http://192.168.1.10/login",
        "http://secure-account-verification.example.com/login",
        "http://paypal-login-security.com/verify",
        "http://bank-account-verification-required.com/login",
    ]

    for test_url in test_urls:
        result = engine.analyze(test_url)

        print()
        print("=" * 80)
        print(test_url)
        print("-" * 80)
        print(
            f"Prediction : {result['prediction']}"
        )
        print(
            f"Risk score : {result['risk_score']}"
        )
        print(
            f"Probability: {result['phishing_probability']}"
        )
        print(
            f"Confidence : {result['confidence']}"
        )
        print(
            f"Risk band  : {result['risk_band']}"
        )

        print("Reasons:")

        if result["reasons"]:
            for reason in result["reasons"]:
                print(f"  - {reason}")
        else:
            print("  - No major structural warning signal detected.")
