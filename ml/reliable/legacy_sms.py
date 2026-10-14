from backend.app.services.heuristic_signals import sms_signals
from pathlib import Path
from typing import Any

import joblib
from scipy.sparse import hstack


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "sms_detector.joblib"

WORD_VECTORIZER_PATH = ROOT / "models" / "sms_word_vectorizer.joblib"

CHAR_VECTORIZER_PATH = ROOT / "models" / "sms_char_vectorizer.joblib"

THRESHOLD = 0.50


class SMSRiskEngine:
    """
    ThreatGuard production SMS scam inference engine.

    Production model:

        Word TF-IDF
             +
        Character TF-IDF
             +
        Logistic Regression

    Labels:

        0 = legitimate
        1 = scam
    """

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)

        self.word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)

        self.char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)

        expected_features = len(self.word_vectorizer.vocabulary_) + len(
            self.char_vectorizer.vocabulary_
        )

        actual_features = self.model.n_features_in_

        if actual_features != expected_features:
            raise RuntimeError(
                "SMS model feature contract mismatch: "
                f"model expects {actual_features}, "
                f"pipeline produces {expected_features}."
            )

        if list(self.model.classes_) != [0, 1]:
            raise RuntimeError("Unexpected SMS model class ordering.")

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        text = str(text).strip()

        if not text:
            raise ValueError("SMS text cannot be empty.")

        return text

    def _predict_probability(
        self,
        text: str,
    ) -> float:
        word_features = self.word_vectorizer.transform([text])

        char_features = self.char_vectorizer.transform([text])

        combined = hstack(
            [
                word_features,
                char_features,
            ],
            format="csr",
        )

        probability = self.model.predict_proba(combined)[:, 1][0]

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
    def _signals(value: str) -> dict[str, Any]:
        return sms_signals(value)

    def analyze(
        self,
        text: str,
    ) -> dict[str, Any]:
        normalized_text = self.normalize_text(text)

        probability = self._predict_probability(normalized_text)

        signals = self._signals(normalized_text)

        risk_band = self._risk_band(probability)

        prediction = "scam" if probability >= THRESHOLD else "legitimate"

        # Heuristic signals are explanatory indicators,
        # not model decisions. Keep the explanation aligned
        # with the final classification shown to the user.
        reasons = signals["reasons"] if prediction == "scam" else []

        return {
            "text": normalized_text,
            "prediction": prediction,
            "risk_score": round(
                probability * 100,
                2,
            ),
            "scam_probability": round(
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
                "name": "ThreatGuard SMS Detector",
                "version": "v1",
                "type": ("Word + Character TF-IDF + Logistic Regression"),
            },
            "reasons": reasons,
            "signals": {
                "security_terms": signals["security_terms"],
                "urgency_terms": signals["urgency_terms"],
                "reward_terms": signals["reward_terms"],
                "payment_terms": signals["payment_terms"],
                "contains_url": signals["contains_url"],
            },
        }
