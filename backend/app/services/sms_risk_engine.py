from pathlib import Path
from typing import Any

import joblib
from scipy.sparse import hstack


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "models"
    / "sms_detector.joblib"
)

WORD_VECTORIZER_PATH = (
    ROOT
    / "models"
    / "sms_word_vectorizer.joblib"
)

CHAR_VECTORIZER_PATH = (
    ROOT
    / "models"
    / "sms_char_vectorizer.joblib"
)

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
        self.model = joblib.load(
            MODEL_PATH
        )

        self.word_vectorizer = joblib.load(
            WORD_VECTORIZER_PATH
        )

        self.char_vectorizer = joblib.load(
            CHAR_VECTORIZER_PATH
        )

        expected_features = (
            len(
                self.word_vectorizer.vocabulary_
            )
            +
            len(
                self.char_vectorizer.vocabulary_
            )
        )

        actual_features = (
            self.model.n_features_in_
        )

        if actual_features != expected_features:
            raise RuntimeError(
                "SMS model feature contract mismatch: "
                f"model expects {actual_features}, "
                f"pipeline produces {expected_features}."
            )

        if list(self.model.classes_) != [0, 1]:
            raise RuntimeError(
                "Unexpected SMS model class ordering."
            )

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        text = str(text).strip()

        if not text:
            raise ValueError(
                "SMS text cannot be empty."
            )

        return text

    def _predict_probability(
        self,
        text: str,
    ) -> float:
        word_features = (
            self.word_vectorizer.transform(
                [text]
            )
        )

        char_features = (
            self.char_vectorizer.transform(
                [text]
            )
        )

        combined = hstack(
            [
                word_features,
                char_features,
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
    def _signals(
        text: str,
    ) -> dict[str, Any]:
        lower = text.lower()

        reasons: list[str] = []

        security_terms = [
            "verify",
            "verification",
            "kyc",
            "account",
            "bank",
            "blocked",
            "suspended",
            "password",
            "otp",
            "pin",
        ]

        urgency_terms = [
            "urgent",
            "immediately",
            "today",
            "now",
            "act now",
            "expires",
            "expired",
        ]

        reward_terms = [
            "winner",
            "won",
            "prize",
            "reward",
            "cashback",
            "congratulations",
            "claim",
        ]

        payment_terms = [
            "payment",
            "pay",
            "fee",
            "refund",
            "money",
            "rs",
            "₹",
        ]

        found_security = [
            term
            for term in security_terms
            if term in lower
        ]

        found_urgency = [
            term
            for term in urgency_terms
            if term in lower
        ]

        found_reward = [
            term
            for term in reward_terms
            if term in lower
        ]

        found_payment = [
            term
            for term in payment_terms
            if term in lower
        ]

        if found_security:
            reasons.append(
                "Message contains security or account-related language."
            )

        if found_urgency:
            reasons.append(
                "Message uses urgency or time-pressure language."
            )

        if found_reward:
            reasons.append(
                "Message contains reward, prize, or claim language."
            )

        if found_payment:
            reasons.append(
                "Message contains payment or money-related language."
            )

        if "http://" in lower or "https://" in lower:
            reasons.append(
                "Message contains a web link."
            )

        return {
            "security_terms": found_security,
            "urgency_terms": found_urgency,
            "reward_terms": found_reward,
            "payment_terms": found_payment,
            "contains_url": (
                "http://" in lower
                or "https://" in lower
            ),
            "reasons": reasons,
        }

    def analyze(
        self,
        text: str,
    ) -> dict[str, Any]:
        normalized_text = (
            self.normalize_text(text)
        )

        probability = (
            self._predict_probability(
                normalized_text
            )
        )

        signals = self._signals(
            normalized_text
        )

        risk_band = self._risk_band(
            probability
        )

        prediction = (
            "scam"
            if probability >= THRESHOLD
            else "legitimate"
        )

        # Heuristic signals are explanatory indicators,
        # not model decisions. Keep the explanation aligned
        # with the final classification shown to the user.
        reasons = (
            signals["reasons"]
            if prediction == "scam"
            else []
        )

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
                "type": (
                    "Word + Character TF-IDF "
                    "+ Logistic Regression"
                ),
            },
            "reasons": reasons,
            "signals": {
                "security_terms": signals[
                    "security_terms"
                ],
                "urgency_terms": signals[
                    "urgency_terms"
                ],
                "reward_terms": signals[
                    "reward_terms"
                ],
                "payment_terms": signals[
                    "payment_terms"
                ],
                "contains_url": signals[
                    "contains_url"
                ],
            },
        }


if __name__ == "__main__":
    engine = SMSRiskEngine()

    test_messages = [
        (
            "Hey bro, are you coming to class tomorrow?"
        ),
        (
            "URGENT! Your bank account will be "
            "blocked today. Verify KYC immediately."
        ),
        (
            "Aapka SBI account block ho jayega. "
            "KYC update karne ke liye turant click karein."
        ),
        (
            "Congratulations! You won Rs 25 lakh. "
            "Pay processing fee to claim."
        ),
    ]

    for text in test_messages:
        result = engine.analyze(text)

        print()
        print("=" * 80)
        print(text)
        print("-" * 80)
        print(
            "Prediction :",
            result["prediction"],
        )
        print(
            "Probability:",
            result["scam_probability"],
        )
        print(
            "Risk score :",
            result["risk_score"],
        )
        print(
            "Risk band  :",
            result["risk_band"],
        )

        print("Reasons:")

        if result["reasons"]:
            for reason in result["reasons"]:
                print("  -", reason)
        else:
            print(
                "  - No major heuristic signal detected."
            )
