from typing import Any

from .sms_risk_engine import SMSRiskEngine
from .url_risk_engine import URLRiskEngine


class ThreatAnalysisService:
    """
    Unified ThreatGuard inference service.

    Supported analysis types:

        URL -> URLRiskEngine
        SMS -> SMSRiskEngine

    This layer deliberately does not modify the underlying
    model probabilities. It only provides a common interface
    for the API layer.
    """

    def __init__(self) -> None:
        self.url_engine = URLRiskEngine()
        self.sms_engine = SMSRiskEngine()

    def analyze_url(
        self,
        url: str,
    ) -> dict[str, Any]:
        result = self.url_engine.analyze(url)

        return {
            "input_type": "url",
            **result,
        }

    def analyze_sms(
        self,
        text: str,
    ) -> dict[str, Any]:
        result = self.sms_engine.analyze(text)

        return {
            "input_type": "sms",
            **result,
        }

    def analyze(
        self,
        input_type: str,
        value: str,
    ) -> dict[str, Any]:
        normalized_type = (
            input_type.strip().lower()
        )

        if normalized_type == "url":
            return self.analyze_url(value)

        if normalized_type == "sms":
            return self.analyze_sms(value)

        raise ValueError(
            "Unsupported input type. "
            "Expected 'url' or 'sms'."
        )


if __name__ == "__main__":
    service = ThreatAnalysisService()

    print()
    print("=" * 100)
    print("THREATGUARD UNIFIED THREAT ANALYSIS SERVICE")
    print("=" * 100)

    print()
    print("===== URL ANALYSIS =====")

    url_result = service.analyze_url(
        "https://secure-account-verification.example.com/login"
    )

    print(
        "Input type :",
        url_result["input_type"],
    )

    print(
        "Prediction :",
        url_result["prediction"],
    )

    print(
        "Risk score :",
        url_result["risk_score"],
    )

    print(
        "Risk band  :",
        url_result["risk_band"],
    )

    print()
    print("===== SMS ANALYSIS =====")

    sms_result = service.analyze_sms(
        "URGENT! Your bank account will be blocked today. "
        "Verify KYC immediately."
    )

    print(
        "Input type :",
        sms_result["input_type"],
    )

    print(
        "Prediction :",
        sms_result["prediction"],
    )

    print(
        "Risk score :",
        sms_result["risk_score"],
    )

    print(
        "Risk band  :",
        sms_result["risk_band"],
    )

    print()
    print("UNIFIED SERVICE TEST: PASS")
