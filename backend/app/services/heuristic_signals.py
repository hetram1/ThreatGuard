"""Descriptive heuristics, independent of model probability and evidence."""

from typing import Any
import pandas as pd
from ml.url_detection.url_features_v2 import extract_features_from_series


def url_signals(
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
        reasons.append("URL uses an IP address instead of a hostname.")

    if features["has_port"]:
        reasons.append("URL specifies a non-default port.")

    if features["has_userinfo"]:
        reasons.append("URL contains embedded user information.")

    if features["encoded_char_count"] > 3:
        reasons.append("URL contains multiple percent-encoded characters.")

    if features["excessive_subdomains"]:
        reasons.append("URL contains an unusually deep subdomain structure.")

    if features["hyphenated_host"]:
        reasons.append("Hostname contains multiple hyphen separators.")

    if features["very_long_host"]:
        reasons.append("Hostname is unusually long.")

    if features["very_long_url"]:
        reasons.append("URL is unusually long.")

    if features["suspicious_keyword_count"] > 0:
        reasons.append("URL contains security-sensitive keywords.")

    if features["suspicious_tld"]:
        reasons.append(
            "Top-level domain is present in the model's higher-risk TLD set."
        )

    if features["has_hex_encoding"]:
        reasons.append("URL contains hexadecimal or encoded character patterns.")

    if features["has_double_slash_path"]:
        reasons.append("URL path contains repeated slash separators.")

    if features["numeric_host"]:
        reasons.append("Hostname contains a numeric-only host component.")

    return {
        "features": features,
        "reasons": reasons,
    }


def sms_signals(
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

    found_security = [term for term in security_terms if term in lower]

    found_urgency = [term for term in urgency_terms if term in lower]

    found_reward = [term for term in reward_terms if term in lower]

    found_payment = [term for term in payment_terms if term in lower]

    if found_security:
        reasons.append("Message contains security or account-related language.")

    if found_urgency:
        reasons.append("Message uses urgency or time-pressure language.")

    if found_reward:
        reasons.append("Message contains reward, prize, or claim language.")

    if found_payment:
        reasons.append("Message contains payment or money-related language.")

    if "http://" in lower or "https://" in lower:
        reasons.append("Message contains a web link.")

    return {
        "security_terms": found_security,
        "urgency_terms": found_urgency,
        "reward_terms": found_reward,
        "payment_terms": found_payment,
        "contains_url": ("http://" in lower or "https://" in lower),
        "reasons": reasons,
    }
