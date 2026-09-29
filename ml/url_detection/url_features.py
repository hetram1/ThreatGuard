from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "secure",
    "security",
    "update",
    "confirm",
    "confirmation",
    "password",
    "credential",
    "wallet",
    "payment",
    "bank",
    "paypal",
    "crypto",
    "bitcoin",
    "invoice",
    "refund",
    "bonus",
    "gift",
    "free",
    "claim",
    "recover",
    "unlock",
    "suspend",
    "suspended",
    "urgent",
    "alert",
    "support",
    "admin",
]


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def safe_parse(url: str):
    url = str(url).strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    try:
        return urlparse(url)
    except Exception:
        return urlparse("http://invalid")


def extract_url_features(url: str) -> dict[str, float]:
    url = str(url).strip()
    parsed = safe_parse(url)

    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    lower_url = url.lower()
    lower_host = host.lower()

    digits = sum(char.isdigit() for char in url)
    letters = sum(char.isalpha() for char in url)

    special_chars = sum(
        not char.isalnum() for char in url
    )

    suspicious_keyword_count = sum(
        keyword in lower_url
        for keyword in SUSPICIOUS_KEYWORDS
    )

    encoded_count = len(
        re.findall(r"%[0-9a-fA-F]{2}", url)
    )

    repeated_separator_count = len(
        re.findall(r"(?:\.{2,}|-{2,}|/{3,})", url)
    )

    subdomain_count = max(
        len(host.split(".")) - 2,
        0
    )

    path_segments = [
        segment for segment in path.split("/")
        if segment
    ]

    return {
        "url_length": len(url),
        "domain_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),

        "num_digits": digits,
        "num_letters": letters,
        "num_special_chars": special_chars,

        "digit_ratio": digits / max(len(url), 1),
        "letter_ratio": letters / max(len(url), 1),
        "special_char_ratio": special_chars / max(len(url), 1),

        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_ampersands": url.count("&"),
        "num_equals": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_percent_symbols": url.count("%"),
        "num_colons": url.count(":"),
        "num_semicolons": url.count(";"),

        "subdomain_count": subdomain_count,
        "path_segment_count": len(path_segments),

        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": int(
            bool(re.fullmatch(
                r"(?:\d{1,3}\.){3}\d{1,3}",
                host,
            ))
        ),

        "has_port": int(parsed.port is not None)
        if parsed.hostname
        else 0,

        "has_userinfo": int(
            parsed.username is not None
            or parsed.password is not None
        ),

        "has_fragment": int(bool(fragment)),
        "has_query": int(bool(query)),

        "has_double_slash_path": int(
            "//" in path
        ),

        "has_hex_encoding": int(
            encoded_count > 0
        ),

        "encoded_char_count": encoded_count,

        "suspicious_keyword_count":
            suspicious_keyword_count,

        "has_suspicious_keyword": int(
            suspicious_keyword_count > 0
        ),

        "repeated_separator_count":
            repeated_separator_count,

        "url_entropy": shannon_entropy(url),

        "domain_entropy": shannon_entropy(host),

        "longest_token_length": max(
            [len(token) for token in re.split(
                r"[^a-zA-Z0-9]+",
                url,
            ) if token] or [0]
        ),

        "digit_run_count": len(
            re.findall(r"\d+", url)
        ),

        "letter_run_count": len(
            re.findall(r"[A-Za-z]+", url)
        ),

        "hostname_has_www": int(
            lower_host.startswith("www.")
        ),

        "hostname_has_com": int(
            lower_host.endswith(".com")
        ),

        "hostname_has_org": int(
            lower_host.endswith(".org")
        ),

        "hostname_has_net": int(
            lower_host.endswith(".net")
        ),

        "hostname_has_co": int(
            ".co" in lower_host
        ),
    }


def extract_features_from_series(
    urls: pd.Series,
) -> pd.DataFrame:
    rows = [
        extract_url_features(url)
        for url in urls
    ]

    return pd.DataFrame(rows)


FEATURE_NAMES = list(
    extract_url_features("https://example.com").keys()
)
