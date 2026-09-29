from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification",
    "account", "secure", "security", "update",
    "confirm", "confirmation", "password",
    "credential", "wallet", "payment", "bank",
    "paypal", "crypto", "bitcoin", "invoice",
    "refund", "bonus", "gift", "free", "claim",
    "recover", "unlock", "suspend", "suspended",
    "urgent", "alert", "support", "admin",
]


def entropy(text: str) -> float:
    if not text:
        return 0.0

    counts = Counter(text)
    n = len(text)

    return -sum(
        (count / n) * math.log2(count / n)
        for count in counts.values()
    )


def parse_url(url: str):
    url = str(url).strip()

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        url,
    ):
        url = "http://" + url

    try:
        return urlparse(url)
    except Exception:
        return urlparse("http://invalid")


def extract_url_features(url: str) -> dict[str, float]:

    url = str(url).strip()
    parsed = parse_url(url)

    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    lower_url = url.lower()
    lower_host = host.lower()

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)

    special = sum(
        not c.isalnum()
        for c in url
    )

    keyword_hits = [
        word
        for word in SUSPICIOUS_KEYWORDS
        if word in lower_url
    ]

    encoded = re.findall(
        r"%[0-9a-fA-F]{2}",
        url,
    )

    host_tokens = [
        x for x in re.split(
            r"[^a-zA-Z0-9]+",
            host,
        ) if x
    ]

    path_tokens = [
        x for x in re.split(
            r"[^a-zA-Z0-9]+",
            path,
        ) if x
    ]

    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 1

    ip_match = bool(
        re.fullmatch(
            r"(?:\d{1,3}\.){3}\d{1,3}",
            host,
        )
    )

    # Stronger signals for suspicious URL construction.
    hyphenated_host = int(
        "-" in host
    )

    numeric_host = int(
        bool(host) and
        sum(c.isdigit() for c in host)
        / max(len(host), 1) > 0.30
    )

    very_long_host = int(
        len(host) > 40
    )

    very_long_url = int(
        len(url) > 100
    )

    excessive_subdomains = int(
        max(len(host.split(".")) - 2, 0) >= 3
    )

    suspicious_tld = int(
        any(
            lower_host.endswith(tld)
            for tld in [
                ".tk", ".ml", ".ga", ".cf",
                ".gq", ".click", ".top",
                ".xyz", ".zip"
            ]
        )
    )

    return {
        # Length / structure
        "url_length": len(url),
        "domain_length": len(host),
        "path_length": len(path),
        "query_length": len(query),

        # Character composition
        "num_digits": digits,
        "num_letters": letters,
        "num_special_chars": special,
        "digit_ratio": digits / max(len(url), 1),
        "letter_ratio": letters / max(len(url), 1),
        "special_char_ratio": special / max(len(url), 1),

        # Separators
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

        # Host structure
        "subdomain_count": max(
            len(host.split(".")) - 2,
            0,
        ),
        "host_token_count": len(host_tokens),
        "path_token_count": len(path_tokens),

        # Network/address properties
        "has_ip_address": int(ip_match),
        "has_port": has_port,
        "has_userinfo": int(
            parsed.username is not None
            or parsed.password is not None
        ),

        # URL encoding / obfuscation
        "encoded_char_count": len(encoded),
        "has_hex_encoding": int(
            len(encoded) > 0
        ),
        "url_entropy": entropy(url),
        "domain_entropy": entropy(host),

        # Suspicious lexical indicators
        "suspicious_keyword_count": len(
            keyword_hits
        ),
        "has_suspicious_keyword": int(
            len(keyword_hits) > 0
        ),

        # Structural anomalies
        "repeated_separator_count": len(
            re.findall(
                r"(?:\.{2,}|-{2,}|/{3,})",
                url,
            )
        ),
        "has_double_slash_path": int(
            "//" in path
        ),

        # Host-specific risk signals
        "hyphenated_host": hyphenated_host,
        "numeric_host": numeric_host,
        "very_long_host": very_long_host,
        "very_long_url": very_long_url,
        "excessive_subdomains": excessive_subdomains,
        "suspicious_tld": suspicious_tld,

        # Token complexity
        "longest_host_token": max(
            [len(x) for x in host_tokens]
            or [0]
        ),
        "longest_path_token": max(
            [len(x) for x in path_tokens]
            or [0]
        ),
        "digit_run_count": len(
            re.findall(r"\d+", url)
        ),
        "letter_run_count": len(
            re.findall(r"[A-Za-z]+", url)
        ),
    }


def extract_features_from_series(
    urls: pd.Series,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            extract_url_features(url)
            for url in urls
        ]
    )


FEATURE_NAMES = list(
    extract_url_features(
        "https://example.com"
    ).keys()
)
