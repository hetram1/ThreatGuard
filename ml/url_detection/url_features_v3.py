from urllib.parse import urlparse
import math
import re

import pandas as pd


FEATURE_NAMES = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "num_digits",
    "num_letters",
    "num_special_chars",
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "host_digits",
    "host_letters",
    "host_special_chars",
    "num_dots",
    "num_hyphens",
    "num_slashes",
    "num_question_marks",
    "num_ampersands",
    "num_equals",
    "num_at_symbols",
    "num_percent_symbols",
    "subdomain_count",
    "host_token_count",
    "path_token_count",
    "has_https",
    "has_ip_address",
    "has_port",
    "has_userinfo",
    "has_query",
    "has_fragment",
    "encoded_char_count",
    "has_hex_encoding",
    "url_entropy",
    "host_entropy",
    "suspicious_keyword_count",
    "has_suspicious_keyword",
    "suspicious_host_keyword_count",
    "suspicious_path_keyword_count",
    "hyphenated_host",
    "numeric_host",
    "very_long_host",
    "very_long_url",
    "excessive_subdomains",
    "suspicious_tld",
    "longest_host_token",
    "longest_path_token",
    "host_digit_ratio",
    "host_letter_ratio",
    "path_depth",
    "query_parameter_count",
    "root_url",
]


SUSPICIOUS_KEYWORDS = {
    "login",
    "signin",
    "sign-in",
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
    "payment",
    "wallet",
    "bank",
    "invoice",
    "unlock",
    "suspended",
    "expired",
    "alert",
    "gift",
    "bonus",
    "claim",
    "recover",
    "authentication",
    "auth",
    "session",
}


SUSPICIOUS_TLDS = {
    "tk",
    "ml",
    "ga",
    "cf",
    "gq",
    "top",
    "xyz",
    "click",
    "work",
    "support",
    "country",
    "download",
    "zip",
    "review",
}


def entropy(value: str) -> float:

    if not value:
        return 0.0

    counts = {}

    for char in value:
        counts[char] = counts.get(char, 0) + 1

    n = len(value)

    return -sum(
        (count / n) * math.log2(count / n)
        for count in counts.values()
    )


def safe_ratio(numerator: int, denominator: int) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def tokenize(value: str):

    if not value:
        return []

    return [
        token
        for token in re.split(
            r"[^a-zA-Z0-9]+",
            value.lower(),
        )
        if token
    ]


def extract_features(url: str):

    url = str(url).strip()

    try:

        parsed = urlparse(
            url
            if "://" in url
            else "http://" + url
        )

    except Exception:

        parsed = urlparse(
            "http://" + url
        )

    hostname = (
        parsed.hostname or ""
    ).lower()

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    url_length = len(url)
    host_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    digits = sum(
        c.isdigit()
        for c in url
    )

    letters = sum(
        c.isalpha()
        for c in url
    )

    special = (
        url_length
        - digits
        - letters
    )

    host_digits = sum(
        c.isdigit()
        for c in hostname
    )

    host_letters = sum(
        c.isalpha()
        for c in hostname
    )

    host_special = (
        host_length
        - host_digits
        - host_letters
    )

    host_tokens = tokenize(
        hostname
    )

    path_tokens = tokenize(
        path
    )

    subdomains = max(
        len(hostname.split(".")) - 2,
        0,
    )

    host_parts = [
        p
        for p in hostname.split(".")
        if p
    ]

    tld = (
        host_parts[-1]
        if host_parts
        else ""
    )

    all_tokens = (
        tokenize(hostname)
        + tokenize(path)
        + tokenize(query)
    )

    suspicious_tokens = [
        token
        for token in all_tokens
        if token in SUSPICIOUS_KEYWORDS
    ]

    suspicious_host_tokens = [
        token
        for token in tokenize(hostname)
        if token in SUSPICIOUS_KEYWORDS
    ]

    suspicious_path_tokens = [
        token
        for token in tokenize(path)
        if token in SUSPICIOUS_KEYWORDS
    ]

    has_https = int(
        parsed.scheme.lower()
        == "https"
    )

    has_ip_address = int(
        bool(
            re.fullmatch(
                r"\d{1,3}(?:\.\d{1,3}){3}",
                hostname,
            )
        )
    )

    has_port = int(
        parsed.port is not None
        if hostname
        else False
    )

    has_userinfo = int(
        parsed.username is not None
        or parsed.password is not None
    )

    has_query = int(
        bool(query)
    )

    has_fragment = int(
        bool(fragment)
    )

    encoded_char_count = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            url,
        )
    )

    has_hex_encoding = int(
        encoded_char_count > 0
    )

    query_parameter_count = 0

    if query:

        query_parameter_count = len(
            [
                p
                for p in query.split("&")
                if p
            ]
        )

    path_depth = len(
        [
            segment
            for segment in path.split("/")
            if segment
        ]
    )

    root_url = int(
        path in ("", "/")
        and not query
        and not fragment
    )

    longest_host_token = max(
        (
            len(token)
            for token in host_tokens
        ),
        default=0,
    )

    longest_path_token = max(
        (
            len(token)
            for token in path_tokens
        ),
        default=0,
    )

    hyphenated_host = int(
        "-" in hostname
    )

    numeric_host = int(
        bool(hostname)
        and all(
            c.isdigit() or c == "."
            for c in hostname
        )
    )

    very_long_host = int(
        host_length >= 40
    )

    very_long_url = int(
        url_length >= 100
    )

    excessive_subdomains = int(
        subdomains >= 4
    )

    suspicious_tld = int(
        tld in SUSPICIOUS_TLDS
    )

    return [
        url_length,
        host_length,
        path_length,
        query_length,

        digits,
        letters,
        special,

        safe_ratio(digits, url_length),
        safe_ratio(letters, url_length),
        safe_ratio(special, url_length),

        host_digits,
        host_letters,
        host_special,

        url.count("."),
        url.count("-"),
        url.count("/"),

        url.count("?"),
        url.count("&"),
        url.count("="),
        url.count("@"),
        url.count("%"),

        subdomains,
        len(host_tokens),
        len(path_tokens),

        has_https,
        has_ip_address,
        has_port,
        has_userinfo,

        has_query,
        has_fragment,

        encoded_char_count,
        has_hex_encoding,

        entropy(url),
        entropy(hostname),

        len(suspicious_tokens),
        int(bool(suspicious_tokens)),

        len(suspicious_host_tokens),
        len(suspicious_path_tokens),

        hyphenated_host,
        numeric_host,
        very_long_host,
        very_long_url,

        excessive_subdomains,
        suspicious_tld,

        longest_host_token,
        longest_path_token,

        safe_ratio(host_digits, host_length),
        safe_ratio(host_letters, host_length),

        path_depth,
        query_parameter_count,

        root_url,
    ]


def extract_features_from_series(
    urls: pd.Series,
) -> pd.DataFrame:

    rows = [
        extract_features(url)
        for url in urls
    ]

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES,
    )
