"""Strict request authority validation layered over shared canonicalization."""

import re
from urllib.parse import urlsplit
from ml.reliable.preprocessing import normalize_url


def validated_url(value):
    # normalize_url handles type, syntax, scheme, hostname, port and IDNA.
    normalized = normalize_url(value)
    raw = value.strip()
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    authority = urlsplit(raw).netloc
    if authority.count("@") > 1:
        raise ValueError("User information must percent-encode embedded @ characters.")
    hostport = authority.rsplit("@", 1)[-1]
    if hostport.startswith("["):
        if not re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", hostport):
            raise ValueError("Invalid bracketed IP authority.")
    elif not re.fullmatch(r"[^:\[\]]+(?::[0-9]+)?", hostport):
        raise ValueError("Invalid hostname or port syntax.")
    return normalized
