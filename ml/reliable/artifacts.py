"""Local artifact integrity helpers; no training libraries or network access."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()
