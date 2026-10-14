"""Download source-labeled URL benchmark bytes; never train or select on them."""

import hashlib, json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/external/zenodo-8041387"
if OUT.exists():
    raise SystemExit("External snapshot already exists; do not overwrite.")
OUT.mkdir(parents=True)
record = requests.get("https://zenodo.org/api/records/8041387", timeout=30)
record.raise_for_status()
metadata = record.json()
(OUT / "source_record.json").write_text(json.dumps(metadata, indent=2))
files = []
for entry in metadata["files"]:
    if entry["key"] not in ["phishing.csv", "not-phishing.csv"]:
        continue
    dest = OUT / entry["key"]
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with requests.get(
        entry["links"]["self"], stream=True, timeout=(20, 90)
    ) as response:
        response.raise_for_status()
        with dest.open("xb") as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
                md5.update(block)
                sha.update(block)
    if "md5:" + md5.hexdigest() != entry["checksum"]:
        raise RuntimeError("Source checksum mismatch")
    files.append(
        {
            "file": entry["key"],
            "sha256": sha.hexdigest(),
            "source_checksum": entry["checksum"],
            "bytes": dest.stat().st_size,
        }
    )
    print("Downloaded and checksum-verified", entry["key"], flush=True)
(OUT / "sealed_manifest.json").write_text(
    json.dumps(
        {
            "source": "https://doi.org/10.5281/zenodo.8041387",
            "license": metadata["metadata"]["license"],
            "publication_date": metadata["metadata"].get("publication_date"),
            "files": files,
            "role": "external evaluation only; no training or selection",
            "labels": "phishing.csv=1; not-phishing.csv=0, as published by authors",
            "content_inspected": False,
            "independence": "Not yet established; require normalized URL and registrable-domain overlap exclusion against ALL development datasets before evaluation.",
        },
        indent=2,
    )
)
