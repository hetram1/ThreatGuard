"""One-shot external evaluation AFTER freezing development and model selection."""

import argparse, json
from pathlib import Path
import joblib, pandas as pd
from ml.reliable.prepare import ROOT, sha
from ml.reliable.preprocessing import normalize_url, url_group
from ml.reliable.train import metrics
from ml.reliable.evaluate import legacy_prob


def run(release):
    output = release / "external_evaluation.json"
    if output.exists():
        raise SystemExit(
            "External test already opened for this release. Preserve results and use fresh data for future development."
        )
    manifest = json.loads((release / "manifest.json").read_text())
    internal = json.loads((release / "evaluation.json").read_text())
    if internal["release_manifest_sha256"] != sha(release / "manifest.json"):
        raise RuntimeError("Release was modified after internal evaluation")
    for path, value in manifest["code_hashes"].items():
        if sha(ROOT / path) != value:
            raise RuntimeError("Frozen inference code changed")
    external = ROOT / "data/external/zenodo-8041387"
    sealed = json.loads((external / "sealed_manifest.json").read_text())
    for entry in sealed["files"]:
        if sha(external / entry["file"]) != entry["sha256"]:
            raise RuntimeError("External snapshot changed")
    # A durable opening receipt prevents silently claiming this dataset stayed untouched.
    receipt = external / "opened_for_evaluation.json"
    if receipt.exists():
        raise SystemExit(
            "External set already opened. Review the existing receipt; do not reuse it for model selection."
        )
    receipt.write_text(
        json.dumps(
            {
                "release_manifest_sha256": sha(release / "manifest.json"),
                "purpose": "final evaluation only",
                "model_selection_complete": True,
            },
            indent=2,
        )
    )
    parts = []
    for filename, label in [("not-phishing.csv", 0), ("phishing.csv", 1)]:
        d = pd.read_csv(external / filename, usecols=["url", "scan_date"])
        d["label"] = label
        parts.append(d)
    frame = pd.concat(parts, ignore_index=True)
    raw_count = len(frame)
    normalized = []
    invalid = 0
    for value in frame.url:
        try:
            normalized.append(normalize_url(value))
        except ValueError:
            normalized.append(None)
            invalid += 1
    frame["normalized"] = normalized
    frame = frame.dropna(subset=["normalized"])
    conflicts = frame.groupby("normalized").label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    frame = frame[~frame.normalized.isin(bad)].drop_duplicates("normalized").copy()
    data = ROOT / manifest["data_directory"]
    legacy = pd.read_csv(data / "legacy_url_membership.csv")
    development = pd.read_csv(data / "url.csv")
    probes = pd.read_csv(ROOT / "audit/fresh_probes.csv")
    values = (
        list(legacy.value)
        + list(development.value)
        + list(probes.loc[probes.kind == "url", "value"])
    )
    seen_urls = set()
    seen_groups = set()
    for value in values:
        try:
            normalized = normalize_url(value)
            seen_urls.add(normalized)
            seen_groups.add(url_group(normalized))
        except ValueError:
            pass
    frame["group"] = frame.normalized.map(url_group)
    exact = int(frame.normalized.isin(seen_urls).sum())
    overlap = frame.group.isin(seen_groups)
    test = frame[~overlap].copy()
    if test.empty or test.label.nunique() != 2:
        raise RuntimeError(
            "No adequate two-class independent subset; do not claim external accuracy."
        )
    info = manifest["models"]["url"]
    if sha(release / info["file"]) != info["sha256"]:
        raise RuntimeError("Model hash mismatch")
    model = joblib.load(release / info["file"])
    after = model.probability(test.url.tolist())
    before = legacy_prob("url", test.url.tolist())
    result = {
        "source": sealed["source"],
        "license": sealed["license"],
        "publication_date": sealed["publication_date"],
        "scope": "Published historical website labels; no current reputation assertion. New source, normalized URL and registrable-domain disjoint from all available URL development data including audit probes. This cannot prove absence of all upstream/source dependencies.",
        "release_manifest_sha256": sha(release / "manifest.json"),
        "sealed_manifest_sha256": sha(external / "sealed_manifest.json"),
        "raw_rows": raw_count,
        "invalid_urls": invalid,
        "conflicting_urls": len(bad),
        "unique_valid_rows": len(frame),
        "exact_development_overlaps": exact,
        "domain_overlap_rows_removed": int(overlap.sum()),
        "retained_domains": test.group.nunique(),
        "retained_rows": len(test),
        "remaining_domain_overlap": len(set(test.group) & seen_groups),
        "remaining_exact_overlap": len(set(test.normalized) & seen_urls),
        "scan_date_min": str(test.scan_date.min()),
        "scan_date_max": str(test.scan_date.max()),
        "before": metrics(test.label, before),
        "after": metrics(test.label, after),
        "sms_external_status": "No verified independent SMS corpus acquired; Mendeley source may overlap existing Smishing_Dataset and was not assumed independent.",
    }
    test.assign(before_probability=before, after_probability=after).to_csv(
        release / "external_url_predictions.csv", index=False
    )
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, default=ROOT / "models/releases/reliable-v1")
    a = p.parse_args()
    run(a.release)
