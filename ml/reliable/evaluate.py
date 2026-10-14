"""Evaluate a frozen release, with provenance checks and fair legacy comparison."""

import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy.sparse import hstack
from ml.reliable.prepare import ROOT, sha
from ml.reliable.train import metrics
from ml.reliable.preprocessing import sms_template
from ml.url_detection.url_features_v2 import FEATURE_NAMES, extract_features_from_series


def legacy_prob(kind, values):
    directory = ROOT / "models"
    if kind == "url":
        model = joblib.load(directory / "url_detector.joblib")
        char = joblib.load(directory / "url_detector_vectorizer.joblib")
        scaler = joblib.load(directory / "url_detector_scaler.joblib")
    else:
        model = joblib.load(directory / "sms_detector.joblib")
        char = joblib.load(directory / "sms_char_vectorizer.joblib")
        word = joblib.load(directory / "sms_word_vectorizer.joblib")
    probabilities = []
    for i in range(0, len(values), 1000):
        x = list(values[i : i + 1000])
        if kind == "url":
            # Original inference prepended https only when a scheme was absent.
            x = [v.strip() if "://" in v else "https://" + v.strip() for v in x]
            features = hstack(
                [
                    char.transform(x),
                    scaler.transform(
                        extract_features_from_series(pd.Series(x))[FEATURE_NAMES]
                    ),
                ],
                format="csr",
            )
        else:
            features = hstack([word.transform(x), char.transform(x)], format="csr")
        probabilities.extend(model.predict_proba(features)[:, 1])
    return np.asarray(probabilities)


def run(release, compare_legacy=False):
    manifest = json.loads((release / "manifest.json").read_text())
    data = ROOT / manifest["data_directory"]
    if sha(data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed")
    for path, expected in manifest["code_hashes"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError("Frozen preprocessing code changed")
    prep = json.loads((data / "manifest.json").read_text())
    if compare_legacy:
        required = [
            "url_detector.joblib",
            "url_detector_vectorizer.joblib",
            "url_detector_scaler.joblib",
            "sms_detector.joblib",
            "sms_word_vectorizer.joblib",
            "sms_char_vectorizer.joblib",
        ]
        for filename in required:
            path = "models/" + filename
            expected = prep["legacy_artifacts"].get(path)
            if (
                expected is None
                or not (ROOT / path).exists()
                or sha(ROOT / path) != expected
            ):
                raise RuntimeError(
                    "Verified legacy comparison artifacts unavailable: " + path
                )
    output = release / "evaluation.json"
    if output.exists():
        raise SystemExit(
            "Evaluation exists. Preserve it; do not repeatedly tune against test results."
        )
    results = {
        "release_manifest_sha256": sha(release / "manifest.json"),
        "note": "Historical internal grouped holdouts. Not external evidence. Legacy comparison restricted to rows/groups absent from legacy training where reconstructable.",
        "models": {},
    }
    probes = pd.read_csv(ROOT / "audit/fresh_probes.csv")
    probe_outputs = []
    for kind, info in manifest["models"].items():
        if sha(release / info["file"]) != info["sha256"]:
            raise RuntimeError("Model hash mismatch")
        if sha(data / f"{kind}.csv") != prep["dataset_hashes"][kind]:
            raise RuntimeError("Dataset hash mismatch")
        model = joblib.load(release / info["file"])
        df = pd.read_csv(data / f"{kind}.csv", keep_default_na=False)
        test = df[df.split == "test"].copy()
        pred = model.probability(test.value.tolist())
        r = {
            "test": metrics(test.label, pred),
            "test_uncalibrated": metrics(
                test.label, model.probability(test.value.tolist(), False)
            ),
            "by_language": {
                str(lang): metrics(
                    test.label[test.language == lang], pred[test.language == lang]
                )
                for lang in test.language.unique()
            },
            "by_source": {
                str(source): metrics(
                    test.label[test.source == source], pred[test.source == source]
                )
                for source in test.source.unique()
            },
            "calibration_bins": [],
        }
        for lo in np.arange(0, 1, 0.1):
            mask = (pred >= lo) & (pred < lo + 0.1 if lo < 0.9 else pred <= 1)
            r["calibration_bins"].append(
                {
                    "lower": round(float(lo), 1),
                    "count": int(mask.sum()),
                    "mean_probability": float(pred[mask].mean())
                    if mask.any()
                    else None,
                    "observed_threat_fraction": float(
                        test.label.to_numpy()[mask].mean()
                    )
                    if mask.any()
                    else None,
                }
            )
        # Require entire current group to be absent from legacy train/validation.
        eligible = set(df.groupby("group").legacy_unseen.all().loc[lambda x: x].index)
        if kind == "sms":
            legacy = pd.read_csv(data / "legacy_sms_membership.csv")
            templates = set(
                legacy.loc[legacy.legacy_split != "test", "text"]
                .dropna()
                .astype(str)
                .map(sms_template)
            )
            eligible -= set(df.loc[df.template.isin(templates), "group"])
        common = test[test.group.isin(eligible)].copy()
        r["common_holdout"] = {
            "rows": len(common),
            "legacy_comparison": "requested"
            if compare_legacy
            else "not requested; use --compare-legacy with verified historical artifacts",
            "caveat": "URL legacy row membership reconstructed; current domain groups excluded if any retained row was seen. SMS exact/template and current near-group exclusions, not exhaustive semantic matching.",
        }
        if compare_legacy and len(common):
            r["common_holdout"]["before"] = metrics(
                common.label, legacy_prob(kind, common.value.tolist())
            )
            r["common_holdout"]["after"] = metrics(
                common.label, model.probability(common.value.tolist())
            )
        test.assign(probability=pred).to_csv(
            release / f"{kind}_test_predictions.csv", index=False
        )
        cases = probes[probes.kind == kind].copy()
        p = model.probability(cases.value.tolist())
        cases["after_probability"] = p
        cases["after_correct"] = (p >= 0.5) == cases.expected.to_numpy()
        r["audit_regression"] = {
            "rows": len(cases),
            "before_correct": int(cases.correct.sum()),
            "after_correct": int(cases.after_correct.sum()),
            "status": "developer diagnostics only; never independent test evidence",
        }
        probe_outputs.append(cases)
        results["models"][kind] = r
    pd.concat(probe_outputs).to_csv(release / "audit_regressions.csv", index=False)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, default=ROOT / "models/releases/reliable-v1")
    p.add_argument("--compare-legacy", action="store_true")
    a = p.parse_args()
    run(a.release, a.compare_legacy)
