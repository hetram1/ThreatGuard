import json
import pandas as pd
import pytest
from ml.reliable.prepare import ROOT, sha
from backend.app.services.url_risk_engine import URLRiskEngine
from backend.app.services.sms_risk_engine import SMSRiskEngine


@pytest.fixture(scope="module")
def engines():
    return {"url": URLRiskEngine(), "sms": SMSRiskEngine()}


def test_release_integrity(engines):
    pointer = json.loads((ROOT / "models/active_release.json").read_text())
    release = ROOT / pointer["directory"]
    assert sha(release / "manifest.json") == pointer["manifest_sha256"]
    for kind, engine in engines.items():
        assert engine.metadata["artifact_sha256"] == sha(release / f"{kind}.joblib")


def test_url_equivalence_inference(engines):
    scores = [
        engines["url"].analyze(x)["probability"]
        for x in [
            "github.com",
            "github.com/",
            "https://github.com/",
            "https://github.com:443",
        ]
    ]
    assert len(set(scores)) == 1


def test_no_domain_allowlist(engines):
    assert "allowlist" not in vars(engines["url"].detector)
    assert engines["url"].analyze("https://github.com")["probability"] > 0


def test_dataset_hashes_and_group_separation():
    release = (
        ROOT
        / json.loads((ROOT / "models/active_release.json").read_text())["directory"]
    )
    manifest = json.loads((release / "manifest.json").read_text())
    directory = ROOT / manifest["data_directory"]
    if not (directory / "manifest.json").exists():
        pytest.skip(
            "Full split-membership verification requires local source datasets; published model/inference tests still run."
        )
    prep = json.loads((directory / "manifest.json").read_text())
    for kind in ["url", "sms"]:
        assert sha(directory / f"{kind}.csv") == prep["dataset_hashes"][kind]
        d = pd.read_csv(directory / f"{kind}.csv")
        assert d.groupby("group").split.nunique().max() == 1
        assert d.groupby("row_sha256").split.nunique().max() == 1
        assert set(d.label) == {0, 1}


PROBES = pd.read_csv(ROOT / "audit/fresh_probes.csv").to_dict("records")


@pytest.mark.parametrize(
    "case", PROBES, ids=[f"{p['kind']}-{i}" for i, p in enumerate(PROBES)]
)
def test_all_audit_probes_have_finite_explained_outputs(engines, case):
    result = engines[case["kind"]].analyze(case["value"])
    assert 0 <= result["probability"] <= 1
    assert result["warnings"] and result["score_description"]
    assert result["model_evidence"]
    assert result["prediction"] == (
        ("phishing" if case["kind"] == "url" else "scam")
        if result["probability"] >= result["threshold"]
        else "legitimate"
    )


@pytest.mark.parametrize(
    "case", PROBES, ids=[f"{p['kind']}-label-{i}" for i, p in enumerate(PROBES)]
)
def test_audit_classification_regressions(engines, case):
    # Fixed known limitations are enumerated in the frozen release's regression report.
    # XFAIL is explicit and counted, not a passing accuracy assertion.
    release = (
        ROOT
        / json.loads((ROOT / "models/active_release.json").read_text())["directory"]
    )
    report = pd.read_csv(release / "audit_regressions.csv")
    known = report.loc[
        (report.kind == case["kind"]) & (report.value == case["value"])
    ].iloc[0]
    result = engines[case["kind"]].analyze(case["value"])
    correct = int(result["prediction"] in ["scam", "phishing"]) == case["expected"]
    if not correct and not bool(known.after_correct):
        pytest.xfail(
            "Unresolved diagnostic classification; see frozen audit_regressions.csv. Not used to tune model."
        )
    assert correct
