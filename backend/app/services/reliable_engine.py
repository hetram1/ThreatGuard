"""Load only checksum-verified releases; keep inference identical to training."""

import json
from pathlib import Path
import joblib
from ml.reliable.artifacts import sha
from .heuristic_signals import url_signals, sms_signals
from ml.reliable.preprocessing import normalize_sms
from backend.app.schemas.url_validation import validated_url as normalize_url

ROOT = Path(__file__).resolve().parents[3]


class ReliableRiskEngine:
    def __init__(self, kind):
        self.kind = kind
        pointer = json.loads((ROOT / "models/active_release.json").read_text())
        release = (ROOT / pointer["directory"]).resolve()
        if not release.is_relative_to((ROOT / "models/releases").resolve()):
            raise RuntimeError("Invalid release path")
        if sha(release / "manifest.json") != pointer["manifest_sha256"]:
            raise RuntimeError("Release manifest checksum mismatch")
        manifest = json.loads((release / "manifest.json").read_text())
        self.info = manifest["models"][kind]
        path = release / self.info["file"]
        if sha(path) != self.info["sha256"]:
            raise RuntimeError(f"{kind} model checksum mismatch")
        for filename, expected in manifest.get("code_hashes", {}).items():
            if sha(ROOT / filename) != expected:
                raise RuntimeError(f"Inference code checksum mismatch: {filename}")
        self.detector = joblib.load(path)
        self.metadata = {
            "name": f"ThreatGuard {kind.upper()} Detector",
            "version": self.info["version"],
            "type": self.info["type"],
            "artifact_sha256": self.info["sha256"],
        }

    def analyze(self, value):
        value = (normalize_url if self.kind == "url" else normalize_sms)(value)
        p = float(self.detector.probability([value])[0])
        threshold = self.info["threshold"]
        threat = "phishing" if self.kind == "url" else "scam"
        prediction = threat if p >= threshold else "legitimate"
        if self.kind == "url":
            reasons = url_signals(value)["reasons"]
            warnings = [
                "Hostname-only lexical assessment. Page content, redirects, URL paths and reputation are not evaluated; a low score does not establish safety."
            ]
        else:
            reasons = sms_signals(value)["reasons"]
            warnings = [
                "Text-only assessment; sender identity and linked pages are not verified. Hindi/Hinglish evidence is limited."
            ]
        return {
            "prediction": prediction,
            "risk_score": round(p * 100, 2),
            "probability": p,
            "confidence": max(p, 1 - p),
            "risk_band": "LOW"
            if p < 0.2
            else "MEDIUM"
            if p < 0.5
            else "HIGH"
            if p < 0.8
            else "CRITICAL",
            "threshold": threshold,
            "model": self.metadata,
            "reasons": reasons,
            "model_evidence": self.detector.evidence(value),
            "warnings": warnings,
            "calibration": self.info["calibration_method"],
            "score_description": "Model estimate under the training distribution, not probability of real-world safety or certainty.",
            "normalized_input": value,
        }
