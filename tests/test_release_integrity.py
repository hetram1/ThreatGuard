import json
import pytest
from ml.reliable.prepare import sha


def test_artifact_tampering_is_rejected(tmp_path, monkeypatch):
    import backend.app.services.reliable_engine as module

    release = tmp_path / "models/releases/test"
    release.mkdir(parents=True)
    model = release / "url.joblib"
    model.write_bytes(b"not a trusted artifact")
    manifest = {"models": {"url": {"file": "url.joblib", "sha256": "0" * 64}}}
    (release / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "models/active_release.json").write_text(
        json.dumps(
            {
                "directory": "models/releases/test",
                "manifest_sha256": sha(release / "manifest.json"),
            }
        )
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="model checksum mismatch"):
        module.ReliableRiskEngine("url")


def test_manifest_tampering_is_rejected(tmp_path, monkeypatch):
    import backend.app.services.reliable_engine as module

    release = tmp_path / "models/releases/test"
    release.mkdir(parents=True)
    (release / "manifest.json").write_text("{}")
    (tmp_path / "models/active_release.json").write_text(
        json.dumps({"directory": "models/releases/test", "manifest_sha256": "0" * 64})
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="manifest checksum mismatch"):
        module.ReliableRiskEngine("url")
