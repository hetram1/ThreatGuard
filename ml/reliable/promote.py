"""Promote an evaluated release without overwriting legacy binaries."""

import argparse, json
from pathlib import Path
from ml.reliable.prepare import ROOT, sha


def promote(release):
    manifest = json.loads((release / "manifest.json").read_text())
    evaluation = json.loads((release / "evaluation.json").read_text())
    if evaluation["release_manifest_sha256"] != sha(release / "manifest.json"):
        raise RuntimeError("Evaluation does not match release")
    for name, info in manifest["models"].items():
        if sha(release / info["file"]) != info["sha256"]:
            raise RuntimeError("Artifact mismatch")
    for name, value in manifest["code_hashes"].items():
        if sha(ROOT / name) != value:
            raise RuntimeError("Preprocessing mismatch")
    pointer = {
        "directory": str(release.relative_to(ROOT)),
        "manifest_sha256": sha(release / "manifest.json"),
        "evaluation_sha256": sha(release / "evaluation.json"),
    }
    path = ROOT / "models/active_release.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(pointer, indent=2))
    tmp.replace(path)
    print(json.dumps(pointer, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, default=ROOT / "models/releases/reliable-v1")
    a = p.parse_args()
    promote(a.release.resolve())
