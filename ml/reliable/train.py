"""Train on train, select on selection, calibrate on calibration. Never read test rows here."""

import argparse, json, platform
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from ml.reliable.model import Detector
from ml.reliable.prepare import sha, ROOT


def metrics(y, p, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = p >= threshold
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "confusion_matrix": cm.tolist(),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
    }


def run(data, out):
    if out.exists():
        raise SystemExit(
            "Release already exists. Choose a fresh --output; immutable releases are never overwritten."
        )
    source = json.loads((data / "manifest.json").read_text())
    for kind in ["url", "sms"]:
        if sha(data / f"{kind}.csv") != source["dataset_hashes"][kind]:
            raise RuntimeError("Prepared dataset hash mismatch.")
    out.mkdir(parents=True)
    summary = {
        "schema": 1,
        "data_manifest_sha256": sha(data / "manifest.json"),
        "data_directory": str(data.relative_to(ROOT)),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "selection_rule": "highest selection F1, tie lower Brier; fixed 0.5 threshold. Calibration chosen by selection Brier, fitted exclusively on separate calibration split.",
        "models": {},
        "external_test": "not used for development",
        "preprocessing_version": "2026-10-14-v1",
        "code_hashes": {
            name: sha(ROOT / name)
            for name in ["ml/reliable/preprocessing.py", "ml/reliable/model.py"]
        },
    }
    for kind in ["url", "sms"]:
        # CSV parsing sees file bytes; immediately discard held-out rows without computing features or metrics.
        df = pd.read_csv(data / f"{kind}.csv")
        df = df[df.split != "test"].copy()
        train = df[df.split == "train"]
        selection = df[df.split == "selection"]
        cal = df[df.split == "calibration"]
        candidates = []
        best = None
        best_key = None
        for C, weight in [(1.0, None), (4.0, "balanced")]:
            print(f"Training {kind} C={C} weight={weight}", flush=True)
            model = Detector(kind, C, weight).fit(train.value.tolist(), train.label)
            raw = model.probability(selection.value.tolist(), calibrated=False)
            m = metrics(selection.label, raw)
            key = (m["f1"], -m["brier"])
            candidates.append({"C": C, "class_weight": weight, "selection": m})
            if best_key is None or key > best_key:
                best = model
                best_key = key
        best.calibrate(cal.value.tolist(), cal.label)
        raw = metrics(
            selection.label, best.probability(selection.value.tolist(), False)
        )
        calibrated = metrics(
            selection.label, best.probability(selection.value.tolist())
        )
        use_calibration = calibrated["brier"] < raw["brier"]
        if not use_calibration:
            best.calibrator = None
        filename = f"{kind}.joblib"
        joblib.dump(best, out / filename, compress=3)
        summary["models"][kind] = {
            "file": filename,
            "sha256": sha(out / filename),
            "version": "reliable-v1",
            "type": "Hostname character TF-IDF + logistic regression"
            if kind == "url"
            else "Word + character TF-IDF + logistic regression",
            "C": best.C,
            "class_weight": best.class_weight,
            "threshold": 0.5,
            "calibrated": use_calibration,
            "calibration_method": "held-out sigmoid"
            if use_calibration
            else "none; sigmoid did not improve selection Brier",
            "selection_uncalibrated": raw,
            "selection_calibrated": calibrated,
            "candidates": candidates,
            "training_rows": len(train),
            "selection_rows": len(selection),
            "calibration_rows": len(cal),
        }
        (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/reliable/v2")
    p.add_argument("--output", type=Path, default=ROOT / "models/releases/reliable-v1")
    a = p.parse_args()
    run(a.data, a.output)
