from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "sms"
MODEL_DIR = ROOT / "models"
AUDIT_DIR = MODEL_DIR / "audit"


# ============================================================
# LOAD V1
# ============================================================

print("=" * 70)
print("THREATGUARD SMS V1 — THRESHOLD CALIBRATION")
print("=" * 70)

model = joblib.load(
    MODEL_DIR / "sms_detector.joblib"
)

word_vectorizer = joblib.load(
    MODEL_DIR / "sms_word_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    MODEL_DIR / "sms_char_vectorizer.joblib"
)


validation = pd.read_csv(
    DATA_DIR / "validation.csv"
)

test = pd.read_csv(
    DATA_DIR / "test.csv"
)


def make_features(
    texts
):

    word = word_vectorizer.transform(
        texts
    )

    char = char_vectorizer.transform(
        texts
    )

    return hstack(
        [
            word,
            char,
        ],
        format="csr",
    )


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print()
print("===== VALIDATION =====")

X_val = make_features(
    validation["text"]
    .fillna("")
    .astype(str)
)

y_val = (
    validation["is_scam"]
    .astype(int)
    .values
)

val_probability = (
    model
    .predict_proba(X_val)[:, 1]
)


# ============================================================
# TEST PROBABILITIES
# ============================================================

print()
print("===== TEST =====")

X_test = make_features(
    test["text"]
    .fillna("")
    .astype(str)
)

y_test = (
    test["is_scam"]
    .astype(int)
    .values
)

test_probability = (
    model
    .predict_proba(X_test)[:, 1]
)


# ============================================================
# THRESHOLD TABLE
# ============================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


def calculate(
    probability,
    labels,
    threshold,
):

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            prediction,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "threshold": threshold,

        "accuracy": float(
            accuracy_score(
                labels,
                prediction
            )
        ),

        "precision": float(
            precision_score(
                labels,
                prediction,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                prediction,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                labels,
                prediction,
                zero_division=0
            )
        ),

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


print()
print("=" * 70)
print("VALIDATION THRESHOLD TABLE")
print("=" * 70)

validation_results = []

for threshold in thresholds:

    result = calculate(
        val_probability,
        y_val,
        threshold,
    )

    validation_results.append(
        result
    )

    print(
        f"{threshold:.2f}"
        f" | acc={result['accuracy']:.4f}"
        f" | precision={result['precision']:.4f}"
        f" | recall={result['recall']:.4f}"
        f" | F1={result['f1']:.4f}"
        f" | FP={result['false_positives']:3d}"
        f" | FN={result['false_negatives']:3d}"
    )


# ============================================================
# CHOOSE THRESHOLD FROM VALIDATION ONLY
# ============================================================

# Primary criterion:
# maximize F1 on validation.
#
# Tie-break:
# higher precision.
#
# This selection NEVER looks at test labels.

selected = sorted(
    validation_results,
    key=lambda x: (
        x["f1"],
        x["precision"],
    ),
    reverse=True,
)[0]


selected_threshold = (
    selected["threshold"]
)


print()
print("=" * 70)
print("SELECTED THRESHOLD")
print("=" * 70)

print(
    "Threshold:",
    selected_threshold
)

print(
    "Validation F1:",
    selected["f1"]
)

print(
    "Validation precision:",
    selected["precision"]
)

print(
    "Validation recall:",
    selected["recall"]
)


# ============================================================
# ONE-TIME TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION AT VALIDATION-SELECTED THRESHOLD")
print("=" * 70)

test_result = calculate(
    test_probability,
    y_test,
    selected_threshold,
)


for key, value in test_result.items():

    print(
        f"{key:20s}: {value}"
    )


# ============================================================
# COMPARE AGAINST DEFAULT 0.50
# ============================================================

default_result = calculate(
    test_probability,
    y_test,
    0.50,
)


print()
print("=" * 70)
print("DEFAULT 0.50 VS SELECTED THRESHOLD")
print("=" * 70)

print()
print("DEFAULT 0.50")

for key in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_positives",
    "false_negatives",
]:

    print(
        f"{key:20s}:",
        default_result[key]
    )


print()
print(
    f"SELECTED {selected_threshold:.2f}"
)

for key in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_positives",
    "false_negatives",
]:

    print(
        f"{key:20s}:",
        test_result[key]
    )


# ============================================================
# ROBUSTNESS AUDIT AT SELECTED THRESHOLD
# ============================================================

print()
print("=" * 70)
print("ROBUSTNESS AUDIT AT SELECTED THRESHOLD")
print("=" * 70)


audit_path = (
    AUDIT_DIR
    / "sms_robustness_audit.csv"
)

audit = pd.read_csv(
    audit_path
)


audit_features = make_features(
    audit["text"]
    .fillna("")
    .astype(str)
)

audit_probability = (
    model
    .predict_proba(
        audit_features
    )[:, 1]
)

audit[
    "calibrated_probability"
] = audit_probability

audit[
    "calibrated_prediction"
] = np.where(
    audit_probability
    >= selected_threshold,
    "SCAM",
    "LEGITIMATE",
)


for category in [
    "BENIGN",
    "SCAM",
]:

    subset = audit[
        audit["expected_type"]
        == category
    ]

    expected = (
        0
        if category == "BENIGN"
        else 1
    )

    predicted = (
        subset[
            "calibrated_probability"
        ]
        >= selected_threshold
    ).astype(int)

    correct = (
        predicted
        == expected
    )

    print(
        f"{category:12s}"
        f" {correct.sum():2d}/{len(subset):2d}"
        f" ({correct.mean():.2%})"
    )


# ============================================================
# SAVE
# ============================================================

calibration = {
    "model": (
        "SMS V1 word + character TF-IDF "
        "+ Logistic Regression"
    ),

    "selection_method": (
        "maximize validation F1; "
        "test labels not used for selection"
    ),

    "selected_threshold": (
        float(selected_threshold)
    ),

    "validation_selected_result": selected,

    "test_result": test_result,

    "default_0_50_test_result": (
        default_result
    ),

    "validation_results": (
        validation_results
    ),

    "note": (
        "Threshold is an operating point, "
        "not a probability calibration guarantee."
    ),
}


output_path = (
    MODEL_DIR
    / "sms_threshold_calibration.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        calibration,
        f,
        indent=2,
    )


audit_output = (
    AUDIT_DIR
    / "sms_robustness_calibrated.csv"
)

audit.to_csv(
    audit_output,
    index=False,
)


print()
print("=" * 70)
print("STEP 22 COMPLETE")
print("=" * 70)

print()
print(
    "Calibration saved:",
    output_path
)

print(
    "Audit saved:",
    audit_output
)

print()
print(
    "DO NOT COMMIT OR PUSH."
)

