from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from sms_features import (
    FEATURE_NAMES,
    extract_sms_features,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "sms"
)

MODEL_DIR = (
    ROOT
    / "models"
)

AUDIT_DIR = (
    MODEL_DIR
    / "audit"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 70)
print("THREATGUARD SMS DETECTOR V2")
print("TF-IDF + SECURITY FEATURES")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

train = pd.read_csv(
    DATA_DIR / "train.csv"
)

validation = pd.read_csv(
    DATA_DIR / "validation.csv"
)

test = pd.read_csv(
    DATA_DIR / "test.csv"
)


x_train = (
    train["text"]
    .fillna("")
    .astype(str)
    .values
)

x_val = (
    validation["text"]
    .fillna("")
    .astype(str)
    .values
)

x_test = (
    test["text"]
    .fillna("")
    .astype(str)
    .values
)

y_train = (
    train["is_scam"]
    .astype(int)
    .values
)

y_val = (
    validation["is_scam"]
    .astype(int)
    .values
)

y_test = (
    test["is_scam"]
    .astype(int)
    .values
)


print()
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# WORD TF-IDF
# ============================================================

print()
print("===== WORD TF-IDF =====")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.995,
    max_features=120_000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)

X_train_word = (
    word_vectorizer
    .fit_transform(x_train)
)

X_val_word = (
    word_vectorizer
    .transform(x_val)
)

X_test_word = (
    word_vectorizer
    .transform(x_test)
)

print(
    "Vocabulary:",
    len(
        word_vectorizer.vocabulary_
    )
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print()
print("===== CHARACTER TF-IDF =====")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=120_000,
    sublinear_tf=True,
    lowercase=True,
)

X_train_char = (
    char_vectorizer
    .fit_transform(x_train)
)

X_val_char = (
    char_vectorizer
    .transform(x_val)
)

X_test_char = (
    char_vectorizer
    .transform(x_test)
)

print(
    "Vocabulary:",
    len(
        char_vectorizer.vocabulary_
    )
)


# ============================================================
# STRUCTURED SECURITY FEATURES
# ============================================================

print()
print("===== SECURITY FEATURES =====")

train_security = extract_sms_features(
    x_train
)

val_security = extract_sms_features(
    x_val
)

test_security = extract_sms_features(
    x_test
)


print(
    "Feature count:",
    len(FEATURE_NAMES)
)

print(
    "Feature names:"
)

print(
    FEATURE_NAMES
)


# ============================================================
# SCALE STRUCTURED FEATURES
# ============================================================

scaler = StandardScaler()

train_security_scaled = (
    scaler.fit_transform(
        train_security
    )
)

val_security_scaled = (
    scaler.transform(
        val_security
    )
)

test_security_scaled = (
    scaler.transform(
        test_security
    )
)


# ============================================================
# COMBINE
# ============================================================

print()
print("===== COMBINING FEATURES =====")

X_train = hstack(
    [
        X_train_word,
        X_train_char,
        csr_matrix(
            train_security_scaled
        ),
    ],
    format="csr",
)

X_val = hstack(
    [
        X_val_word,
        X_val_char,
        csr_matrix(
            val_security_scaled
        ),
    ],
    format="csr",
)

X_test = hstack(
    [
        X_test_word,
        X_test_char,
        csr_matrix(
            test_security_scaled
        ),
    ],
    format="csr",
)


print(
    "Train matrix:",
    X_train.shape
)

print(
    "Validation matrix:",
    X_val.shape
)

print(
    "Test matrix:",
    X_test.shape
)


# ============================================================
# TRAIN
# ============================================================

print()
print("===== TRAINING =====")

model = LogisticRegression(
    C=3.0,
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=42,
)

model.fit(
    X_train,
    y_train
)


print(
    "Training complete."
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    name,
    X,
    y,
):

    probability = (
        model
        .predict_proba(X)[:, 1]
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                y,
                prediction
            )
        ),

        "precision": float(
            precision_score(
                y,
                prediction,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                prediction,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y,
                prediction,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y,
                probability
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y,
                probability
            )
        ),
    }

    cm = confusion_matrix(
        y,
        prediction,
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    for key, value in metrics.items():

        print(
            f"{key:12s}: {value:.6f}"
        )

    print()
    print("Confusion matrix:")
    print(cm)

    return (
        metrics,
        probability,
        prediction,
        cm,
    )


val_metrics, val_probability, val_prediction, val_cm = evaluate(
    "VALIDATION",
    X_val,
    y_val,
)

test_metrics, test_probability, test_prediction, test_cm = evaluate(
    "TEST",
    X_test,
    y_test,
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = {}

for threshold in [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:

    prediction = (
        test_probability >= threshold
    ).astype(int)

    result = {
        "accuracy": float(
            accuracy_score(
                y_test,
                prediction
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
    }

    threshold_results[
        str(threshold)
    ] = result

    print(
        f"threshold={threshold:.2f}"
        f" | accuracy={result['accuracy']:.4f}"
        f" | precision={result['precision']:.4f}"
        f" | recall={result['recall']:.4f}"
        f" | f1={result['f1']:.4f}"
    )


# ============================================================
# PER-LANGUAGE
# ============================================================

print()
print("=" * 70)
print("PER-LANGUAGE TEST")
print("=" * 70)

language_results = {}

for language in sorted(
    test["language"]
    .dropna()
    .unique()
):

    mask = (
        test["language"]
        .values
        == language
    )

    labels = y_test[mask]

    probability = (
        test_probability[mask]
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    result = {
        "samples": int(
            mask.sum()
        ),

        "scam_samples": int(
            labels.sum()
        ),

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
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                prediction,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                labels,
                prediction,
                zero_division=0,
            )
        ),
    }

    language_results[
        language
    ] = result

    print()
    print(language)

    for key, value in result.items():

        print(
            f"{key:12s}: {value}"
        )


# ============================================================
# SAVE TEST ERRORS
# ============================================================

results = test.copy()

results[
    "scam_probability"
] = test_probability

results[
    "predicted_is_scam"
] = test_prediction

results[
    "correct"
] = (
    results["is_scam"]
    ==
    results["predicted_is_scam"]
)


false_positives = results[
    (results["is_scam"] == 0)
    &
    (results["predicted_is_scam"] == 1)
].copy()

false_negatives = results[
    (results["is_scam"] == 1)
    &
    (results["predicted_is_scam"] == 0)
].copy()


fp_path = (
    AUDIT_DIR
    / "sms_v2_false_positives.csv"
)

fn_path = (
    AUDIT_DIR
    / "sms_v2_false_negatives.csv"
)


false_positives.to_csv(
    fp_path,
    index=False
)

false_negatives.to_csv(
    fn_path,
    index=False
)


print()
print(
    "False positives:",
    len(false_positives)
)

print(
    "False negatives:",
    len(false_negatives)
)


# ============================================================
# SECURITY FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("SECURITY FEATURE COEFFICIENTS")
print("=" * 70)


feature_count = len(
    FEATURE_NAMES
)

coefficients = (
    model.coef_[0]
    [-feature_count:]
)

feature_importance = (
    pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(
                coefficients
            ),
        }
    )
    .sort_values(
        "abs_coefficient",
        ascending=False
    )
)


print(
    feature_importance
    .to_string(
        index=False
    )
)


feature_importance.to_csv(
    AUDIT_DIR
    / "sms_v2_security_feature_importance.csv",
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": (
        "word + character TF-IDF "
        "+ structured SMS security features "
        "+ Logistic Regression"
    ),

    "label_convention": {
        "0": "legitimate",
        "1": "scam",
    },

    "train_samples": int(
        len(train)
    ),

    "validation_samples": int(
        len(validation)
    ),

    "test_samples": int(
        len(test)
    ),

    "security_feature_count": int(
        len(FEATURE_NAMES)
    ),

    "security_features": FEATURE_NAMES,

    "validation": val_metrics,

    "test": test_metrics,

    "threshold_analysis": threshold_results,

    "language_results": language_results,

    "false_positives": int(
        len(false_positives)
    ),

    "false_negatives": int(
        len(false_negatives)
    ),
}


metrics_path = (
    MODEL_DIR
    / "sms_v2_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_DIR
    / "sms_detector_v2.joblib"
)

joblib.dump(
    word_vectorizer,
    MODEL_DIR
    / "sms_v2_word_vectorizer.joblib"
)

joblib.dump(
    char_vectorizer,
    MODEL_DIR
    / "sms_v2_char_vectorizer.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR
    / "sms_v2_security_scaler.joblib"
)


print()
print("=" * 70)
print("SMS V2 COMPLETE")
print("=" * 70)

print()
print("TEST RESULTS")

for key, value in test_metrics.items():

    print(
        f"{key:12s}: {value:.6f}"
    )

print()
print(
    "False positives:",
    len(false_positives)
)

print(
    "False negatives:",
    len(false_negatives)
)

print()
print(
    "Saved:",
    metrics_path
)

