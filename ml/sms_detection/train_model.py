from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

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
    classification_report,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "sms"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 70)
print("THREATGUARD SMS DETECTOR")
print("WORD + CHARACTER TF-IDF + LOGISTIC REGRESSION")
print("=" * 70)


# ============================================================
# LOAD DATA
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


print()
print("===== DATASET =====")

print(
    "Train:",
    train.shape
)

print(
    "Validation:",
    validation.shape
)

print(
    "Test:",
    test.shape
)


# ============================================================
# LABEL CONVENTION
# ============================================================

# ThreatGuard convention:
#
# 0 = legitimate
# 1 = scam
#
# This exactly matches the dataset's is_scam field.

y_train = train["is_scam"].astype(int).values
y_val = validation["is_scam"].astype(int).values
y_test = test["is_scam"].astype(int).values

x_train = train["text"].fillna("").astype(str).values
x_val = validation["text"].fillna("").astype(str).values
x_test = test["text"].fillna("").astype(str).values


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("===== CLASS DISTRIBUTION =====")

for name, labels in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test),
]:

    values, counts = np.unique(
        labels,
        return_counts=True
    )

    print()
    print(name)

    for value, count in zip(
        values,
        counts
    ):

        percentage = (
            count / len(labels) * 100
        )

        label_name = (
            "LEGITIMATE"
            if value == 0
            else "SCAM"
        )

        print(
            f"{value} = {label_name:10s}"
            f" {count:6d}"
            f" ({percentage:6.2f}%)"
        )


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


X_train_word = word_vectorizer.fit_transform(
    x_train
)

X_val_word = word_vectorizer.transform(
    x_val
)

X_test_word = word_vectorizer.transform(
    x_test
)


print(
    "Word vocabulary:",
    len(word_vectorizer.vocabulary_)
)

print(
    "Train word matrix:",
    X_train_word.shape
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


X_train_char = char_vectorizer.fit_transform(
    x_train
)

X_val_char = char_vectorizer.transform(
    x_val
)

X_test_char = char_vectorizer.transform(
    x_test
)


print(
    "Character vocabulary:",
    len(char_vectorizer.vocabulary_)
)

print(
    "Train character matrix:",
    X_train_char.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print()
print("===== COMBINING FEATURES =====")

X_train = hstack(
    [
        X_train_word,
        X_train_char,
    ],
    format="csr",
)

X_val = hstack(
    [
        X_val_word,
        X_val_char,
    ],
    format="csr",
)

X_test = hstack(
    [
        X_test_word,
        X_test_char,
    ],
    format="csr",
)


print(
    "Combined train matrix:",
    X_train.shape
)

print(
    "Combined validation matrix:",
    X_val.shape
)

print(
    "Combined test matrix:",
    X_test.shape
)


# ============================================================
# MODEL
# ============================================================

print()
print("===== TRAINING MODEL =====")

model = LogisticRegression(
    C=4.0,
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
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    X,
    y,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                y,
                predictions
            )
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probabilities
            )
        ),
    }

    cm = confusion_matrix(
        y,
        predictions
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

    print()
    print("Classification report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "legitimate",
                "scam",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return (
        metrics,
        probabilities,
        predictions,
        cm,
    )


# ============================================================
# VALIDATION
# ============================================================

val_metrics, val_prob, val_pred, val_cm = evaluate(
    "VALIDATION",
    X_val,
    y_val,
)


# ============================================================
# TEST
# ============================================================

test_metrics, test_prob, test_pred, test_cm = evaluate(
    "TEST",
    X_test,
    y_test,
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS — TEST")
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

    predictions = (
        test_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    threshold_results[
        str(threshold)
    ] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    print(
        f"threshold={threshold:.2f}"
        f" | accuracy={accuracy:.4f}"
        f" | precision={precision:.4f}"
        f" | recall={recall:.4f}"
        f" | f1={f1:.4f}"
    )


# ============================================================
# PER-LANGUAGE EVALUATION
# ============================================================

print()
print("=" * 70)
print("PER-LANGUAGE TEST EVALUATION")
print("=" * 70)

language_results = {}

for language in sorted(
    test["language"].dropna().unique()
):

    mask = (
        test["language"]
        .values
        == language
    )

    labels = y_test[mask]
    probabilities = test_prob[mask]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    result = {
        "samples": int(mask.sum()),
        "scam_samples": int(
            labels.sum()
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
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
# SAVE MISCLASSIFICATIONS
# ============================================================

print()
print("===== SAVING MISCLASSIFICATIONS =====")

results_df = test.copy()

results_df[
    "scam_probability"
] = test_prob

results_df[
    "predicted_is_scam"
] = test_pred

results_df[
    "correct"
] = (
    results_df["is_scam"]
    == results_df["predicted_is_scam"]
)


false_positives = results_df[
    (results_df["is_scam"] == 0)
    &
    (results_df["predicted_is_scam"] == 1)
].copy()

false_negatives = results_df[
    (results_df["is_scam"] == 1)
    &
    (results_df["predicted_is_scam"] == 0)
].copy()


audit_dir = MODEL_DIR / "audit"

audit_dir.mkdir(
    parents=True,
    exist_ok=True,
)


fp_path = (
    audit_dir
    / "sms_false_positives.csv"
)

fn_path = (
    audit_dir
    / "sms_false_negatives.csv"
)


false_positives.to_csv(
    fp_path,
    index=False
)

false_negatives.to_csv(
    fn_path,
    index=False
)


print(
    "False positives:",
    len(false_positives)
)

print(
    "False negatives:",
    len(false_negatives)
)

print(
    "Saved:",
    fp_path
)

print(
    "Saved:",
    fn_path
)


# ============================================================
# DISPLAY MOST CONFIDENT ERRORS
# ============================================================

print()
print("=" * 70)
print("MOST CONFIDENT FALSE POSITIVES")
print("=" * 70)

if len(false_positives) > 0:

    print(
        false_positives
        .sort_values(
            "scam_probability",
            ascending=False
        )[
            [
                "text",
                "language",
                "scam_probability",
                "head2_scam_intent",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No false positives."
    )


print()
print("=" * 70)
print("MOST CONFIDENT FALSE NEGATIVES")
print("=" * 70)

if len(false_negatives) > 0:

    print(
        false_negatives
        .sort_values(
            "scam_probability",
            ascending=True
        )[
            [
                "text",
                "language",
                "scam_probability",
                "head2_scam_intent",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No false negatives."
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": (
        "word + character TF-IDF "
        "+ Logistic Regression"
    ),

    "label_convention": {
        "0": "legitimate",
        "1": "scam",
    },

    "training_samples": int(
        len(train)
    ),

    "validation_samples": int(
        len(validation)
    ),

    "test_samples": int(
        len(test)
    ),

    "word_tfidf": {
        "analyzer": "word",
        "ngram_range": [1, 2],
        "max_features": 120000,
        "min_df": 2,
        "max_df": 0.995,
        "sublinear_tf": True,
    },

    "char_tfidf": {
        "analyzer": "char",
        "ngram_range": [3, 5],
        "max_features": 120000,
        "min_df": 2,
        "sublinear_tf": True,
    },

    "logistic_regression": {
        "C": 4.0,
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": 1000,
    },

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

    "random_state": 42,
}


metrics_path = (
    MODEL_DIR
    / "sms_baseline_metrics.json"
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

print()
print("===== SAVING MODEL =====")

joblib.dump(
    model,
    MODEL_DIR
    / "sms_detector.joblib"
)

joblib.dump(
    word_vectorizer,
    MODEL_DIR
    / "sms_word_vectorizer.joblib"
)

joblib.dump(
    char_vectorizer,
    MODEL_DIR
    / "sms_char_vectorizer.joblib"
)


print(
    "Model saved."
)

print(
    "Metrics saved:"
)

print(
    metrics_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SMS BASELINE COMPLETE")
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
print("Production candidate:")
print(
    "sms_detector.joblib"
)

print()
print("IMPORTANT:")
print(
    "These are baseline results."
)

print(
    "Do not claim them on the resume yet."
)

