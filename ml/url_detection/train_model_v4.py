from pathlib import Path
import json

import joblib
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
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

UCI_PATH = (
    ROOT
    / "data"
    / "raw"
    / "phiusiil.csv"
)

BENIGN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "benign_url_augmentation.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "url_char_logreg_v4.joblib"
)

VECTORIZER_PATH = (
    ROOT
    / "models"
    / "url_char_vectorizer_v4.joblib"
)

SCALER_PATH = (
    ROOT
    / "models"
    / "url_char_scaler_v4.joblib"
)

METRICS_PATH = (
    ROOT
    / "models"
    / "url_v4_metrics.json"
)


print("=" * 60)
print("ThreatGuard — URL Model V4")
print("=" * 60)


# ============================================================
# LOAD ORIGINAL UCI DATA
# ============================================================

print("\n===== LOADING UCI DATA =====")

uci = pd.read_csv(
    UCI_PATH
)

uci = (
    uci
    .drop_duplicates(
        subset=["URL"],
        keep="first",
    )
    .reset_index(drop=True)
)

uci_urls = (
    uci["URL"]
    .fillna("")
    .astype(str)
)

# UCI:
# 0 = phishing
# 1 = legitimate
#
# ThreatGuard:
# 0 = legitimate
# 1 = phishing

uci_labels = (
    uci["label"] == 0
).astype(int)


print(
    "UCI rows:",
    len(uci),
)

print(
    "UCI phishing:",
    int(uci_labels.sum()),
)

print(
    "UCI legitimate:",
    int((uci_labels == 0).sum()),
)


# ============================================================
# LOAD BENIGN AUGMENTATION
# ============================================================

print(
    "\n===== LOADING BENIGN AUGMENTATION ====="
)

benign = pd.read_csv(
    BENIGN_PATH
)

benign = (
    benign
    .drop_duplicates(
        subset=["URL"],
        keep="first",
    )
    .reset_index(drop=True)
)

benign_urls = (
    benign["URL"]
    .fillna("")
    .astype(str)
)

benign_labels = (
    pd.Series(
        0,
        index=benign.index,
        dtype="int64",
    )
)


print(
    "Augmented rows:",
    len(benign),
)

print(
    "Augmented legitimate:",
    len(benign),
)


# ============================================================
# COMBINE
# ============================================================

urls = pd.concat(
    [
        uci_urls,
        benign_urls,
    ],
    ignore_index=True,
)

labels = pd.concat(
    [
        uci_labels.reset_index(drop=True),
        benign_labels,
    ],
    ignore_index=True,
)


combined = pd.DataFrame(
    {
        "URL": urls,
        "label": labels,
    }
)


combined = (
    combined
    .drop_duplicates(
        subset=["URL"],
        keep="first",
    )
    .reset_index(drop=True)
)


print(
    "\n===== COMBINED DATASET ====="
)

print(
    "Rows:",
    len(combined),
)

print(
    "Phishing:",
    int(
        (combined["label"] == 1).sum()
    ),
)

print(
    "Legitimate:",
    int(
        (combined["label"] == 0).sum()
    ),
)


# ============================================================
# SPLIT
# ============================================================

X_urls = combined["URL"].astype(str)
y = combined["label"].astype(int)


X_train_urls, X_test_urls, y_train, y_test = (
    train_test_split(
        X_urls,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)


print(
    "\n===== SPLIT ====="
)

print(
    "Train:",
    len(X_train_urls),
)

print(
    "Test:",
    len(X_test_urls),
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print(
    "\n===== CHARACTER TF-IDF ====="
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=150_000,
    min_df=2,
    sublinear_tf=True,
)


X_train_char = vectorizer.fit_transform(
    X_train_urls
)

X_test_char = vectorizer.transform(
    X_test_urls
)


print(
    "Vocabulary:",
    len(vectorizer.vocabulary_),
)

print(
    "Character train matrix:",
    X_train_char.shape,
)


# ============================================================
# ENGINEERED V2 FEATURES
# ============================================================

print(
    "\n===== ENGINEERED FEATURES ====="
)

train_engineered = (
    extract_features_from_series(
        X_train_urls
    )
)

test_engineered = (
    extract_features_from_series(
        X_test_urls
    )
)


train_engineered = train_engineered[
    FEATURE_NAMES
]

test_engineered = test_engineered[
    FEATURE_NAMES
]


scaler = StandardScaler()

train_engineered_scaled = (
    scaler.fit_transform(
        train_engineered
    )
)

test_engineered_scaled = (
    scaler.transform(
        test_engineered
    )
)


# ============================================================
# COMBINE
# ============================================================

X_train = hstack(
    [
        X_train_char,
        train_engineered_scaled,
    ]
)

X_test = hstack(
    [
        X_test_char,
        test_engineered_scaled,
    ]
)


print(
    "Combined train matrix:",
    X_train.shape,
)

print(
    "Combined test matrix:",
    X_test.shape,
)


# ============================================================
# CLASS BALANCE
# ============================================================

phishing = int(
    (y_train == 1).sum()
)

legitimate = int(
    (y_train == 0).sum()
)

class_weight = {
    0: 1.0,
    1: legitimate / phishing,
}


print(
    "\nClass weights:",
    class_weight,
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\n===== TRAINING V4 ====="
)

model = LogisticRegression(
    C=4.0,
    class_weight=class_weight,
    max_iter=1000,
    solver="liblinear",
    random_state=42,
)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n===== EVALUATION ====="
)

probabilities = (
    model
    .predict_proba(X_test)[:, 1]
)

predictions = (
    probabilities >= 0.50
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions,
)

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

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)

cm = confusion_matrix(
    y_test,
    predictions,
)


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# SAVE
# ============================================================

print(
    "\n===== SAVING V4 ====="
)

joblib.dump(
    model,
    MODEL_PATH,
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH,
)

joblib.dump(
    scaler,
    SCALER_PATH,
)


metrics = {
    "model":
        "character_tfidf_logistic_regression_v4",

    "uci_rows":
        int(len(uci)),

    "benign_augmentation_rows":
        int(len(benign)),

    "combined_rows":
        int(len(combined)),

    "train_samples":
        int(len(X_train_urls)),

    "test_samples":
        int(len(X_test_urls)),

    "character_ngram_range":
        [3, 5],

    "character_max_features":
        150000,

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "pr_auc":
        float(pr_auc),

    "confusion_matrix":
        cm.tolist(),
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


print(
    "\nSaved model:",
    MODEL_PATH,
)

print(
    "Saved vectorizer:",
    VECTORIZER_PATH,
)

print(
    "Saved scaler:",
    SCALER_PATH,
)

print(
    "Saved metrics:",
    METRICS_PATH,
)


print(
    "\n=========================================="
)

print(
    "V4 TRAINING COMPLETE"
)

print(
    "=========================================="
)
