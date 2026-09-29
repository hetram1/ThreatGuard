from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "url_features.csv"
)

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "url_feature_names.json"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "url_xgboost.joblib"
METRICS_PATH = MODEL_DIR / "url_model_metrics.json"


print("=" * 60)
print("ThreatGuard — XGBoost URL Classifier")
print("=" * 60)

print("\n===== LOADING DATA =====")

df = pd.read_csv(DATA_PATH)

with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

X = df[feature_names].copy()

# UCI:
# 1 = legitimate
# 0 = phishing
#
# ThreatGuard convention:
# 1 = phishing
# 0 = legitimate
y = (df["label"] == 0).astype(int)

print("Total samples:", len(df))
print("Features:", len(feature_names))

print("\nThreatGuard labels:")
print("0 = legitimate")
print("1 = phishing")

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\n===== CHECKING DATA =====")

print("Missing:", X.isnull().sum().sum())
print(
    "Infinite:",
    np.isinf(X.to_numpy()).sum(),
)

print("\n===== SPLITTING DATA =====")

# First: 80% development, 20% final test.
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Second: 75% train, 25% validation of development set.
# Final proportions:
# train = 60%
# validation = 20%
# test = 20%
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

print(
    f"Train:      {len(X_train):,} "
    f"({len(X_train) / len(X):.1%})"
)

print(
    f"Validation: {len(X_val):,} "
    f"({len(X_val) / len(X):.1%})"
)

print(
    f"Test:       {len(X_test):,} "
    f"({len(X_test) / len(X):.1%})"
)

print("\n===== TRAINING CLASS BALANCE =====")

print(y_train.value_counts().sort_index())

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / positive

print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4),
)

print("\n===== TRAINING XGBOOST =====")

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],
    verbose=False,
)

print("Training complete.")

print("\n===== EVALUATING =====")


def evaluate_split(name, X_split, y_split):
    probabilities = model.predict_proba(
        X_split
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_split,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_split,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_split,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_split,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_split,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_split,
                probabilities,
            )
        ),
    }

    print(f"\n--- {name.upper()} ---")

    for key, value in metrics.items():
        print(
            f"{key:12}: {value:.4f}"
        )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_split,
            predictions,
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_split,
            predictions,
            target_names=[
                "legitimate",
                "phishing",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return metrics


train_metrics = evaluate_split(
    "train",
    X_train,
    y_train,
)

val_metrics = evaluate_split(
    "validation",
    X_val,
    y_val,
)

test_metrics = evaluate_split(
    "test",
    X_test,
    y_test,
)

print("\n===== FEATURE IMPORTANCE =====")

importance = pd.Series(
    model.feature_importances_,
    index=feature_names,
).sort_values(
    ascending=False
)

print(
    importance.head(20).to_string()
)

print("\n===== SAVING MODEL =====")

joblib.dump(
    model,
    MODEL_PATH,
)

print("Model saved:")
print(MODEL_PATH)

print("\n===== SAVING METRICS =====")

metrics_output = {
    "model": "XGBoost",
    "random_state": 42,
    "features": feature_names,
    "feature_count": len(feature_names),
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "classification_threshold": 0.50,
    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics,
    "top_features": {
        str(k): float(v)
        for k, v in importance.head(20).items()
    },
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics_output,
        f,
        indent=2,
    )

print("Metrics saved:")
print(METRICS_PATH)

print("\n===== COMPLETE =====")
