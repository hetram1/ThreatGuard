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
)

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "data"
    / "processed"
    / "url_features_v2.csv"
)

FEATURES = (
    ROOT
    / "data"
    / "processed"
    / "url_feature_names_v2.json"
)

MODEL_DIR = ROOT / "models"

df = pd.read_csv(DATA)

with open(
    FEATURES,
    "r",
    encoding="utf-8",
) as f:
    feature_names = json.load(f)

X = df[feature_names]
y = df["label"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)

negative = int(
    (y_train == 0).sum()
)

positive = int(
    (y_train == 1).sum()
)

model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_weight=3,
    reg_alpha=0.25,
    reg_lambda=1.5,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=negative / positive,
)

print("=" * 60)
print("ThreatGuard — V2 URL Model")
print("=" * 60)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))
print("Features:", len(feature_names))

print("\nTraining...")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_test, y_test),
    ],
    verbose=False,
)

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)

metrics = {
    "accuracy": accuracy_score(
        y_test,
        predictions,
    ),
    "precision": precision_score(
        y_test,
        predictions,
        zero_division=0,
    ),
    "recall": recall_score(
        y_test,
        predictions,
        zero_division=0,
    ),
    "f1": f1_score(
        y_test,
        predictions,
        zero_division=0,
    ),
    "roc_auc": roc_auc_score(
        y_test,
        probabilities,
    ),
    "pr_auc": average_precision_score(
        y_test,
        probabilities,
    ),
}

print("\n===== V2 TEST METRICS =====")

for key, value in metrics.items():
    print(
        f"{key:12}: {value:.4f}"
    )

print("\n===== TOP FEATURES =====")

importance = (
    pd.Series(
        model.feature_importances_,
        index=feature_names,
    )
    .sort_values(
        ascending=False
    )
)

print(
    importance.head(20).to_string()
)

model_path = (
    MODEL_DIR / "url_xgboost_v2.joblib"
)

metrics_path = (
    MODEL_DIR / "url_model_metrics_v2.json"
)

joblib.dump(
    model,
    model_path,
)

output = {
    "model": "XGBoost URL V2",
    "features": feature_names,
    "test_metrics": {
        k: float(v)
        for k, v in metrics.items()
    },
    "top_features": {
        k: float(v)
        for k, v in importance.head(20).items()
    },
}

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        indent=2,
    )

print("\nModel:")
print(model_path)

print("\nMetrics:")
print(metrics_path)
