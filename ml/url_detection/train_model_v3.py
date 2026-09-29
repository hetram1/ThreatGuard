from pathlib import Path
import json

import joblib
import pandas as pd

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

from xgboost import XGBClassifier

from url_features_v3 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = (
    ROOT
    / "data"
    / "raw"
    / "phiusiil.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "url_xgboost_v3.joblib"
)

METRICS_PATH = (
    ROOT
    / "models"
    / "url_v3_metrics.json"
)

FEATURE_PATH = (
    ROOT
    / "models"
    / "url_v3_feature_names.json"
)


print("=" * 60)
print("ThreatGuard — URL Model V3")
print("=" * 60)


print("\n===== LOADING DATA =====")

df = pd.read_csv(
    RAW_DATA
)

df = (
    df
    .drop_duplicates(
        subset=["URL"],
        keep="first",
    )
    .reset_index(drop=True)
)

urls = (
    df["URL"]
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

y = (
    df["label"] == 0
).astype(int)


print(
    "Rows:",
    len(df),
)

print(
    "Phishing:",
    int(y.sum()),
)

print(
    "Legitimate:",
    int((y == 0).sum()),
)


print(
    "\n===== BUILDING V3 FEATURES ====="
)

X = extract_features_from_series(
    urls
)

X = X[
    FEATURE_NAMES
]

print(
    "Feature count:",
    len(FEATURE_NAMES),
)

print(
    "Matrix:",
    X.shape,
)


print(
    "\n===== SPLITTING ====="
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)


scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)


print(
    "scale_pos_weight:",
    scale_pos_weight,
)


model = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_weight=3,
    reg_alpha=0.20,
    reg_lambda=1.50,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)


print(
    "\n===== TRAINING ====="
)

model.fit(
    X_train,
    y_train,
)


print(
    "\n===== TESTING ====="
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
    "\n===== V3 TEST RESULTS ====="
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


print(
    "\n===== TOP FEATURES ====="
)

importance = (
    pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance":
                model.feature_importances_,
        }
    )
    .sort_values(
        "importance",
        ascending=False,
    )
)

print(
    importance
    .head(25)
    .to_string(index=False)
)


print(
    "\n===== SAVING ====="
)

joblib.dump(
    model,
    MODEL_PATH,
)

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "model": "xgboost_url_v3",
            "dataset_rows": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "feature_count": int(len(FEATURE_NAMES)),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist(),
        },
        f,
        indent=2,
    )

with open(
    FEATURE_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        FEATURE_NAMES,
        f,
        indent=2,
    )


print(
    "\nSaved model:",
    MODEL_PATH,
)

print(
    "Saved metrics:",
    METRICS_PATH,
)

print(
    "Saved features:",
    FEATURE_PATH,
)

print(
    "\n=========================================="
)

print(
    "V3 TRAINING COMPLETE"
)

print(
    "=========================================="
)
