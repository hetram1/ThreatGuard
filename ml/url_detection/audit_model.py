from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap

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

from url_features import (
    extract_features_from_series,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phiusiil.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "url_xgboost.joblib"
)

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "url_feature_names.json"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "models"
    / "audit"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 60)
print("ThreatGuard — URL Model Audit")
print("=" * 60)

print("\n===== LOADING MODEL =====")

model = joblib.load(MODEL_PATH)

with open(
    FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8",
) as f:
    feature_names = json.load(f)

print("Model loaded.")
print("Feature count:", len(feature_names))

print("\n===== LOADING RAW DATA =====")

df = pd.read_csv(RAW_PATH)

# Remove exact duplicate URLs before the audit.
# This tests generalization beyond repeated identical URLs.
df_unique = df.drop_duplicates(
    subset=["URL"]
).reset_index(drop=True)

print("Original rows:", len(df))
print("Unique URLs:", len(df_unique))
print(
    "Removed duplicate URLs:",
    len(df) - len(df_unique),
)

print("\n===== BUILDING AUDIT SET =====")

# UCI label:
# 1 = legitimate
# 0 = phishing
#
# ThreatGuard:
# 0 = legitimate
# 1 = phishing
y = (
    df_unique["label"] == 0
).astype(int)

X = extract_features_from_series(
    df_unique["URL"]
)

X = X[feature_names]

print("Audit samples:", len(X))

# Fixed random split, separate from training script.
X_dev, X_test, y_dev, y_test, urls_dev, urls_test = (
    train_test_split(
        X,
        y,
        df_unique["URL"],
        test_size=0.20,
        random_state=123,
        stratify=y,
    )
)

print(
    "Independent audit test samples:",
    len(X_test),
)

print("\n===== AUDIT PERFORMANCE =====")

probabilities = model.predict_proba(
    X_test
)[:, 1]


def print_threshold_metrics(
    threshold: float,
):
    predictions = (
        probabilities >= threshold
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

    print(
        f"\nThreshold: {threshold:.2f}"
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

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )


print_threshold_metrics(0.30)
print_threshold_metrics(0.40)
print_threshold_metrics(0.50)
print_threshold_metrics(0.60)
print_threshold_metrics(0.70)

print("\nROC-AUC:")
print(
    f"{roc_auc_score(y_test, probabilities):.4f}"
)

print("\nPR-AUC:")
print(
    f"{average_precision_score(y_test, probabilities):.4f}"
)

print("\n===== FINDING MODEL ERRORS =====")

predictions = (
    probabilities >= 0.50
).astype(int)

results = pd.DataFrame(
    {
        "url": urls_test.values,
        "actual": y_test.values,
        "probability_phishing": probabilities,
        "prediction": predictions,
    }
)

false_positives = results[
    (results["actual"] == 0)
    & (results["prediction"] == 1)
].sort_values(
    "probability_phishing",
    ascending=False,
)

false_negatives = results[
    (results["actual"] == 1)
    & (results["prediction"] == 0)
].sort_values(
    "probability_phishing",
    ascending=True,
)

print(
    "\nFalse positives:",
    len(false_positives),
)

print(
    "False negatives:",
    len(false_negatives),
)

print("\nTop false positives:")

if len(false_positives):
    print(
        false_positives.head(10).to_string(
            index=False
        )
    )
else:
    print("None.")

print("\nTop false negatives:")

if len(false_negatives):
    print(
        false_negatives.head(10).to_string(
            index=False
        )
    )
else:
    print("None.")

false_positives.to_csv(
    AUDIT_DIR / "false_positives.csv",
    index=False,
)

false_negatives.to_csv(
    AUDIT_DIR / "false_negatives.csv",
    index=False,
)

print("\n===== SHAP EXPLAINABILITY =====")

# Use a manageable sample for SHAP.
sample_size = min(
    1000,
    len(X_test),
)

X_shap = X_test.iloc[
    :sample_size
].copy()

print(
    "SHAP sample size:",
    len(X_shap),
)

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_shap
)

# Handle possible SHAP output formats.
if isinstance(shap_values, list):
    shap_matrix = np.asarray(
        shap_values[-1]
    )
else:
    shap_matrix = np.asarray(
        shap_values
    )

mean_abs_shap = np.mean(
    np.abs(shap_matrix),
    axis=0,
)

shap_importance = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    )
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
)

print("\nTop SHAP features:")
print(
    shap_importance.head(20).to_string(
        index=False
    )
)

shap_importance.to_csv(
    AUDIT_DIR / "shap_feature_importance.csv",
    index=False,
)

print("\n===== MANUAL THREAT TESTS =====")

manual_urls = [
    "https://www.google.com",
    "https://www.amazon.com",
    "https://github.com",
    "https://paypal.com/login",
    "http://192.168.1.10/login",
    "http://secure-account-verification.example.com/login",
    "https://example.com",
    "http://verify-account-security.example.com/update",
]

manual_features = extract_features_from_series(
    pd.Series(manual_urls)
)

manual_features = manual_features[
    feature_names
]

manual_probabilities = model.predict_proba(
    manual_features
)[:, 1]

manual_results = pd.DataFrame(
    {
        "url": manual_urls,
        "phishing_probability":
            manual_probabilities,
        "risk_score":
            manual_probabilities * 100,
    }
)

print(
    manual_results.to_string(
        index=False
    )
)

manual_results.to_csv(
    AUDIT_DIR / "manual_url_tests.csv",
    index=False,
)

print("\n===== SAVING AUDIT SUMMARY =====")

audit_summary = {
    "original_samples": int(len(df)),
    "unique_urls": int(len(df_unique)),
    "independent_test_samples": int(
        len(X_test)
    ),
    "roc_auc": float(
        roc_auc_score(
            y_test,
            probabilities,
        )
    ),
    "pr_auc": float(
        average_precision_score(
            y_test,
            probabilities,
        )
    ),
    "false_positives_at_0_5": int(
        len(false_positives)
    ),
    "false_negatives_at_0_5": int(
        len(false_negatives)
    ),
}

with open(
    AUDIT_DIR / "audit_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        audit_summary,
        f,
        indent=2,
    )

print(
    AUDIT_DIR / "audit_summary.json"
)

print("\n===== AUDIT COMPLETE =====")
