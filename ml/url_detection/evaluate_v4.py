from pathlib import Path
import json

import joblib
import numpy as np
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

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

UCI_PATH = ROOT / "data" / "raw" / "phiusiil.csv"

MODEL_PATH = ROOT / "models" / "url_char_logreg_v4.joblib"
VECTORIZER_PATH = ROOT / "models" / "url_char_vectorizer_v4.joblib"
SCALER_PATH = ROOT / "models" / "url_char_scaler_v4.joblib"

SUMMARY_PATH = ROOT / "models" / "audit" / "v4_evaluation_summary.json"
RESULTS_PATH = ROOT / "models" / "audit" / "v4_sanity_results.csv"
FP_PATH = ROOT / "models" / "audit" / "v4_false_positives.csv"
FN_PATH = ROOT / "models" / "audit" / "v4_false_negatives.csv"

SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 60)
print("ThreatGuard — V4 Evaluation")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n===== LOADING V4 =====")

model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

print("Model loaded.")
print("Vectorizer loaded.")
print("Scaler loaded.")


def predict_urls(urls):
    """
    V4 prediction pipeline.

    Returns phishing probability.
    """

    urls = pd.Series(
        urls,
        dtype="string",
    ).fillna("")

    X_char = vectorizer.transform(
        urls
    )

    engineered = extract_features_from_series(
        urls
    )

    engineered = engineered[
        FEATURE_NAMES
    ]

    engineered_scaled = scaler.transform(
        engineered
    )

    from scipy.sparse import hstack

    X = hstack(
        [
            X_char,
            engineered_scaled,
        ]
    )

    probabilities = (
        model
        .predict_proba(X)[:, 1]
    )

    return probabilities


# ============================================================
# 1. CURATED SANITY SET
# ============================================================

print(
    "\n===== CURATED SANITY TEST ====="
)

# IMPORTANT:
# These are developer sanity expectations,
# NOT dataset-derived ground truth.

sanity_data = [
    # --------------------------------------------------------
    # BENIGN / EXPECTED LEGITIMATE
    # --------------------------------------------------------

    (
        "https://google.com",
        0,
        "benign",
    ),

    (
        "https://www.google.com",
        0,
        "benign",
    ),

    (
        "https://amazon.com",
        0,
        "benign",
    ),

    (
        "https://www.microsoft.com",
        0,
        "benign",
    ),

    (
        "https://apple.com",
        0,
        "benign",
    ),

    (
        "https://github.com",
        0,
        "benign",
    ),

    (
        "https://example.com",
        0,
        "benign",
    ),

    (
        "https://stackoverflow.com",
        0,
        "benign",
    ),

    (
        "https://docs.python.org/3/",
        0,
        "benign",
    ),

    (
        "https://developer.mozilla.org/en-US/",
        0,
        "benign",
    ),

    (
        "https://wikipedia.org/wiki/India",
        0,
        "benign",
    ),

    (
        "https://python.org",
        0,
        "benign",
    ),

    (
        "https://mozilla.org",
        0,
        "benign",
    ),

    (
        "https://amazon.in",
        0,
        "benign",
    ),

    (
        "https://flipkart.com",
        0,
        "benign",
    ),

    (
        "https://gov.in",
        0,
        "benign",
    ),

    (
        "https://uidai.gov.in",
        0,
        "benign",
    ),

    (
        "https://rbi.org.in",
        0,
        "benign",
    ),

    (
        "https://docs.github.com/en",
        0,
        "benign",
    ),

    (
        "https://www.nasa.gov/mission",
        0,
        "benign",
    ),

    # --------------------------------------------------------
    # SUSPICIOUS / EXPECTED PHISHING
    # --------------------------------------------------------

    (
        "http://paypal-login-security.com/verify",
        1,
        "suspicious",
    ),

    (
        "http://secure-account-verification.com/login",
        1,
        "suspicious",
    ),

    (
        "http://verify-account-security.com/confirm",
        1,
        "suspicious",
    ),

    (
        "http://amazon-security-check.com/login",
        1,
        "suspicious",
    ),

    (
        "http://microsoft-account-verify.com/signin",
        1,
        "suspicious",
    ),

    (
        "http://appleid-security-check.com/login",
        1,
        "suspicious",
    ),

    (
        "http://paypal.com.user-login-security.com/verify",
        1,
        "suspicious",
    ),

    (
        "http://192.168.1.10/login",
        1,
        "suspicious",
    ),

    (
        "http://10.0.0.25/account/verify",
        1,
        "suspicious",
    ),

    (
        "http://secure-login-account-payment.com/update",
        1,
        "suspicious",
    ),

    (
        "http://bank-account-verification-required.com/login",
        1,
        "suspicious",
    ),

    (
        "http://wallet-security-verification.com/confirm",
        1,
        "suspicious",
    ),

    (
        "http://crypto-wallet-verify.com/login",
        1,
        "suspicious",
    ),

    (
        "http://free-prize-claim-now.com/winner",
        1,
        "suspicious",
    ),

    (
        "http://account-suspended-security-check.com/restore",
        1,
        "suspicious",
    ),

    (
        "http://signin-verification-required.com/auth",
        1,
        "suspicious",
    ),

    (
        "http://payment-confirmation-security.com/verify",
        1,
        "suspicious",
    ),

    (
        "http://login-verification-account.com/secure",
        1,
        "suspicious",
    ),

    (
        "http://update-billing-information-security.com/login",
        1,
        "suspicious",
    ),

    (
        "http://urgent-account-verification.com/confirm",
        1,
        "suspicious",
    ),
]


sanity_df = pd.DataFrame(
    sanity_data,
    columns=[
        "URL",
        "expected",
        "category",
    ],
)


sanity_probabilities = predict_urls(
    sanity_df["URL"]
)

sanity_df["phishing_probability"] = (
    sanity_probabilities
)

sanity_df["prediction"] = (
    sanity_df["phishing_probability"] >= 0.50
).astype(int)

sanity_df["correct"] = (
    sanity_df["prediction"]
    == sanity_df["expected"]
)


sanity_df.to_csv(
    RESULTS_PATH,
    index=False,
)


print(
    "\nSanity results:"
)

print(
    sanity_df[
        [
            "URL",
            "category",
            "expected",
            "phishing_probability",
            "prediction",
            "correct",
        ]
    ].to_string(index=False)
)


sanity_accuracy = (
    sanity_df["correct"]
    .mean()
)

print(
    f"\nSanity accuracy: "
    f"{sanity_accuracy:.4f}"
)

print(
    "Correct:",
    int(sanity_df["correct"].sum()),
    "/",
    len(sanity_df),
)


print(
    "\n===== SANITY BY CATEGORY ====="
)

for category, group in sanity_df.groupby(
    "category"
):

    print(
        category,
        ":",
        int(group["correct"].sum()),
        "/",
        len(group),
    )


# ============================================================
# 2. UNSEEN-DOMAIN UCI EVALUATION
# ============================================================

print(
    "\n===== UNSEEN-DOMAIN EVALUATION ====="
)

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


def extract_domain(url):
    from urllib.parse import urlparse

    value = str(url).strip()

    parsed = urlparse(
        value
        if "://" in value
        else "http://" + value
    )

    host = (
        parsed.hostname
        or ""
    ).lower()

    if host.startswith("www."):
        host = host[4:]

    return host


uci["domain"] = (
    uci["URL"]
    .fillna("")
    .astype(str)
    .map(extract_domain)
)


# Remove empty domains.
uci = uci[
    uci["domain"].str.len() > 0
].copy()


# UCI:
# 0 = phishing
# 1 = legitimate
#
# ThreatGuard:
# 1 = phishing
# 0 = legitimate

uci["threatguard_label"] = (
    uci["label"] == 0
).astype(int)


domains = (
    uci["domain"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(
    42
)

rng.shuffle(
    domains
)

split_index = int(
    len(domains) * 0.80
)

train_domains = set(
    domains[:split_index]
)

test_domains = set(
    domains[split_index:]
)


test_df = uci[
    uci["domain"].isin(
        test_domains
    )
].copy()


print(
    "Unique domains:",
    len(domains),
)

print(
    "Training domains:",
    len(train_domains),
)

print(
    "Unseen test domains:",
    len(test_domains),
)

print(
    "Unseen-domain test URLs:",
    len(test_df),
)


test_probabilities = predict_urls(
    test_df["URL"]
)

test_predictions = (
    test_probabilities >= 0.50
).astype(int)

y_true = (
    test_df["threatguard_label"]
    .to_numpy()
)


accuracy = accuracy_score(
    y_true,
    test_predictions,
)

precision = precision_score(
    y_true,
    test_predictions,
    zero_division=0,
)

recall = recall_score(
    y_true,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    test_predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_true,
    test_probabilities,
)

pr_auc = average_precision_score(
    y_true,
    test_probabilities,
)

cm = confusion_matrix(
    y_true,
    test_predictions,
)


print(
    "\nUnseen-domain metrics:"
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
# 3. FALSE POSITIVES / FALSE NEGATIVES
# ============================================================

print(
    "\n===== ERROR ANALYSIS ====="
)

error_df = test_df[
    [
        "URL",
        "domain",
        "threatguard_label",
    ]
].copy()

error_df["phishing_probability"] = (
    test_probabilities
)

error_df["prediction"] = (
    test_predictions
)

false_positives = error_df[
    (
        error_df["threatguard_label"] == 0
    )
    &
    (
        error_df["prediction"] == 1
    )
].sort_values(
    "phishing_probability",
    ascending=False,
)


false_negatives = error_df[
    (
        error_df["threatguard_label"] == 1
    )
    &
    (
        error_df["prediction"] == 0
    )
].sort_values(
    "phishing_probability",
    ascending=True,
)


false_positives.to_csv(
    FP_PATH,
    index=False,
)

false_negatives.to_csv(
    FN_PATH,
    index=False,
)


print(
    "\nFalse positives:",
    len(false_positives),
)

print(
    false_positives
    .head(20)
    .to_string(index=False)
)


print(
    "\nFalse negatives:",
    len(false_negatives),
)

print(
    false_negatives
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 4. SUMMARY
# ============================================================

summary = {
    "model":
        "character_tfidf_logistic_regression_v4",

    "sanity": {
        "total":
            int(len(sanity_df)),

        "correct":
            int(sanity_df["correct"].sum()),

        "accuracy":
            float(sanity_accuracy),

        "benign_correct":
            int(
                sanity_df[
                    sanity_df["category"] == "benign"
                ]["correct"].sum()
            ),

        "benign_total":
            int(
                (
                    sanity_df["category"]
                    == "benign"
                ).sum()
            ),

        "suspicious_correct":
            int(
                sanity_df[
                    sanity_df["category"] == "suspicious"
                ]["correct"].sum()
            ),

        "suspicious_total":
            int(
                (
                    sanity_df["category"]
                    == "suspicious"
                ).sum()
            ),
    },

    "unseen_domain": {
        "unique_domains":
            int(len(domains)),

        "train_domains":
            int(len(train_domains)),

        "test_domains":
            int(len(test_domains)),

        "test_urls":
            int(len(test_df)),

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

        "false_positives":
            int(len(false_positives)),

        "false_negatives":
            int(len(false_negatives)),
    },
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\nSaved:",
    SUMMARY_PATH,
)

print(
    "Saved:",
    RESULTS_PATH,
)

print(
    "Saved:",
    FP_PATH,
)

print(
    "Saved:",
    FN_PATH,
)


print(
    "\n=========================================="
)

print(
    "STEP 13 COMPLETE"
)

print(
    "=========================================="
)

print(
    "\nDO NOT COMMIT."
)

print(
    "DO NOT PUSH."
)

print(
    "\nUse the complete output to decide whether V4"
)

print(
    "is suitable for ThreatGuard."
)
