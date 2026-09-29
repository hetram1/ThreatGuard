from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"
AUDIT_DIR = MODEL_DIR / "audit"

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 70)
print("THREATGUARD SMS DETECTOR")
print("ROBUSTNESS / ADVERSARIAL AUDIT")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_DIR / "sms_detector.joblib"
)

word_vectorizer = joblib.load(
    MODEL_DIR / "sms_word_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    MODEL_DIR / "sms_char_vectorizer.joblib"
)


print()
print("===== MODEL LOADED =====")

print(
    "Model:",
    type(model).__name__
)

print(
    "Word vocabulary:",
    len(word_vectorizer.vocabulary_)
)

print(
    "Character vocabulary:",
    len(char_vectorizer.vocabulary_)
)


# ============================================================
# CURATED SECURITY TEST SET
# ============================================================

samples = [

    # --------------------------------------------------------
    # CLEARLY BENIGN
    # --------------------------------------------------------

    (
        "BENIGN",
        "Hey bro, are you coming to class tomorrow?"
    ),

    (
        "BENIGN",
        "Your order has been delivered. Thanks for shopping with us."
    ),

    (
        "BENIGN",
        "Meeting is scheduled for 5 PM today."
    ),

    (
        "BENIGN",
        "Mom please call me when you reach home."
    ),

    (
        "BENIGN",
        "Your appointment is confirmed for Monday at 10 AM."
    ),

    (
        "BENIGN",
        "Happy birthday! Have a great day."
    ),

    (
        "BENIGN",
        "The train will arrive at platform 2."
    ),

    (
        "BENIGN",
        "Your electricity bill of Rs 840 has been paid successfully."
    ),

    (
        "BENIGN",
        "Your Amazon order is out for delivery."
    ),

    (
        "BENIGN",
        "Google security alert: a new device signed in to your account."
    ),

    # --------------------------------------------------------
    # CLEAR SCAM
    # --------------------------------------------------------

    (
        "SCAM",
        "URGENT! Your bank account will be blocked today. Verify KYC immediately at https://secure-verify-account.example.com"
    ),

    (
        "SCAM",
        "Congratulations! You have won Rs 25,00,000. Pay Rs 499 processing fee to claim your prize."
    ),

    (
        "SCAM",
        "Your UPI account is suspended. Send OTP immediately to reactivate your account."
    ),

    (
        "SCAM",
        "Your SBI account will be closed within 24 hours. Update PAN and KYC now."
    ),

    (
        "SCAM",
        "Electricity connection will be disconnected tonight. Pay your pending bill immediately using this link."
    ),

    (
        "SCAM",
        "Dear customer, your parcel is on hold. Pay Rs 25 customs fee here to receive it."
    ),

    (
        "SCAM",
        "You have been selected for a guaranteed government job. Pay the registration fee now."
    ),

    (
        "SCAM",
        "Your credit card has been compromised. Call this number immediately to secure your account."
    ),

    (
        "SCAM",
        "KYC expired. Click the link below immediately or your bank account will be frozen."
    ),

    (
        "SCAM",
        "You won a lottery prize of Rs 50 lakh. Send your Aadhaar and bank details to claim it."
    ),

    # --------------------------------------------------------
    # INDIAN / HINGLISH
    # --------------------------------------------------------

    (
        "SCAM",
        "Aapka SBI account block ho jayega. KYC update karne ke liye link par turant click karein."
    ),

    (
        "SCAM",
        "Aapka UPI band hone wala hai. OTP share karke account activate karein."
    ),

    (
        "SCAM",
        "Aapko Rs 10 lakh ka lottery prize mila hai. Claim karne ke liye processing fee pay karein."
    ),

    (
        "SCAM",
        "Bijli connection aaj raat cut ho jayega. Pending bill abhi pay karein."
    ),

    (
        "SCAM",
        "Sir aapka KYC pending hai, verification ke liye Aadhaar details send karein."
    ),

    (
        "SCAM",
        "Congratulations sir, aapka number lucky draw me select hua hai. Prize lene ke liye registration fee pay karein."
    ),

    # --------------------------------------------------------
    # OBFUSCATED / EVASION
    # --------------------------------------------------------

    (
        "SCAM",
        "URGENT!!! y0ur bank acc0unt will be bl0cked. verify KYC n0w."
    ),

    (
        "SCAM",
        "C0ngratulati0ns! Y0u w0n Rs 50 lakh. Click n0w t0 claim."
    ),

    (
        "SCAM",
        "Your K.Y.C is exp1red. Upd8te your acc0unt immediately."
    ),

    (
        "SCAM",
        "Aapka acc0unt bl0ck h0 jayega, KYC abhi upd8te karein."
    ),

    # --------------------------------------------------------
    # URL HEAVY
    # --------------------------------------------------------

    (
        "SCAM",
        "Important security notice: verify your account now https://account-security-verification.example.com/login"
    ),

    (
        "SCAM",
        "Click https://tinyurl.com/abc123 to claim your cashback reward before midnight."
    ),

    (
        "SCAM",
        "Your refund is waiting. Open http://paypa1-secure-login.example.com immediately."
    ),

    # --------------------------------------------------------
    # BENIGN BUT SUSPICIOUS-LOOKING
    # --------------------------------------------------------

    (
        "BENIGN",
        "Your bank statement is available online. Please review it at your usual banking website."
    ),

    (
        "BENIGN",
        "Your package tracking link is https://www.example.com/tracking/12345."
    ),

    (
        "BENIGN",
        "Your OTP for login is 482913. Do not share it with anyone."
    ),

    (
        "BENIGN",
        "Your UPI payment of Rs 250 to the merchant was successful."
    ),

    (
        "BENIGN",
        "Your KYC verification was completed successfully."
    ),

    # --------------------------------------------------------
    # SHORT / AMBIGUOUS
    # --------------------------------------------------------

    (
        "AMBIGUOUS",
        "OTP 482913"
    ),

    (
        "AMBIGUOUS",
        "KYC update"
    ),

    (
        "AMBIGUOUS",
        "Verify account"
    ),

    (
        "AMBIGUOUS",
        "Call me urgently"
    ),

    (
        "AMBIGUOUS",
        "Claim reward"
    ),

]


df = pd.DataFrame(
    samples,
    columns=[
        "expected_type",
        "text",
    ],
)


# ============================================================
# PREDICT
# ============================================================

print()
print("===== RUNNING AUDIT =====")


word_features = word_vectorizer.transform(
    df["text"]
)

char_features = char_vectorizer.transform(
    df["text"]
)

features = hstack(
    [
        word_features,
        char_features,
    ],
    format="csr",
)


probabilities = model.predict_proba(
    features
)[:, 1]


df[
    "scam_probability"
] = probabilities

df[
    "prediction"
] = np.where(
    probabilities >= 0.50,
    "SCAM",
    "LEGITIMATE",
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()

for expected_type in [
    "BENIGN",
    "SCAM",
    "AMBIGUOUS",
]:

    print()
    print("=" * 70)
    print(expected_type)
    print("=" * 70)

    subset = df[
        df["expected_type"]
        == expected_type
    ]

    for _, row in subset.iterrows():

        print()
        print(
            "Expected:",
            row["expected_type"]
        )

        print(
            "Prediction:",
            row["prediction"]
        )

        print(
            "Probability:",
            f"{row['scam_probability']:.6f}"
        )

        print(
            "Text:",
            row["text"]
        )


# ============================================================
# CATEGORY SUMMARY
# ============================================================

print()
print("=" * 70)
print("CATEGORY SUMMARY")
print("=" * 70)


summary = {}


for category in [
    "BENIGN",
    "SCAM",
]:

    subset = df[
        df["expected_type"]
        == category
    ]

    expected_label = (
        0
        if category == "BENIGN"
        else 1
    )

    predicted_label = (
        subset["scam_probability"]
        >= 0.50
    ).astype(int)

    correct = (
        predicted_label
        == expected_label
    )

    summary[
        category
    ] = {
        "samples": int(
            len(subset)
        ),
        "correct": int(
            correct.sum()
        ),
        "accuracy": float(
            correct.mean()
        ),
    }

    print(
        f"{category:12s}"
        f" {correct.sum():2d}/{len(subset):2d}"
        f" ({correct.mean():.2%})"
    )


# ============================================================
# OBFUSCATION TEST
# ============================================================

print()
print("=" * 70)
print("OBFUSCATION TEST")
print("=" * 70)


obfuscation = [

    "URGENT your bank account will be blocked verify KYC now",

    "URGENT!!! y0ur bank acc0unt will be bl0cked. verify KYC n0w.",

    "C0ngratulati0ns! Y0u w0n Rs 50 lakh. Click n0w t0 claim.",

    "Congratulations! You won Rs 50 lakh. Click now to claim.",

    "Aapka account block ho jayega KYC abhi update karein.",

    "Aapka acc0unt bl0ck h0 jayega KYC abhi upd8te karein.",

]


obf_features = hstack(
    [
        word_vectorizer.transform(
            obfuscation
        ),
        char_vectorizer.transform(
            obfuscation
        ),
    ],
    format="csr",
)


obf_probabilities = (
    model.predict_proba(
        obf_features
    )[:, 1]
)


for text, probability in zip(
    obfuscation,
    obf_probabilities,
):

    print()
    print(
        f"{probability:.6f}",
        "|",
        text
    )


# ============================================================
# SAVE AUDIT
# ============================================================

audit_csv = (
    AUDIT_DIR
    / "sms_robustness_audit.csv"
)

df.to_csv(
    audit_csv,
    index=False,
)


audit_json = (
    AUDIT_DIR
    / "sms_robustness_summary.json"
)

summary_data = {
    "total_samples": int(
        len(df)
    ),
    "category_summary": summary,
    "model_threshold": 0.50,
    "note": (
        "Curated security sanity test. "
        "Expected labels are developer-defined "
        "for robustness testing and are not "
        "part of the training dataset."
    ),
}


with open(
    audit_json,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_data,
        f,
        indent=2,
    )


print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(audit_csv)
print(audit_json)

