from pathlib import Path
import json

import pandas as pd

from url_features import (
    FEATURE_NAMES,
    extract_features_from_series,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phiusiil.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "url_features.csv"
)

FEATURE_NAMES_PATH = (
    PROCESSED_DIR
    / "url_feature_names.json"
)


print("=" * 60)
print("ThreatGuard — URL Feature Engineering")
print("=" * 60)

print("\nLoading raw dataset:")
print(RAW_PATH)

df = pd.read_csv(RAW_PATH)

print("\nRaw shape:", df.shape)

if "URL" not in df.columns:
    raise ValueError("URL column not found.")

if "label" not in df.columns:
    raise ValueError("label column not found.")

print("\nExtracting URL-only features...")

features = extract_features_from_series(
    df["URL"]
)

features["label"] = df["label"].astype(int).values

print("\nFeature matrix shape:")
print(features.shape)

print("\nNumber of URL features:")
print(len(FEATURE_NAMES))

print("\nFeature names:")
for index, name in enumerate(FEATURE_NAMES, 1):
    print(f"{index:2}. {name}")

print("\n===== LABEL DISTRIBUTION =====")

label_counts = features["label"].value_counts()

print(label_counts.to_string())

print("\n===== LABEL MEANING =====")
print("UCI label 1 = legitimate")
print("UCI label 0 = phishing")

print("\n===== FEATURE PREVIEW =====")
print(features.head(5).to_string())

print("\n===== FEATURE QUALITY =====")

missing = features.isnull().sum().sum()
infinite = features.replace(
    [float("inf"), float("-inf")],
    pd.NA,
).isna().sum().sum()

print("Missing values:", missing)
print("Infinite values:", infinite)

print("\n===== DUPLICATES =====")
print(
    "Duplicate feature rows:",
    features.duplicated().sum()
)

print("\n===== SAVING PROCESSED DATA =====")

features.to_csv(
    OUTPUT_PATH,
    index=False,
)

with open(
    FEATURE_NAMES_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        FEATURE_NAMES,
        file,
        indent=2,
    )

print("\nSaved feature dataset:")
print(OUTPUT_PATH)

print("\nSaved feature schema:")
print(FEATURE_NAMES_PATH)

print("\n===== COMPLETE =====")
