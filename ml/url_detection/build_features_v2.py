from pathlib import Path
import json

import pandas as pd

from url_features_v2 import (
    FEATURE_NAMES,
    extract_features_from_series,
)


ROOT = Path(__file__).resolve().parents[2]

raw_path = (
    ROOT / "data" / "raw" / "phiusiil.csv"
)

output_dir = (
    ROOT / "data" / "processed"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(raw_path)

features = extract_features_from_series(
    df["URL"]
)

# UCI:
# 1 = legitimate
# 0 = phishing
features["label"] = (
    df["label"] == 0
).astype(int)

features.to_csv(
    output_dir / "url_features_v2.csv",
    index=False,
)

with open(
    output_dir / "url_feature_names_v2.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        FEATURE_NAMES,
        f,
        indent=2,
    )

print(
    "Raw samples:",
    len(df),
)

print(
    "V2 features:",
    len(FEATURE_NAMES),
)

print(
    "Output:",
    output_dir / "url_features_v2.csv",
)

print("\nFeatures:")
for i, name in enumerate(
    FEATURE_NAMES,
    1,
):
    print(
        f"{i:2}. {name}"
    )
