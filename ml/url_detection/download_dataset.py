from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ThreatGuard — UCI PhiUSIIL Dataset Downloader")
print("=" * 60)

print("\nFetching UCI dataset ID 967...")

dataset = fetch_ucirepo(id=967)

X = dataset.data.features
y = dataset.data.targets

print("\n===== FEATURES =====")
print("Shape:", X.shape)
print("Columns:")
for i, column in enumerate(X.columns, 1):
    print(f"{i:3}. {column}")

print("\n===== TARGET =====")
print("Shape:", y.shape)
print("Columns:", list(y.columns))

print("\n===== TARGET DISTRIBUTION =====")
for column in y.columns:
    print(y[column].value_counts(dropna=False))

print("\n===== COMBINING DATA =====")

df = X.copy()

for column in y.columns:
    df[column] = y[column]

output_path = RAW_DIR / "phiusiil.csv"
df.to_csv(output_path, index=False)

print("\nSaved:")
print(output_path)

print("\n===== DATASET INFO =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Memory:", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n===== FIRST 5 ROWS =====")
print(df.head().to_string())

print("\n===== MISSING VALUES =====")
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing.to_string())

print("\n===== DUPLICATES =====")
print("Duplicate rows:", df.duplicated().sum())

print("\n===== COMPLETE =====")
