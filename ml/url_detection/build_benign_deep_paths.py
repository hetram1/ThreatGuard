from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "majestic_million.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "benign_deep_path_augmentation.csv"
)

SEED = 42
DOMAIN_SAMPLE_SIZE = 50_000

# Keep the augmentation controlled.
#
# 50,000 domains x 4 paths = 200,000 rows.
#
# These four structures specifically target the
# false-positive patterns observed during Step 13.

PATHS = [
    "/docs/",
    "/wiki/",
    "/en-US/",
    "/research/",
]


print("=" * 60)
print("ThreatGuard — Controlled Deep-Path Augmentation")
print("=" * 60)


df = pd.read_csv(
    INPUT,
    usecols=["GlobalRank", "Domain"],
)


domains = (
    df["Domain"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
)

domains = domains[
    domains.str.len() >= 4
]

domains = domains[
    ~domains.str.contains(
        r"[^a-z0-9.\-_]",
        regex=True,
    )
]

domains = (
    domains
    .drop_duplicates()
)


domains = domains.sample(
    n=min(
        DOMAIN_SAMPLE_SIZE,
        len(domains),
    ),
    random_state=SEED,
)


print(
    "Selected domains:",
    len(domains),
)

print(
    "Selected paths:",
    len(PATHS),
)


rows = []


for domain in domains:

    clean_domain = (
        domain[4:]
        if domain.startswith("www.")
        else domain
    )

    for path in PATHS:

        rows.append(
            {
                "URL":
                    f"https://{clean_domain}{path}",

                # ThreatGuard:
                # 0 = legitimate
                # 1 = phishing
                "label":
                    0,

                "source":
                    "majestic_benign_deep_path",
            }
        )


augmented = pd.DataFrame(
    rows
)


augmented = (
    augmented
    .drop_duplicates(
        subset=["URL"]
    )
    .reset_index(drop=True)
)


print(
    "\nGenerated rows:",
    len(augmented),
)

print(
    "Expected rows:",
    DOMAIN_SAMPLE_SIZE * len(PATHS),
)

print(
    "Labels:",
    augmented["label"]
    .value_counts()
    .to_dict(),
)

print(
    "Unique URLs:",
    augmented["URL"].nunique(),
)


print(
    "\nPath distribution:"
)

print(
    augmented["URL"]
    .str.extract(
        r"https?://[^/]+(/.*)"
    )[0]
    .value_counts()
    .to_string()
)


print(
    "\nSample:"
)

print(
    augmented
    .head(20)
    .to_string(index=False)
)


augmented.to_csv(
    OUTPUT,
    index=False,
)


print(
    "\nSaved:",
    OUTPUT,
)

print(
    "\n=========================================="
)

print(
    "CONTROLLED DEEP-PATH AUGMENTATION COMPLETE"
)

print(
    "=========================================="
)
