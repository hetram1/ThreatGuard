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
    / "benign_url_augmentation.csv"
)


SEED = 42

# We want enough domain diversity to teach the model
# that legitimate URLs can contain paths, queries,
# subdomains and normal application structures.
#
# 50,000 domains x 2 URLs = 100,000 benign examples.

DOMAIN_SAMPLE_SIZE = 50_000


print("=" * 60)
print("ThreatGuard — Controlled Benign Augmentation")
print("=" * 60)


df = pd.read_csv(
    INPUT,
    usecols=["GlobalRank", "Domain"],
)


print(
    "\nAvailable domains:",
    len(df),
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


print(
    "Valid unique domains:",
    len(domains),
)


# Prefer a broad rank-based sample rather than
# only the absolute top domains.
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


# ------------------------------------------------------------
# Two deliberately different URL structures per domain.
#
# Root URL teaches the model that normal domains are benign.
#
# Path URL directly addresses the false-positive problem
# discovered with:
#
# docs.python.org/3/
# developer.mozilla.org/en-US/
# wikipedia.org/wiki/India
# ------------------------------------------------------------

rows = []


for domain in domains:

    clean_domain = (
        domain[4:]
        if domain.startswith("www.")
        else domain
    )

    # Root URL
    rows.append(
        {
            "URL":
                f"https://{clean_domain}",
            "label": 1,
            "source":
                "majestic_benign_root",
        }
    )

    # Normal application/documentation path
    rows.append(
        {
            "URL":
                f"https://{clean_domain}/docs/getting-started",
            "label": 1,
            "source":
                "majestic_benign_path",
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
    "Labels:",
    augmented["label"]
    .value_counts()
    .to_dict(),
)


print(
    "\nSource distribution:"
)

print(
    augmented["source"]
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
    "CONTROLLED AUGMENTATION COMPLETE"
)

print(
    "=========================================="
)
