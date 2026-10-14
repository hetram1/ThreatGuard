"""Build immutable, auditable datasets; never overwrite an existing preparation."""

import argparse
import json
from pathlib import Path
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split
from ml.reliable.preprocessing import (
    normalize_url,
    normalize_sms,
    url_group,
    sms_template,
    digest,
    split_for,
)
from ml.reliable.artifacts import ROOT, sha


def near_groups(templates):
    """Union exact templates and MinHash candidates with >=.85 word-bigram Jaccard.

    Candidate generation is approximate; it cannot guarantee detection of all
    paraphrases. Transitive components stay together in one split.
    """
    unique = sorted(set(templates))
    parent = list(range(len(unique)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lsh = MinHashLSH(threshold=0.7, num_perm=128)
    tokens = []
    edges = 0
    for i, t in enumerate(unique):
        words = t.split()
        shingles = set(" ".join(words[j : j + 2]) for j in range(len(words) - 1)) or {t}
        tokens.append(shingles)
        mh = MinHash(num_perm=128, seed=42)
        mh.update_batch([x.encode() for x in sorted(shingles)])
        for key in sorted(lsh.query(mh), key=int):
            k = int(key)
            other = tokens[k]
            if len(shingles & other) / len(shingles | other) >= 0.85:
                a, b = root(i), root(k)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                    edges += 1
        lsh.insert(str(i), mh)
    mapping = {t: digest(unique[root(i)]) for i, t in enumerate(unique)}
    return [mapping[t] for t in templates], {
        "templates": len(unique),
        "near_duplicate_unions": edges,
        "algorithm": "128-permutation MinHash LSH candidates (.7), exact word-bigram Jaccard >= .85; connected components",
    }


def prepare(out):
    if out.exists():
        raise SystemExit(
            f"Refusing to overwrite {out}; choose a new --output or reuse its manifest."
        )
    out.mkdir(parents=True)
    source_paths = [
        ROOT / "data/raw/phiusiil.csv",
        ROOT / "data/processed/benign_url_augmentation.csv",
        ROOT / "data/processed/benign_deep_path_augmentation.csv",
    ] + [ROOT / f"data/processed/sms/{s}.csv" for s in ["train", "validation", "test"]]
    inventory = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    artifacts = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted((ROOT / "models").glob("*.joblib"))
    }
    (out / "legacy_artifacts.json").write_text(json.dumps(artifacts, indent=2))
    raw = (
        pd.read_csv(source_paths[0], usecols=["URL", "label"])
        .drop_duplicates("URL")
        .reset_index(drop=True)
    )
    uci = pd.DataFrame(
        {
            "value": raw.URL.fillna("").astype(str),
            "label": (raw.label == 0).astype(int),
            "source": "UCI PhiUSIIL",
            "language": "n/a",
        }
    )
    # Preserve the historical V5 row-split membership before any normalization.
    parts = [uci[["value", "label"]]]
    for p in source_paths[1:3]:
        d = (
            pd.read_csv(p, usecols=["URL"])
            .drop_duplicates("URL")
            .reset_index(drop=True)
        )
        parts.append(pd.DataFrame({"value": d.URL.fillna("").astype(str), "label": 0}))
    combined = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("value")
        .reset_index(drop=True)
    )
    old_train, old_test = train_test_split(
        combined, test_size=0.2, random_state=42, stratify=combined.label
    )
    combined["split"] = "train"
    combined.loc[old_test.index, "split"] = "test"
    combined["row_sha256"] = combined.value.map(digest)
    combined.to_csv(out / "legacy_url_membership.csv", index=False)
    old_train_values = set(old_train.value)
    url = uci.copy()
    invalid = []
    clean = []
    for i, row in url.iterrows():
        try:
            clean.append(normalize_url(row.value))
        except ValueError as e:
            clean.append(None)
            invalid.append({"row": i, "reason": str(e)})
    url["normalized"] = clean
    url = url.dropna(subset=["normalized"]).copy()
    conflicts = url.groupby("normalized").label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    url = url[~url.normalized.isin(bad)].drop_duplicates("normalized").copy()
    url["group"] = url.normalized.map(url_group)
    url["split"] = url.group.map(split_for)
    url["row_sha256"] = url.normalized.map(digest)
    url["legacy_unseen"] = ~url.value.isin(old_train_values)
    url.to_csv(out / "url.csv", index=False)
    pd.DataFrame(invalid).to_csv(out / "invalid_url_rows.csv", index=False)
    frames = []
    for split in ["train", "validation", "test"]:
        f = pd.read_csv(ROOT / f"data/processed/sms/{split}.csv")
        f["legacy_split"] = split
        frames.append(f)
    sms = pd.concat(frames, ignore_index=True)
    sms.to_csv(out / "legacy_sms_membership.csv", index=False)
    invalid_sms = (
        sms.text.isna()
        | sms.text.fillna("").str.strip().eq("")
        | sms.text.fillna("").str.len().gt(10000)
    )
    sms.loc[invalid_sms].to_csv(out / "invalid_sms_rows.csv", index=False)
    sms = sms[~invalid_sms].copy()
    sms["value"] = sms.text.map(normalize_sms)
    sms["normalized"] = sms.value.str.casefold()
    old_seen = set(sms.loc[sms.legacy_split != "test", "normalized"])
    source_counts = (
        sms.groupby(["source_dataset", "language", "is_scam"])
        .size()
        .reset_index(name="count")
        .to_dict("records")
    )
    # Spam != scam; generated/benchmark examples are not independent real observations.
    excluded = sms.source_dataset.isin(
        ["UCI_SMS_Spam", "Synthetic_Tier_C", "Benchmark_Ground_Truth"]
    )
    sms = sms[~excluded].copy()
    conflicts_sms = sms.groupby("normalized").is_scam.nunique()
    bad_sms = set(conflicts_sms[conflicts_sms > 1].index)
    sms = sms[~sms.normalized.isin(bad_sms)].drop_duplicates("normalized").copy()
    templates = sms.value.map(sms_template).tolist()
    sms["group"], group_info = near_groups(templates)
    sms["template"] = templates
    sms["label"] = sms.is_scam.astype(int)
    sms["source"] = sms.source_dataset
    sms["split"] = sms.group.map(split_for)
    sms["row_sha256"] = sms.normalized.map(digest)
    sms["legacy_unseen"] = ~sms.normalized.isin(old_seen)
    sms[
        [
            "value",
            "normalized",
            "label",
            "source",
            "language",
            "group",
            "template",
            "split",
            "row_sha256",
            "legacy_unseen",
        ]
    ].to_csv(out / "sms.csv", index=False)
    checks = {}
    for name, frame in [("url", url), ("sms", sms)]:
        assert frame.groupby("group").split.nunique().max() == 1
        assert frame.groupby("row_sha256").split.nunique().max() == 1
        checks[name] = {
            "rows": len(frame),
            "groups": frame.group.nunique(),
            "group_overlap": 0,
            "normalized_row_overlap": 0,
            "split_counts": frame.groupby(["split", "label"])
            .size()
            .reset_index(name="count")
            .to_dict("records"),
        }
    manifest = {
        "schema": 1,
        "seed": 42,
        "source_hashes": inventory,
        "legacy_artifacts": artifacts,
        "legacy_membership_status": "URL reconstructed from V5 source code and current hashed inputs; SMS original file membership. Historical training execution not attested.",
        "source_counts_sms": source_counts,
        "invalid_sms_rows": int(invalid_sms.sum()),
        "excluded_sms_rows": int(excluded.sum()),
        "url_invalid_rows": len(invalid),
        "url_conflicting_normalized_values": len(bad),
        "sms_conflicting_normalized_values": len(bad_sms),
        "sms_grouping": group_info,
        "checks": checks,
        "dataset_hashes": {name: sha(out / f"{name}.csv") for name in ["url", "sms"]},
        "external_test": {
            "status": "unavailable",
            "used_for_training": False,
            "note": "No independently verified external corpus acquired. Internal test is never used for model selection.",
        },
        "url_scope": "Hostname only; original path labels are confounded. Synthetic benign augmentations excluded from new training.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "data/reliable/v2")
    args = p.parse_args()
    prepare(args.output)
