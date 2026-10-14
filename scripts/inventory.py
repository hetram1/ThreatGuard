"""Read-only project inventory, including ignored files and local Git storage."""

import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    totals = {}
    logical = 0
    allocated = 0
    seen = set()
    largest = []
    source_lines = 0
    for directory, dirs, files in os.walk(ROOT):
        for name in files:
            p = Path(directory) / name
            st = p.lstat()
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            size = st.st_size
            blocks = st.st_blocks * 512
            rel = p.relative_to(ROOT)
            bucket = rel.parts[0]
            totals[bucket] = totals.get(bucket, 0) + blocks
            logical += size
            allocated += blocks
            if bucket not in [".git", ".venv"] and "node_modules" not in rel.parts:
                largest.append({"path": str(rel), "bytes": size})
                if p.suffix in [
                    ".py",
                    ".js",
                    ".jsx",
                    ".ts",
                    ".tsx",
                    ".css",
                ] and not any(
                    x in rel.parts for x in ["__pycache__", "dist", ".pytest_cache"]
                ):
                    source_lines += len(p.read_bytes().splitlines())
    return {
        "allocated_file_bytes": allocated,
        "logical_file_bytes": logical,
        "directory_allocated_file_bytes": dict(
            sorted(totals.items(), key=lambda x: x[1], reverse=True)
        ),
        "largest_project_files": sorted(
            largest, key=lambda x: x["bytes"], reverse=True
        )[:35],
        "source_lines_excluding_generated": source_lines,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Inventory already exists; choose another output.")
    result = inventory()
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
