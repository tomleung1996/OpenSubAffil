"""Attach each NER row to its canonical department name."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import CANONICAL_DEPT_CSV, DEDUPE_OUTPUT_CSV, NER_PROCESSED_CSV

OUTPUT_COLUMNS = [
    "raw_affiliation_id", "institution_id",
    "raw_dept_str", "clean_dept_str",
    "start", "end", "frequency",
    "canonical_dept_name", "cluster_total_frequency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-detailed", type=Path, default=NER_PROCESSED_CSV)
    parser.add_argument("--clusters", type=Path, default=DEDUPE_OUTPUT_CSV)
    parser.add_argument("--output", type=Path, default=CANONICAL_DEPT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ner = pd.read_csv(args.ner_detailed)
    clusters = pd.read_csv(args.clusters, dtype={"institution_id": str})
    print(f"Loaded {len(ner):,} NER rows and {len(clusters):,} cluster rows.")

    merged = ner.merge(
        clusters,
        left_on=["institution_ids", "clean_dept_str"],
        right_on=["institution_id", "member_dept_name"],
        how="left",
    )
    matched = merged["institution_id"].notnull()
    print(f"Matched {matched.sum():,} rows ({matched.mean():.2%}).")
    merged = merged[matched]

    merged = merged.rename(columns={"raw_aff_id": "raw_affiliation_id"})
    merged["raw_affiliation_id"] = merged["raw_affiliation_id"].astype(int)
    merged["frequency"] = merged["frequency"].astype(int)
    merged["cluster_total_frequency"] = merged["cluster_total_frequency"].astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged[OUTPUT_COLUMNS].to_csv(args.output, index=False)
    print(f"Wrote {len(merged):,} rows to {args.output}")


if __name__ == "__main__":
    main()
