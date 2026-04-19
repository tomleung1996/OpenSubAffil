"""Tag each raw affiliation string with its detected language."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

from config import LANG_FREQ_CSV, RAW_AFFILIATION_CSV, SUPPORTED_LANGUAGES


def build_detector() -> "LanguageDetectorBuilder":
    languages = [getattr(Language, name) for name in SUPPORTED_LANGUAGES]
    return LanguageDetectorBuilder.from_languages(*languages).build()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_AFFILIATION_CSV,
                        help="CSV produced by sql/01_raw_affiliation_strings.sql.")
    parser.add_argument("--output", type=Path, default=LANG_FREQ_CSV,
                        help="Destination CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    detector = build_detector()
    detections = detector.detect_languages_in_parallel_of(df["raw_affiliation_string"].tolist())
    df["language"] = [d.name if d is not None else "UNKNOWN" for d in detections]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
