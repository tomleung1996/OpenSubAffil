"""Clean NER entities and attribute each department to a single parent institution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from config import (
    ABBR_LOOKUP_CSV,
    DEDUPE_INPUT_CSV,
    INSTITUTION_ID_NAME_CSV,
    NER_OUTPUT_JSONL,
    NER_PROCESSED_CSV,
    RAW_AFF_TO_INSTITUTIONS_CSV,
)
from text_utils import clean_department_string, is_abbreviation

ORG_MATCH_SCORE_THRESHOLD = 75
RIGHT_ORG_MAX_DISTANCE = 64
RIGHT_ORG_MAX_WINDOW = 3
FLUSH_ROWS = 1_000_000

DETAILED_COLUMNS = [
    "raw_aff_id", "institution_ids", "inst_count",
    "raw_dept_str", "clean_dept_str", "start", "end", "frequency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-jsonl", type=Path, default=NER_OUTPUT_JSONL)
    parser.add_argument("--raw-aff-institutions", type=Path, default=RAW_AFF_TO_INSTITUTIONS_CSV,
                        help="CSV produced by sql/02_raw_aff_to_institutions.sql.")
    parser.add_argument("--institution-names", type=Path, default=INSTITUTION_ID_NAME_CSV)
    parser.add_argument("--abbr-lookup", type=Path, default=ABBR_LOOKUP_CSV)
    parser.add_argument("--detailed-output", type=Path, default=NER_PROCESSED_CSV)
    parser.add_argument("--dedupe-output", type=Path, default=DEDUPE_INPUT_CSV)
    return parser.parse_args()


def load_raw_aff_to_institutions(path: Path) -> dict[int, list[str]]:
    """Return {raw_affiliation_string_id: [institution_id, ...]}.

    The CSV has columns ``raw_affiliation_string_id`` and ``institution_ids_str``
    (semicolon-separated institution IDs produced by STRING_AGG in SQL).
    """
    df = pd.read_csv(path)
    mapping: dict[int, list[str]] = {}
    for raw_id, ids_str in zip(df["raw_affiliation_string_id"], df["institution_ids_str"]):
        if pd.isna(ids_str):
            continue
        mapping[int(raw_id)] = [s.strip() for s in str(ids_str).split(";") if s.strip()]
    return mapping


def load_institution_names(path: Path) -> tuple[dict[str, str], set[str]]:
    """Return (cleaned-name lookup, set of known institution names)."""
    df = pd.read_csv(path)
    df["institution_id"] = df["institution_id"].astype(str)
    df["clean_inst_name"] = df["institution_name"].apply(clean_department_string)
    id_to_clean = dict(zip(df["institution_id"], df["clean_inst_name"]))
    known_names = set(df["institution_name"].astype(str).str.lower().tolist())
    return id_to_clean, known_names


def load_abbreviation_lookup(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(
        df["abbr_name"].astype(str).str.lower(),
        df["full_name"].astype(str).str.lower(),
    ))


def split_entities_by_reset(entities: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a row's entities into segments whenever ``start`` decreases.

    The NER model emits entities per span, so start offsets reset at each span.
    """
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_start: int | None = None
    for ent in entities:
        start = ent.get("start")
        if prev_start is not None and start is not None and start < prev_start:
            if current:
                segments.append(current)
                current = []
        current.append(ent)
        if start is not None:
            prev_start = start
    if current:
        segments.append(current)
    return segments


def is_monotonic_starts(segment: list[dict[str, Any]]) -> bool:
    last: int | None = None
    for ent in segment:
        start = ent.get("start")
        if start is None:
            continue
        if last is not None and start < last:
            return False
        last = start
    return True


def match_org_to_candidate(
    org_text: str,
    candidate_ids: list[str],
    inst_id_to_name: dict[str, str],
    threshold: int = ORG_MATCH_SCORE_THRESHOLD,
) -> str | None:
    """Fuzzy-match an ORG entity text to one of the candidate institution names."""
    if not org_text:
        return None
    best_id, best_score = None, -1
    for inst_id in candidate_ids:
        name = inst_id_to_name.get(inst_id)
        if not name:
            continue
        score = fuzz.token_sort_ratio(org_text, name)
        if score > best_score:
            best_id, best_score = inst_id, score
    return best_id if best_score >= threshold else None


def pick_right_org_by_distance(sub_end: int | None, org_matches: list[dict[str, Any]]) -> str | None:
    if sub_end is None:
        return None
    rights = [o for o in org_matches if o.get("start") is not None and o["start"] >= sub_end]
    if not rights:
        return None
    rights.sort(key=lambda x: x["start"])
    closest = rights[0]
    if (closest["start"] - sub_end) <= RIGHT_ORG_MAX_DISTANCE:
        return closest["inst_id"]
    return None


def pick_right_org_by_order(sub_order: int, org_matches: list[dict[str, Any]]) -> str | None:
    rights = [o for o in org_matches if o.get("order") is not None and o["order"] > sub_order]
    if not rights:
        return None
    rights.sort(key=lambda x: x["order"])
    closest = rights[0]
    if (closest["order"] - sub_order) <= RIGHT_ORG_MAX_WINDOW:
        return closest["inst_id"]
    return None


def _flush(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=DETAILED_COLUMNS)
    header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=header, index=False,
              encoding="utf-8", quoting=csv.QUOTE_ALL)


def iter_segment_rows(
    segment: list[dict[str, Any]],
    candidate_ids: list[str],
    inst_id_to_name: dict[str, str],
    known_inst_names: set[str],
    abbr_lookup: dict[str, str],
    raw_aff_id: int,
    inst_count: int,
    frequency: int,
) -> Iterable[dict[str, Any]]:
    monotonic = is_monotonic_starts(segment)
    candidate_ids_str = ";".join(candidate_ids)

    org_matches: list[dict[str, Any]] = []
    for order, ent in enumerate(segment):
        if ent.get("entity_group") != "ORG":
            continue
        start, end = ent.get("start"), ent.get("end")
        if start is None or end is None:
            continue
        cleaned = clean_department_string(ent.get("word", ""))
        if is_abbreviation(cleaned) and cleaned in abbr_lookup:
            cleaned = abbr_lookup[cleaned]
        if not cleaned:
            continue
        matched = match_org_to_candidate(cleaned, candidate_ids, inst_id_to_name)
        if matched:
            org_matches.append({"start": start, "end": end, "inst_id": matched, "order": order})

    for order, ent in enumerate(segment):
        if ent.get("entity_group") != "SUB":
            continue
        cleaned = clean_department_string(ent.get("word", ""))
        if not cleaned or len(cleaned.split()) < 2:
            continue
        if is_abbreviation(cleaned) and cleaned in abbr_lookup:
            cleaned = abbr_lookup[cleaned]
        if cleaned in known_inst_names:
            continue  # matches an institution name -> not a sub-unit

        matched_id: str | None = None
        if monotonic:
            matched_id = pick_right_org_by_distance(ent.get("end"), org_matches)
        if matched_id is None:
            matched_id = pick_right_org_by_order(order, org_matches)

        if matched_id:
            inst_ids_field = str(matched_id)
            inst_count_field = 1
        else:
            inst_ids_field = candidate_ids_str
            inst_count_field = inst_count

        yield {
            "raw_aff_id": raw_aff_id,
            "institution_ids": inst_ids_field,
            "inst_count": inst_count_field,
            "raw_dept_str": ent.get("word"),
            "clean_dept_str": cleaned,
            "start": ent.get("start"),
            "end": ent.get("end"),
            "frequency": frequency,
        }


def stream_detailed_csv(
    jsonl_path: Path,
    output_path: Path,
    raw_aff_to_inst: dict[int, list[str]],
    inst_id_to_name: dict[str, str],
    known_inst_names: set[str],
    abbr_lookup: dict[str, str],
) -> None:
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer: list[dict[str, Any]] = []
    total_rows = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Processing NER JSONL"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_aff_id = row.get("raw_affiliation_string_id")
            if raw_aff_id is None:
                continue
            candidate_ids = raw_aff_to_inst.get(int(raw_aff_id), [])
            if not candidate_ids:
                continue
            entities = row.get("entities", [])
            if not entities:
                continue

            frequency = row.get("frequency", 1)
            inst_count = len(candidate_ids)
            for segment in split_entities_by_reset(entities):
                buffer.extend(iter_segment_rows(
                    segment, candidate_ids, inst_id_to_name, known_inst_names,
                    abbr_lookup, int(raw_aff_id), inst_count, frequency,
                ))

            if len(buffer) >= FLUSH_ROWS:
                _flush(buffer, output_path)
                total_rows += len(buffer)
                buffer = []

    if buffer:
        _flush(buffer, output_path)
        total_rows += len(buffer)
    print(f"Wrote {total_rows:,} detailed rows to {output_path}")


def build_dedupe_input(detailed_path: Path, dedupe_path: Path) -> None:
    df = pd.read_csv(detailed_path)
    print(f"Loaded {len(df):,} detailed rows")

    single_mask = df["inst_count"] == 1
    kept = int(single_mask.sum())
    print(f"Keeping {kept:,} rows with inst_count == 1 ({kept / len(df):.2%})")

    df = df.loc[single_mask]
    aggregated = (
        df.groupby(["institution_ids", "clean_dept_str"], as_index=False)["frequency"]
        .sum()
        .sort_values(by=["institution_ids", "frequency"], ascending=[True, False])
    )
    dedupe_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(dedupe_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(aggregated):,} aggregated rows to {dedupe_path}")


def main() -> None:
    args = parse_args()

    print("Loading lookup tables...")
    raw_aff_to_inst = load_raw_aff_to_institutions(args.raw_aff_institutions)
    inst_id_to_name, known_inst_names = load_institution_names(args.institution_names)
    abbr_lookup = load_abbreviation_lookup(args.abbr_lookup)
    print(
        f"  {len(raw_aff_to_inst):,} raw_affiliation_string_ids, "
        f"{len(inst_id_to_name):,} institution names, "
        f"{len(abbr_lookup):,} abbreviation entries"
    )

    stream_detailed_csv(
        args.ner_jsonl, args.detailed_output,
        raw_aff_to_inst, inst_id_to_name, known_inst_names, abbr_lookup,
    )
    build_dedupe_input(args.detailed_output, args.dedupe_output)


if __name__ == "__main__":
    main()
