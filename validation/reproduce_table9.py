"""Reproduce the GERiT hierarchy validation results reported in Table 9.

Run from the public pipeline directory:

uv run --with pandas --with rapidfuzz --with lingua-language-detector \
  python validation/reproduce_table9.py --gerit-path /path/to/gerit.csv

GERiT data are not distributed with this repository. The OpenAlex education-
institution ROR mapping included in ``data/`` is used by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from lingua import Language, LanguageDetectorBuilder
from rapidfuzz import fuzz

DEFAULT_MATCH_THRESHOLD = 85

STOP_WORDS = [
    "of",
    "the",
    "and",
    "for",
    "in",
    "on",
    "at",
    "to",
    "amp",
]

TYPE_WORDS = {
    "department",
    "dept",
    "dpt",
    "school",
    "sch",
    "college",
    "coll",
    "faculty",
    "fac",
    "institute",
    "inst",
    "institution",
    "center",
    "centre",
    "ctr",
    "division",
    "div",
    "program",
    "prog",
    "laboratory",
    "lab",
    "academy",
    "acad",
    "university",
    "univ",
    "hospital",
    "hosp",
    "administration",
    "admin",
    "association",
    "assoc",
    "unit",
    "section",
    "sect",
    "sec",
    "branch",
    "group",
    "grp",
}


def first_existing_path(*candidates: Path) -> Path:
    """Use the first available input while retaining public-pipeline defaults."""
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


class BenchmarkResults:
    def __init__(
        self,
        *,
        gerit_row_count: int,
        gerit_ror_row_count: int,
        gerit_mapped_row_count: int,
        gerit_clean_row_count: int,
        gerit_filtered_row_count: int,
        gerit_institution_count: int,
        gerit_filtered_institution_count: int,
        our_hierarchy_row_count: int,
        our_institution_count: int,
        our_named_edge_count: int,
        matched_institution_count: int,
        comparable_predicted_edge_match_total: int,
        gold_edge_total: int,
        direct_match_total: int,
        comparable_predicted_edge_total: int,
        micro_recall: float,
        micro_precision: float | None,
        micro_f1: float | None,
        macro_recall: float,
        macro_precision: float | None,
        macro_f1: float | None,
        mid_matched_institution_count: int,
        mid_comparable_predicted_edge_match_total: int,
        mid_gold_edge_total: int,
        mid_direct_match_total: int,
        mid_comparable_predicted_edge_total: int,
        mid_micro_recall: float,
        mid_micro_precision: float | None,
        mid_micro_f1: float | None,
        mid_macro_recall: float,
        mid_macro_precision: float | None,
        mid_macro_f1: float | None,
        institution_stats: pd.DataFrame,
        mid_institution_stats: pd.DataFrame,
    ) -> None:
        self.gerit_row_count = gerit_row_count
        self.gerit_ror_row_count = gerit_ror_row_count
        self.gerit_mapped_row_count = gerit_mapped_row_count
        self.gerit_clean_row_count = gerit_clean_row_count
        self.gerit_filtered_row_count = gerit_filtered_row_count
        self.gerit_institution_count = gerit_institution_count
        self.gerit_filtered_institution_count = gerit_filtered_institution_count
        self.our_hierarchy_row_count = our_hierarchy_row_count
        self.our_institution_count = our_institution_count
        self.our_named_edge_count = our_named_edge_count
        self.matched_institution_count = matched_institution_count
        self.comparable_predicted_edge_match_total = comparable_predicted_edge_match_total
        self.gold_edge_total = gold_edge_total
        self.direct_match_total = direct_match_total
        self.comparable_predicted_edge_total = comparable_predicted_edge_total
        self.micro_recall = micro_recall
        self.micro_precision = micro_precision
        self.micro_f1 = micro_f1
        self.macro_recall = macro_recall
        self.macro_precision = macro_precision
        self.macro_f1 = macro_f1
        self.mid_matched_institution_count = mid_matched_institution_count
        self.mid_comparable_predicted_edge_match_total = (
            mid_comparable_predicted_edge_match_total
        )
        self.mid_gold_edge_total = mid_gold_edge_total
        self.mid_direct_match_total = mid_direct_match_total
        self.mid_comparable_predicted_edge_total = mid_comparable_predicted_edge_total
        self.mid_micro_recall = mid_micro_recall
        self.mid_micro_precision = mid_micro_precision
        self.mid_micro_f1 = mid_micro_f1
        self.mid_macro_recall = mid_macro_recall
        self.mid_macro_precision = mid_macro_precision
        self.mid_macro_f1 = mid_macro_f1
        self.institution_stats = institution_stats
        self.mid_institution_stats = mid_institution_stats


def similarity(a: str, b: str) -> float:
    return float(fuzz.token_set_ratio(a, b))


def f1_score(precision: float | None, recall: float) -> float | None:
    if precision is None or pd.isna(precision):
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_unit_f1_score(precision: float | None, recall: float) -> float | None:
    if precision is None or pd.isna(precision):
        return 0.0 if recall == 0 else None
    return f1_score(precision, recall)


def clean_str(raw_str: object) -> str:
    if not raw_str:
        return ""

    import re

    text_value = str(raw_str).lower()
    text_value = re.sub(r"\(.*$", " ", text_value)
    text_value = re.sub(r"[^\w\s]", " ", text_value)
    text_value = re.sub(r"\bamp\b", " ", text_value)
    text_value = re.sub(r"\s+", " ", text_value).strip()

    words = text_value.split()
    while words and words[0] in STOP_WORDS:
        words.pop(0)
    while words and words[-1] in STOP_WORDS:
        words.pop()
    text_value = " ".join(words)

    if not any(type_word in text_value for type_word in TYPE_WORDS):
        return ""

    text_value = re.sub(r"^\d+\s*", "", text_value)
    return text_value.strip()


def build_language_detector():
    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.CHINESE,
        Language.JAPANESE,
        Language.RUSSIAN,
    ).build()


def substitute_root(name: object, root_name: object, root_id: object) -> str | None:
    if pd.isna(name):
        return None

    name_str = str(name).strip()
    if not name_str:
        return None

    if pd.isna(root_name):
        return name_str

    root_str = str(root_name).strip()
    if root_str and name_str.lower() == root_str.lower():
        return str(int(root_id))
    return name_str


def best_match(
    name: str | None,
    candidates: Iterable[str],
    threshold: int = DEFAULT_MATCH_THRESHOLD,
) -> tuple[str | None, float]:
    if not name:
        return None, 0.0

    candidate_list = sorted({candidate for candidate in candidates if candidate})
    if not candidate_list:
        return None, 0.0

    best_candidate: str | None = None
    best_score = 0.0
    for candidate in candidate_list:
        score = similarity(name, candidate)
        if score > best_score or (
            score == best_score and best_candidate is not None and candidate < best_candidate
        ):
            best_candidate = candidate
            best_score = score

    if best_score >= threshold:
        return best_candidate, best_score
    return None, best_score


def build_named_hierarchy(
    institutions_df: pd.DataFrame,
    sub_institutions_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
) -> pd.DataFrame:
    parent_lookup = sub_institutions_df.rename(
        columns={
            "sub_institution_id": "parent_sub_institution_id",
            "canonical_name": "parent_name",
        }
    )
    child_lookup = sub_institutions_df.rename(
        columns={
            "sub_institution_id": "child_sub_institution_id",
            "canonical_name": "child_name",
        }
    )

    named_hierarchy_df = (
        hierarchy_df.merge(
            institutions_df[["institution_id", "institution_name"]],
            on="institution_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            parent_lookup[["institution_id", "parent_sub_institution_id", "parent_name"]],
            on=["institution_id", "parent_sub_institution_id"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            child_lookup[["institution_id", "child_sub_institution_id", "child_name"]],
            on=["institution_id", "child_sub_institution_id"],
            how="left",
            validate="many_to_one",
        )
    )

    if named_hierarchy_df["institution_name"].isna().any():
        raise ValueError("Missing institution_name for one or more hierarchy rows")
    if named_hierarchy_df["child_name"].isna().any():
        raise ValueError("Missing child canonical_name for one or more hierarchy rows")

    named_hierarchy_df["parent_name"] = named_hierarchy_df["parent_name"].fillna(
        named_hierarchy_df["institution_name"]
    )
    named_hierarchy_df["parent_norm"] = named_hierarchy_df.apply(
        lambda row: substitute_root(
            row["parent_name"], row["institution_name"], row["institution_id"]
        ),
        axis=1,
    )
    named_hierarchy_df["child_norm"] = named_hierarchy_df.apply(
        lambda row: substitute_root(
            row["child_name"], row["institution_name"], row["institution_id"]
        ),
        axis=1,
    )

    return named_hierarchy_df[
        [
            "institution_id",
            "institution_name",
            "parent_sub_institution_id",
            "child_sub_institution_id",
            "parent_name",
            "child_name",
            "parent_norm",
            "child_norm",
        ]
    ].copy()


def load_named_hierarchy_from_files(
    institutions_path: str | Path,
    sub_institutions_path: str | Path,
    hierarchy_path: str | Path,
) -> pd.DataFrame:
    institutions_df = pd.read_csv(institutions_path)
    sub_institutions_df = pd.read_csv(sub_institutions_path)
    hierarchy_df = pd.read_csv(
        hierarchy_path,
        dtype={
            "institution_id": "int64",
            "parent_sub_institution_id": "Int64",
            "child_sub_institution_id": "Int64",
        },
    )
    return build_named_hierarchy(
        institutions_df=institutions_df,
        sub_institutions_df=sub_institutions_df,
        hierarchy_df=hierarchy_df,
    )


def build_edge_index(
    our_named_hierarchy_df: pd.DataFrame,
) -> tuple[dict[int, set[tuple[str, str]]], dict[int, set[str]]]:
    our_edges: dict[int, set[tuple[str, str]]] = {}
    our_nodes: dict[int, set[str]] = {}

    for row in our_named_hierarchy_df.itertuples(index=False):
        inst_id = int(row.institution_id)
        parent_norm = row.parent_norm
        child_norm = row.child_norm
        if not parent_norm or not child_norm:
            continue
        our_edges.setdefault(inst_id, set()).add((parent_norm, child_norm))
        our_nodes.setdefault(inst_id, set()).update([parent_norm, child_norm])

    return our_edges, our_nodes


def prepare_gerit_gold(
    gerit_df: pd.DataFrame,
    openalex_ror_df: pd.DataFrame,
    *,
    english_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    gerit_df = gerit_df.copy()
    stats = {
        "gerit_row_count": len(gerit_df),
        "gerit_ror_row_count": 0,
        "gerit_mapped_row_count": 0,
        "gerit_clean_row_count": 0,
        "gerit_filtered_row_count": 0,
        "gerit_institution_count": int(gerit_df["rorid"].dropna().nunique()),
        "gerit_filtered_institution_count": 0,
    }

    gerit_df = gerit_df[gerit_df["rorid"].notna()].copy()
    stats["gerit_ror_row_count"] = len(gerit_df)

    ror_to_openalex = dict(
        zip(openalex_ror_df["ror_id"], openalex_ror_df["institution_id"])
    )
    gerit_df["openalex_id"] = gerit_df["rorid"].map(ror_to_openalex)
    gerit_df = gerit_df[gerit_df["openalex_id"].notna()].copy()
    gerit_df["openalex_id"] = gerit_df["openalex_id"].astype(int)
    stats["gerit_mapped_row_count"] = len(gerit_df)

    gerit_df["clean_root_name"] = gerit_df["root_institution"].apply(clean_str)
    gerit_df["clean_parent_name"] = gerit_df["parent_name"].apply(clean_str)
    gerit_df["clean_child_name"] = gerit_df["child_name"].apply(clean_str)
    gerit_df = gerit_df[
        (gerit_df["clean_root_name"] != "")
        & (gerit_df["clean_parent_name"] != "")
        & (gerit_df["clean_child_name"] != "")
    ].copy()
    stats["gerit_clean_row_count"] = len(gerit_df)

    if english_only and not gerit_df.empty:
        detector = build_language_detector()
        child_names = gerit_df["clean_child_name"].tolist()
        child_languages = detector.detect_languages_in_parallel_of(child_names)
        english_mask = [
            language is not None and language.name == "ENGLISH"
            for language in child_languages
        ]
        gerit_df = gerit_df[english_mask].copy()

    gerit_df = gerit_df[
        ["openalex_id", "clean_root_name", "clean_parent_name", "clean_child_name"]
    ].reset_index(drop=True)
    stats["gerit_filtered_row_count"] = len(gerit_df)
    stats["gerit_filtered_institution_count"] = int(gerit_df["openalex_id"].nunique())
    return gerit_df, stats


def _compute_metric_block(
    gold_df: pd.DataFrame,
    our_edges: dict[int, set[tuple[str, str]]],
    our_nodes: dict[int, set[str]],
    match_threshold: int,
    include_mid_only: bool,
) -> tuple[int, int, int, int, float, float | None, float | None, float, float | None, float | None, pd.DataFrame]:
    if include_mid_only:
        gold_df = gold_df[
            (gold_df["parent_norm"].notna())
            & (gold_df["child_norm"].notna())
            & (gold_df["parent_norm"] != gold_df["openalex_id"].astype(str))
        ].copy()

    gold_groups = gold_df.groupby("openalex_id")
    recall_values: list[float] = []
    precision_values: list[float | None] = []
    f1_values: list[float | None] = []
    institution_stats: list[dict[str, object]] = []

    gold_edge_total = 0
    direct_match_total = 0
    comparable_predicted_edge_total = 0
    comparable_predicted_edge_match_total = 0
    macro_institution_count = 0
    matched_institution_count = 0

    stat_prefix = "mid_" if include_mid_only else ""

    for inst_id, group in gold_groups:
        gold_edges = [(row.parent_norm, row.child_norm) for row in group.itertuples(index=False)]
        gold_edges = [(parent, child) for parent, child in gold_edges if parent and child]
        if not gold_edges:
            continue

        inst_id = int(inst_id)
        macro_institution_count += 1
        gold_edge_total += len(gold_edges)
        our_edge_set = our_edges.get(inst_id, set())
        if not our_edge_set:
            recall_values.append(0.0)
            precision_values.append(None)
            f1_values.append(0.0)
            institution_stats.append(
                {
                    "institution_id": inst_id,
                    f"{stat_prefix}gold_edges": len(gold_edges),
                    f"{stat_prefix}direct_match": 0,
                    f"{stat_prefix}recall": 0.0,
                    f"{stat_prefix}precision_num": 0,
                    f"{stat_prefix}precision_denom": 0,
                    f"{stat_prefix}precision": None,
                }
            )
            continue

        node_pool = our_nodes.get(inst_id, set())
        gold_nodes = {node for edge in gold_edges for node in edge}
        gold_edge_set = set(gold_edges)

        inst_direct_match = 0
        for gold_parent, gold_child in gold_edges:
            matched_parent, _ = best_match(gold_parent, node_pool, threshold=match_threshold)
            matched_child, _ = best_match(gold_child, node_pool, threshold=match_threshold)
            if matched_parent is None or matched_child is None:
                continue
            if (matched_parent, matched_child) in our_edge_set:
                inst_direct_match += 1

        direct_match_total += inst_direct_match
        if inst_direct_match > 0:
            matched_institution_count += 1
        inst_recall = inst_direct_match / len(gold_edges)
        recall_values.append(inst_recall)

        precision_num = 0
        precision_denom = 0
        for parent_norm, child_norm in our_edge_set:
            matched_parent, _ = best_match(parent_norm, gold_nodes, threshold=match_threshold)
            matched_child, _ = best_match(child_norm, gold_nodes, threshold=match_threshold)
            if matched_parent is None or matched_child is None:
                continue
            precision_denom += 1
            if (matched_parent, matched_child) in gold_edge_set:
                precision_num += 1

        inst_precision = precision_num / precision_denom if precision_denom else None
        precision_values.append(inst_precision)
        f1_values.append(macro_unit_f1_score(inst_precision, inst_recall))
        comparable_predicted_edge_total += precision_denom
        comparable_predicted_edge_match_total += precision_num

        institution_stats.append(
            {
                "institution_id": inst_id,
                f"{stat_prefix}gold_edges": len(gold_edges),
                f"{stat_prefix}direct_match": inst_direct_match,
                f"{stat_prefix}recall": inst_recall,
                f"{stat_prefix}precision_num": precision_num,
                f"{stat_prefix}precision_denom": precision_denom,
                f"{stat_prefix}precision": inst_precision,
            }
        )

    micro_recall = direct_match_total / gold_edge_total if gold_edge_total else 0.0
    micro_precision = (
        comparable_predicted_edge_match_total / comparable_predicted_edge_total
        if comparable_predicted_edge_total
        else None
    )
    micro_f1 = f1_score(micro_precision, micro_recall)
    macro_recall = (
        sum(recall_values) / macro_institution_count if macro_institution_count else 0.0
    )
    precision_candidates = [value for value in precision_values if value is not None]
    macro_precision = (
        sum(precision_candidates) / len(precision_candidates)
        if precision_candidates
        else None
    )
    macro_f1_candidates = [value for value in f1_values if value is not None]
    macro_f1 = (
        sum(macro_f1_candidates) / len(macro_f1_candidates)
        if macro_f1_candidates
        else None
    )

    institution_stats_df = pd.DataFrame(institution_stats)
    if not institution_stats_df.empty:
        institution_stats_df = institution_stats_df.sort_values("institution_id").reset_index(
            drop=True
        )

    return (
        gold_edge_total,
        direct_match_total,
        comparable_predicted_edge_total,
        comparable_predicted_edge_match_total,
        matched_institution_count,
        micro_recall,
        micro_precision,
        micro_f1,
        macro_recall,
        macro_precision,
        macro_f1,
        institution_stats_df,
    )


def evaluate_benchmark(
    gerit_gold_df: pd.DataFrame,
    our_named_hierarchy_df: pd.DataFrame,
    gerit_stats: dict[str, int],
    match_threshold: int = DEFAULT_MATCH_THRESHOLD,
) -> BenchmarkResults:
    our_named_hierarchy_df = our_named_hierarchy_df.copy()
    our_named_hierarchy_df["institution_id"] = our_named_hierarchy_df["institution_id"].astype(
        "Int64"
    )
    valid_inst_ids = set(our_named_hierarchy_df["institution_id"].dropna().astype(int).tolist())
    our_hierarchy_row_count = len(our_named_hierarchy_df)
    our_institution_count = int(our_named_hierarchy_df["institution_id"].nunique())

    gold_df = gerit_gold_df.dropna(subset=["openalex_id"]).copy()
    gold_df["openalex_id"] = gold_df["openalex_id"].astype("Int64")
    gold_df = gold_df[gold_df["openalex_id"].isin(valid_inst_ids)].reset_index(drop=True)
    gold_df["parent_norm"] = gold_df.apply(
        lambda row: substitute_root(
            row["clean_parent_name"], row["clean_root_name"], row["openalex_id"]
        ),
        axis=1,
    )
    gold_df["child_norm"] = gold_df.apply(
        lambda row: substitute_root(
            row["clean_child_name"], row["clean_root_name"], row["openalex_id"]
        ),
        axis=1,
    )

    our_edges, our_nodes = build_edge_index(our_named_hierarchy_df)
    our_named_edge_count = sum(len(edge_set) for edge_set in our_edges.values())

    (
        gold_edge_total,
        direct_match_total,
        comparable_predicted_edge_total,
        comparable_predicted_edge_match_total,
        matched_institution_count,
        micro_recall,
        micro_precision,
        micro_f1,
        macro_recall,
        macro_precision,
        macro_f1,
        institution_stats_df,
    ) = _compute_metric_block(
        gold_df=gold_df,
        our_edges=our_edges,
        our_nodes=our_nodes,
        match_threshold=match_threshold,
        include_mid_only=False,
    )

    (
        mid_gold_edge_total,
        mid_direct_match_total,
        mid_comparable_predicted_edge_total,
        mid_comparable_predicted_edge_match_total,
        mid_matched_institution_count,
        mid_micro_recall,
        mid_micro_precision,
        mid_micro_f1,
        mid_macro_recall,
        mid_macro_precision,
        mid_macro_f1,
        mid_institution_stats_df,
    ) = _compute_metric_block(
        gold_df=gold_df,
        our_edges=our_edges,
        our_nodes=our_nodes,
        match_threshold=match_threshold,
        include_mid_only=True,
    )

    return BenchmarkResults(
        gerit_row_count=gerit_stats["gerit_row_count"],
        gerit_ror_row_count=gerit_stats["gerit_ror_row_count"],
        gerit_mapped_row_count=gerit_stats["gerit_mapped_row_count"],
        gerit_clean_row_count=gerit_stats["gerit_clean_row_count"],
        gerit_filtered_row_count=len(gold_df),
        gerit_institution_count=gerit_stats["gerit_institution_count"],
        gerit_filtered_institution_count=int(gold_df["openalex_id"].nunique()),
        our_hierarchy_row_count=our_hierarchy_row_count,
        our_institution_count=our_institution_count,
        our_named_edge_count=our_named_edge_count,
        matched_institution_count=matched_institution_count,
        comparable_predicted_edge_match_total=comparable_predicted_edge_match_total,
        gold_edge_total=gold_edge_total,
        direct_match_total=direct_match_total,
        comparable_predicted_edge_total=comparable_predicted_edge_total,
        micro_recall=micro_recall,
        micro_precision=micro_precision,
        micro_f1=micro_f1,
        macro_recall=macro_recall,
        macro_precision=macro_precision,
        macro_f1=macro_f1,
        mid_matched_institution_count=mid_matched_institution_count,
        mid_comparable_predicted_edge_match_total=mid_comparable_predicted_edge_match_total,
        mid_gold_edge_total=mid_gold_edge_total,
        mid_direct_match_total=mid_direct_match_total,
        mid_comparable_predicted_edge_total=mid_comparable_predicted_edge_total,
        mid_micro_recall=mid_micro_recall,
        mid_micro_precision=mid_micro_precision,
        mid_micro_f1=mid_micro_f1,
        mid_macro_recall=mid_macro_recall,
        mid_macro_precision=mid_macro_precision,
        mid_macro_f1=mid_macro_f1,
        institution_stats=institution_stats_df,
        mid_institution_stats=mid_institution_stats_df,
    )


def format_metric(value: object) -> str:
    """Format a metric to three decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def build_table9(results: BenchmarkResults) -> pd.DataFrame:
    """Build the two OpenSubAffil rows reported in Table 9."""
    return pd.DataFrame(
        [
            {
                "level": "All edges",
                "institutions": results.gerit_filtered_institution_count,
                "micro_precision": results.micro_precision,
                "micro_recall": results.micro_recall,
                "micro_f1": results.micro_f1,
                "macro_precision": results.macro_precision,
                "macro_recall": results.macro_recall,
                "macro_f1": results.macro_f1,
            },
            {
                "level": "Nested",
                "institutions": results.gerit_filtered_institution_count,
                "micro_precision": results.mid_micro_precision,
                "micro_recall": results.mid_micro_recall,
                "micro_f1": results.mid_micro_f1,
                "macro_precision": results.mid_macro_precision,
                "macro_recall": results.mid_macro_recall,
                "macro_f1": results.mid_macro_f1,
            },
        ]
    )


def print_table9(table9_df: pd.DataFrame) -> None:
    """Print the GERiT rows of Table 9."""
    display_df = table9_df.rename(
        columns={
            "level": "Level",
            "institutions": "Institutions",
            "micro_precision": "Micro P",
            "micro_recall": "Micro R",
            "micro_f1": "Micro F1",
            "macro_precision": "Macro P",
            "macro_recall": "Macro R",
            "macro_f1": "Macro F1",
        }
    )
    print("Table 9. Hierarchy evaluation against GERiT")
    print(
        display_df.to_string(
            index=False,
            formatters={
                column: format_metric
                for column in [
                    "Micro P",
                    "Micro R",
                    "Micro F1",
                    "Macro P",
                    "Macro R",
                    "Macro F1",
                ]
            },
        )
    )


def print_summary(results: BenchmarkResults) -> None:
    print(f"GERiT rows loaded: {results.gerit_row_count:,}")
    print(f"GERiT rows with ROR IDs: {results.gerit_ror_row_count:,}")
    print(
        "GERiT rows matched to OpenAlex education institutions: "
        f"{results.gerit_mapped_row_count:,}"
    )
    print(f"GERiT rows after name cleaning: {results.gerit_clean_row_count:,}")
    print(f"GERiT rows after final filtering: {results.gerit_filtered_row_count:,}")
    print(f"GERiT institutions before final filtering: {results.gerit_institution_count:,}")
    print(f"GERiT institutions after final filtering: {results.gerit_filtered_institution_count:,}")
    print(f"Our hierarchy rows loaded: {results.our_hierarchy_row_count:,}")
    print(f"Our institutions loaded: {results.our_institution_count:,}")
    print(f"Our named edge count: {results.our_named_edge_count:,}")
    print()
    print(f"Gold edges (GERiT): {results.gold_edge_total:,}")
    print(f"Direct matches: {results.direct_match_total:,}")
    print(
        "Institutions with at least one direct match: "
        f"{results.matched_institution_count:,}"
    )
    print(
        "Comparable edges (ours, both ends in GERiT nodes): "
        f"{results.comparable_predicted_edge_total:,}"
    )
    print(
        "Comparable predicted matches: "
        f"{results.comparable_predicted_edge_match_total:,}"
    )
    print(f"Micro recall: {results.micro_recall:.4f}")
    print(
        f"Micro precision: {results.micro_precision:.4f}"
        if results.micro_precision is not None
        else "Micro precision: N/A"
    )
    print(f"Micro F1: {results.micro_f1:.4f}" if results.micro_f1 is not None else "Micro F1: N/A")
    print(f"Macro recall: {results.macro_recall:.4f}")
    print(
        f"Macro precision: {results.macro_precision:.4f}"
        if results.macro_precision is not None
        else "Macro precision: N/A"
    )
    print(f"Macro F1: {results.macro_f1:.4f}" if results.macro_f1 is not None else "Macro F1: N/A")
    print()
    print(f"Mid-level gold edges: {results.mid_gold_edge_total:,}")
    print(f"Mid-level direct matches: {results.mid_direct_match_total:,}")
    print(
        "Mid-level institutions with at least one direct match: "
        f"{results.mid_matched_institution_count:,}"
    )
    print(
        "Mid-level comparable edges (ours, both ends in GERiT nodes): "
        f"{results.mid_comparable_predicted_edge_total:,}"
    )
    print(
        "Mid-level comparable predicted matches: "
        f"{results.mid_comparable_predicted_edge_match_total:,}"
    )
    print(f"Mid-level micro recall: {results.mid_micro_recall:.4f}")
    print(
        f"Mid-level micro precision: {results.mid_micro_precision:.4f}"
        if results.mid_micro_precision is not None
        else "Mid-level micro precision: N/A"
    )
    print(
        f"Mid-level micro F1: {results.mid_micro_f1:.4f}"
        if results.mid_micro_f1 is not None
        else "Mid-level micro F1: N/A"
    )
    print(f"Mid-level macro recall: {results.mid_macro_recall:.4f}")
    print(
        f"Mid-level macro precision: {results.mid_macro_precision:.4f}"
        if results.mid_macro_precision is not None
        else "Mid-level macro precision: N/A"
    )
    print(
        f"Mid-level macro F1: {results.mid_macro_f1:.4f}"
        if results.mid_macro_f1 is not None
        else "Mid-level macro F1: N/A"
    )


def run_benchmark(
    *,
    gerit_path: str | Path,
    ror_mapping_path: str | Path,
    institutions_path: str | Path,
    sub_institutions_path: str | Path,
    hierarchy_path: str | Path,
    match_threshold: int = DEFAULT_MATCH_THRESHOLD,
    english_only: bool = True,
) -> BenchmarkResults:
    gerit_df = pd.read_csv(gerit_path)
    openalex_ror_df = pd.read_csv(ror_mapping_path, dtype={"ror_id": "string"})
    gerit_gold_df, gerit_stats = prepare_gerit_gold(
        gerit_df=gerit_df,
        openalex_ror_df=openalex_ror_df,
        english_only=english_only,
    )
    our_named_hierarchy_df = load_named_hierarchy_from_files(
        institutions_path=institutions_path,
        sub_institutions_path=sub_institutions_path,
        hierarchy_path=hierarchy_path,
    )
    results = evaluate_benchmark(
        gerit_gold_df=gerit_gold_df,
        our_named_hierarchy_df=our_named_hierarchy_df,
        gerit_stats=gerit_stats,
        match_threshold=match_threshold,
    )

    print_summary(results)
    print()
    print_table9(build_table9(results))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    pipeline_root = Path(__file__).resolve().parents[1]
    repository_root = pipeline_root.parent
    pipeline_final_output = pipeline_root / "data" / "final_output"
    repository_final_output = repository_root / "data" / "final_output"
    repository_fixed_final_output = (
        repository_root / "data" / "final_output_html_entity_fixed"
    )
    parser = argparse.ArgumentParser(
        description="Reproduce the GERiT hierarchy validation reported in Table 9."
    )
    parser.add_argument(
        "--gerit-path",
        type=Path,
        help="Path to a locally obtained GERiT hierarchy CSV (not distributed).",
    )
    parser.add_argument(
        "--ror-mapping-path",
        default=pipeline_root
        / "data"
        / "openalex_education_institution_ror_mapping.csv",
        type=Path,
    )
    parser.add_argument(
        "--institutions-path",
        default=first_existing_path(
            pipeline_final_output / "opensubaffil_institutions.csv",
            pipeline_final_output / "OpenSubAffil_institutions.csv",
            repository_final_output / "opensubaffil_institutions.csv",
            repository_final_output / "OpenSubAffil_institutions.csv",
        ),
        type=Path,
    )
    parser.add_argument(
        "--sub-institutions-path",
        default=first_existing_path(
            pipeline_final_output / "opensubaffil_sub_institutions.csv",
            pipeline_final_output / "OpenSubAffil_sub_institutions.csv",
            repository_fixed_final_output / "opensubaffil_sub_institutions.csv",
            repository_fixed_final_output / "OpenSubAffil_sub_institutions.csv",
            repository_final_output / "opensubaffil_sub_institutions.csv",
            repository_final_output / "OpenSubAffil_sub_institutions.csv",
        ),
        type=Path,
    )
    parser.add_argument(
        "--hierarchy-path",
        default=first_existing_path(
            pipeline_final_output / "opensubaffil_hierarchy.csv",
            pipeline_final_output / "OpenSubAffil_hierarchy.csv",
            repository_fixed_final_output / "opensubaffil_hierarchy.csv",
            repository_fixed_final_output / "OpenSubAffil_hierarchy.csv",
            repository_final_output / "opensubaffil_hierarchy.csv",
            repository_final_output / "OpenSubAffil_hierarchy.csv",
        ),
        type=Path,
    )
    parser.add_argument(
        "--match-threshold",
        default=DEFAULT_MATCH_THRESHOLD,
        type=int,
    )
    parser.add_argument(
        "--no-english-only",
        action="store_true",
        help="Disable GERiT child-name English filtering.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    gerit_path = args.gerit_path
    if gerit_path is None:
        local_gerit_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "gerit_institution_hierarchy.csv"
        )
        if local_gerit_path.is_file():
            gerit_path = local_gerit_path
        else:
            parser.error(
                "--gerit-path is required because GERiT data are not distributed "
                "with this repository"
            )

    run_benchmark(
        gerit_path=gerit_path,
        ror_mapping_path=args.ror_mapping_path,
        institutions_path=args.institutions_path,
        sub_institutions_path=args.sub_institutions_path,
        hierarchy_path=args.hierarchy_path,
        match_threshold=args.match_threshold,
        english_only=not args.no_english_only,
    )


if __name__ == "__main__":
    main()
