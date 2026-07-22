"""Reproduce the Wikidata hierarchy validation results reported in Table 8.

Run from the public pipeline directory after generating the final output files:

uv run --with pandas --with rapidfuzz python validation/reproduce_table8.py

The released Wikidata benchmark is used by default. Alternative benchmark and
OpenSubAffil paths can be supplied through the command-line options.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz


DEFAULT_MATCH_THRESHOLD = 85
SIZE_BUCKET_ORDER = ("large", "medium", "small")
TABLE8_METRIC_COLUMNS = (
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def first_existing_path(*candidates: Path) -> Path:
    """Use the first available input while retaining the public-pipeline default."""
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


class BenchmarkResults:
    def __init__(
        self,
        *,
        wikidata_row_count: int,
        wikidata_institution_count: int,
        wikidata_filtered_row_count: int,
        wikidata_filtered_institution_count: int,
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
        size_bucket_summary: pd.DataFrame,
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
        mid_size_bucket_summary: pd.DataFrame,
        institution_stats: pd.DataFrame,
        mid_institution_stats: pd.DataFrame,
    ) -> None:
        self.wikidata_row_count = wikidata_row_count
        self.wikidata_institution_count = wikidata_institution_count
        self.wikidata_filtered_row_count = wikidata_filtered_row_count
        self.wikidata_filtered_institution_count = wikidata_filtered_institution_count
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
        self.size_bucket_summary = size_bucket_summary
        self.mid_matched_institution_count = mid_matched_institution_count
        self.mid_comparable_predicted_edge_match_total = mid_comparable_predicted_edge_match_total
        self.mid_gold_edge_total = mid_gold_edge_total
        self.mid_direct_match_total = mid_direct_match_total
        self.mid_comparable_predicted_edge_total = mid_comparable_predicted_edge_total
        self.mid_micro_recall = mid_micro_recall
        self.mid_micro_precision = mid_micro_precision
        self.mid_micro_f1 = mid_micro_f1
        self.mid_macro_recall = mid_macro_recall
        self.mid_macro_precision = mid_macro_precision
        self.mid_macro_f1 = mid_macro_f1
        self.mid_size_bucket_summary = mid_size_bucket_summary
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

    candidate_list = sorted({cand for cand in candidates if cand})
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
    institutions_df = institutions_df.copy()
    sub_institutions_df = sub_institutions_df.copy()
    hierarchy_df = hierarchy_df.copy()

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

    missing_inst = named_hierarchy_df["institution_name"].isna()
    if missing_inst.any():
        raise ValueError("Missing institution_name for one or more hierarchy rows")

    missing_child = named_hierarchy_df["child_name"].isna()
    if missing_child.any():
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


def prepare_wikidata_edges(
    wikidata_df: pd.DataFrame,
    valid_institution_ids: set[int],
) -> pd.DataFrame:
    wk_df = wikidata_df.dropna(subset=["openalex_id"]).copy()
    wk_df["openalex_id"] = wk_df["openalex_id"].astype("Int64")
    wk_df = wk_df[wk_df["openalex_id"].isin(valid_institution_ids)].reset_index(drop=True)
    wk_df["parent_norm"] = wk_df.apply(
        lambda row: substitute_root(
            row["clean_parent_name"], row["clean_root_name"], row["openalex_id"]
        ),
        axis=1,
    )
    wk_df["child_norm"] = wk_df.apply(
        lambda row: substitute_root(
            row["clean_child_name"], row["clean_root_name"], row["openalex_id"]
        ),
        axis=1,
    )
    return wk_df


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


def _build_all_edges_view(
    edge_set: set[tuple[str, str]], institution_id: str
) -> set[tuple[str, str]]:
    """Return the edge set used for the all-edges evaluation."""
    all_edges = set(edge_set)
    changed = True
    while changed:
        changed = False
        snapshot = list(all_edges)
        for parent_a, child_a in snapshot:
            for parent_b, child_b in snapshot:
                if (
                    child_a == child_b
                    and parent_a == institution_id
                    and (institution_id, parent_b) not in all_edges
                ):
                    all_edges.add((institution_id, parent_b))
                    changed = True
                if (
                    child_a == parent_b
                    and parent_a == institution_id
                    and (institution_id, child_b) not in all_edges
                ):
                    all_edges.add((institution_id, child_b))
                    changed = True
    return all_edges


def build_institution_size_metadata(wk_df: pd.DataFrame) -> pd.DataFrame:
    if wk_df.empty:
        return pd.DataFrame(columns=["institution_id", "wikidata_edge_count", "size_bucket"])

    size_df = (
        wk_df.groupby("openalex_id")
        .size()
        .rename("wikidata_edge_count")
        .reset_index()
        .rename(columns={"openalex_id": "institution_id"})
    )
    size_df["institution_id"] = size_df["institution_id"].astype(int)
    size_df["wikidata_edge_count"] = size_df["wikidata_edge_count"].astype(int)
    size_df = size_df.sort_values(
        ["wikidata_edge_count", "institution_id"], ascending=[False, True]
    ).reset_index(drop=True)

    institution_count = len(size_df)
    large_cutoff = math.ceil(institution_count * 0.1)
    medium_cutoff = math.ceil(institution_count * 0.5)

    size_buckets: list[str] = []
    for rank in range(1, institution_count + 1):
        if rank <= large_cutoff:
            size_buckets.append("large")
        elif rank <= medium_cutoff:
            size_buckets.append("medium")
        else:
            size_buckets.append("small")
    size_df["size_bucket"] = size_buckets

    return size_df.sort_values("institution_id").reset_index(drop=True)


def _finalize_institution_stats(
    institution_stats: list[dict[str, object]],
    institution_size_df: pd.DataFrame,
    *,
    stat_prefix: str,
) -> pd.DataFrame:
    metric_columns = [
        "institution_id",
        f"{stat_prefix}gold_edges",
        f"{stat_prefix}direct_match",
        f"{stat_prefix}recall",
        f"{stat_prefix}precision_num",
        f"{stat_prefix}precision_denom",
        f"{stat_prefix}precision",
    ]
    ordered_columns = [
        "institution_id",
        "wikidata_edge_count",
        "size_bucket",
        *metric_columns[1:],
    ]

    institution_stats_df = pd.DataFrame(institution_stats, columns=metric_columns)
    if institution_stats_df.empty:
        return pd.DataFrame(columns=ordered_columns)

    institution_stats_df = institution_stats_df.merge(
        institution_size_df,
        on="institution_id",
        how="left",
        validate="one_to_one",
    )
    return institution_stats_df[ordered_columns].sort_values("institution_id").reset_index(
        drop=True
    )


def build_size_bucket_summary(
    institution_stats_df: pd.DataFrame,
    *,
    stat_prefix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gold_col = f"{stat_prefix}gold_edges"
    direct_col = f"{stat_prefix}direct_match"
    recall_col = f"{stat_prefix}recall"
    precision_num_col = f"{stat_prefix}precision_num"
    precision_denom_col = f"{stat_prefix}precision_denom"
    precision_col = f"{stat_prefix}precision"

    for size_bucket in SIZE_BUCKET_ORDER:
        if institution_stats_df.empty:
            bucket_df = institution_stats_df
        else:
            bucket_df = institution_stats_df[institution_stats_df["size_bucket"] == size_bucket]

        institution_count = len(bucket_df)
        edge_counts = (
            bucket_df["wikidata_edge_count"].dropna() if "wikidata_edge_count" in bucket_df else pd.Series(dtype="float64")
        )
        gold_edge_total = int(bucket_df[gold_col].sum()) if institution_count else 0
        direct_match_total = int(bucket_df[direct_col].sum()) if institution_count else 0
        comparable_predicted_edge_total = (
            int(bucket_df[precision_denom_col].sum()) if institution_count else 0
        )
        comparable_predicted_edge_match_total = (
            int(bucket_df[precision_num_col].sum()) if institution_count else 0
        )
        matched_institution_count = (
            int((bucket_df[direct_col] > 0).sum()) if institution_count else 0
        )
        micro_recall = direct_match_total / gold_edge_total if gold_edge_total else 0.0
        micro_precision = (
            comparable_predicted_edge_match_total / comparable_predicted_edge_total
            if comparable_predicted_edge_total
            else None
        )
        micro_f1 = f1_score(micro_precision, micro_recall)
        macro_recall = float(bucket_df[recall_col].mean()) if institution_count else 0.0
        precision_values = bucket_df[precision_col].dropna() if institution_count else pd.Series(dtype="float64")
        macro_precision = (
            float(precision_values.mean()) if not precision_values.empty else None
        )
        f1_values = [
            macro_unit_f1_score(row.precision, row.recall)
            for row in bucket_df[[precision_col, recall_col]]
            .rename(columns={precision_col: "precision", recall_col: "recall"})
            .itertuples(index=False)
        ] if institution_count else []
        f1_candidates = [value for value in f1_values if value is not None]
        macro_f1 = (
            sum(f1_candidates) / len(f1_candidates)
            if f1_candidates
            else None
        )

        rows.append(
            {
                "size_bucket": size_bucket,
                "institution_count": institution_count,
                "matched_institution_count": matched_institution_count,
                "wikidata_edge_count_min": int(edge_counts.min()) if not edge_counts.empty else None,
                "wikidata_edge_count_max": int(edge_counts.max()) if not edge_counts.empty else None,
                f"{stat_prefix}gold_edge_total": gold_edge_total,
                f"{stat_prefix}direct_match_total": direct_match_total,
                f"{stat_prefix}comparable_predicted_edge_total": comparable_predicted_edge_total,
                f"{stat_prefix}comparable_predicted_edge_match_total": comparable_predicted_edge_match_total,
                f"{stat_prefix}micro_recall": micro_recall,
                f"{stat_prefix}micro_precision": micro_precision,
                f"{stat_prefix}micro_f1": micro_f1,
                f"{stat_prefix}macro_recall": macro_recall,
                f"{stat_prefix}macro_precision": macro_precision,
                f"{stat_prefix}macro_f1": macro_f1,
            }
        )

    return pd.DataFrame(rows)


def _format_optional_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def _print_size_bucket_summary(summary_df: pd.DataFrame, *, title: str, stat_prefix: str = "") -> None:
    gold_col = f"{stat_prefix}gold_edge_total"
    micro_f1_col = f"{stat_prefix}micro_f1"

    print(title)
    for size_bucket in SIZE_BUCKET_ORDER:
        if summary_df.empty:
            row = None
        else:
            bucket_rows = summary_df[summary_df["size_bucket"] == size_bucket]
            row = bucket_rows.iloc[0] if not bucket_rows.empty else None

        institution_count = int(row["institution_count"]) if row is not None else 0
        gold_edge_total = int(row[gold_col]) if row is not None and gold_col in row else 0
        micro_f1 = row[micro_f1_col] if row is not None and micro_f1_col in row else None
        print(
            f"{size_bucket}: institutions={institution_count}, "
            f"gold_edges={gold_edge_total}, micro_f1={_format_optional_metric(micro_f1)}"
        )


def _compute_metric_block(
    wk_df: pd.DataFrame,
    our_edges: dict[int, set[tuple[str, str]]],
    our_nodes: dict[int, set[str]],
    institution_size_df: pd.DataFrame,
    match_threshold: int,
    include_mid_only: bool,
) -> tuple[int, int, int, int, float, float | None, float | None, float, float | None, float | None, pd.DataFrame]:
    if include_mid_only:
        wk_df = wk_df[
            (wk_df["parent_norm"].notna())
            & (wk_df["child_norm"].notna())
            & (wk_df["parent_norm"] != wk_df["openalex_id"].astype(str))
        ].copy()

    wk_groups = wk_df.groupby("openalex_id")
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

    for inst_id, group in wk_groups:
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
        wk_nodes = {node for edge in gold_edges for node in edge}
        gold_edge_set = set(gold_edges)
        evaluation_our_edge_set = (
            _build_all_edges_view(our_edge_set, str(inst_id))
            if not include_mid_only
            else our_edge_set
        )
        evaluation_gold_edge_set = (
            _build_all_edges_view(gold_edge_set, str(inst_id))
            if not include_mid_only
            else gold_edge_set
        )

        inst_direct_match = 0
        for gold_parent, gold_child in gold_edges:
            matched_parent, _ = best_match(gold_parent, node_pool, threshold=match_threshold)
            matched_child, _ = best_match(gold_child, node_pool, threshold=match_threshold)
            if matched_parent is None or matched_child is None:
                continue
            if (matched_parent, matched_child) in evaluation_our_edge_set:
                inst_direct_match += 1

        direct_match_total += inst_direct_match
        if inst_direct_match > 0:
            matched_institution_count += 1
        inst_recall = inst_direct_match / len(gold_edges)
        recall_values.append(inst_recall)

        precision_num = 0
        precision_denom = 0
        for parent_norm, child_norm in our_edge_set:
            matched_parent, _ = best_match(parent_norm, wk_nodes, threshold=match_threshold)
            matched_child, _ = best_match(child_norm, wk_nodes, threshold=match_threshold)
            if matched_parent is None or matched_child is None:
                continue
            precision_denom += 1
            if (matched_parent, matched_child) in evaluation_gold_edge_set:
                precision_num += 1

        inst_precision = (
            precision_num / precision_denom if precision_denom else None
        )
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
    macro_precision_candidates = [value for value in precision_values if value is not None]
    macro_precision = (
        sum(macro_precision_candidates) / len(macro_precision_candidates)
        if macro_precision_candidates
        else None
    )
    macro_f1_candidates = [value for value in f1_values if value is not None]
    macro_f1 = (
        sum(macro_f1_candidates) / len(macro_f1_candidates)
        if macro_f1_candidates
        else None
    )

    institution_stats_df = _finalize_institution_stats(
        institution_stats=institution_stats,
        institution_size_df=institution_size_df,
        stat_prefix=stat_prefix,
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
    wikidata_df: pd.DataFrame,
    our_named_hierarchy_df: pd.DataFrame,
    match_threshold: int = DEFAULT_MATCH_THRESHOLD,
) -> BenchmarkResults:
    wikidata_row_count = len(wikidata_df)
    wikidata_institution_count = int(
        wikidata_df["openalex_id"].dropna().astype("Int64").nunique()
    )
    our_hierarchy_row_count = len(our_named_hierarchy_df)
    our_named_hierarchy_df = our_named_hierarchy_df.copy()
    our_named_hierarchy_df["institution_id"] = our_named_hierarchy_df["institution_id"].astype(
        "Int64"
    )
    valid_inst_ids = set(our_named_hierarchy_df["institution_id"].dropna().astype(int).tolist())
    wk_df = prepare_wikidata_edges(wikidata_df=wikidata_df, valid_institution_ids=valid_inst_ids)
    institution_size_df = build_institution_size_metadata(wk_df)
    wikidata_filtered_row_count = len(wk_df)
    wikidata_filtered_institution_count = int(wk_df["openalex_id"].nunique())
    our_institution_count = int(our_named_hierarchy_df["institution_id"].nunique())

    our_edges, our_nodes = build_edge_index(our_named_hierarchy_df)
    all_edge_our_edges = {
        inst_id: _build_all_edges_view(edge_set, str(inst_id))
        for inst_id, edge_set in our_edges.items()
    }
    all_edge_our_nodes = {
        inst_id: {node for edge in edge_set for node in edge}
        for inst_id, edge_set in all_edge_our_edges.items()
    }
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
        wk_df=wk_df,
        our_edges=all_edge_our_edges,
        our_nodes=all_edge_our_nodes,
        institution_size_df=institution_size_df,
        match_threshold=match_threshold,
        include_mid_only=False,
    )
    size_bucket_summary_df = build_size_bucket_summary(institution_stats_df)

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
        wk_df=wk_df,
        our_edges=our_edges,
        our_nodes=our_nodes,
        institution_size_df=institution_size_df,
        match_threshold=match_threshold,
        include_mid_only=True,
    )
    mid_size_bucket_summary_df = build_size_bucket_summary(
        mid_institution_stats_df,
        stat_prefix="mid_",
    )

    return BenchmarkResults(
        wikidata_row_count=wikidata_row_count,
        wikidata_institution_count=wikidata_institution_count,
        wikidata_filtered_row_count=wikidata_filtered_row_count,
        wikidata_filtered_institution_count=wikidata_filtered_institution_count,
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
        size_bucket_summary=size_bucket_summary_df,
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
        mid_size_bucket_summary=mid_size_bucket_summary_df,
        institution_stats=institution_stats_df,
        mid_institution_stats=mid_institution_stats_df,
    )


def build_table8(results: BenchmarkResults) -> pd.DataFrame:
    """Build the five rows reported in the manuscript's Table 8."""
    rows: list[dict[str, object]] = [
        {
            "level": "All edges",
            "subset": "Overall",
            "institutions": results.wikidata_filtered_institution_count,
            "micro_precision": results.micro_precision,
            "micro_recall": results.micro_recall,
            "micro_f1": results.micro_f1,
            "macro_precision": results.macro_precision,
            "macro_recall": results.macro_recall,
            "macro_f1": results.macro_f1,
        }
    ]

    for size_bucket in SIZE_BUCKET_ORDER:
        bucket_rows = results.size_bucket_summary[
            results.size_bucket_summary["size_bucket"] == size_bucket
        ]
        if len(bucket_rows) != 1:
            raise ValueError(
                f"Expected one summary row for size bucket {size_bucket!r}"
            )
        bucket = bucket_rows.iloc[0]
        rows.append(
            {
                "level": "All edges",
                "subset": size_bucket.title(),
                "institutions": int(bucket["institution_count"]),
                "micro_precision": bucket["micro_precision"],
                "micro_recall": bucket["micro_recall"],
                "micro_f1": bucket["micro_f1"],
                "macro_precision": bucket["macro_precision"],
                "macro_recall": bucket["macro_recall"],
                "macro_f1": bucket["macro_f1"],
            }
        )

    rows.append(
        {
            "level": "Nested",
            "subset": "Overall",
            "institutions": len(results.mid_institution_stats),
            "micro_precision": results.mid_micro_precision,
            "micro_recall": results.mid_micro_recall,
            "micro_f1": results.mid_micro_f1,
            "macro_precision": results.mid_macro_precision,
            "macro_recall": results.mid_macro_recall,
            "macro_f1": results.mid_macro_f1,
        }
    )
    return pd.DataFrame(
        rows,
        columns=["level", "subset", "institutions", *TABLE8_METRIC_COLUMNS],
    )


def format_reported_metric(value: object) -> str:
    """Format a metric to three decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def print_table8(table8_df: pd.DataFrame) -> None:
    """Print a publication-style Table 8."""
    print("Table 8. Sub-institutional hierarchy evaluation against Wikidata")
    display_df = table8_df.rename(
        columns={
            "level": "Level",
            "subset": "Subset",
            "institutions": "Institutions",
            "micro_precision": "Micro P",
            "micro_recall": "Micro R",
            "micro_f1": "Micro F1",
            "macro_precision": "Macro P",
            "macro_recall": "Macro R",
            "macro_f1": "Macro F1",
        }
    )
    print(
        display_df.to_string(
            index=False,
            formatters={
                column: format_reported_metric
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
    print(f"Wikidata rows loaded: {results.wikidata_row_count:,}")
    print(f"Wikidata institutions loaded: {results.wikidata_institution_count:,}")
    print(
        "Wikidata rows after filtering to our institutions: "
        f"{results.wikidata_filtered_row_count:,}"
    )
    print(
        "Wikidata institutions after filtering to our institutions: "
        f"{results.wikidata_filtered_institution_count:,}"
    )
    print(f"Our hierarchy rows loaded: {results.our_hierarchy_row_count:,}")
    print(f"Our institutions loaded: {results.our_institution_count:,}")
    print(f"Our named edge count: {results.our_named_edge_count:,}")
    print()
    print(f"Gold edges (wikidata): {results.gold_edge_total:,}")
    print(f"Direct matches: {results.direct_match_total:,}")
    print(
        "Institutions with at least one direct match: "
        f"{results.matched_institution_count:,}"
    )
    print(
        "Comparable edges (ours, both ends in wikidata nodes): "
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
    print(
        f"Micro F1: {results.micro_f1:.4f}"
        if results.micro_f1 is not None
        else "Micro F1: N/A"
    )
    print(f"Macro recall: {results.macro_recall:.4f}")
    print(
        f"Macro precision: {results.macro_precision:.4f}"
        if results.macro_precision is not None
        else "Macro precision: N/A"
    )
    print(
        f"Macro F1: {results.macro_f1:.4f}"
        if results.macro_f1 is not None
        else "Macro F1: N/A"
    )
    print()
    _print_size_bucket_summary(
        results.size_bucket_summary,
        title="Results by institution size:",
    )
    print()
    print(f"Mid-level gold edges: {results.mid_gold_edge_total:,}")
    print(f"Mid-level direct matches: {results.mid_direct_match_total:,}")
    print(
        "Mid-level institutions with at least one direct match: "
        f"{results.mid_matched_institution_count:,}"
    )
    print(
        "Mid-level comparable edges (ours, both ends in wikidata nodes): "
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
    print()
    _print_size_bucket_summary(
        results.mid_size_bucket_summary,
        title="Mid-level results by institution size:",
        stat_prefix="mid_",
    )


def run_benchmark(
    wikidata_path: str | Path,
    institutions_path: str | Path,
    sub_institutions_path: str | Path,
    hierarchy_path: str | Path,
    match_threshold: int = DEFAULT_MATCH_THRESHOLD,
) -> BenchmarkResults:
    wikidata_df = pd.read_csv(wikidata_path)
    our_named_hierarchy_df = load_named_hierarchy_from_files(
        institutions_path=institutions_path,
        sub_institutions_path=sub_institutions_path,
        hierarchy_path=hierarchy_path,
    )
    results = evaluate_benchmark(
        wikidata_df=wikidata_df,
        our_named_hierarchy_df=our_named_hierarchy_df,
        match_threshold=match_threshold,
    )

    table8_df = build_table8(results)
    print_summary(results)
    print()
    print_table8(table8_df)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    pipeline_root = Path(__file__).resolve().parents[1]
    repository_root = pipeline_root.parent
    pipeline_final_output = pipeline_root / "data" / "final_output"
    repository_final_output = repository_root / "data" / "final_output"
    parser = argparse.ArgumentParser(
        description="Reproduce the Wikidata hierarchy validation reported in Table 8."
    )
    parser.add_argument(
        "--wikidata-path",
        default=pipeline_root / "data" / "wikidata_institution_hierarchy.csv",
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_benchmark(
        wikidata_path=args.wikidata_path,
        institutions_path=args.institutions_path,
        sub_institutions_path=args.sub_institutions_path,
        hierarchy_path=args.hierarchy_path,
        match_threshold=args.match_threshold,
    )


if __name__ == "__main__":
    main()
