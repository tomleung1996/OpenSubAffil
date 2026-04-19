"""Build a parent-child hierarchy over the canonical departments of each institution."""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm.auto import tqdm

from config import (
    CANONICAL_DEPT_CSV,
    HIERARCHY_CSV,
    INSTITUTION_ID_NAME_CSV,
    INSTITUTION_NODE_PREFIX,
    LEXICAL_RANKS,
    ROOT_LEXICAL_RANK,
)

# Edge-scoring weights (see module docstring).
MIN_COOC_COUNT = 2
MIN_ASSOC_SCORE = 0.01
SCORE_THRESHOLD = 0.0
SCORE_QUANTILE = 0.5
TOP_K_PARENTS_PER_CHILD = 3

W_COVERAGE = 0.4
W_FREQ = 0.2
W_DEGREE = 0.1
W_POSITION = 0.3
W_POSITION_AVG = 0.5
W_POSITION_CONSISTENCY = 0.5
LEXICAL_CAP = 0.25
RANK_GAP_PENALTY = 0.15
PARENT_FREQ_DAMPING = 1.0

DEFAULT_TOP_COVERAGE = 0.95

_TOKEN_RE = re.compile(r"[a-z]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CANONICAL_DEPT_CSV)
    parser.add_argument("--institution-names", type=Path, default=INSTITUTION_ID_NAME_CSV)
    parser.add_argument("--output", type=Path, default=HIERARCHY_CSV)
    parser.add_argument("--top-coverage", type=float, default=DEFAULT_TOP_COVERAGE,
                        help="Coverage ratio of canonical departments to keep per institution.")
    return parser.parse_args()


# --------------------------------------------------------------------- helpers

def is_root_node(name: str) -> bool:
    return str(name).startswith(INSTITUTION_NODE_PREFIX)


def infer_lexical_rank(name: str) -> int | None:
    if is_root_node(name):
        return ROOT_LEXICAL_RANK
    tokens = set(_TOKEN_RE.findall(str(name).lower()))
    for rank in sorted(LEXICAL_RANKS):
        if tokens & LEXICAL_RANKS[rank]:
            return rank
    return None


def lexical_direction_bonus(parent: str, child: str) -> float:
    if is_root_node(parent) or is_root_node(child):
        return 0.0
    parent_rank = infer_lexical_rank(parent)
    child_rank = infer_lexical_rank(child)
    if parent_rank is None or child_rank is None or parent_rank == child_rank:
        return 0.0
    bonus = (child_rank - parent_rank) / len(LEXICAL_RANKS)
    return float(max(-LEXICAL_CAP, min(LEXICAL_CAP, bonus)))


def rank_gap_penalty(parent: str, child: str) -> float:
    fallback = len(LEXICAL_RANKS) + 1
    parent_rank = infer_lexical_rank(parent) or fallback
    child_rank = infer_lexical_rank(child) or fallback
    gap = max(child_rank - parent_rank - 1, 0)
    return -RANK_GAP_PENALTY * gap if gap else 0.0


def parent_freq_scale(freq: int) -> float:
    if freq <= 0:
        return 1.0
    scale = 1.0 / (1.0 + math.log1p(freq))
    return scale ** PARENT_FREQ_DAMPING if PARENT_FREQ_DAMPING != 1.0 else scale


def deduplicate_departments(group_df: pd.DataFrame) -> list[tuple[str, float]]:
    """Keep the earliest-``start`` occurrence of each canonical name in a record."""
    earliest: dict[str, float] = {}
    for _, row in group_df.iterrows():
        name = str(row.get("canonical_dept_name", "")).strip()
        if not name:
            continue
        try:
            start = float(row.get("start"))
        except (TypeError, ValueError):
            start = math.inf
        if math.isnan(start):
            start = math.inf
        if name not in earliest or start < earliest[name]:
            earliest[name] = start
    return sorted(earliest.items(), key=lambda kv: (kv[1], kv[0]))


# ------------------------------------------------------ top-coverage filtering

def keep_top_canonicals(df: pd.DataFrame, coverage: float) -> pd.DataFrame:
    freq = df.groupby(["institution_id", "canonical_dept_name"], as_index=False)["frequency"].sum()
    keep: list[tuple[Any, str]] = []
    for _, group in tqdm(freq.groupby("institution_id", sort=False),
                         desc="Selecting head canonicals", total=freq["institution_id"].nunique()):
        g = group.sort_values("frequency", ascending=False)
        total = g["frequency"].sum()
        if total == 0:
            continue
        cutoff_idx = (g["frequency"].cumsum() / total >= coverage).idxmax()
        keep.extend(g.loc[:cutoff_idx, ["institution_id", "canonical_dept_name"]].itertuples(index=False, name=None))
    keep_df = pd.DataFrame(keep, columns=["institution_id", "canonical_dept_name"])
    return df.merge(keep_df, on=["institution_id", "canonical_dept_name"], how="inner")


# -------------------------------------------------------------- co-occurrence

def build_cooccurrence_graph(inst_df: pd.DataFrame, root_node: str):
    pair_counter: Counter = Counter()
    node_freq: Counter = Counter()
    neighbor_sets: dict[str, set[str]] = defaultdict(set)
    position_sum: dict[str, float] = defaultdict(float)
    position_count: Counter = Counter()
    pair_order: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    for _, raw_group in inst_df.groupby("raw_affiliation_id"):
        records = deduplicate_departments(raw_group)
        if not records:
            continue
        names = [root_node] + [name for name, _ in records]
        n = len(names)
        norm_pos = {names[i]: (i / (n - 1) if n > 1 else 0.5) for i in range(n)}

        for name in names:
            node_freq[name] += 1
            position_sum[name] += norm_pos[name]
            position_count[name] += 1

        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                first, second = names[i], names[j]
                key = tuple(sorted((first, second)))
                pair_counter[key] += 1
                neighbor_sets[first].add(second)
                neighbor_sets[second].add(first)
                if first == key[0]:
                    pair_order[key][0] += 1
                else:
                    pair_order[key][1] += 1
    return pair_counter, node_freq, neighbor_sets, position_sum, position_count, pair_order


def _position_signal(parent: str, child: str, avg_position, pair_order) -> float:
    avg_component = max(-1.0, min(1.0, avg_position(parent) - avg_position(child)))
    key = tuple(sorted((parent, child)))
    counts = pair_order.get(key)
    if not counts:
        consistency = 0.0
    else:
        total = counts[0] + counts[1]
        if total == 0:
            consistency = 0.0
        else:
            if parent == key[0]:
                desired, other = counts[1], counts[0]
            else:
                desired, other = counts[0], counts[1]
            consistency = (desired - other) / total
    return W_POSITION * (W_POSITION_AVG * avg_component + W_POSITION_CONSISTENCY * consistency)


def score_candidate_edges(inst_id: Any, pair_counter, node_freq, neighbor_sets,
                          position_sum, position_count, pair_order) -> list[dict]:
    def avg_position(name: str) -> float:
        count = position_count.get(name, 0)
        return position_sum[name] / count if count else 0.5

    edges: list[dict] = []
    for (a, b), pair_count in pair_counter.items():
        freq_a, freq_b = node_freq[a], node_freq[b]
        if pair_count < MIN_COOC_COUNT:
            continue
        assoc = pair_count / float(min(freq_a, freq_b))
        if assoc < MIN_ASSOC_SCORE:
            continue

        coverage_delta = (pair_count / freq_b) - (pair_count / freq_a)
        degree_balance = (len(neighbor_sets.get(a, ())) - len(neighbor_sets.get(b, ()))) / max(
            len(neighbor_sets.get(a, ())) + len(neighbor_sets.get(b, ())), 1
        )
        freq_balance = (freq_a - freq_b) / max(freq_a + freq_b, 1)
        base = W_COVERAGE * coverage_delta + W_DEGREE * degree_balance + W_FREQ * freq_balance

        position_ab = _position_signal(a, b, avg_position, pair_order)
        position_ba = _position_signal(b, a, avg_position, pair_order)
        lex_ab = lexical_direction_bonus(a, b)
        lex_ba = lexical_direction_bonus(b, a)
        gap_ab = rank_gap_penalty(a, b)
        gap_ba = rank_gap_penalty(b, a)

        score_ab = (base + position_ab + lex_ab + gap_ab) * parent_freq_scale(freq_a)
        score_ba = (-base + position_ba + lex_ba + gap_ba) * parent_freq_scale(freq_b)

        edges.append({"institution_id": inst_id, "parent_dept": a, "child_dept": b,
                      "direction_score": score_ab})
        edges.append({"institution_id": inst_id, "parent_dept": b, "child_dept": a,
                      "direction_score": score_ba})
    return edges


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower, upper = math.floor(pos), math.ceil(pos)
    if lower == upper:
        return values[lower]
    return values[lower] * (1 - (pos - lower)) + values[upper] * (pos - lower)


def _forms_cycle(parent_map: dict[str, str], parent: str, child: str) -> bool:
    cursor = parent
    while cursor:
        if cursor == child:
            return True
        cursor = parent_map.get(cursor)
    return False


def select_acyclic_edges(candidates: list[dict]) -> list[dict]:
    selected: list[dict] = []
    parent_map: dict[str, str] = {}
    for edge in sorted(candidates, key=lambda e: e["direction_score"], reverse=True):
        child, parent = edge["child_dept"], edge["parent_dept"]
        if child in parent_map:
            continue
        if _forms_cycle(parent_map, parent, child):
            continue
        parent_map[child] = parent
        selected.append(edge)
    return selected


def process_institution(inst_id: Any, inst_df: pd.DataFrame) -> list[dict]:
    root = f"{INSTITUTION_NODE_PREFIX}{inst_id}"
    graph = build_cooccurrence_graph(inst_df, root)
    pair_counter, node_freq, neighbors, pos_sum, pos_count, pair_order = graph
    if not pair_counter:
        return []

    edges = score_candidate_edges(inst_id, pair_counter, node_freq, neighbors,
                                  pos_sum, pos_count, pair_order)
    if not edges:
        return []

    scores = sorted(e["direction_score"] for e in edges)
    threshold = max(SCORE_THRESHOLD, _quantile(scores, SCORE_QUANTILE) or SCORE_THRESHOLD)
    edges = [e for e in edges if e["direction_score"] >= threshold and not is_root_node(e["child_dept"])]

    if TOP_K_PARENTS_PER_CHILD and edges:
        per_child: dict[str, list[dict]] = defaultdict(list)
        for edge in edges:
            per_child[edge["child_dept"]].append(edge)
        edges = []
        for bucket in per_child.values():
            bucket.sort(key=lambda e: e["direction_score"], reverse=True)
            edges.extend(bucket[:TOP_K_PARENTS_PER_CHILD])

    return select_acyclic_edges(edges)


# ------------------------------------------------------ post-processing / I/O

def _build_hierarchy_df(all_edges: list[dict], all_depts: pd.DataFrame,
                        inst_name_map: dict[Any, str]) -> pd.DataFrame:
    hierarchy = pd.DataFrame(all_edges, columns=["institution_id", "parent_dept", "child_dept"])
    root_edges = hierarchy[hierarchy["parent_dept"].astype(str).str.startswith(INSTITUTION_NODE_PREFIX)]
    attached = (
        root_edges[["institution_id", "child_dept"]]
        .rename(columns={"child_dept": "canonical_dept_name"})
        .drop_duplicates()
    )
    missing = (
        all_depts.merge(attached, on=["institution_id", "canonical_dept_name"],
                        how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
    )

    def _replace_root(name: Any, inst_id: Any) -> Any:
        if isinstance(name, str) and name.startswith(INSTITUTION_NODE_PREFIX):
            return inst_name_map.get(inst_id, name)
        return name

    hierarchy["parent_dept"] = [
        _replace_root(p, i) for p, i in zip(hierarchy["parent_dept"], hierarchy["institution_id"])
    ]
    hierarchy["child_dept"] = [
        _replace_root(c, i) for c, i in zip(hierarchy["child_dept"], hierarchy["institution_id"])
    ]

    completion_edges = missing.assign(
        parent_dept=missing["institution_id"].map(inst_name_map),
        child_dept=missing["canonical_dept_name"],
    )[["institution_id", "parent_dept", "child_dept"]]

    combined = pd.concat([hierarchy, completion_edges], ignore_index=True, sort=False)
    combined["institution_name"] = combined["institution_id"].map(inst_name_map)
    combined["parent_name"] = combined["parent_dept"].astype(str).str.lower()
    combined["child_name"] = combined["child_dept"].astype(str).str.lower()
    combined["institution_name"] = combined["institution_name"].astype(str).str.lower()
    combined = combined[["institution_id", "institution_name", "parent_name", "child_name"]]
    return combined.drop_duplicates().sort_values(by=["institution_id", "parent_name", "child_name"])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    before = len(df)
    df = keep_top_canonicals(df, args.top_coverage)
    print(f"Retained {len(df):,} / {before:,} rows "
          f"({len(df) / before:.2%}) after top-{args.top_coverage:.0%} filtering")

    all_edges: list[dict] = []
    grouped = df.groupby("institution_id", sort=False)
    for inst_id, inst_df in tqdm(grouped, total=grouped.ngroups, desc="Institutions"):
        all_edges.extend(process_institution(inst_id, inst_df))
    print(f"Constructed {len(all_edges):,} scored edges across {grouped.ngroups:,} institutions")

    inst_name_df = pd.read_csv(args.institution_names)
    inst_name_map = dict(zip(inst_name_df["institution_id"], inst_name_df["institution_name"]))

    all_depts = df[["institution_id", "canonical_dept_name"]].drop_duplicates()
    final = _build_hierarchy_df(all_edges, all_depts, inst_name_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"Final edges (including direct root completions): {len(final):,}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
