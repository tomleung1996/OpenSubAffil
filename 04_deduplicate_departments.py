"""Cluster near-duplicate department names within each institution and pick a canonical form."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

try:
    import torch
except ImportError:  # optional dependency used for GPU similarity
    torch = None  # type: ignore[assignment]

from config import DEDUPE_INPUT_CSV, DEDUPE_OUTPUT_CSV, DEPARTMENT_CUE_TOKENS
from text_utils import extract_core_name, split_prefix_suffix


@dataclass
class Record:
    institution_id: str
    dept_name: str
    frequency: int
    embedding: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEDUPE_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEDUPE_OUTPUT_CSV)
    parser.add_argument("--borderline-output", type=Path, default=None,
                        help="Optional CSV of borderline pairs for human review.")
    parser.add_argument("--output-style", choices=["compact", "detailed"], default="compact",
                        help="compact: one row per cluster member; detailed: one row per cluster.")
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--institution-id", action="append", dest="institution_filters",
                        help="Restrict to specific institution IDs (repeatable or comma-separated).")
    parser.add_argument("--limit-institutions", type=int, default=None,
                        help="Only process the first N institutions (for debugging).")

    # Stage-1 prefix stripping.
    parser.add_argument("--min-suffix-freq-in-inst", type=int, default=3)
    parser.add_argument("--max-prefixed-ratio", type=float, default=0.25)
    parser.add_argument("--max-full-freq", type=int, default=5)
    # Stage-2 suffix variant unification.
    parser.add_argument("--min-suffix-variant-freq", type=int, default=500)
    parser.add_argument("--min-prefix-variants", type=int, default=2)
    # Stage-3 core-name variant unification.
    parser.add_argument("--min-core-group-freq", type=int, default=200)

    # Embedding + clustering.
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--distance-threshold", type=float, default=0.15,
                        help="Cosine distance threshold (0=identical, 2=opposite).")
    parser.add_argument("--borderline-margin", type=float, default=0.05,
                        help="Similarity margin below the merge threshold for borderline reporting.")
    parser.add_argument("--linkage", choices=["average", "complete"], default="average")
    parser.add_argument("--similarity-device", default=None,
                        help="Device for similarity computation (e.g. 'cuda'). CPU by default.")
    return parser.parse_args()


# --------------------------------------------------------------------------- I/O

def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"institution_ids": str},
        usecols=["institution_ids", "clean_dept_str", "frequency"],
        low_memory=False,
    )
    df["institution_ids"] = df["institution_ids"].fillna("").astype(str).str.strip()
    df = df[df["institution_ids"].str.len() > 0]
    df["clean_dept_str"] = df["clean_dept_str"].fillna("").astype(str).str.strip()
    df = df[df["clean_dept_str"].str.len() > 0]
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)
    return df


def filter_institutions(df: pd.DataFrame, filters: list[str] | None,
                        limit: int | None) -> pd.DataFrame:
    if filters:
        tokens: set[str] = set()
        for raw in filters:
            tokens.update(p.strip() for p in raw.split(",") if p.strip())
        df = df[df["institution_ids"].isin(tokens)]
        if df.empty:
            raise ValueError("No rows left after --institution-id filter.")
    if limit is not None:
        keep = df["institution_ids"].drop_duplicates().iloc[:limit].tolist()
        df = df[df["institution_ids"].isin(keep)]
    return df


# --------------------------------------------------------- deterministic stages

def strip_person_prefixes(df: pd.DataFrame, column: str, min_suffix_freq: int,
                          max_prefixed_ratio: float, max_full_freq: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    splits = [split_prefix_suffix(name, DEPARTMENT_CUE_TOKENS) for name in df[column]]
    df["_suffix"] = [s[1] for s in splits]
    df["_valid"] = [s[2] for s in splits]

    has_suffix = df["_suffix"].str.len() > 0
    df["_suffix_freq"] = 0
    if has_suffix.any():
        df.loc[has_suffix, "_suffix_freq"] = (
            df.loc[has_suffix].groupby(["institution_ids", "_suffix"])["frequency"].transform("sum")
        )
    df["_full_freq"] = df.groupby(["institution_ids", column])["frequency"].transform("sum")

    ratio = pd.Series(np.inf, index=df.index, dtype=float)
    non_zero = df["_suffix_freq"] > 0
    ratio.loc[non_zero] = df.loc[non_zero, "_full_freq"] / df.loc[non_zero, "_suffix_freq"]

    strip_mask = (
        df["_valid"]
        & has_suffix
        & (df["_suffix_freq"] >= min_suffix_freq)
        & (ratio <= max_prefixed_ratio)
        & (df["_full_freq"] <= max_full_freq)
    )
    df.loc[strip_mask, column] = df.loc[strip_mask, "_suffix"]
    df.drop(columns=["_suffix", "_valid", "_suffix_freq", "_full_freq"], inplace=True)
    return df


def normalize_high_freq_suffix_variants(df: pd.DataFrame, column: str,
                                        min_group_freq: int, min_prefix_variants: int) -> pd.DataFrame:
    if df.empty:
        return df

    split_lookup = {
        name: split_prefix_suffix(name, DEPARTMENT_CUE_TOKENS, max_prefix_tokens=4)
        for name in df[column].unique()
    }

    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby("institution_ids", sort=False):
        inst = group.copy()
        splits = inst[column].map(split_lookup)
        inst["_prefix"] = [s[0] for s in splits]
        inst["_suffix"] = [s[1] for s in splits]
        inst["_valid"] = [s[2] for s in splits]

        usable = (inst["_suffix"].str.len() > 0) & inst["_valid"]
        if usable.any():
            grouped = inst.loc[usable].groupby("_suffix")
            inst.loc[usable, "_group_freq"] = grouped["frequency"].transform("sum")
            inst.loc[usable, "_variants"] = grouped["_prefix"].transform("nunique")
        else:
            inst["_group_freq"] = 0
            inst["_variants"] = 0

        targets = (
            usable
            & (inst["_group_freq"] >= min_group_freq)
            & (inst["_variants"] >= min_prefix_variants)
        )
        for suffix in inst.loc[targets, "_suffix"].drop_duplicates():
            mask = (inst["_suffix"] == suffix) | (inst[column] == suffix)
            candidates = inst.loc[mask].copy()
            candidates["_len"] = candidates[column].str.len()
            canonical = candidates.sort_values(
                by=["frequency", "_len", column],
                ascending=[False, False, True],
            )[column].iloc[0]
            inst.loc[mask, column] = canonical

        inst.drop(columns=["_prefix", "_suffix", "_valid", "_group_freq", "_variants"],
                  inplace=True, errors="ignore")
        pieces.append(inst)

    return pd.concat(pieces, ignore_index=True) if pieces else df


def normalize_core_name_variants(df: pd.DataFrame, column: str,
                                 min_group_freq: int) -> pd.DataFrame:
    if df.empty:
        return df
    core_lookup = {name: extract_core_name(name) for name in df[column].unique()}

    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby("institution_ids", sort=False):
        inst = group.copy()
        inst["_core"] = inst[column].map(core_lookup)
        usable = inst["_core"].str.len() > 0
        if usable.any():
            grouped = inst.loc[usable].groupby("_core")
            inst.loc[usable, "_group_freq"] = grouped["frequency"].transform("sum")
            inst.loc[usable, "_variants"] = grouped[column].transform("nunique")
        else:
            inst["_group_freq"] = 0
            inst["_variants"] = 0

        targets = (
            usable
            & (inst["_group_freq"] >= min_group_freq)
            & (inst["_variants"] >= 2)
        )
        for core in inst.loc[targets, "_core"].drop_duplicates():
            mask = inst["_core"] == core
            candidates = inst.loc[mask].copy()
            if candidates[column].nunique() < 2:
                continue
            candidates["_len"] = candidates[column].str.len()
            canonical = candidates.sort_values(
                by=["frequency", "_len", column],
                ascending=[False, False, True],
            )[column].iloc[0]
            inst.loc[mask, column] = canonical

        inst.drop(columns=["_core", "_group_freq", "_variants"], inplace=True, errors="ignore")
        pieces.append(inst)

    return pd.concat(pieces, ignore_index=True) if pieces else df


# --------------------------------------------------------- embedding + clustering

def build_embedding_lookup(texts: list[str], model_name: str, batch_size: int,
                           device: str | None) -> dict[str, np.ndarray]:
    unique = sorted(set(texts))
    if not unique:
        return {}
    kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **kwargs)
    embeddings = model.encode(
        unique,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return {text: embeddings[i].astype(np.float32) for i, text in enumerate(unique)}


def build_member_lookup(df: pd.DataFrame) -> dict[tuple[str, str], list[dict]]:
    grouped = (
        df.groupby(["institution_ids", "normalized_dept_str", "clean_dept_str"], as_index=False)
        ["frequency"].sum()
    )
    lookup: dict[tuple[str, str], list[dict]] = {}
    for row in grouped.itertuples(index=False):
        key = (row.institution_ids, row.normalized_dept_str)
        lookup.setdefault(key, []).append({
            "raw_name": row.clean_dept_str,
            "frequency": int(row.frequency),
        })
    for entries in lookup.values():
        entries.sort(key=lambda e: (-e["frequency"], e["raw_name"]))
    return lookup


def build_records(group: pd.DataFrame, embedding_lookup: dict[str, np.ndarray]) -> list[Record]:
    inst_id = group["institution_ids"].iloc[0]
    collapsed = group.groupby("normalized_dept_str", as_index=False)["frequency"].sum()
    records: list[Record] = []
    for row in collapsed.itertuples(index=False):
        vec = embedding_lookup.get(row.normalized_dept_str)
        if vec is None:
            continue
        records.append(Record(inst_id, row.normalized_dept_str, int(row.frequency), vec))
    return records


def compute_similarity_matrix(vectors: np.ndarray, device: str | None,
                              chunk_size: int = 2048) -> np.ndarray:
    if device is not None:
        if torch is None:
            raise RuntimeError("--similarity-device requires PyTorch.")
        tensor = torch.from_numpy(vectors).to(torch.device(device))
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
        chunks: list[np.ndarray] = []
        for start in range(0, tensor.shape[0], chunk_size):
            block = torch.matmul(tensor[start:start + chunk_size], tensor.t())
            chunks.append(block.detach().cpu().numpy())
        return np.clip(np.vstack(chunks), -1.0, 1.0)

    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.nan_to_num(normalized)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def cluster_institution(records: list[Record], distance_threshold: float, linkage: str,
                        similarity_device: str | None, borderline_margin: float,
                        collect_borderline: bool) -> tuple[list[list[Record]], list[tuple[Record, Record, float]]]:
    if not records:
        return [], []
    if len(records) == 1:
        return [records], []

    vectors = np.stack([r.embedding for r in records]).astype(np.float32)
    sim_matrix: np.ndarray | None = None
    if similarity_device is not None or collect_borderline:
        sim_matrix = compute_similarity_matrix(vectors, device=similarity_device)

    if sim_matrix is not None:
        distance = np.clip(1.0 - sim_matrix, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        clusterer = AgglomerativeClustering(
            metric="precomputed", linkage=linkage,
            distance_threshold=distance_threshold, n_clusters=None,
        )
        labels = clusterer.fit_predict(distance)
    else:
        clusterer = AgglomerativeClustering(
            metric="cosine", linkage=linkage,
            distance_threshold=distance_threshold, n_clusters=None,
        )
        labels = clusterer.fit_predict(vectors)

    clusters: dict[int, list[Record]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(records[idx])
    cluster_list = list(clusters.values())

    borderline: list[tuple[Record, Record, float]] = []
    if collect_borderline and sim_matrix is not None and borderline_margin > 0:
        merge_similarity = 1.0 - distance_threshold
        lower = max(0.0, merge_similarity - borderline_margin)
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if labels[i] == labels[j]:
                    continue
                s = float(sim_matrix[i, j])
                if lower <= s < merge_similarity:
                    borderline.append((records[i], records[j], s))
    return cluster_list, borderline


# --------------------------------------------------------- output construction

def aggregate_cluster(inst_id: str, members: list[Record],
                      member_lookup: dict[tuple[str, str], list[dict]]) -> dict:
    canonical = sorted(
        members,
        key=lambda r: (-r.frequency, len(r.dept_name), r.dept_name.lower()),
    )[0]
    details: list[dict] = []
    for rec in sorted(members, key=lambda r: (-r.frequency, r.dept_name.lower())):
        raw_members = member_lookup.get((inst_id, rec.dept_name), [
            {"raw_name": rec.dept_name, "frequency": rec.frequency},
        ])
        for item in raw_members:
            details.append({
                "raw_name": item["raw_name"],
                "frequency": item["frequency"],
                "normalized_name": rec.dept_name,
            })
    return {
        "institution_id": inst_id,
        "canonical_clean_dept": canonical.dept_name,
        "total_frequency": sum(r.frequency for r in members),
        "member_count": len(members),
        "member_details": details,
    }


def build_cluster_rows(clusters: list[dict], style: str) -> list[dict]:
    if style == "detailed":
        return [{
            "institution_id": c["institution_id"],
            "canonical_dept_name": c["canonical_clean_dept"],
            "total_frequency": c["total_frequency"],
            "member_count": c["member_count"],
            "member_details": json.dumps(c["member_details"], ensure_ascii=False),
        } for c in clusters]

    rows: list[dict] = []
    for c in clusters:
        for member in c["member_details"]:
            rows.append({
                "institution_id": c["institution_id"],
                "canonical_dept_name": c["canonical_clean_dept"],
                "cluster_total_frequency": c["total_frequency"],
                "member_dept_name": member["raw_name"],
            })
    return rows


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.input)
    df = df[df["frequency"] >= args.min_frequency]
    df = filter_institutions(df, args.institution_filters, args.limit_institutions)
    if df.empty:
        raise ValueError("No data remains after filtering.")

    df["normalized_dept_str"] = df["clean_dept_str"]

    print("Stage 1: strip person prefixes...")
    df = strip_person_prefixes(
        df, column="normalized_dept_str",
        min_suffix_freq=args.min_suffix_freq_in_inst,
        max_prefixed_ratio=args.max_prefixed_ratio,
        max_full_freq=args.max_full_freq,
    )
    print("Stage 2: unify high-frequency suffix variants...")
    df = normalize_high_freq_suffix_variants(
        df, column="normalized_dept_str",
        min_group_freq=args.min_suffix_variant_freq,
        min_prefix_variants=args.min_prefix_variants,
    )
    print("Stage 3: unify core-name variants...")
    df = normalize_core_name_variants(
        df, column="normalized_dept_str",
        min_group_freq=args.min_core_group_freq,
    )
    print(f"Post-normalisation: {len(df):,} rows across {df['institution_ids'].nunique():,} institutions.")

    print("Encoding department names...")
    embedding_lookup = build_embedding_lookup(
        texts=df["normalized_dept_str"].tolist(),
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
    )
    if not embedding_lookup:
        raise RuntimeError("Failed to encode any department strings.")

    member_lookup = build_member_lookup(df)
    clusters_output: list[dict] = []
    borderline_rows: list[dict] = []
    collect_borderline = args.borderline_output is not None

    grouped = df.groupby("institution_ids")
    for inst_id, group in tqdm(grouped, total=grouped.ngroups, desc="Clustering institutions"):
        records = build_records(group, embedding_lookup)
        clusters, borderline = cluster_institution(
            records,
            distance_threshold=args.distance_threshold,
            linkage=args.linkage,
            similarity_device=args.similarity_device,
            borderline_margin=args.borderline_margin,
            collect_borderline=collect_borderline,
        )
        for members in clusters:
            clusters_output.append(aggregate_cluster(inst_id, members, member_lookup))
        if collect_borderline:
            for rec_a, rec_b, score in borderline:
                borderline_rows.append({
                    "institution_id": inst_id,
                    "dept_a": rec_a.dept_name,
                    "dept_b": rec_b.dept_name,
                    "similarity": round(score, 4),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_cluster_rows(clusters_output, args.output_style)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows):,} cluster rows to {args.output}")

    if collect_borderline:
        args.borderline_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(borderline_rows).to_csv(args.borderline_output, index=False)
        print(f"Wrote {len(borderline_rows):,} borderline rows to {args.borderline_output}")


if __name__ == "__main__":
    main()
