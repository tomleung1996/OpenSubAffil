"""Assemble the five OpenSubAffil release tables from the pipeline intermediates."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    CANONICAL_DEPT_CSV,
    EDUCATION_INSTITUTIONS_CSV,
    FINAL_OUTPUT_DIR,
    HIERARCHY_CSV,
    INSTITUTION_ID_NAME_CSV,
    LANG_FREQ_CSV,
)

RAW_AFF_STRINGS_OUTPUT = "opensubaffil_raw_affiliation_strings.csv"
INSTITUTIONS_OUTPUT = "opensubaffil_institutions.csv"
SUB_INSTITUTIONS_OUTPUT = "opensubaffil_sub_institutions.csv"
MAPPINGS_OUTPUT = "opensubaffil_mappings.csv"
HIERARCHY_OUTPUT = "opensubaffil_hierarchy.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dept", type=Path, default=CANONICAL_DEPT_CSV)
    parser.add_argument("--hierarchy", type=Path, default=HIERARCHY_CSV)
    parser.add_argument("--institution-names", type=Path, default=INSTITUTION_ID_NAME_CSV)
    parser.add_argument("--raw-aff-strings", type=Path, default=LANG_FREQ_CSV)
    parser.add_argument("--education-institutions", type=Path, default=EDUCATION_INSTITUTIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=FINAL_OUTPUT_DIR)
    return parser.parse_args()


def load_education_institution_ids(path: Path) -> set[int]:
    df = pd.read_csv(path, usecols=["institution_id"])
    return set(df["institution_id"].astype(int).tolist())


def load_raw_affiliation_strings(path: Path, raw_ids: list[int]) -> pd.DataFrame:
    target_ids = {int(i) for i in raw_ids}
    if not target_ids:
        return pd.DataFrame(columns=["raw_affiliation_string_id", "raw_affiliation_string"])

    df = pd.read_csv(
        path,
        usecols=["raw_affiliation_string_id", "raw_affiliation_string"],
    )
    df = df[df["raw_affiliation_string_id"].isin(target_ids)]
    df = df.drop_duplicates(subset=["raw_affiliation_string_id"])

    missing = target_ids - set(df["raw_affiliation_string_id"].astype(int).tolist())
    if missing:
        preview = ", ".join(str(i) for i in sorted(missing)[:10])
        raise ValueError(
            f"Missing raw_affiliation_string rows for {len(missing)} ids: {preview}"
        )
    return df.sort_values("raw_affiliation_string_id").reset_index(drop=True)


def _strip_root_like_names(
    filtered_raw: pd.DataFrame,
    filtered_hier: pd.DataFrame,
    institution_name_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop canonical names that coincide (case-insensitively) with their institution's name.

    Such "root-like" names otherwise leak into ``sub_institutions`` as
    self-referential entries. Rows referencing them are removed from the raw
    mappings and from the hierarchy (child side, plus the non-root parent
    side -- genuine root-parent edges where ``parent_name == institution_name``
    are kept intact).
    """
    lookup = institution_name_df[["institution_id", "institution_name"]].copy()
    lookup["institution_name"] = lookup["institution_name"].astype(str).str.lower()

    candidates = pd.concat(
        [
            filtered_hier[["institution_id", "parent_name"]].rename(columns={"parent_name": "canonical_name"}),
            filtered_hier[["institution_id", "child_name"]].rename(columns={"child_name": "canonical_name"}),
            filtered_raw[["institution_id", "canonical_dept_name"]].rename(columns={"canonical_dept_name": "canonical_name"}),
        ],
        ignore_index=True,
    ).dropna(subset=["canonical_name"]).drop_duplicates()
    candidates = candidates.merge(lookup, on="institution_id", how="left", validate="many_to_one")
    root_like = candidates[
        candidates["canonical_name"].astype(str).str.lower() == candidates["institution_name"]
    ][["institution_id", "canonical_name"]].drop_duplicates()
    if root_like.empty:
        return filtered_raw, filtered_hier

    def _anti_join(df: pd.DataFrame, keys: pd.DataFrame, on: list[str]) -> pd.DataFrame:
        merged = df.merge(keys, on=on, how="left", indicator=True)
        return merged[merged["_merge"] == "left_only"].drop(columns="_merge")

    filtered_raw = _anti_join(
        filtered_raw,
        root_like.rename(columns={"canonical_name": "canonical_dept_name"}),
        ["institution_id", "canonical_dept_name"],
    )
    filtered_hier = _anti_join(
        filtered_hier,
        root_like.rename(columns={"canonical_name": "child_name"}),
        ["institution_id", "child_name"],
    )
    is_root_parent = filtered_hier["parent_name"] == filtered_hier["institution_name"]
    non_root = _anti_join(
        filtered_hier[~is_root_parent],
        root_like.rename(columns={"canonical_name": "parent_name"}),
        ["institution_id", "parent_name"],
    )
    filtered_hier = pd.concat([filtered_hier[is_root_parent], non_root], ignore_index=True)
    return filtered_raw, filtered_hier


def build_core_tables(
    raw_aff_dept_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    institution_name_df: pd.DataFrame,
    education_ids: set[int],
) -> dict[str, pd.DataFrame | list[int]]:
    filtered_raw = raw_aff_dept_df[
        raw_aff_dept_df["institution_id"].isin(education_ids)
    ].copy()
    filtered_hier = hierarchy_df[
        hierarchy_df["institution_id"].isin(education_ids)
    ].copy()

    # Only canonical names that participate in the hierarchy survive.
    valid_names_df = pd.concat(
        [
            filtered_hier[["institution_id", "parent_name"]].rename(
                columns={"parent_name": "canonical_dept_name"}
            ),
            filtered_hier[["institution_id", "child_name"]].rename(
                columns={"child_name": "canonical_dept_name"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["canonical_dept_name"]).drop_duplicates().reset_index(drop=True)

    filtered_raw = filtered_raw.merge(
        valid_names_df,
        on=["institution_id", "canonical_dept_name"],
        how="inner",
    )

    # Defensive cleanup: drop canonical names that coincide (case-insensitively)
    # with their institution's own name, along with every edge/mapping that
    # references them.
    filtered_raw, filtered_hier = _strip_root_like_names(
        filtered_raw, filtered_hier, institution_name_df,
    )

    # Canonical name pool: every non-root parent (root parents carry the
    # institution's own name and must not become a sub-institution entry),
    # every child, and every name still referenced by the mappings.
    root_parent_mask = filtered_hier["parent_name"] == filtered_hier["institution_name"]
    sub_institutions_df = (
        pd.concat(
            [
                filtered_hier.loc[~root_parent_mask, ["institution_id", "parent_name"]].rename(
                    columns={"parent_name": "canonical_name"}
                ),
                filtered_hier[["institution_id", "child_name"]].rename(
                    columns={"child_name": "canonical_name"}
                ),
                filtered_raw[["institution_id", "canonical_dept_name"]].rename(
                    columns={"canonical_dept_name": "canonical_name"}
                ),
            ],
            ignore_index=True,
        )
        .dropna(subset=["canonical_name"])
        .drop_duplicates()
        .sort_values(["institution_id", "canonical_name"])
        .reset_index(drop=True)
    )
    sub_institutions_df.insert(0, "sub_institution_id", range(1, len(sub_institutions_df) + 1))

    # Mappings: raw affiliation string -> sub_institution_id.
    mappings_df = filtered_raw.merge(
        sub_institutions_df.rename(columns={"canonical_name": "canonical_dept_name"}),
        on=["institution_id", "canonical_dept_name"],
        how="left",
        validate="many_to_one",
    )
    if mappings_df["sub_institution_id"].isna().any():
        raise ValueError("Failed to assign sub_institution_id for one or more mappings")

    mappings_df = mappings_df.rename(
        columns={
            "raw_affiliation_id": "raw_affiliation_string_id",
            "raw_dept_str": "raw_sub_institution_name",
        }
    )
    mappings_df = (
        mappings_df[[
            "raw_affiliation_string_id", "institution_id",
            "sub_institution_id", "raw_sub_institution_name",
        ]]
        .drop_duplicates()
        .sort_values([
            "raw_affiliation_string_id", "institution_id",
            "sub_institution_id", "raw_sub_institution_name",
        ])
        .reset_index(drop=True)
    )

    # Hierarchy with sub_institution_id pairs. A ``parent_name`` that equals the
    # institution name is the virtual root — emit NULL for parent_sub_institution_id.
    parent_lookup = sub_institutions_df.rename(
        columns={"sub_institution_id": "parent_sub_institution_id",
                 "canonical_name": "parent_name"}
    )
    child_lookup = sub_institutions_df.rename(
        columns={"sub_institution_id": "child_sub_institution_id",
                 "canonical_name": "child_name"}
    )
    hierarchy_output_df = (
        filtered_hier
        .merge(
            parent_lookup[["institution_id", "parent_name", "parent_sub_institution_id"]],
            on=["institution_id", "parent_name"], how="left", validate="many_to_one",
        )
        .merge(
            child_lookup[["institution_id", "child_name", "child_sub_institution_id"]],
            on=["institution_id", "child_name"], how="left", validate="many_to_one",
        )
    )
    hierarchy_output_df.loc[
        hierarchy_output_df["parent_name"] == hierarchy_output_df["institution_name"],
        "parent_sub_institution_id",
    ] = pd.NA
    if hierarchy_output_df["child_sub_institution_id"].isna().any():
        raise ValueError("Failed to assign child_sub_institution_id for one or more hierarchy rows")

    hierarchy_output_df = (
        hierarchy_output_df[[
            "institution_id", "parent_sub_institution_id", "child_sub_institution_id",
        ]]
        .drop_duplicates()
        .astype({
            "institution_id": "int64",
            "parent_sub_institution_id": "Int64",
            "child_sub_institution_id": "Int64",
        })
        .sort_values([
            "institution_id", "child_sub_institution_id", "parent_sub_institution_id",
        ])
        .reset_index(drop=True)
    )

    involved_ids = sorted(
        set(sub_institutions_df["institution_id"].astype(int))
        | set(hierarchy_output_df["institution_id"].astype(int))
        | set(mappings_df["institution_id"].astype(int))
    )
    institutions_df = (
        institution_name_df[institution_name_df["institution_id"].isin(involved_ids)]
        [["institution_id", "institution_name"]]
        .drop_duplicates()
        .sort_values("institution_id")
        .reset_index(drop=True)
    )

    return {
        "institutions": institutions_df,
        "sub_institutions": sub_institutions_df,
        "mappings": mappings_df,
        "hierarchy": hierarchy_output_df,
        "raw_affiliation_string_ids": mappings_df["raw_affiliation_string_id"].astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()

    raw_aff_dept_df = pd.read_csv(
        args.canonical_dept,
        usecols=["raw_affiliation_id", "institution_id", "raw_dept_str", "canonical_dept_name"],
    )
    hierarchy_df = pd.read_csv(
        args.hierarchy,
        usecols=["institution_id", "institution_name", "parent_name", "child_name"],
    )
    institution_name_df = pd.read_csv(
        args.institution_names,
        usecols=["institution_id", "institution_name"],
    )
    print(f"Canonical department rows: {len(raw_aff_dept_df):,}")
    print(f"Hierarchy rows: {len(hierarchy_df):,}")
    print(f"Institution name rows: {len(institution_name_df):,}")

    education_ids = load_education_institution_ids(args.education_institutions)
    print(f"Education institutions: {len(education_ids):,}")

    core = build_core_tables(
        raw_aff_dept_df=raw_aff_dept_df,
        hierarchy_df=hierarchy_df,
        institution_name_df=institution_name_df,
        education_ids=education_ids,
    )
    print(f"opensubaffil_institutions rows: {len(core['institutions']):,}")
    print(f"opensubaffil_sub_institutions rows: {len(core['sub_institutions']):,}")
    print(f"opensubaffil_mappings rows: {len(core['mappings']):,}")
    print(f"opensubaffil_hierarchy rows: {len(core['hierarchy']):,}")

    raw_aff_strings_df = load_raw_affiliation_strings(
        args.raw_aff_strings, core["raw_affiliation_string_ids"],
    )
    print(f"opensubaffil_raw_affiliation_strings rows: {len(raw_aff_strings_df):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_aff_strings_df.to_csv(args.output_dir / RAW_AFF_STRINGS_OUTPUT, index=False)
    core["institutions"].to_csv(args.output_dir / INSTITUTIONS_OUTPUT, index=False)
    core["sub_institutions"].to_csv(args.output_dir / SUB_INSTITUTIONS_OUTPUT, index=False)
    core["mappings"].to_csv(args.output_dir / MAPPINGS_OUTPUT, index=False)
    core["hierarchy"].to_csv(args.output_dir / HIERARCHY_OUTPUT, index=False, na_rep="")
    print(f"Wrote submission files to {args.output_dir}")


if __name__ == "__main__":
    main()
