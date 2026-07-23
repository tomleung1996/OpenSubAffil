"""Reproduce the disambiguation validation results reported in Table 7.

The script is self-contained apart from pandas and uses only the two expert
annotation CSV files supplied on the command line. In each file, ``label=1``
means that ``member_name`` refers to the same sub-institution as
``canonical_name`` and ``label=0`` means that it does not.

``is_original_member=1`` identifies a name already assigned to the cluster;
``is_original_member=0`` identifies an unmerged similarity candidate. The
institution and entity tier fields are stored directly in both released CSVs,
so no unpublished sampling metadata are required.

Example
-------
uv run --with pandas python validation/reproduce_table7.py \
    data/sub_institution_cluster_membership_annotation_expert1.csv \
    data/sub_institution_cluster_membership_annotation_expert2.csv \
    --output table7_reproduced.csv
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


TIER_ORDER = ("large", "medium", "small")
KEY_COLUMNS = [
    "evaluation_id",
    "institution_id",
    "institution_name",
    "institution_size_tier",
    "sub_institution_id",
    "canonical_name",
    "entity_size_tier",
    "member_name",
    "is_original_member",
]
REQUIRED_COLUMNS = [*KEY_COLUMNS, "label"]
METRIC_COLUMNS = [
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]


def load_expert_annotations(path: str | Path) -> pd.DataFrame:
    """Load and validate one expert annotation file."""
    path = Path(path)
    df = pd.read_csv(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["is_original_member"] = pd.to_numeric(
        df["is_original_member"], errors="raise"
    ).astype(int)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not df["is_original_member"].isin([0, 1]).all():
        raise ValueError(f"{path} contains is_original_member values outside {{0, 1}}")
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{path} contains label values outside {{0, 1}}")

    for column in ["institution_size_tier", "entity_size_tier"]:
        df[column] = df[column].astype(str).str.strip().str.lower()
        invalid_tiers = sorted(set(df[column]) - set(TIER_ORDER))
        if invalid_tiers:
            raise ValueError(f"{path} contains invalid {column} values: {invalid_tiers}")

    if df.duplicated(subset=KEY_COLUMNS, keep=False).any():
        raise ValueError(f"{path} contains duplicate annotation keys")
    return df


def align_expert_annotations(
    expert1_df: pd.DataFrame,
    expert2_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align the two expert files and require identical annotation rows."""
    aligned_df = expert1_df.rename(columns={"label": "expert1_label"}).merge(
        expert2_df.rename(columns={"label": "expert2_label"}),
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = aligned_df["_merge"] != "both"
    if unmatched.any():
        counts = aligned_df.loc[unmatched, "_merge"].value_counts().to_dict()
        raise ValueError(f"The expert files do not contain identical rows: {counts}")

    return aligned_df.drop(columns="_merge").sort_values(
        ["evaluation_id", "is_original_member", "member_name"],
        kind="stable",
    ).reset_index(drop=True)


def compute_agreement(aligned_df: pd.DataFrame) -> tuple[float, float]:
    """Return raw agreement and Cohen's kappa for the two experts."""
    labels1 = aligned_df["expert1_label"]
    labels2 = aligned_df["expert2_label"]
    observed_agreement = float((labels1 == labels2).mean())

    probability1_positive = float(labels1.mean())
    probability2_positive = float(labels2.mean())
    expected_agreement = (
        probability1_positive * probability2_positive
        + (1.0 - probability1_positive) * (1.0 - probability2_positive)
    )
    if expected_agreement >= 1.0:
        cohen_kappa = 1.0 if observed_agreement >= 1.0 else float("nan")
    else:
        cohen_kappa = (
            observed_agreement - expected_agreement
        ) / (1.0 - expected_agreement)
    return observed_agreement, cohen_kappa


def build_entity_metrics(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the unanimous-positive consensus and calculate TP, FP, and FN."""
    working_df = aligned_df.copy()

    working_df["final_label"] = working_df[
        ["expert1_label", "expert2_label"]
    ].min(axis=1).astype(int)

    working_df["tp"] = (
        (working_df["is_original_member"] == 1)
        & (working_df["final_label"] == 1)
    ).astype(int)
    working_df["fp"] = (
        (working_df["is_original_member"] == 1)
        & (working_df["final_label"] == 0)
    ).astype(int)
    working_df["fn"] = (
        (working_df["is_original_member"] == 0)
        & (working_df["final_label"] == 1)
    ).astype(int)

    entity_columns = [
        "evaluation_id",
        "institution_size_tier",
        "entity_size_tier",
    ]
    entity_metrics_df = (
        working_df.groupby(entity_columns, as_index=False)[["tp", "fp", "fn"]]
        .sum()
        .sort_values("evaluation_id", kind="stable")
        .reset_index(drop=True)
    )

    precision_denominator = entity_metrics_df["tp"] + entity_metrics_df["fp"]
    recall_denominator = entity_metrics_df["tp"] + entity_metrics_df["fn"]
    f1_denominator = (
        2 * entity_metrics_df["tp"]
        + entity_metrics_df["fp"]
        + entity_metrics_df["fn"]
    )
    entity_metrics_df["precision"] = (
        entity_metrics_df["tp"]
        / precision_denominator.where(precision_denominator > 0)
    )
    entity_metrics_df["recall"] = (
        entity_metrics_df["tp"] / recall_denominator.where(recall_denominator > 0)
    )
    entity_metrics_df["f1"] = (
        2 * entity_metrics_df["tp"] / f1_denominator.where(f1_denominator > 0)
    )
    return entity_metrics_df


def summarize_subset(entity_metrics_df: pd.DataFrame) -> dict[str, float | int]:
    """Calculate micro- and macro-averaged metrics for one subset."""
    if entity_metrics_df.empty:
        raise ValueError("Cannot calculate metrics for an empty subset")

    total_tp = int(entity_metrics_df["tp"].sum())
    total_fp = int(entity_metrics_df["fp"].sum())
    total_fn = int(entity_metrics_df["fn"].sum())

    micro_precision = total_tp / (total_tp + total_fp)
    micro_recall = total_tp / (total_tp + total_fn)
    micro_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": float(entity_metrics_df["precision"].mean()),
        "macro_recall": float(entity_metrics_df["recall"].mean()),
        "macro_f1": float(entity_metrics_df["f1"].mean()),
    }


def build_table7(entity_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build the overall and tier-specific rows shown in Table 7."""
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "grouping": "Overall",
            "subset": "/",
            **summarize_subset(entity_metrics_df),
        }
    )

    for grouping, column in [
        ("Institution", "institution_size_tier"),
        ("Entity", "entity_size_tier"),
    ]:
        for tier in TIER_ORDER:
            subset_df = entity_metrics_df[entity_metrics_df[column] == tier]
            rows.append(
                {
                    "grouping": grouping,
                    "subset": tier.title(),
                    **summarize_subset(subset_df),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["grouping", "subset", *METRIC_COLUMNS],
    )


def format_table7_metric(value: float) -> str:
    """Match Table 7's reporting from four-decimal results to three decimals."""
    four_decimal_value = Decimal(f"{float(value):.4f}")
    return str(
        four_decimal_value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


def build_reported_table7(table7_df: pd.DataFrame) -> pd.DataFrame:
    """Return the three-decimal values displayed in the manuscript table."""
    reported_df = table7_df.copy()
    for column in METRIC_COLUMNS:
        reported_df[column] = reported_df[column].map(
            lambda value: float(format_table7_metric(value))
        )
    return reported_df


def print_results(
    aligned_df: pd.DataFrame,
    table7_df: pd.DataFrame,
    observed_agreement: float,
    cohen_kappa: float,
) -> None:
    """Print agreement statistics and a publication-style Table 7."""
    original_members = int((aligned_df["is_original_member"] == 1).sum())
    candidate_members = int((aligned_df["is_original_member"] == 0).sum())

    print(f"Aligned annotation rows: {len(aligned_df):,}")
    print(f"Original members: {original_members:,}")
    print(f"Candidate members: {candidate_members:,}")
    print(f"Raw agreement: {observed_agreement:.2%}")
    print(f"Cohen's kappa: {cohen_kappa:.4f}")
    print("Consensus rule: final label is 1 only when both experts assign label 1.")
    print()
    print("Table 7. Disambiguation quality evaluation results")

    display_df = table7_df.rename(
        columns={
            "grouping": "Grouping",
            "subset": "Subset",
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
                "Micro P": format_table7_metric,
                "Micro R": format_table7_metric,
                "Micro F1": format_table7_metric,
                "Macro P": format_table7_metric,
                "Macro R": format_table7_metric,
                "Macro F1": format_table7_metric,
            },
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce expert agreement and Table 7 from two expert annotation CSV files."
        )
    )
    parser.add_argument("expert1_path", type=Path)
    parser.add_argument("expert2_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for writing the reproduced Table 7 as CSV.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    expert1_df = load_expert_annotations(args.expert1_path)
    expert2_df = load_expert_annotations(args.expert2_path)
    aligned_df = align_expert_annotations(expert1_df, expert2_df)
    observed_agreement, cohen_kappa = compute_agreement(aligned_df)
    entity_metrics_df = build_entity_metrics(aligned_df)
    table7_df = build_table7(entity_metrics_df)

    print_results(
        aligned_df=aligned_df,
        table7_df=table7_df,
        observed_agreement=observed_agreement,
        cohen_kappa=cohen_kappa,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        build_reported_table7(table7_df).to_csv(
            args.output,
            index=False,
            float_format="%.3f",
        )
        print()
        print(f"Table 7 CSV: {args.output}")


if __name__ == "__main__":
    main()
