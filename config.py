"""Shared constants for the sub-institution disambiguation pipeline."""
from __future__ import annotations

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_ROOT / "data"
FINAL_OUTPUT_DIR = DATA_DIR / "final_output"

# Default file paths.
RAW_AFFILIATION_CSV = DATA_DIR / "raw_affiliation_string.csv"
LANG_FREQ_CSV = DATA_DIR / "raw_aff_str_lang_freq.csv"
NER_OUTPUT_JSONL = DATA_DIR / "raw_affiliation_ner_outputs.jsonl"
RAW_AFF_TO_INSTITUTIONS_CSV = DATA_DIR / "raw_aff_to_institutions.csv"
INSTITUTION_ID_NAME_CSV = DATA_DIR / "institution_id_name.csv"
ABBR_LOOKUP_CSV = DATA_DIR / "raw_department_str_abbr_full.csv"
NER_PROCESSED_CSV = DATA_DIR / "raw_affiliation_ner_outputs_with_processed_inst.csv"
DEDUPE_INPUT_CSV = DATA_DIR / "institution_raw_department_str.csv"
DEDUPE_OUTPUT_CSV = DATA_DIR / "institution_department_clusters.csv"
CANONICAL_DEPT_CSV = DATA_DIR / "raw_affiliation_canonical_department.csv"
HIERARCHY_CSV = DATA_DIR / "raw_affiliation_canonical_department_hierarchy.csv"
EDUCATION_INSTITUTIONS_CSV = DATA_DIR / "education_institutions.csv"

# Hugging Face model identifiers.
SPAN_MODEL = "SIRIS-Lab/affilgood-span"
NER_MODEL = "SIRIS-Lab/affilgood-ner"

# Languages supported by the lingua detector.
SUPPORTED_LANGUAGES = (
    "ENGLISH",
    "SPANISH",
    "FRENCH",
    "GERMAN",
    "CHINESE",
    "JAPANESE",
    "RUSSIAN",
)

# Department/organization type words and abbreviations.
TYPE_WORDS: frozenset[str] = frozenset({
    "department", "dept", "dpt",
    "school", "sch",
    "college", "coll", "col",
    "faculty", "fac",
    "institute", "inst", "institution",
    "center", "centre", "ctr",
    "division", "div",
    "program", "prog", "programme",
    "laboratory", "laboratories", "lab",
    "academy", "acad",
    "university", "univ",
    "hospital", "hosp",
    "administration", "admin",
    "association", "assoc",
    "unit", "office",
    "section", "sect", "sec",
    "branch",
    "group", "grp",
})

TYPE_WORDS_ABBR: tuple[str, ...] = (
    "dept", "dpt", "sch", "coll", "fac", "inst", "ctr",
    "div", "prog", "lab", "acad", "univ", "admin", "assoc",
    "sect", "sec", "grp",
)

STOP_WORDS: tuple[str, ...] = (
    "of", "the", "and", "for", "in", "on", "at", "to", "amp",
)

# Cue tokens used by the deduplication pre-processing (see text_utils.split_prefix_suffix).
DEPARTMENT_CUE_TOKENS: tuple[str, ...] = (
    "school", "department", "college", "faculty",
    "center", "centre", "institute", "laboratory", "laboratories",
    "lab", "division", "program", "programme", "office", "unit",
)

PREFIX_EXCLUSION_TOKENS: frozenset[str] = frozenset({
    "school", "department", "dept", "college", "faculty",
    "center", "centre", "institute", "laboratory", "laboratories",
    "lab", "program", "programme", "office", "unit",
    "phd", "msc", "ms", "mba", "ba", "bs", "bsc", "jd", "llm",
})

# Lexical hierarchy ranks used when scoring parent/child candidate edges.
# Lower rank = higher level in the hierarchy.
LEXICAL_RANKS: dict[int, frozenset[str]] = {
    1: frozenset({"university", "univ"}),
    2: frozenset({"faculty", "fac"}),
    3: frozenset({"school", "sch", "college", "coll", "col"}),
    4: frozenset({
        "department", "dept", "center", "centre", "ctr",
        "laboratory", "laboratories", "lab",
        "institute", "institution", "inst",
    }),
}

INSTITUTION_NODE_PREFIX = "__INST__"
ROOT_LEXICAL_RANK = 0
