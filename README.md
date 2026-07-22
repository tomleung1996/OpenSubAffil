# OpenSubAffil

Reference implementation for the paper *OpenSubAffil: A large-scale dataset of sub-institutional name disambiguation and hierarchical structures from OpenAlex*. The released dataset is available at Zenodo: [https://doi.org/10.5281/zenodo.19602783](https://doi.org/10.5281/zenodo.19602783).

Given the OpenAlex raw affiliation strings, the pipeline extracts department-level entities (schools, faculties, centers, labs, ...), deduplicates them within every parent institution, and arranges them into a parent/child hierarchy. The final deliverables are the `opensubaffil_*.csv` tables written to `data/final_output/`.

## Layout

```
.
├── data/                              # All inputs / intermediates / outputs live here
├── sql/                               # OpenAlex extraction queries (SQL Server)
│   ├── 01_raw_affiliation_strings.sql
│   ├── 02_institution_id_name.sql
│   ├── 03_raw_aff_to_institutions.sql
│   └── 04_education_institutions.sql
├── config.py                          # Shared constants and default paths
├── text_utils.py                      # String cleaning / normalisation helpers
├── 01_detect_language.py              # Lingua-based language tagging
├── 02_run_ner.py                      # Two-stage NER (span + entity) on GPU
├── 03a_generate_abbreviation_lookup.py # Optional Gemini abbreviation expansion
├── 03_process_ner_output.py           # NER JSONL → flat CSV, org attribution
├── 04_deduplicate_departments.py      # Per-institution canonical clustering
├── 05_merge_canonical.py              # Attach canonical names to every NER row
├── 06_build_hierarchy.py              # Build parent/child hierarchy per institution
└── 07_final_output.py                 # Assemble the release tables
```

All scripts resolve default paths through `config.py`, which points inside `data/`. Override any of them with the `--input` / `--output` flags exposed by each script.

## Dependencies

Python 3.10+ with `pandas`, `numpy`, `tqdm`, `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `rapidfuzz`, `lingua-language-detector`, `orjson`. Regenerating the abbreviation lookup additionally requires `google-genai`. Step 02 requires CUDA; it fans out across all visible devices via `torch.nn.DataParallel`.

## Inputs

Before running anything, populate `data/` with the CSVs produced by the queries in `sql/`. They target a local OpenAlex mirror (SQL Server) and each one already ends with an `ORDER BY`, so the export is deterministic:

- `sql/01_raw_affiliation_strings.sql` → `data/raw_affiliation_string.csv`
- `sql/02_institution_id_name.sql` → `data/institution_id_name.csv`
- `sql/03_raw_aff_to_institutions.sql` → `data/raw_aff_to_institutions.csv`
- `sql/04_education_institutions.sql` → `data/education_institutions.csv`

Step 03 additionally expects `data/raw_department_str_abbr_full.csv`, a two-column (`abbr_name`, `full_name`) lookup. The lookup used for the paper should be retained for exact reproduction because LLM outputs may vary. Alternatively, it can be regenerated from the step-02 NER output with `03a_generate_abbreviation_lookup.py`, which includes the original prompt and uses the Gemini Batch API. Set `GEMINI_API_KEY` before running the script.

## Running

```bash
python 01_detect_language.py
python 02_run_ner.py
# Optional: regenerate data/raw_department_str_abbr_full.csv
python 03a_generate_abbreviation_lookup.py
python 03_process_ner_output.py
python 04_deduplicate_departments.py
python 05_merge_canonical.py
python 06_build_hierarchy.py
python 07_final_output.py
```

Useful debugging flags:

- `python 02_run_ner.py --max-rows 1000` — quick end-to-end smoke test.
- `python 04_deduplicate_departments.py --limit-institutions 10` or `--institution-id 37461747` — restrict clustering to a few institutions.
- `python 06_build_hierarchy.py --top-coverage 0.9` — change the long-tail coverage cutoff.

## Outputs

Step 07 writes the five release tables to `data/final_output/`:

| File | Columns | Description |
|------|---------|-------------|
| `opensubaffil_raw_affiliation_strings.csv` | `raw_affiliation_string_id, raw_affiliation_string` | Every raw string referenced by the mappings table. |
| `opensubaffil_institutions.csv` | `institution_id, institution_name` | Parent institutions covered by the release. |
| `opensubaffil_sub_institutions.csv` | `sub_institution_id, institution_id, canonical_name` | One row per canonical department within a parent institution. |
| `opensubaffil_mappings.csv` | `raw_affiliation_string_id, institution_id, sub_institution_id, raw_sub_institution_name` | Links each raw affiliation string to a canonical sub-institution. |
| `opensubaffil_hierarchy.csv` | `institution_id, parent_sub_institution_id, child_sub_institution_id` | Parent/child edges within each institution; `parent_sub_institution_id` is empty when the parent is the institution itself (root edge). |
