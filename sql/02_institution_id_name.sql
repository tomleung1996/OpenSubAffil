-- Source query for `data/institution_id_name.csv`.
--
-- Pulls the canonical English display name for every institution in OpenAlex.
-- Used by `pipeline/03_process_ner_output.py` to match ORG entities against
-- candidate institution names, by `pipeline/06_build_hierarchy.py` to label
-- the virtual root node of each institution, and by
-- `pipeline/07_final_output.py` when assembling the final release tables.
--
-- Expected output columns: institution_id, institution_name.
SELECT
    institution_id,
    institution_name
FROM institution
ORDER BY institution_id;
