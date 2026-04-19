-- Source query for `data/raw_aff_to_institutions.csv`.
--
-- For every raw affiliation string, emits the list of distinct institution ids
-- it has been linked to across OpenAlex works, joined into a single
-- semicolon-separated string. Consumed by `pipeline/03_process_ner_output.py`,
-- which needs to know the institution candidates for each string in order to
-- attribute SUB entities to a single institution when possible.
--
-- Expected output columns: raw_affiliation_string_id, institution_count,
-- institution_ids_str.
WITH unique_pairs AS (
    SELECT DISTINCT
        wa.raw_affiliation_string_id,
        wai.institution_id
    FROM work_affiliation AS wa
    INNER JOIN work_affiliation_institution AS wai
        ON  wa.work_id        = wai.work_id
        AND wa.affiliation_seq = wai.affiliation_seq
    WHERE wa.raw_affiliation_string_id IS NOT NULL
)
SELECT
    raw_affiliation_string_id,
    COUNT(DISTINCT institution_id) AS institution_count,
    STRING_AGG(CAST(institution_id AS VARCHAR(MAX)), ';')
        WITHIN GROUP (ORDER BY institution_id) AS institution_ids_str
FROM unique_pairs
GROUP BY raw_affiliation_string_id
ORDER BY raw_affiliation_string_id;
