-- Source query for `data/raw_affiliation_string.csv`.
--
-- Lists every raw affiliation string in OpenAlex that is linked to at least
-- one institution, together with how many works mention it and how many
-- distinct institutions it is associated with. The downstream pipeline reads
-- the resulting CSV in `pipeline/01_detect_language.py` and in
-- `pipeline/07_final_output.py`.
--
-- Expected output columns: raw_affiliation_string_id, institution_count,
-- frequency, raw_affiliation_string.
WITH raw_aff_inst_count AS (
    SELECT
        wa.raw_affiliation_string_id,
        COUNT(DISTINCT wai.institution_id) AS institution_count,
        COUNT(DISTINCT wa.work_id)         AS frequency
    FROM work_affiliation_institution AS wai
    INNER JOIN work_affiliation AS wa
        ON  wai.work_id        = wa.work_id
        AND wai.affiliation_seq = wa.affiliation_seq
    GROUP BY wa.raw_affiliation_string_id
)
SELECT
    raic.raw_affiliation_string_id,
    raic.institution_count,
    raic.frequency,
    ras.raw_affiliation_string
FROM raw_aff_inst_count AS raic
INNER JOIN raw_affiliation_string AS ras
    ON raic.raw_affiliation_string_id = ras.raw_affiliation_string_id
ORDER BY raic.raw_affiliation_string_id;
