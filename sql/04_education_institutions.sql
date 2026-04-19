-- Source query for `data/education_institutions.csv`.
--
-- Returns the set of institutions classified as "education" in OpenAlex
-- (institution_type_id = 3). `pipeline/07_final_output.py` restricts the
-- Scientific Data release to this subset so that the paper's evaluation is
-- scoped to universities and colleges.
--
-- Expected output columns: institution_id.
SELECT
    institution_id
FROM institution
WHERE institution_type_id = 3
ORDER BY institution_id;
