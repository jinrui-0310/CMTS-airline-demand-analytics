-- 1. Total observations
SELECT COUNT(*) AS total_rows
FROM route_quarter;

-- 2. Time coverage
SELECT
    MIN(Year) AS first_year,
    MAX(Year) AS last_year,
    MIN(Quarter) AS min_quarter,
    MAX(Quarter) AS max_quarter
FROM route_quarter;

-- 3. Missing values
SELECT
    SUM(CASE WHEN HHI_TK IS NULL THEN 1 ELSE 0 END)
        AS missing_hhi,
    SUM(CASE WHEN passengers IS NULL THEN 1 ELSE 0 END)
        AS missing_passengers,
    SUM(CASE WHEN avg_fare IS NULL THEN 1 ELSE 0 END)
        AS missing_fare
FROM route_quarter;

-- 4. Invalid numeric values
SELECT
    SUM(CASE WHEN passengers < 0 THEN 1 ELSE 0 END)
        AS negative_passengers,
    SUM(CASE WHEN avg_fare <= 0 THEN 1 ELSE 0 END)
        AS nonpositive_fares,
    SUM(CASE WHEN route_distance <= 0 THEN 1 ELSE 0 END)
        AS nonpositive_distances
FROM route_quarter;

-- 5. Observations by year
SELECT
    Year,
    COUNT(*) AS observations,
    SUM(passengers) AS total_passengers
FROM route_quarter
GROUP BY Year
ORDER BY Year;

-- 6. Quarterly data coverage
SELECT
    Year,
    Quarter,
    COUNT(*) AS observations
FROM route_quarter
GROUP BY Year, Quarter
ORDER BY Year, Quarter;

-- 7. HHI missing rate by year
SELECT
    Year,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN HHI_TK IS NULL THEN 1 ELSE 0 END)
        AS missing_hhi,
    ROUND(
        100.0 * SUM(CASE WHEN HHI_TK IS NULL THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS missing_hhi_pct
FROM route_quarter
GROUP BY Year
ORDER BY Year;

-- 8. Verify quarterly coverage
SELECT
    Year,
    GROUP_CONCAT(DISTINCT Quarter) AS quarters,
    COUNT(DISTINCT Quarter) AS quarter_count,
    COUNT(*) AS observations
FROM route_quarter
GROUP BY Year
ORDER BY Year;

-- 9. Check route panel ID uniqueness
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT route_panel_id) AS unique_ids,
    COUNT(*) - COUNT(DISTINCT route_panel_id) AS duplicate_or_null_ids,
    SUM(CASE WHEN route_panel_id IS NULL THEN 1 ELSE 0 END)
        AS null_ids
FROM route_quarter;

-- 10. Check Route x Quarter uniqueness
SELECT
    route_panel_id,
    Year,
    Quarter,
    COUNT(*) AS occurrences
FROM route_quarter
GROUP BY
    route_panel_id,
    Year,
    Quarter
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20;