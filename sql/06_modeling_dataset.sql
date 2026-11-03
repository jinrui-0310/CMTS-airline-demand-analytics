
-- 1. Count eligible observations

SELECT
    COUNT(*) AS eligible_observations
FROM route_features
WHERE passengers_lag1 IS NOT NULL
  AND passengers_lag4 IS NOT NULL;


-- 2. Count eligible observations by year

SELECT
    Year,
    COUNT(*) AS eligible_observations
FROM route_features
WHERE passengers_lag1 IS NOT NULL
  AND passengers_lag4 IS NOT NULL
GROUP BY Year
ORDER BY Year;


-- 3. Inspect sample records

SELECT
    Year,
    Quarter,
    route_id,
    target_passengers,
    passengers_lag1,
    passengers_lag4,
    fare_lag1,
    hhi_lag1,
    carriers_lag1
FROM route_features
WHERE passengers_lag1 IS NOT NULL
  AND passengers_lag4 IS NOT NULL
LIMIT 10;

