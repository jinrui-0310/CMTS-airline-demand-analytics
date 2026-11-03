-- ============================================================
-- Step 1: Validate Route × Quarter uniqueness
-- ============================================================

SELECT
    Year,
    Quarter,
    OriginCityMarketID,
    DestCityMarketID,
    route_id,
    COUNT(*) AS n_records
FROM route_quarter
GROUP BY
    Year,
    Quarter,
    OriginCityMarketID,
    DestCityMarketID,
    route_id
HAVING COUNT(*) > 1
LIMIT 10;

-- ============================================================
-- Step 2: Route History Length Distribution
-- ============================================================

WITH route_history AS (
    SELECT
        OriginCityMarketID,
        DestCityMarketID,
        route_id,

        COUNT(*) AS observed_quarters

    FROM route_quarter

    GROUP BY
        OriginCityMarketID,
        DestCityMarketID,
        route_id
)

SELECT
    CASE
        WHEN observed_quarters < 4 THEN '01: <4 quarters'
        WHEN observed_quarters < 8 THEN '02: 4-7 quarters'
        WHEN observed_quarters < 16 THEN '03: 8-15 quarters'
        WHEN observed_quarters < 32 THEN '04: 16-31 quarters'
        ELSE '05: 32+ quarters'
    END AS history_group,

    COUNT(*) AS n_routes,

    ROUND(AVG(observed_quarters), 2) AS avg_observed_quarters

FROM route_history

GROUP BY history_group

ORDER BY history_group;

-- ============================================================
-- Step 3: Route Quarter Continuity
-- ============================================================

WITH ordered_routes AS (
    SELECT
        OriginCityMarketID,
        DestCityMarketID,
        route_id,

        Year,
        Quarter,

        Year * 4 + Quarter AS quarter_index,

        LAG(Year * 4 + Quarter) OVER (
            PARTITION BY
                OriginCityMarketID,
                DestCityMarketID,
                route_id
            ORDER BY Year, Quarter
        ) AS previous_quarter_index

    FROM route_quarter
)

SELECT
    COUNT(*) AS total_observations,

    SUM(
        CASE
            WHEN previous_quarter_index IS NULL THEN 1
            ELSE 0
        END
    ) AS first_observations,

    SUM(
        CASE
            WHEN previous_quarter_index IS NOT NULL
             AND quarter_index - previous_quarter_index = 1
            THEN 1
            ELSE 0
        END
    ) AS consecutive_observations,

    SUM(
        CASE
            WHEN previous_quarter_index IS NOT NULL
             AND quarter_index - previous_quarter_index > 1
            THEN 1
            ELSE 0
        END
    ) AS observations_after_gap

FROM ordered_routes;