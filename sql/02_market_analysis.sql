-- 1. Quarterly Passenger Demand Trends

SELECT
    Year,
    Quarter,
    Year || '-Q' || Quarter AS period,
    COUNT(*) AS active_route_observations,
    SUM(passengers) AS total_route_passengers,
    ROUND(AVG(passengers), 2) AS avg_passengers_per_route

FROM route_quarter
GROUP BY Year, Quarter
ORDER BY Year, Quarter;


-- 2. Quarterly Passenger-Weighted Average Fare

SELECT
    Year,
    Quarter,
    Year || '-Q' || Quarter AS period,

    ROUND(
        SUM(avg_fare * passengers)
        / NULLIF(SUM(passengers), 0),
        2
    ) AS weighted_avg_fare,

    ROUND(
        AVG(avg_fare),
        2
    ) AS unweighted_avg_fare

FROM route_quarter

GROUP BY Year, Quarter

ORDER BY Year, Quarter;


-- 3. Quarterly Market Concentration (HHI)

WITH market_quarter AS (
    SELECT
        Year,
        Quarter,
        OriginCityMarketID,
        DestCityMarketID,
        MAX(HHI_TK) AS HHI_TK
    FROM route_quarter
    GROUP BY
        Year,
        Quarter,
        OriginCityMarketID,
        DestCityMarketID
)

SELECT
    Year,
    Quarter,
    Year || '-Q' || Quarter AS period,

    COUNT(*) AS total_markets,

    COUNT(HHI_TK) AS markets_with_hhi,

    ROUND(
        100.0 * COUNT(HHI_TK) / COUNT(*),
        2
    ) AS hhi_coverage_pct,

    ROUND(AVG(HHI_TK), 4) AS avg_market_hhi

FROM market_quarter

GROUP BY Year, Quarter

ORDER BY Year, Quarter;

-- 4. Top 20 O&D Markets by Passenger Demand (2024)

SELECT
    OriginCityMarketID,
    DestCityMarketID,

    SUM(passengers) AS total_passengers,

    ROUND(
        SUM(avg_fare * passengers)
        / NULLIF(SUM(passengers), 0),
        2
    ) AS weighted_avg_fare

FROM route_quarter

WHERE Year = 2024

GROUP BY
    OriginCityMarketID,
    DestCityMarketID

ORDER BY total_passengers DESC

LIMIT 20;