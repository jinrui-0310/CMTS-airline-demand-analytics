-- Build route-quarter lookup table.
-- Quarter index ensures exact calendar-quarter matching.

CREATE OR REPLACE TABLE route_time_lookup AS
SELECT
    OriginCityMarketID,
    DestCityMarketID,
    route_id,
    Year,
    Quarter,
    Year * 4 + Quarter AS quarter_index,
    passengers,
    avg_fare,
    HHI_TK,
    n_carriers_TK,
    route_distance,
    nonstop_distance
FROM route_quarter;
