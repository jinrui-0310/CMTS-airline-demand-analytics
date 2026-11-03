CREATE OR REPLACE TABLE route_features AS

SELECT
    -- Route identification
    curr.Year,
    curr.Quarter,
    curr.OriginCityMarketID,
    curr.DestCityMarketID,
    curr.route_id,

    -- Prediction target
    curr.passengers AS target_passengers,

    -- Historical demand
    lag1.passengers AS passengers_lag1,
    lag4.passengers AS passengers_lag4,

    -- Historical market conditions
    lag1.avg_fare AS fare_lag1,
    lag1.HHI_TK AS hhi_lag1,
    lag1.n_carriers_TK AS carriers_lag1,

    -- Route characteristics
    curr.route_distance,
    curr.nonstop_distance,

    -- Time features
    curr.quarter_index

FROM route_time_lookup AS curr

LEFT JOIN route_time_lookup AS lag1
    ON curr.OriginCityMarketID = lag1.OriginCityMarketID
    AND curr.DestCityMarketID = lag1.DestCityMarketID
    AND curr.route_id = lag1.route_id
    AND lag1.quarter_index = curr.quarter_index - 1

LEFT JOIN route_time_lookup AS lag4
    ON curr.OriginCityMarketID = lag4.OriginCityMarketID
    AND curr.DestCityMarketID = lag4.DestCityMarketID
    AND curr.route_id = lag4.route_id
    AND lag4.quarter_index = curr.quarter_index - 4;