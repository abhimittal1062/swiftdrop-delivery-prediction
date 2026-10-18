# Preprocessing contract

## Time values
`parse_time_total_minutes` supports:
- `HH:MM`
- `HH:MM:SS`
- Excel fractional day values (`0.833333333` ≈ 20:00)
- hours greater than 23 (`24:05:00` parses to 1445 total minutes)

For `Time_Orderd`, we derive numeric minute-of-day plus sine/cosine cyclic features. For `Time_Order_picked`, we parse and normalize for auditing, but do not use it as a predictor because it may be unavailable at order placement.

## Missing values
The source's missing age, ratings, and order-time values are preserved as missing during model-frame preparation. Train/validation split happens before the sklearn imputer is fitted. Therefore, validation statistics are not used to learn imputation values.

A separate no-null CSV is generated for inspection/sharing. It imputes using the whole dataset and is not used for validation scoring.

## Encoding
Categorical columns (including `Weather_conditions`, `Road_traffic_density`, `Type_of_order`, `Type_of_vehicle`, `Festival`, `City`, and discrete `Vehicle_condition` / `multiple_deliveries`) use `SimpleImputer(strategy="most_frequent")` and `OneHotEncoder(handle_unknown="ignore")` inside the pipeline.

## Leakage control
- `delivery_speed` is excluded because it is derived from `Time_taken (min)`.
- `ID` and `Delivery_person_ID` are excluded.
- `Time_Order_picked` is excluded from predictors because it is not known at order placement.
