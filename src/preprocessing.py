"""Reusable data cleaning, feature engineering, and sklearn preprocessing.

Important design:
- `prepare_model_frame` cleans values and engineers features but preserves missing
  feature values as NaN until after the train/validation split.
- Imputation and one-hot encoding are fitted only on training rows by the sklearn
  Pipeline, preventing validation leakage.
- `make_fully_clean_export` creates a separate no-null CSV for inspection/sharing.
  Do NOT use that full-data-imputed export to report model validation metrics.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Time_taken (min)"
ID_COLUMNS = ("ID", "Delivery_person_ID")
LEAKAGE_COLUMNS = ("delivery_speed",)
PICKUP_COLUMNS = (
    "Time_Order_picked",
    "Time_Order_picked_parse_failed",
    "Time_Order_picked_was_missing",
)

def parse_time_total_minutes(value):
    """Parse HH:MM[:SS] or Excel fraction-of-day to total minutes.

    Examples: 0.833333333 -> about 20:00; 24:05:00 -> 1445 minutes.
    Invalid/missing values return np.nan. Hours > 23 are accepted intentionally.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return np.nan

    try:
        numeric = float(s)
        if 0 <= numeric < 1:
            return numeric * 1440.0
    except (TypeError, ValueError):
        pass

    match = re.fullmatch(r"(\d{1,3}):(\d{1,2})(?::(\d{1,2})(?:\.\d+)?)?", s)
    if not match:
        return np.nan

    hours, minutes = int(match.group(1)), int(match.group(2))
    seconds = int(match.group(3) or 0)
    if minutes >= 60 or seconds >= 60:
        return np.nan
    return hours * 60.0 + minutes + seconds / 60.0

def _normalise_text_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_tokens = {"", "na", "n/a", "null", "none", "nan"}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.str.lower().isin(missing_tokens), pd.NA)
    return df

def prepare_model_frame(
    raw_df: pd.DataFrame,
    *,
    require_target: bool = True,
) -> pd.DataFrame:
    """Clean and engineer features; preserve NaNs for train-fitted imputation.

    Args:
        raw_df: Raw delivery-order rows.
        require_target: True for supervised training/evaluation (valid target rows
            only); False for inference, where the target must not be present or used.
    """
    df = raw_df.copy()
    df.columns = df.columns.str.strip()
    df = _normalise_text_missing(df)

    if require_target and TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is missing.")

    # A supplied target must never become an inference feature.
    if not require_target and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    numeric_cols = [
        "Delivery_person_Age", "Delivery_person_Ratings",
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
        "Vehicle_condition", "multiple_deliveries", "distance_km", TARGET,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supervised training requires a valid target; never impute the target.
    if require_target:
        valid_y = df[TARGET].notna() & np.isfinite(df[TARGET]) & (df[TARGET] > 0)
        df = df.loc[valid_y].copy()

    if "Delivery_person_Age" in df:
        df.loc[~df["Delivery_person_Age"].between(18, 60), "Delivery_person_Age"] = np.nan
    if "Delivery_person_Ratings" in df:
        df.loc[~df["Delivery_person_Ratings"].between(1, 5), "Delivery_person_Ratings"] = np.nan

    # Broad regional sanity bounds; invalid coordinates are imputed later.
    bounds = {
        "Restaurant_latitude": (8, 33),
        "Delivery_location_latitude": (8, 33),
        "Restaurant_longitude": (68, 90),
        "Delivery_location_longitude": (68, 90),
    }
    for col, (lo, hi) in bounds.items():
        if col in df:
            df.loc[~df[col].between(lo, hi), col] = np.nan

    if "distance_km" in df:
        bad = (~np.isfinite(df["distance_km"])) | (df["distance_km"] <= 0)
        df.loc[bad, "distance_km"] = np.nan

    # Date features.
    if "Order_Date" in df:
        parsed_date = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
        df["order_day_of_week"] = parsed_date.dt.dayofweek.astype(float)
        df["order_month"] = parsed_date.dt.month.astype(float)
        df["order_day"] = parsed_date.dt.day.astype(float)
        df["order_is_weekend"] = parsed_date.dt.dayofweek.isin([5, 6]).astype(float)
        df["Order_Date_parse_failed"] = parsed_date.isna().astype("int8")
        df = df.drop(columns=["Order_Date"])

    # Order time: turn both normal clock strings and Excel fractions into numeric features.
    if "Time_Orderd" in df:
        original = df["Time_Orderd"]
        total = original.map(parse_time_total_minutes)
        df["Time_Orderd_was_missing"] = original.isna().astype("int8")
        df["Time_Orderd_parse_failed"] = (original.notna() & total.isna()).astype("int8")
        minute_of_day = total.mod(1440)
        df["order_time_minutes"] = minute_of_day
        # Cyclic encoding captures midnight wrap-around more naturally than raw clock labels.
        angle = 2 * np.pi * minute_of_day / 1440.0
        df["order_time_sin"] = np.sin(angle)
        df["order_time_cos"] = np.cos(angle)
        df = df.drop(columns=["Time_Orderd"])

    # Pickup time is parsed for quality/audit but is deliberately not a predictor:
    # at order placement, pickup time is not yet known.
    if "Time_Order_picked" in df:
        original = df["Time_Order_picked"]
        total = original.map(parse_time_total_minutes)
        df["Time_Order_picked_was_missing"] = original.isna().astype("int8")
        df["Time_Order_picked_parse_failed"] = (original.notna() & total.isna()).astype("int8")
        df["Time_Order_picked_normalized"] = total.map(
            lambda x: f"{int(round(x)) % 1440 // 60:02d}:{int(round(x)) % 60:02d}"
            if pd.notna(x) else np.nan
        )
        df = df.drop(columns=["Time_Order_picked"])

    # Discrete ordinal/count values are treated as categorical for a robust baseline.
    for col in ("Vehicle_condition", "multiple_deliveries"):
        if col in df:
            df[col] = df[col].map(
                lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer()
                else (str(v) if pd.notna(v) else np.nan)
            )

    # Remove identifiers and target-derived feature.
    df = df.drop(columns=[c for c in (*ID_COLUMNS, *LEAKAGE_COLUMNS) if c in df], errors="ignore")
    
    # Pickup-time information is unavailable when the order is placed.
    # Exclude both the pickup-time value and its audit flags from model features.
    pickup_columns_to_drop = [
        "Time_Order_picked",
        "Time_Order_picked_normalized",
        "Time_Order_picked_parse_failed",
        "Time_Order_picked_was_missing",
    ]

    df = df.drop(
        columns=[
            col for col in pickup_columns_to_drop
            if col in df.columns
        ]
    )

    return df.reset_index(drop=True)

def prepare_inference_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw order data for prediction without requiring a target column."""
    return prepare_model_frame(raw_df, require_target=False)


def make_fully_clean_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a no-null export for inspection; separate from model evaluation data."""
    out = frame.copy().replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if col == TARGET:
            # Target should already be valid; never fabricate target values.
            out = out.loc[out[TARGET].notna()].copy()
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            median = out[col].median()
            out[col] = out[col].fillna(median if pd.notna(median) else 0)
        else:
            modes = out[col].dropna().mode()
            fill = modes.iloc[0] if not modes.empty else "Unknown"
            out[col] = out[col].fillna(fill).astype("string").fillna(fill).astype(str)
            out[col] = out[col].replace({"<NA>": fill, "nan": fill, "None": fill, "": fill})
    return out.reset_index(drop=True)

def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build train-fitted imputation/encoding. Fit only after splitting."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
