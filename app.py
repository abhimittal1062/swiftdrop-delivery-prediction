
"""SwiftDrop delivery-time prediction app."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort

from src.preprocessing import TARGET, prepare_inference_frame


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"

XGBOOST_PIPELINE_PATH = MODEL_DIR / "xgboost_pipeline.joblib"
TABNET_PREPROCESSOR_PATH = MODEL_DIR / "tabnet_preprocessor.joblib"
TABNET_ONNX_PATH = MODEL_DIR / "tabnet.onnx"


st.set_page_config(
    page_title="SwiftDrop | Delivery Prediction",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 SwiftDrop Delivery Prediction")
st.caption("Estimate food delivery time using trained machine-learning models.")


@st.cache_resource
def load_xgboost_pipeline():
    if not XGBOOST_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost pipeline not found: {XGBOOST_PIPELINE_PATH}"
        )
    return joblib.load(XGBOOST_PIPELINE_PATH)


@st.cache_resource
def load_tabnet_resources():
    if not TABNET_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"TabNet preprocessor not found: {TABNET_PREPROCESSOR_PATH}"
        )
    if not TABNET_ONNX_PATH.exists():
        raise FileNotFoundError(f"TabNet ONNX model not found: {TABNET_ONNX_PATH}")

    preprocessor = joblib.load(TABNET_PREPROCESSOR_PATH)
    session = ort.InferenceSession(
        str(TABNET_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    return preprocessor, session


def make_raw_input(
    age,
    ratings,
    restaurant_lat,
    restaurant_lon,
    delivery_lat,
    delivery_lon,
    vehicle_condition,
    multiple_deliveries,
    distance_km,
    order_date,
    order_time,
    weather,
    traffic,
    order_type,
    vehicle_type,
    festival,
    city,
):
    # Use the original source column names expected by preprocessing.py.
    return pd.DataFrame([{
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": ratings,
        "Restaurant_latitude": restaurant_lat,
        "Restaurant_longitude": restaurant_lon,
        "Delivery_location_latitude": delivery_lat,
        "Delivery_location_longitude": delivery_lon,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "distance_km": distance_km,
        "Order_Date": order_date,
        "Time_Orderd": order_time,
        "Weather_conditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "Festival": festival,
        "City": city,
    }])


def prepare_features(raw_df, preprocessor):
    frame = prepare_inference_frame(raw_df)

    # Keep the same feature columns and ordering used during training.
    expected_columns = list(preprocessor.feature_names_in_)
    frame = frame.reindex(columns=expected_columns)

    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed, dtype=np.float32)

def predict_xgboost(raw_df):
    pipeline = load_xgboost_pipeline()

    # Generate the same engineered features used during training.
    frame = prepare_inference_frame(raw_df)

    # Match the training feature schema and column order.
    expected_columns = list(
        pipeline.named_steps["preprocessor"].feature_names_in_
    )
    frame = frame.reindex(columns=expected_columns)

    return float(pipeline.predict(frame)[0])


def predict_tabnet(raw_df):
    preprocessor, session = load_tabnet_resources()
    features = prepare_features(raw_df, preprocessor)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run(
        [output_name],
        {input_name: features},
    )[0]
    return float(np.asarray(result).reshape(-1)[0])


with st.sidebar:
    st.header("Model settings")
    selected_model = st.selectbox(
        "Prediction model",
        ["XGBoost (recommended)", "TabNet"],
    )
    compare_models = st.checkbox("Compare both models", value=True)

st.subheader("Delivery details")
st.write("Enter the order and delivery information below.")

with st.form("delivery_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Delivery partner age",
            min_value=18,
            max_value=60,
            value=30,
        )
        ratings = st.number_input(
            "Delivery partner rating",
            min_value=1.0,
            max_value=5.0,
            value=4.6,
            step=0.1,
        )
        vehicle_condition = st.selectbox(
            "Vehicle condition",
            [0, 1, 2, 3],
            index=1,
        )
        multiple_deliveries = st.selectbox(
            "Multiple deliveries",
            [0, 1, 2, 3],
            index=1,
        )
        vehicle_type = st.selectbox(
            "Vehicle type",
            ["motorcycle", "scooter", "electric_scooter", "bicycle"],
        )

    with col2:
        restaurant_lat = st.number_input(
            "Restaurant latitude",
            min_value=8.0,
            max_value=33.0,
            value=12.9716,
            format="%.6f",
        )
        restaurant_lon = st.number_input(
            "Restaurant longitude",
            min_value=68.0,
            max_value=90.0,
            value=77.5946,
            format="%.6f",
        )
        delivery_lat = st.number_input(
            "Delivery latitude",
            min_value=8.0,
            max_value=33.0,
            value=12.9352,
            format="%.6f",
        )
        delivery_lon = st.number_input(
            "Delivery longitude",
            min_value=68.0,
            max_value=90.0,
            value=77.6245,
            format="%.6f",
        )
        distance_km = st.number_input(
            "Delivery distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )

    with col3:
        order_date = st.date_input("Order date")
        order_time = st.text_input(
            "Order time (HH:MM)",
            value="18:30",
            help="Use 24-hour time, for example 18:30.",
        )
        weather = st.selectbox(
            "Weather",
            ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"],
        )
        traffic = st.selectbox(
            "Road traffic",
            ["Low", "Medium", "High", "Jam"],
            index=2,
        )
        order_type = st.selectbox(
            "Order type",
            ["Snack", "Meal", "Drinks", "Buffet"],
        )
        festival = st.selectbox("Festival", ["No", "Yes"])
        city = st.selectbox(
            "City type",
            ["Metropolitian", "Urban", "Semi-Urban"],
        )

    submitted = st.form_submit_button(
        "Predict delivery time",
        type="primary",
        use_container_width=True,
    )


if submitted:
    try:
        # The preprocessing code expects DD-MM-YYYY.
        raw_df = make_raw_input(
            age=age,
            ratings=ratings,
            restaurant_lat=restaurant_lat,
            restaurant_lon=restaurant_lon,
            delivery_lat=delivery_lat,
            delivery_lon=delivery_lon,
            vehicle_condition=vehicle_condition,
            multiple_deliveries=multiple_deliveries,
            distance_km=distance_km,
            order_date=order_date.strftime("%d-%m-%Y"),
            order_time=order_time.strip(),
            weather=weather,
            traffic=traffic,
            order_type=order_type,
            vehicle_type=vehicle_type,
            festival=festival,
            city=city,
        )

        if not order_time.strip():
            st.error("Please enter an order time, such as 18:30.")
            st.stop()

        with st.spinner("Predicting delivery time..."):
            if selected_model == "XGBoost (recommended)":
                prediction = predict_xgboost(raw_df)
            else:
                prediction = predict_tabnet(raw_df)

            comparison = {}
            if compare_models:
                comparison["XGBoost"] = predict_xgboost(raw_df)
                comparison["TabNet"] = predict_tabnet(raw_df)

        st.success("Prediction completed.")

        st.metric(
            label=f"Estimated delivery time ({selected_model})",
            value=f"{prediction:.1f} minutes",
        )

        if comparison:
            st.subheader("Model comparison")
            c1, c2 = st.columns(2)
            c1.metric("XGBoost prediction", f"{comparison['XGBoost']:.1f} min")
            c2.metric("TabNet prediction", f"{comparison['TabNet']:.1f} min")
            st.caption(
                "These are model estimates, not guaranteed delivery times."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the model files exist and the input values match "
            "the dataset's expected format."
        )
