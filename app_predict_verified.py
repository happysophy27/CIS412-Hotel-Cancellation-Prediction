"""CIS 412 hotel reservation cancellation predictor.

Change booking inputs, then click Predict to calculate a fresh probability.
The optional verification panel shows the exact values passed to the model.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hotel Cancellation Risk", layout="centered")
st.title("Hotel Reservation Cancellation Risk")
st.caption("Educational prototype; do not use as the sole basis for booking decisions.")

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "hotel_cancellation_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "Hotel Reservations.csv")


try:
    df = load_data()
    model = load_model()
except (FileNotFoundError, OSError) as exc:
    st.error(f"Required project file could not be loaded: {exc}")
    st.stop()

features = df.drop(columns=["Booking_ID", "booking_status"], errors="ignore").columns

with st.form("booking_form"):
    values = {}
    for col in features:
        series = df[col]
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            choices = sorted(series.dropna().unique().tolist())
            values[col] = st.selectbox(col, choices)
        elif pd.api.types.is_integer_dtype(series):
            values[col] = st.number_input(col, value=int(series.median()), step=1)
        else:
            values[col] = st.number_input(col, value=float(series.median()), step=1.0)

    submitted = st.form_submit_button("Predict cancellation risk", type="primary")

if submitted:
    X = pd.DataFrame([values], columns=features)
    X["total_nights"] = X["no_of_weekend_nights"] + X["no_of_week_nights"]
    try:
        probabilities = model.predict_proba(X)[0]
        # sklearn classifiers expose the class order for predict_proba.
        # For the original binary model, class 1 denotes Canceled.
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        if 1 in classes:
            canceled_index = classes.index(1)
        elif "Canceled" in classes:
            canceled_index = classes.index("Canceled")
        else:
            st.error(f"Cannot identify the cancellation class in model.classes_: {classes}")
            st.stop()
        probability = float(probabilities[canceled_index])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.session_state["prediction"] = {
        "probability": probability,
        "inputs": X.copy(),
    }

if "prediction" in st.session_state:
    result = st.session_state["prediction"]
    probability = result["probability"]
    st.metric("Estimated cancellation probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    st.info("Use for human review and outreach prioritization only. Risk is not certainty.")
    st.caption("This result is from the last time you clicked Predict cancellation risk.")
    with st.expander("Verify the inputs used for this prediction"):
        st.dataframe(result["inputs"].T.rename(columns={0: "Submitted value"}), use_container_width=True)
        st.write(f"Raw model probability: {probability:.8f}")
