import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Hotel Cancellation Risk", layout="centered")
st.title("Hotel Reservation Cancellation Risk")
st.caption("Educational prototype; do not use as the sole basis for booking decisions.")

root = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    return pd.read_csv(root / "Hotel Reservations.csv")

@st.cache_resource
def load_model():
    return joblib.load(root / "hotel_cancellation_model.joblib")

df = load_data()
model = load_model()
features = df.drop(columns=["Booking_ID", "booking_status"]).columns

st.subheader("Reservation details")
values = {}
for col in features:
    if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
        choices = sorted(df[col].dropna().unique().tolist())
        values[col] = st.selectbox(col.replace("_", " ").title(), choices, key=f"input_{col}")
    else:
        median = df[col].median()
        if pd.api.types.is_integer_dtype(df[col]):
            values[col] = st.number_input(
                col.replace("_", " ").title(),
                min_value=0,
                value=max(0, int(median)),
                step=1,
                key=f"input_{col}",
            )
        else:
            values[col] = st.number_input(
                col.replace("_", " ").title(),
                min_value=0.0,
                value=max(0.0, float(median)),
                step=1.0,
                format="%.2f",
                key=f"input_{col}",
            )

# No st.form: every number change automatically reruns the prediction.
X = pd.DataFrame([values])
X["total_nights"] = X["no_of_weekend_nights"] + X["no_of_week_nights"]
try:
    p = float(model.predict_proba(X)[0, 1])
    st.subheader("Prediction result")
    st.metric("Estimated cancellation probability", f"{p:.1%}")
    st.progress(min(1.0, max(0.0, p)))
    st.info("Use for human review and outreach prioritization only. Risk is not certainty.")
except Exception as exc:
    st.error("Prediction could not be calculated. Please check the model and feature columns.")
    st.exception(exc)
