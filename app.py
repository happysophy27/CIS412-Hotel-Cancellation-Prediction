import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
st.set_page_config(page_title="Hotel Cancellation Risk",layout="centered")
st.title("Hotel Reservation Cancellation Risk")
st.caption("Educational prototype; do not use as the sole basis for booking decisions.")
root=Path(__file__).parent
df=pd.read_csv(root/"Hotel Reservations.csv")
model=joblib.load(root/"hotel_cancellation_model.joblib")
features=df.drop(columns=["Booking_ID","booking_status"]).columns
with st.form("booking"):
    values={}
    for col in features:
        if df[col].dtype=="object":
            choices=sorted(df[col].dropna().unique().tolist())
            values[col]=st.selectbox(col,choices,index=0)
        else:
            v=float(df[col].median())
            if pd.api.types.is_integer_dtype(df[col]):
                values[col]=st.number_input(col,value=int(v),step=1)
            else:
                values[col]=st.number_input(col,value=v,step=1.0)
    submitted=st.form_submit_button("Predict cancellation risk")
if submitted:
    X=pd.DataFrame([values]);X["total_nights"]=X["no_of_weekend_nights"]+X["no_of_week_nights"]
    p=float(model.predict_proba(X)[0,1])
    st.metric("Estimated cancellation probability",f"{p:.1%}")
    st.progress(p)
    st.info("Use for human review and outreach prioritization only. Risk is not certainty.")
