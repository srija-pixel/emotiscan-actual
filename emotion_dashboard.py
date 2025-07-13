import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from twilio.rest import Client
import datetime


# Set page config
st.set_page_config(page_title="EmotiScan Dashboard", layout="centered")

# Header
st.title("🧠 EmotiScan Emotion Dashboard")
st.caption("Real-time emotion tracker for non-verbal individuals")

# Load emotion data
try:
    df = pd.read_csv("emotion_log.csv", parse_dates=["Timestamp"])
    st.success("Emotion log loaded successfully!")
except:
    st.error("⚠️ Could not load emotion_log.csv. Please run the webcam detection first.")
    st.stop()

# Date filter
st.sidebar.header("📅 Filter")
start = st.sidebar.date_input("Start Date", df["Timestamp"].min().date())
end = st.sidebar.date_input("End Date", df["Timestamp"].max().date())
mask = (df["Timestamp"].dt.date >= start) & (df["Timestamp"].dt.date <= end)
filtered_df = df.loc[mask]

# Show DataFrame
if st.checkbox("Show Emotion Log Table"):
    st.write(filtered_df)

# Emotion Counts
emotion_counts = filtered_df["Emotion"].value_counts()
st.subheader("📊 Emotion Distribution")
st.bar_chart(emotion_counts)

# Timeline
st.subheader("📈 Emotion Timeline")
timeline = filtered_df.groupby(pd.Grouper(key="Timestamp", freq="5min"))["Emotion"].agg(lambda x: x.mode()[0])
st.line_chart(timeline.value_counts())

# Latest emotion
st.subheader("📍 Latest Detected Emotion")
latest = filtered_df.iloc[-1]
st.metric("Emotion", latest["Emotion"], delta=None)
st.caption(f"Detected at {latest['Timestamp']}")
