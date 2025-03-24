import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# Title
st.title("🔋 Residual Load Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.drop('dataset_id', axis=1, inplace=True)
    return df

df = load_data()

st.sidebar.header("📊 Dashboard Controls")

# Sidebar filters
date_range = st.sidebar.date_input("Select a date range", [df.index.min(), df.index.max()])
if len(date_range) == 2:
    df = df[(df.index.date >= date_range[0]) & (df.index.date <= date_range[1])]

# Model selection
model_choice = st.sidebar.selectbox("🤖 Select a model:", [
    "XGBoost", "KNN", "Linear Regression"
])

# Load selected model
model_map = {
    "XGBoost": "xgboost_model.pkl",
    "KNN": "knn_model.pkl",
    "Linear Regression": "linear_regression_model.pkl"
}

# Time series plot
st.subheader("🔌 Energy Consumption Overview")
st.line_chart(df[['load', 'residual_load']])

# Correlation heatmap
st.subheader("📈 Correlation Matrix")
corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Histogram of residual load
st.subheader("📊 Residual Load Distribution")
fig2 = px.histogram(df, x='residual_load', nbins=50, title="Distribution of Residual Load")
st.plotly_chart(fig2)

# Model prediction
st.subheader(f"📡 Prediction - {model_choice}")
features = df[['Gb(i)', 'Gd(i)', 'H_sun', 'T2m', 'WS10m']]

if st.button(f"Run prediction with {model_choice}"):
    try:
        model = joblib.load(model_map[model_choice])
        predictions = model.predict(features)
        df['Predicted_Residual_Load'] = predictions
        st.line_chart(df[['residual_load', 'Predicted_Residual_Load']])
        rmse = np.sqrt(np.mean((df['residual_load'] - predictions)**2))
        st.success(f"RMSE for {model_choice} on the selected data: {rmse:.2f}")
    except FileNotFoundError:
        st.error(f"⚠️ Model file for {model_choice} not found. Make sure it is exported from your notebook.")

# Peak detection
st.subheader("🚨 Peak Consumption Detection")
thresh = st.slider("Set threshold for residual load peaks", 
                   min_value=0.0, 
                   max_value=float(df['residual_load'].max()), 
                   value=float(df['residual_load'].quantile(0.95)))
peaks = df[df['residual_load'] >= thresh]
fig3 = px.scatter(peaks, x=peaks.index, y='residual_load', color='T2m', title="Detected Residual Load Peaks")
st.plotly_chart(fig3)

# Footer
st.markdown("---")
st.markdown("ESILV Project – Energy Consumption Optimization")
