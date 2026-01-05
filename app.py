import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")
st.title("🛍️ Mall Customer Segmentation (K-Means)")

# File names
SCALER_FILE = "scaler.pkl"
MODEL_FILE = "kmeans_model.pkl"
DATA_FILE = "Mall_Customers.csv"

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

# Train & save model
def train_and_save_model():
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)

    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(kmeans, MODEL_FILE)

    return scaler, kmeans

# Load or train model
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.info("Training model for the first time...")
    scaler, kmeans = train_and_save_model()
else:
    scaler = joblib.load(SCALER_FILE)
    kmeans = joblib.load(MODEL_FILE)

# 🔥 ALWAYS recompute clusters for visualization
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = scaler.transform(X)
df['Cluster'] = kmeans.predict(X_scaled)

# Dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Cluster visualization
st.subheader("📈 Customer Segments")

plt.figure(figsize=(8, 6))
for cluster in range(5):
    plt.scatter(
        df[df['Cluster'] == cluster]['Annual Income (k$)'],
        df[df['Cluster'] == cluster]['Spending Score (1-100)'],
        label=f'Cluster {cluster}'
    )

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.legend()
st.pyplot(plt)

# Prediction section
st.subheader("🔍 Predict Customer Segment")

income = st.number_input("Annual Income (k$)", min_value=0.0, max_value=200.0, value=50.0)
score = st.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0)

if st.button("Predict Cluster"):
    input_data = [[income, score]]
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"🧠 This customer belongs to **Cluster {cluster}**")
