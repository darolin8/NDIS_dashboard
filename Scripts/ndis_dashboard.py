import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import json

# ML Libraries for advanced analytics
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import IsolationForest
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    st.warning("Some ML libraries are missing. Please install scikit-learn and mlxtend for full functionality.")

# Page config
st.set_page_config(
    page_title="Advanced NDIS Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        .reportview-container .main .block-container {
            padding-top: 1rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 1rem;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .metric-card {
            padding: 15px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-label {
            font-size: 16px;
            color: #666;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Helper functions
def process_data(df):
    df['Incident Date'] = pd.to_datetime(df['Incident Date'], errors='coerce')
    df['Report Date'] = pd.to_datetime(df['Report Date'], errors='coerce')
    df['Days to Report'] = (df['Report Date'] - df['Incident Date']).dt.days
    return df

def prepare_ml_features(df):
    df_ml = df.copy()
    le = LabelEncoder()
    for col in ['Category', 'Severity', 'Status']:
        if col in df_ml.columns:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    
    features = ['Days to Report', 'Category', 'Severity', 'Status']
    df_ml = df_ml[features].fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_ml)
    return scaled_features, df_ml

def predict_future_incidents(df, days=30):
    daily_counts = df.groupby(df['Incident Date'].dt.date).size()
    daily_counts = daily_counts.reindex(
        pd.date_range(df['Incident Date'].min(), df['Incident Date'].max()),
        fill_value=0
    )
    
    trend = np.polyfit(range(len(daily_counts)), daily_counts.values, 1)
    future_x = range(len(daily_counts), len(daily_counts) + days)
    future_y = np.polyval(trend, future_x)
    
    return pd.DataFrame({
        'Date': pd.date_range(df['Incident Date'].max() + timedelta(days=1), periods=days),
        'Predicted Incidents': np.maximum(future_y, 0)
    })

# Sidebar controls
st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload Incident Data (CSV)", type="csv")
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["📊 Overview", "🤖 Advanced Analytics", "🔮 Predictive Analytics"]
)

if analysis_mode == "🤖 Advanced Analytics":
    adv_choice = st.sidebar.selectbox(
        "Choose Technique",
        ["Clustering", "Association Rules", "Anomaly Detection"]
    )

prediction_days = st.sidebar.slider("Prediction Horizon (days)", 7, 90, 30)

# Main app
st.title("📊 Advanced NDIS Dashboard")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = process_data(df)
    
    # Overview
    if analysis_mode == "📊 Overview":
        st.header("📋 Incident Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Incidents", len(df))
        with col2:
            st.metric("Avg. Days to Report", f"{df['Days to Report'].mean():.1f}")
        with col3:
            st.metric("Open Incidents", len(df[df['Status'] == 'Open']))
        with col4:
            st.metric("Closed Incidents", len(df[df['Status'] == 'Closed']))
        
        fig = px.histogram(df, x='Incident Date', title='Incident Frequency Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Analytics
    elif analysis_mode == "🤖 Advanced Analytics":
        st.header("🤖 Machine Learning Insights")
        
        try:
            features, df_ml = prepare_ml_features(df)

            # ---- Clustering ----
            if adv_choice == "Clustering":
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(features)
                df['Cluster'] = clusters
                fig = px.scatter(
                    df,
                    x='Days to Report',
                    y='Severity',
                    color='Cluster',
                    title="Incident Clustering"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---- Association Rules ----
            elif adv_choice == "Association Rules":
                if {'Category', 'Severity', 'Status'}.issubset(df.columns):
                    df_rules = pd.get_dummies(df[['Category', 'Severity', 'Status']])
                    frequent = apriori(df_rules, min_support=0.2, use_colnames=True)
                    rules = association_rules(frequent, metric="lift", min_threshold=1)
                    st.subheader("Top Association Rules")
                    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(10))
                else:
                    st.warning("Not enough categorical data for Association Rules.")

            # ---- Anomaly Detection ----
            elif adv_choice == "Anomaly Detection":
                iso = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso.fit_predict(features)
                df['Anomaly'] = anomalies
                fig = px.scatter(
                    df,
                    x='Days to Report',
                    y='Severity',
                    color=df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'}),
                    title="Incident Anomaly Detection"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"ML Analysis failed: {str(e)}")
    
    # Predictive Analytics
    elif analysis_mode == "🔮 Predictive Analytics":
        st.header("🔮 Predictive Analytics")
        
        df_filtered = df.dropna(subset=['Incident Date'])
        
        if len(df_filtered) > 10:
            with st.spinner("🔮 Generating predictions..."):
                predictions = predict_future_incidents(df_filtered, prediction_days)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_filtered['Incident Date'],
                    y=df_filtered.groupby('Incident Date').size(),
                    mode='lines',
                    name='Historical'
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['Date'],
                    y=predictions['Predicted Incidents'],
                    mode='lines',
                    name='Predicted'
                ))
                fig.update_layout(title="Incident Forecast")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for prediction.")
else:
    st.info("👆 Please upload a CSV file to begin analysis.")



         
