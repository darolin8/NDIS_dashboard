import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

from dashboard_pages import (
    plot_metric,
    plot_gauge,
    plot_severity_distribution,
    plot_incident_types_bar,
    plot_location_analysis,
    plot_monthly_trends,
    plot_medical_outcomes,
    plot_incident_trends,
    plot_weekday_analysis,
    plot_time_analysis,
    plot_reportable_analysis,
    plot_reporter_type_metrics,
    plot_reporter_performance_scatter,
    apply_investigation_rules,
    plot_compliance_metrics_poly,
    plot_reporting_delay_by_date,
    plot_24h_compliance_rate_by_location,
    plot_investigation_pipeline,
    plot_serious_injury_age_severity,
    plot_contributing_factors_by_month,
)

# ---- DATA LOADING SECTION ----
@st.cache_data
def load_data():
    df = pd.read_csv("incident_data.csv", parse_dates=["incident_date", "notification_date"])
    if "incident_weekday" not in df.columns and "incident_date" in df.columns:
        df["incident_weekday"] = df["incident_date"].dt.day_name()
    return df

df = load_data()
df = apply_investigation_rules(df)

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filters")
if "incident_date" in df.columns:
    min_date, max_date = df["incident_date"].min(), df["incident_date"].max()
    date_range = st.sidebar.date_input("Incident Date Range", [min_date, max_date])
    filtered_df = df[(df["incident_date"] >= pd.to_datetime(date_range[0])) & (df["incident_date"] <= pd.to_datetime(date_range[1]))]
else:
    filtered_df = df.copy()

# ---- PAGE NAVIGATION ----
page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Executive Summary",
        "Operational Performance & Risk Analysis",
        "Compliance & Investigation",
    ],
)

# ---- EXECUTIVE SUMMARY PAGE ----
if page == "Executive Summary":
    st.title("Executive Summary")

    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plot_metric("Total Incidents", len(filtered_df), color_graph="#5B8FF9")
    with col2:
        plot_metric("High Severity", int((filtered_df['severity'].str.lower() == 'high').sum()), color_graph="#FF2B2B")
    with col3:
        plot_metric("Reportable Incidents", int(filtered_df['reportable'].sum()), color_graph="#F6BD16")
    with col4:
        plot_metric("Avg Age", filtered_df['participant_age'].mean() if 'participant_age' in filtered_df.columns else 0, suffix=" yrs", color_graph="#5AD8A6")

    st.subheader("Severity Distribution")
    plot_severity_distribution(filtered_df)

    st.subheader("Top 10 Incident Types")
    plot_incident_types_bar(filtered_df)

    st.subheader("Location Analysis")
    plot_location_analysis(filtered_df)

    st.subheader("Monthly Trends")
    plot_monthly_trends(filtered_df)

    st.subheader("Medical Outcomes")
    plot_medical_outcomes(filtered_df)

    st.subheader("Daily Incident Trends")
    plot_incident_trends(filtered_df)

    st.subheader("Incidents by Day of Week")
    plot_weekday_analysis(filtered_df)

    st.subheader("Incidents by Hour of Day")
    plot_time_analysis(filtered_df)

    st.subheader("Reportable Analysis")
    plot_reportable_analysis(filtered_df)

# ---- PAGE 2: OPERATIONAL PERFORMANCE & RISK ANALYSIS ----
elif page == "Operational Performance & Risk Analysis":
    st.title("Operational Performance & Risk Analysis")

    st.subheader("Key Metrics")
    plot_reporter_type_metrics(filtered_df)

    st.subheader("Reporter Performance Analysis")
    plot_reporter_performance_scatter(filtered_df)

# ---- PAGE 3: COMPLIANCE & INVESTIGATION ----
elif page == "Compliance & Investigation":
    st.title("Compliance & Investigation")

    st.subheader("Compliance Metrics")
    plot_compliance_metrics_poly(filtered_df)

    st.subheader("Reporting Delay by Incident Date")
    plot_reporting_delay_by_date(filtered_df)

    st.subheader("24hr Compliance Rate by Location")
    plot_24h_compliance_rate_by_location(filtered_df)

    st.subheader("Investigation Pipeline")
    plot_investigation_pipeline(filtered_df)

    st.subheader("Serious Injury: Age and Severity Pattern")
    plot_serious_injury_age_severity(filtered_df)

    st.subheader("Contributing Factors by Month-Year")
    plot_contributing_factors_by_month(filtered_df)
