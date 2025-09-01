import streamlit as st
import pandas as pd
import os

# Import dashboard page functions
from dashboard_pages import (
    display_executive_summary_section,
    display_operational_performance_section,
    display_compliance_investigation_section,
    display_ml_insights_section,
    apply_investigation_rules
)

st.set_page_config(
    page_title="Incident Management Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    file_path = "text data/ndis_incidents_1000.csv"
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found. Please upload or place it in the app directory.")
        st.stop()
    df = pd.read_csv(file_path, parse_dates=["incident_date", "notification_date"])
    if "incident_weekday" not in df.columns and "incident_date" in df.columns:
        df["incident_weekday"] = df["incident_date"].dt.day_name()
    return df

def main():
    st.title("🏥 Incident Management Dashboard")
    st.markdown("### Comprehensive Analysis and Reporting System")

    # Load and preprocess data
    df = load_data()
    df = apply_investigation_rules(df)

    # Sidebar navigation and filters
    st.sidebar.header("📊 Dashboard Navigation")
    filtered_df = df.copy()

    # Date Filter
    if "incident_date" in df.columns:
        min_date, max_date = df["incident_date"].min(), df["incident_date"].max()
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            [min_date, max_date],
            help="Filter incidents by date range"
        )
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df["incident_date"] >= pd.to_datetime(date_range[0])) &
                (filtered_df["incident_date"] <= pd.to_datetime(date_range[1]))
            ]

    # Age Filter
    if "participant_age" in df.columns:
        age_min = int(df["participant_age"].min())
        age_max = int(df["participant_age"].max())
        age_range = st.sidebar.slider(
            "👥 Age Group",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            help="Filter by participant age range"
        )
        filtered_df = filtered_df[
            (filtered_df["participant_age"] >= age_range[0]) &
            (filtered_df["participant_age"] <= age_range[1])
        ]

    # Location Filter
    if "location" in df.columns:
        locations = sorted(df["location"].dropna().unique())
        selected_locations = st.sidebar.multiselect(
            "🏢 Location",
            options=locations,
            default=locations,
            help="Select specific locations"
        )
        if selected_locations:
            filtered_df = filtered_df[filtered_df["location"].isin(selected_locations)]

    # Severity Filter
    if "severity" in df.columns:
        severities = sorted(df["severity"].dropna().unique())
        selected_severities = st.sidebar.multiselect(
            "⚠️ Severity",
            options=severities,
            default=severities,
            help="Filter by incident severity"
        )
        if selected_severities:
            filtered_df = filtered_df[filtered_df["severity"].isin(selected_severities)]

    # Incident Type Filter
    if "incident_type" in df.columns:
        incident_types = sorted(df["incident_type"].dropna().unique())
        selected_incident_types = st.sidebar.multiselect(
            "📋 Incident Type",
            options=incident_types,
            default=incident_types,
            help="Select incident types"
        )
        if selected_incident_types:
            filtered_df = filtered_df[filtered_df["incident_type"].isin(selected_incident_types)]

    # Reporter Type Filter
    if "reported_by" in df.columns:
        reporter_types = sorted(df["reported_by"].dropna().unique())
        selected_reporter_types = st.sidebar.multiselect(
            "👤 Reporter Type",
            options=reporter_types,
            default=reporter_types,
            help="Filter by who reported the incident"
        )
        if selected_reporter_types:
            filtered_df = filtered_df[filtered_df["reported_by"].isin(selected_reporter_types)]

    # Filter summary
    if len(filtered_df) != len(df):
        st.sidebar.success(f"Applied filters: {len(filtered_df)} of {len(df)} records")

    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.experimental_rerun()

    # Data overview in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Data Overview")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Filtered", len(filtered_df))
    with col2:
        st.metric("Total", len(df))
    if len(filtered_df) > 0:
        st.sidebar.metric("Date Range (Days)", (filtered_df['incident_date'].max() - filtered_df['incident_date'].min()).days)
        st.sidebar.metric("Locations", filtered_df['location'].nunique() if 'location' in filtered_df.columns else 0)
        st.sidebar.metric("Incident Types", filtered_df['incident_type'].nunique() if 'incident_type' in filtered_df.columns else 0)
        high_severity_pct = (filtered_df['severity'].str.lower() == 'high').mean() * 100 if 'severity' in filtered_df.columns else 0
        reportable_pct = filtered_df['reportable'].mean() * 100 if 'reportable' in filtered_df.columns else 0
        st.sidebar.markdown("**Quick Stats:**")
        st.sidebar.write(f"🔴 High Severity: {high_severity_pct:.1f}%")
        st.sidebar.write(f"📊 Reportable: {reportable_pct:.1f}%")

    # Page navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Select Dashboard Page",
        [
            "📊 Executive Summary",
            "📈 Operational Performance & Risk Analysis",
            "📋 Compliance & Investigation",
            "🤖 ML Insights",
        ],
    )

    # Display selected page
    if page == "📊 Executive Summary":
        display_executive_summary_section(filtered_df)
    elif page == "📈 Operational Performance & Risk Analysis":
        display_operational_performance_section(filtered_df)
    elif page == "📋 Compliance & Investigation":
        display_compliance_investigation_section(filtered_df)
    elif page == "🤖 ML Insights":
        display_ml_insights_section(filtered_df)

if __name__ == "__main__":
    main()
