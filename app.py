import streamlit as st
import pandas as pd
import os

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

    st.sidebar.header("📊 Dashboard Navigation")

    # ---- Move Dashboard Page Selector to Top of Sidebar ----
    page = st.sidebar.radio(
        "Select Dashboard Page",
        [
            "📊 Executive Summary",
            "📈 Operational Performance & Risk Analysis",
            "📋 Compliance & Investigation",
            "🤖 ML Insights",
        ],
    )

    # ---- Date Filter ----
    st.sidebar.header("Filters")
    filtered_df = df.copy()
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

    # ---- Age Filter ----
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

    # ---- Location Filter with "All" option ----
    if "location" in df.columns:
        locations = sorted(df["location"].dropna().unique())
        locations_with_all = ["All"] + locations
        selected_location = st.sidebar.selectbox(
            "🏢 Location",
            options=locations_with_all,
            index=0,
            help="Select specific location or 'All'"
        )
        if selected_location != "All":
            filtered_df = filtered_df[filtered_df["location"] == selected_location]

    # ---- Severity Filter with "All" option ----
    if "severity" in df.columns:
        severities = sorted(df["severity"].dropna().unique())
        severities_with_all = ["All"] + severities
        selected_severity = st.sidebar.selectbox(
            "⚠️ Severity",
            options=severities_with_all,
            index=0,
            help="Filter by incident severity or 'All'"
        )
        if selected_severity != "All":
            filtered_df = filtered_df[filtered_df["severity"] == selected_severity]

    # ---- Incident Type Filter with "All" option ----
    if "incident_type" in df.columns:
        incident_types = sorted(df["incident_type"].dropna().unique())
        incident_types_with_all = ["All"] + incident_types
        selected_incident_type = st.sidebar.selectbox(
            "📋 Incident Type",
            options=incident_types_with_all,
            index=0,
            help="Select incident type or 'All'"
        )
        if selected_incident_type != "All":
            filtered_df = filtered_df[filtered_df["incident_type"] == selected_incident_type]

    # ---- Reporter Type Filter with "All" option ----
    if "reported_by" in df.columns:
        reporter_types = sorted(df["reported_by"].dropna().unique())
        reporter_types_with_all = ["All"] + reporter_types
        selected_reporter_type = st.sidebar.selectbox(
            "👤 Reporter Type",
            options=reporter_types_with_all,
            index=0,
            help="Filter by who reported the incident or 'All'"
        )
        if selected_reporter_type != "All":
            filtered_df = filtered_df[filtered_df["reported_by"] == selected_reporter_type]

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
    # Already moved to top

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
