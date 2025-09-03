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

from incident_mapping import render_incident_mapping

# ----- CONFIG -----
st.set_page_config(
    page_title="Incident Management Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- DATA LOADING -----
@st.cache_data
def load_data():
    file_path = "text data/ndis_incident_1000.csv"
    url = "https://raw.githubusercontent.com/darolin8/NDIS_dashboard/main/text%20data/ndis_incident_1000.csv"
    # Try local file first
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=["incident_date", "notification_date"])
    else:
        try:
            df = pd.read_csv(url)
        except Exception as e:
            st.error(f"Could not load data from either local file or URL: {e}")
            st.stop()
        # Try to parse dates if present
        if "incident_date" in df.columns:
            df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
        if "notification_date" in df.columns:
            df["notification_date"] = pd.to_datetime(df["notification_date"], errors="coerce")
    # Drop rows with missing incident_date
    if "incident_date" in df.columns:
        df = df.dropna(subset=["incident_date"])
    # Add weekday column if missing
    if "incident_weekday" not in df.columns and "incident_date" in df.columns:
        df["incident_weekday"] = df["incident_date"].dt.day_name()
    return df

# ----- MAIN DASHBOARD -----
def main():
    st.title("🏥 Incident Management Dashboard")
    st.markdown("### Comprehensive Analysis and Reporting System")

    # Load and preprocess data
    df = load_data()
    df = apply_investigation_rules(df)

    # ------ SIDEBAR NAVIGATION AND FILTERS ------
    st.sidebar.header("📊 Dashboard Navigation")
    page = st.sidebar.radio(
        "Select Dashboard Page",
        [
            "📊 Executive Summary",
            "📈 Operational Performance & Risk Analysis",
            "📋 Compliance & Investigation",
            "🤖 ML Insights",
            "🗺️ Incident Map" 
        ],
    )

    # ---- Filters ----
    st.sidebar.header("Filters")
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
        locations_with_all = ["All"] + locations
        selected_location = st.sidebar.selectbox(
            "🏢 Location",
            options=locations_with_all,
            index=0,
            help="Select specific location or 'All'"
        )
        if selected_location != "All":
            filtered_df = filtered_df[filtered_df["location"] == selected_location]
    # Severity Filter
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
    # Incident Type Filter
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
    # Reporter Type Filter
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

    # ---- Filter summary ----
    if len(filtered_df) != len(df):
        st.sidebar.success(f"Applied filters: {len(filtered_df)} of {len(df)} records")
    if st.sidebar.button("🔄 Reset All Filters"):
        st.experimental_rerun()

    # ---- Data overview ----
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

    # ------ PAGE DISPATCH ------
    if page == "📊 Executive Summary":
        display_executive_summary_section(filtered_df)
    elif page == "📈 Operational Performance & Risk Analysis":
        display_operational_performance_section(filtered_df)
    elif page == "📋 Compliance & Investigation":
        display_compliance_investigation_section(filtered_df)
    elif page == "🤖 ML Insights":
        display_ml_insights_section(filtered_df)
        elif page == "🤖 ML Insights":
    st.header("🤖 ML Insights")

    # Page-specific controls (in sidebar)
    st.sidebar.markdown("---")
    forecast_horizon = st.sidebar.slider("Forecast months", 3, 12, 6, 1, key="ml_forecast_months")
    top_n_causes = st.sidebar.slider("Top N causes (time chart)", 3, 10, 5, 1, key="ml_top_n_causes")

    # Forecasting
    incident_volume_forecasting(filtered_df, periods=forecast_horizon)

    # Location Risk
    location_risk_profiling(filtered_df)

    # Seasonal & temporal
    seasonal_temporal_patterns(filtered_df)

    # Time + causes combo
    st.markdown("### ⏰ Time vs Causes (Stacked Bars + Total Line)")
    plot_time_with_causes(filtered_df, cause_col=None, top_n=top_n_causes)

    # Carer performance scatter
    st.markdown("### 🧑‍⚕️ Carer Performance Scatter")
    plot_carer_performance_scatter(filtered_df)

    # Features for correlation & clustering & model comparison
    X, feature_names, features_df = create_comprehensive_features(filtered_df)
    if X is not None and features_df is not None:
        # Correlation
        correlation_analysis(X, feature_names, features_df)

        # Clustering
        clustering_analysis(X, features_df, feature_names)

        # Predictive models (if severity available)
        if "severity" in filtered_df.columns:
            sev_map = {"low":0, "minor":0, "medium":1, "moderate":1, "high":2, "major":2, "critical":2}
            y = (
                filtered_df["severity"]
                .astype(str).str.strip().str.lower()
                .map(sev_map)
                .fillna(0).astype(int)
            )
            predictive_models_comparison(X, y, feature_names, target_name="severity")

    elif page == "🗺️ Incident Map":
        render_incident_mapping(df, filtered_df)

if __name__ == "__main__":
    main()
