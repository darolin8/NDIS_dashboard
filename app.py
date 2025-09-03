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

from ml_helpers import (
    prepare_ml_features,
    compare_models,
    forecast_incident_volume,
    profile_location_risk,
    detect_seasonal_patterns,
    perform_clustering_analysis,
    plot_correlation_heatmap,
    incident_type_risk_profiling as profile_incident_type_risk  # ← No comma here!
)


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
            "🧠 Advanced ML Analytics"
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
    elif page == "🧠 Advanced ML Analytics":
        st.header("🧠 Advanced ML Analytics")
        # Add ML helpers section with subpage selection
        ml_page = st.radio("Select ML Analysis", [
            "Location Risk",
            "Incident Type Risk",
            "Seasonal Patterns",
            "Severity ML Models",
            "Anomaly Detection",
            "Clustering",
            "Correlation Matrix",
            "Incident Forecast"
        ], horizontal=True)
        # Each ML analysis section
        if ml_page == "Location Risk":
            st.subheader("Incident Risk by Location")
            risk_df, fig = profile_location_risk(filtered_df)
            st.dataframe(risk_df, use_container_width=True)
        elif ml_page == "Incident Type Risk":
            st.subheader("Incident Risk by Type")
            risk_df, fig = profile_incident_type_risk(filtered_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.dataframe(risk_df, use_container_width=True)
        elif ml_page == "Seasonal Patterns":
            st.subheader("Seasonal & Temporal Patterns")
            fig = detect_seasonal_patterns(filtered_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
        if ml_page == "Severity ML Models":
            st.subheader("Severity ML Models")
            X, y, feature_names = prepare_ml_features(filtered_df)  
            results, rows, roc_fig = compare_models(X, y, feature_names)
            if rows:
                metrics_df = pd.DataFrame(rows)
                st.dataframe(metrics_df)
            if roc_fig is not None:
                st.plotly_chart(roc_fig, use_container_width=True)
        elif ml_page == "Anomaly Detection":
            st.subheader("Anomaly Detection")
            anomaly_df, feat_names = perform_anomaly_detection(filtered_df)
            if anomaly_df is not None:
                fig = plot_anomaly_scatter(anomaly_df, 'month', 'hour', axis_labels={'month':'Month', 'hour':'Hour'})
                st.pyplot(fig)
                st.dataframe(anomaly_df.head(20), use_container_width=True)
        elif ml_page == "Clustering":
            st.subheader("Clustering Analysis")
            clustered_df, feat_names, sil_score, pca = perform_clustering_analysis(filtered_df)
            if clustered_df is not None:
                fig3d = plot_3d_clusters(clustered_df)
                st.plotly_chart(fig3d, use_container_width=True)
                cluster_info = analyze_cluster_characteristics(clustered_df)
                st.write(cluster_info)
        elif ml_page == "Correlation Matrix":
            st.subheader("Correlation Matrix")
            fig = plot_correlation_heatmap(filtered_df)
            st.pyplot(fig)
        elif ml_page == "Incident Forecast":
            st.subheader("Incident Forecasting")
            actual, forecast = forecast_incident_volume(filtered_df, periods=6)
            import plotly.graph_objs as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines+markers', name='Actual'))
            if not forecast.empty:
                fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines+markers', name='Forecast'))
            fig.update_layout(title="Incident Volume Forecast", xaxis_title="Month", yaxis_title="Incidents")
            st.plotly_chart(fig, use_container_width=True)
            st.write("Forecast Table")
            if not forecast.empty:
                forecast_df = pd.DataFrame({
                    'Month': forecast.index.strftime('%Y-%m'),
                    'Forecast': forecast.values
                })
                st.dataframe(forecast_df, use_container_width=True)

if __name__ == "__main__":
    main()
