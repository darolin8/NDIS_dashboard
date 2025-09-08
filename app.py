# app.py
# ---- BEGIN: robust import bootstrap (top of app.py) ----
import os, sys
import pandas as pd
import streamlit as st

# ✅ MUST be the first Streamlit call
st.set_page_config(
    page_title="Incident Management Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

UTILS_DIR = os.path.join(APP_DIR, "utils")
if os.path.isdir(UTILS_DIR) and UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

# First: load ml_helpers directly and expose any real error
try:
    import ml_helpers as ML
    # Flip to True if you want to see import paths
    DEBUG_IMPORTS = False
    if DEBUG_IMPORTS:
        st.info(f"ml_helpers loaded from: {getattr(ML, '__file__', 'unknown')}")
except Exception as e:
    st.error("Failed to import ml_helpers. Details:")
    st.exception(e)
    st.stop()

# Next: import dashboard_pages and expose any real error
try:
    from dashboard_pages import (
        display_executive_summary_section,
        display_operational_performance_section,
        display_compliance_investigation_section,
        display_ml_insights_section,
        apply_investigation_rules,      
    )
except Exception as e:
    st.error("Failed to import dashboard_pages. Details:")
    st.exception(e)
    # Optional: probe where Python looked
    import importlib.util
    spec = importlib.util.find_spec("dashboard_pages")
    st.caption(f"dashboard_pages spec: {spec}")
    st.stop()
# ---- END: robust import bootstrap ----

# ✅ Your modules
from incident_mapping import render_incident_mapping

# ✅ Prefer wrappers from ml_helpers (they safely fall back if utils is missing)
try:
    from ml_helpers import prepare_ndis_data, create_comprehensive_features
except Exception:
    # Last-resort fallback if someone removed the wrappers
    from utils.ndis_enhanced_prep import prepare_ndis_data, create_comprehensive_features  # type: ignore

# ✅ ML helper (baseline training)
from ml_helpers import predictive_models_comparison

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

# ----- OPTIONAL: ML trainer (sidebar) -----
def sidebar_ml_controls(df_for_training: pd.DataFrame):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 ML — Baselines")

    target_choice = st.sidebar.selectbox(
        "Target",
        ["High/Critical (binary)", "Reportable (binary)"],
        help="High/Critical uses df['high_severity']; Reportable uses df['reportable_bin']",
        index=0,
    )

    test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.25, 0.05)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42, step=1)

    if st.sidebar.button("Train models"):
        use_df = df_for_training.copy()

        # Decide target
        if target_choice.startswith("High/Critical"):
            if "high_severity" not in use_df.columns:
                st.sidebar.error("Column 'high_severity' not found after preparation.")
                return
            target_col = "high_severity"
        else:
            if "reportable_bin" not in use_df.columns:
                st.sidebar.error("Column 'reportable_bin' not found after preparation.")
                return
            target_col = "reportable_bin"

        try:
            models = predictive_models_comparison(
                use_df, target=target_col, test_size=float(test_size), random_state=int(seed)
            )
            st.session_state.trained_models = models

            # Quick success toast
            best_name, best_blob = max(models.items(), key=lambda kv: kv[1]["accuracy"])
            st.sidebar.success(f"Best: {best_name} • acc {best_blob['accuracy']:.2%}")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# ------ MAIN DASHBOARD ------
def main():
    st.title("🏥 Incident Management Dashboard")
    st.markdown("### Comprehensive Analysis and Reporting System")

    # Load and domain rules
    df = load_data()
    df = apply_investigation_rules(df)

    # === ML prep & feature-ready ===
    df = prepare_ndis_data(df)  # wrapper adds severity_numeric, reportable_bin, histories, location_risk, etc.

    # ✅ Global, reusable target for “meaningful” prediction
    if "high_severity" not in df.columns:
        if "severity_numeric" in df.columns:
            df["high_severity"] = (df["severity_numeric"] >= 3).astype(int)
        elif "severity" in df.columns:
            df["high_severity"] = df["severity"].astype(str).isin(["High", "Critical"]).astype(int)
        else:
            df["high_severity"] = 0

    # Keep in session for other pages
    st.session_state.df = df

    # Build features for full dataset (handy for clustering/similarity pages)
    try:
        X_full, feature_names_full, features_df_full = create_comprehensive_features(df)
        st.session_state.features_df_full = features_df_full
        st.session_state.feature_names_full = feature_names_full
    except Exception:
        st.session_state.features_df_full = None
        st.session_state.feature_names_full = None

    # Ensure a place to store trained models
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}

    # ------ SIDEBAR NAVIGATION AND FILTERS ------
    st.sidebar.header("📊 Dashboard Navigation")
    page = st.sidebar.radio(
        "Select Dashboard Page",
        [
            "📊 Executive Summary",
            "📈 Operational Performance & Risk Analysis",
            "📋 Compliance & Investigation",
            "🤖 ML Insights",
            "🗺️ Incident Map",
        ],
        index=0,
    )

    # ---- Filters ----
    st.sidebar.header("Filters")
    filtered_df = df.copy()

    # Date Filter
    if "incident_date" in df.columns and not df["incident_date"].isna().all():
        min_date, max_date = df["incident_date"].min(), df["incident_date"].max()
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            [min_date, max_date],
            help="Filter incidents by date range",
        )
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df["incident_date"] >= pd.to_datetime(date_range[0]))
                & (filtered_df["incident_date"] <= pd.to_datetime(date_range[1]))
            ]

    # Age Filter
    if "participant_age" in df.columns and df["participant_age"].notna().any():
        age_min = int(df["participant_age"].min())
        age_max = int(df["participant_age"].max())
        age_range = st.sidebar.slider(
            "👥 Age Group",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            help="Filter by participant age range",
        )
        filtered_df = filtered_df[
            (filtered_df["participant_age"] >= age_range[0])
            & (filtered_df["participant_age"] <= age_range[1])
        ]

    # Location Filter
    if "location" in df.columns:
        locations = sorted(df["location"].dropna().unique())
        locations_with_all = ["All"] + locations
        selected_location = st.sidebar.selectbox(
            "🏢 Location",
            options=locations_with_all,
            index=0,
            help="Select specific location or 'All'",
        )
        if selected_location != "All":
            filtered_df = filtered_df[filtered_df["location"] == selected_location]

    # Severity Filter
    if "severity" in df.columns:
        severities = sorted(df["severity"].astype(str).dropna().unique())
        severities_with_all = ["All"] + list(severities)
        selected_severity = st.sidebar.selectbox(
            "⚠️ Severity",
            options=severities_with_all,
            index=0,
            help="Filter by incident severity or 'All'",
        )
        if selected_severity != "All":
            filtered_df = filtered_df[filtered_df["severity"].astype(str) == selected_severity]

    # Incident Type Filter
    if "incident_type" in df.columns:
        incident_types = sorted(df["incident_type"].dropna().unique())
        incident_types_with_all = ["All"] + list(incident_types)
        selected_incident_type = st.sidebar.selectbox(
            "📋 Incident Type",
            options=incident_types_with_all,
            index=0,
            help="Select incident type or 'All'",
        )
        if selected_incident_type != "All":
            filtered_df = filtered_df[filtered_df["incident_type"] == selected_incident_type]

    # Reporter Type Filter
    if "reported_by" in df.columns:
        reporter_types = sorted(df["reported_by"].dropna().unique())
        reporter_types_with_all = ["All"] + list(reporter_types)
        selected_reporter_type = st.sidebar.selectbox(
            "👤 Reporter Type",
            options=reporter_types_with_all,
            index=0,
            help="Filter by who reported the incident or 'All'",
        )
        if selected_reporter_type != "All":
            filtered_df = filtered_df[filtered_df["reported_by"] == selected_reporter_type]

    # Page-specific controls (in sidebar)
    st.sidebar.markdown("---")
    st.sidebar.slider("Forecast months", 3, 12, 6, 1, key="ml_forecast_months")
    st.sidebar.slider("Top N causes (time chart)", 3, 10, 5, 1, key="ml_top_n_causes")

    # ---- Filter summary ----
    if len(filtered_df) != len(df):
        st.sidebar.success(f"Applied filters: {len(filtered_df)} of {len(df)} records")

    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()

    # ---- Data overview ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Data Overview")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Filtered", len(filtered_df))
    with col2:
        st.metric("Total", len(df))
    if len(filtered_df) > 0:
        st.sidebar.metric(
            "Date Range (Days)",
            (filtered_df["incident_date"].max() - filtered_df["incident_date"].min()).days
            if "incident_date" in filtered_df.columns else 0,
        )
        st.sidebar.metric("Locations", filtered_df["location"].nunique() if "location" in filtered_df.columns else 0)
        st.sidebar.metric("Incident Types", filtered_df["incident_type"].nunique() if "incident_type" in filtered_df.columns else 0)

        # Quick stats
        high_severity_pct = (filtered_df.get("high_severity", pd.Series([0]*len(filtered_df))).mean() * 100)
        reportable_pct = filtered_df.get("reportable_bin", pd.Series([0]*len(filtered_df))).mean() * 100
        st.sidebar.markdown("**Quick Stats:**")
        st.sidebar.write(f"🔴 High/Critical: {high_severity_pct:.1f}%")
        st.sidebar.write(f"📊 Reportable: {reportable_pct:.1f}%")

    # === ML: filtered features (optional convenience for pages) ===
    try:
        X_filt, feature_names_filt, features_df_filt = create_comprehensive_features(filtered_df)
        st.session_state.features_df_filtered = features_df_filt
        st.session_state.feature_names_filtered = feature_names_filt
    except Exception:
        st.session_state.features_df_filtered = None
        st.session_state.feature_names_filtered = None

    # === ML: trainer on current slice ===
    sidebar_ml_controls(filtered_df)

    # ------ PAGE DISPATCH ------
    if page == "📊 Executive Summary":
        display_executive_summary_section(filtered_df)
    elif page == "📈 Operational Performance & Risk Analysis":
        display_operational_performance_section(filtered_df)
    elif page == "📋 Compliance & Investigation":
        display_compliance_investigation_section(filtered_df)
        st.markdown("---")
        st.subheader("Enhanced Investigation Pipeline")
        group_by = st.sidebar.selectbox(
        "Group pipeline by:",
        ["carer_id", "severity", "incident_type", "location"],
        index=0
        )
        display_enhanced_investigation_pipeline(filtered_df, group_by=group_by)
    elif page == "🤖 ML Insights":
        display_ml_insights_section(filtered_df)
    elif page == "🗺️ Incident Map":
        render_incident_mapping(df, filtered_df)

if __name__ == "__main__":
    main()
