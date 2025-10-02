# app.py
import os, sys
import streamlit as st
import pandas as pd

# ---------- PATHS ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# ---------- PAGE CONFIG (must be first Streamlit call) ----------
st.set_page_config(
    page_title="Incident Management Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- THEME (Option B: apply in app entry) ----------
try:
    from utils.theme import css, set_plotly_theme
    css()
    set_plotly_theme()
except Exception:
    pass

# ---------- PAGES & HELPERS ----------
# Use the section renderers you defined in pages/dashboard_pages.py
try:
    from pages.dashboard_pages import (
        display_executive_summary_section,
        display_operational_performance_section,
        display_compliance_investigation_section,
        display_ml_insights_section,
        apply_investigation_rules,
    )
except Exception as e:
    st.error("Failed to import from pages/dashboard_pages.py — check folder layout and __init__.py.")
    st.exception(e)
    st.stop()

# Your other modules (keep these only if they exist in your repo)
try:
    from incident_mapping import render_incident_mapping
except Exception:
    render_incident_mapping = None  # hide map tab if module missing

try:
    from utils.ndis_enhanced_prep import prepare_ndis_data, create_comprehensive_features
except Exception:
    prepare_ndis_data = lambda df: df
    def create_comprehensive_features(df): return (None, [], pd.DataFrame())

try:
    from ndis_dashboard.utils.generative import generate_summary_and_mitigations
except Exception:
    def generate_summary_and_mitigations(row, narrative=""):
        return "Summary unavailable.", ["Add generative module to enable mitigations."]

# ---------- DATA ----------
@st.cache_data
def load_data():
    file_path = os.path.join(APP_DIR, "text data", "ndis_incident_1000.csv")
    url = "https://raw.githubusercontent.com/darolin8/NDIS_dashboard/main/text%20data/ndis_incident_1000.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=["incident_date", "notification_date"])
    else:
        df = pd.read_csv(url)
        if "incident_date" in df.columns:
            df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
        if "notification_date" in df.columns:
            df["notification_date"] = pd.to_datetime(df["notification_date"], errors="coerce")

    if "incident_date" in df.columns:
        df = df.dropna(subset=["incident_date"])

    if "incident_weekday" not in df.columns and "incident_date" in df.columns:
        df["incident_weekday"] = df["incident_date"].dt.day_name()

    return df

# ---------- SMALL HELPER ----------
def render_ai_summary_section(df, page_key: str):
    st.markdown("---")
    st.subheader("🧠 AI Summary & Mitigations (beta)")
    with st.expander("Show / hide", expanded=True):
        st.caption(f"[debug] rows available: {len(df)}")
        if len(df) > 0:
            idx = st.number_input("Row index", 0, len(df)-1, 0, step=1, key=f"gen_idx_{page_key}")
            row = df.iloc[int(idx)]
            narrative = str(row["narrative"]) if "narrative" in df.columns else ""
            summary, recs = generate_summary_and_mitigations(row, narrative=narrative)
            st.markdown("**Summary**"); st.write(summary)
            st.markdown("**Mitigation Recommendations**")
            for r in recs: st.write(f"- {r}")
        else:
            st.info("No rows available to summarise.")

# ---------- MAIN ----------
def main():
    st.title("🏥 Incident Management Dashboard")
    st.markdown("### Comprehensive Analysis and Reporting System")

    # Load & prepare data
    df = load_data()
    df = apply_investigation_rules(df)
    df = prepare_ndis_data(df)  # adds severity_numeric, reportable_bin, histories, etc.

    # Keep handy for other pages
    st.session_state.df = df

    # Precompute features for ML pages (best effort)
    try:
        X_full, feature_names_full, features_df_full = create_comprehensive_features(df)
        st.session_state.features_df_full = features_df_full
        st.session_state.feature_names_full = feature_names_full
    except Exception:
        st.session_state.features_df_full = None
        st.session_state.feature_names_full = None

    # Ensure place to store trained models
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}

    # -------- Sidebar Navigation --------
    st.sidebar.header("📊 Dashboard Navigation")
    pages = [
        "📊 Executive Summary",
        "📈 Operational Performance & Risk Analysis",
        "📋 Compliance & Investigation",
        "🤖 ML Insights",
    ]
    if render_incident_mapping is not None:
        pages.append("🗺️ Incident Map")

    page = st.sidebar.radio("Select Dashboard Page", pages, index=0)

    # -------- Filters --------
    st.sidebar.header("Filters")
    filtered_df = df.copy()

    # Date
    if "incident_date" in df.columns and len(df):
        min_date, max_date = df["incident_date"].min(), df["incident_date"].max()
        date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date])
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df["incident_date"] >= pd.to_datetime(date_range[0])) &
                (filtered_df["incident_date"] <= pd.to_datetime(date_range[1]))
            ]

    # Age
    if "participant_age" in df.columns and not df["participant_age"].isna().all():
        age_min = int(df["participant_age"].min())
        age_max = int(df["participant_age"].max())
        a0, a1 = st.sidebar.slider("👥 Age Group", age_min, age_max, (age_min, age_max))
        filtered_df = filtered_df[(filtered_df["participant_age"] >= a0) & (filtered_df["participant_age"] <= a1)]

    # Location
    if "location" in df.columns:
        locs = ["All"] + sorted(df["location"].dropna().unique().tolist())
        loc_choice = st.sidebar.selectbox("🏢 Location", options=locs, index=0)
        if loc_choice != "All":
            filtered_df = filtered_df[filtered_df["location"] == loc_choice]

    # Severity
    if "severity" in df.columns:
        sevs = ["All"] + sorted(df["severity"].astype(str).dropna().unique().tolist())
        sev_choice = st.sidebar.selectbox("⚠️ Severity", options=sevs, index=0)
        if sev_choice != "All":
            filtered_df = filtered_df[filtered_df["severity"].astype(str) == sev_choice]

    # Incident type
    if "incident_type" in df.columns:
        types = ["All"] + sorted(df["incident_type"].dropna().unique().tolist())
        type_choice = st.sidebar.selectbox("📋 Incident Type", options=types, index=0)
        if type_choice != "All":
            filtered_df = filtered_df[filtered_df["incident_type"] == type_choice]

    # Carer
    if "carer_id" in df.columns:
        carers = ["All"] + sorted(df["carer_id"].astype(str).dropna().unique().tolist())
        carer_choice = st.sidebar.selectbox("👤 Carer ID", options=carers, index=0)
        if carer_choice != "All":
            filtered_df = filtered_df[filtered_df["carer_id"].astype(str) == carer_choice]

    # Shared controls
    st.sidebar.markdown("---")
    st.sidebar.slider("Forecast months", 3, 12, 6, 1, key="ml_forecast_months")
    st.sidebar.slider("Top N causes (time chart)", 3, 10, 5, 1, key="ml_top_n_causes")

    # Filter summary
    if len(filtered_df) != len(df):
        st.sidebar.success(f"Applied filters: {len(filtered_df)} of {len(df)} records")

    if st.sidebar.button("🔄 Reset All Filters"):
        st.experimental_rerun()

    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Data Overview")
    col1, col2 = st.sidebar.columns(2)
    with col1: st.metric("Filtered", len(filtered_df))
    with col2: st.metric("Total", len(df))

    if len(filtered_df) > 0 and "incident_date" in filtered_df.columns:
        st.sidebar.metric("Date Range (Days)", (filtered_df["incident_date"].max() - filtered_df["incident_date"].min()).days)
    if "location" in filtered_df.columns:
        st.sidebar.metric("Locations", filtered_df["location"].nunique())
    if "incident_type" in filtered_df.columns:
        st.sidebar.metric("Incident Types", filtered_df["incident_type"].nunique())

    high_sev_pct = (
        (filtered_df.get("severity_numeric", pd.Series([0]*len(filtered_df))) >= 3).mean() * 100
        if len(filtered_df) else 0
    )
    reportable_pct = (
        filtered_df.get("reportable_bin", pd.Series([0]*len(filtered_df))).mean() * 100
        if len(filtered_df) else 0
    )
    st.sidebar.markdown("**Quick Stats:**")
    st.sidebar.write(f"🔴 High/Critical: {high_sev_pct:.1f}%")
    st.sidebar.write(f"📊 Reportable: {reportable_pct:.1f}%")

    # Share common bits to other pages if needed
    st.session_state["APP_FILTERED_DF"] = filtered_df

    # -------- Page dispatch --------
    if page == "📊 Executive Summary":
        display_executive_summary_section(filtered_df)
    elif page == "📈 Operational Performance & Risk Analysis":
        display_operational_performance_section(filtered_df)
    elif page == "📋 Compliance & Investigation":
        display_compliance_investigation_section(filtered_df)
        render_ai_summary_section(filtered_df, page_key="comp")
    elif page == "🤖 ML Insights":
        display_ml_insights_section(filtered_df)
    elif page == "🗺️ Incident Map" and render_incident_mapping is not None:
        render_incident_mapping(df, filtered_df)
    else:
        st.info("Module for this page is not available.")

if __name__ == "__main__":
    main()
