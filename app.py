# ---- BEGIN: ultra-robust ml_helpers import (top of app.py) ----
import os, sys, importlib, importlib.util
import streamlit as st
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

UTILS_DIR = os.path.join(APP_DIR, "utils")
if os.path.isdir(UTILS_DIR) and UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

def _diagnose_module(name):
    spec = importlib.util.find_spec(name)
    st.caption(f"🔎 find_spec('{name}') → {spec}")
    st.caption(f"sys.path[0:5] → {sys.path[:5]}")
    try:
        st.caption(f"APP_DIR contents → {sorted(os.listdir(APP_DIR))[:20]}")
    except Exception:
        pass
    if os.path.isdir(UTILS_DIR):
        try:
            st.caption(f"utils/ contents → {sorted(os.listdir(UTILS_DIR))[:20]}")
        except Exception:
            pass

# 1) Try a normal import first
try:
    import ml_helpers as ML
    st.success(f"✅ ml_helpers loaded from: {getattr(ML, '__file__', 'unknown')}")
except Exception as e1:
    st.warning("Normal import failed; diagnosing & attempting direct file load…")
    _diagnose_module("ml_helpers")

    # 2) If ml_helpers.py exists beside app.py, load it directly by path
    ml_path_candidates = [
        os.path.join(APP_DIR, "ml_helpers.py"),
        os.path.join(UTILS_DIR, "ml_helpers.py"),
    ]
    ml_path = next((p for p in ml_path_candidates if os.path.isfile(p)), None)

    if ml_path:
        try:
            loader = importlib.machinery.SourceFileLoader("ml_helpers", ml_path)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            ML = importlib.util.module_from_spec(spec)
            loader.exec_module(ML)
            sys.modules["ml_helpers"] = ML
            st.success(f"✅ Loaded ml_helpers directly from file: {ml_path}")
        except Exception as e2:
            st.error("❌ Could not load ml_helpers even via file loader.")
            st.exception(e2)
            st.stop()
    else:
        st.error("❌ ml_helpers.py not found next to app.py or in utils/.")
        st.info("Make sure the file is named exactly 'ml_helpers.py' and lives in the same folder as app.py.")
        st.exception(e1)
        st.stop()
# ---- END: ultra-robust ml_helpers import ----


# ✅ Your modules
from incident_mapping import render_incident_mapping
from utils.ndis_enhanced_prep import prepare_ndis_data, create_comprehensive_features
from ml_helpers import apply_investigation_rules


# ----- CONFIG -----
st.set_page_config(
    page_title="Incident Management Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
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

    # Load and domain rules
    df = load_data()
    df = apply_investigation_rules(df)

    # === ML: standardise & feature-ready ===
    df = prepare_ndis_data(df)  # adds severity_numeric, reportable_bin, histories, location_risk, etc.

    # 🔒 Build post-notify, intake-only targets (constructed from info available at/after notification timestamps,
    # but trained with only intake-time features). This function is defined in ml_helpers.py.
    try:
        df = ML.build_labels_post_notify(df)
    except AttributeError:
        st.warning(
            "build_labels_post_notify() not found in ml_helpers; "
            "continuing without post-notify targets."
        )

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

    # 📅 Date Filter
    if "incident_date" in df.columns:
        min_date, max_date = df["incident_date"].min(), df["incident_date"].max()
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            [min_date, max_date],
            help="Filter incidents by date range",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df["incident_date"] >= pd.to_datetime(date_range[0]))
                & (filtered_df["incident_date"] <= pd.to_datetime(date_range[1]))
            ]

    # 👥 Age Filter
    if "participant_age" in df.columns and not df["participant_age"].isna().all():
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

    # 🏢 Location Filter
    if "location" in df.columns:
        locations = sorted(df["location"].dropna().unique())
        locations_with_all = ["All"] + list(locations)
        selected_location = st.sidebar.selectbox(
            "🏢 Location",
            options=locations_with_all,
            index=0,
            help="Select specific location or 'All'",
        )
        if selected_location != "All":
            filtered_df = filtered_df[filtered_df["location"] == selected_location]

    # ⚠️ Severity Filter
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

    # 📋 Incident Type Filter
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

    # 👤 Carer ID
    if "carer_id" in df.columns:
        carers = sorted(df["carer_id"].astype(str).dropna().unique())
        carers_with_all = ["All"] + list(carers)
        selected_carer = st.sidebar.selectbox(
            "👤 Carer ID",
            options=carers_with_all,
            index=0,
            help="Filter by carer or 'All'",
        )
        if selected_carer != "All":
            filtered_df = filtered_df[filtered_df["carer_id"].astype(str) == selected_carer]

    # 🧩 Group pipeline by (exposed for pipeline view)
    group_by = st.sidebar.selectbox(
        "Group pipeline by:",
        options=["carer_id", "severity", "incident_type", "location"],
        index=0,
        help="Controls grouping in the Enhanced Investigation Pipeline",
    )

    # Page-specific controls
    st.sidebar.markdown("---")
    forecast_horizon = st.sidebar.slider("Forecast months", 3, 12, 6, 1, key="ml_forecast_months")
    top_n_causes = st.sidebar.slider("Top N causes (time chart)", 3, 10, 5, 1, key="ml_top_n_causes")

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
        st.sidebar.metric(
            "Date Range (Days)",
            (filtered_df["incident_date"].max() - filtered_df["incident_date"].min()).days
            if "incident_date" in filtered_df.columns else 0,
        )
        st.sidebar.metric("Locations", filtered_df["location"].nunique() if "location" in filtered_df.columns else 0)
        st.sidebar.metric("Incident Types", filtered_df["incident_type"].nunique() if "incident_type" in filtered_df.columns else 0)

        # Quick stats using prepared columns (if present)
        high_severity_pct = (
            (filtered_df.get("severity_numeric", pd.Series([0]*len(filtered_df))) >= 3)
            .mean() * 100
            if len(filtered_df) else 0
        )
        reportable_pct = (
            filtered_df.get("reportable_bin", pd.Series([0]*len(filtered_df))).mean() * 100
            if len(filtered_df) else 0
        )

        st.sidebar.markdown("**Quick Stats:**")
        st.sidebar.write(f"🔴 High/Critical: {high_severity_pct:.1f}%")
        st.sidebar.write(f"📊 Reportable: {reportable_pct:.1f}%")

    # === Make filters available to pages ===
    st.session_state["APP_FILTERED_DF"] = filtered_df
    st.session_state["APP_GROUP_BY"] = group_by

    # === ML: filtered features (optional convenience for pages) ===
    try:
        X_filt, feature_names_filt, features_df_filt = create_comprehensive_features(filtered_df)
        st.session_state.features_df_filtered = features_df_filt
        st.session_state.feature_names_filtered = feature_names_filt
    except Exception:
        st.session_state.features_df_filtered = None
        st.session_state.feature_names_filtered = None

    # =========================
    # 🤖 Intake-only modeling
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Intake-only modeling (post-notify targets)")

    TARGETS = {
        "Medical attention (≤72h or final flag)": "medical_attention_required",
        "Investigation opened (≤7 days)": "investigation_required",
        "Notify delay > 24 hours": "delay_over_24h",
    }
    target_label = st.sidebar.selectbox("Target", list(TARGETS.keys()))
    target_col = TARGETS[target_label]

    test_size_pct = st.sidebar.slider("Test size (latest % of time)", 10, 40, 20, step=5)
    leak_corr_threshold = st.sidebar.slider("Leak correlation threshold", 0.60, 0.95, 0.80, 0.01)
    run_training = st.sidebar.button("▶️ Train intake-only models")

    if run_training:
        with st.spinner(f"Training models for **{target_label}**…"):
            try:
                results = ML.predictive_models_comparison(
                    df=df,                         # use full (pre-filter) to keep time ordering intact
                    target=target_col,
                    test_size=test_size_pct / 100.0,
                    split_strategy="time_grouped",      # custom: time split + identity guard
                    time_col="incident_datetime",
                    group_cols=["participant_id", "carer_id"],
                    intake_only=True,                   # hard-ban leaky / post-outcome features
                    leak_corr_threshold=leak_corr_threshold,
                )
                st.session_state.trained_models = results
                st.success("Models trained and stored in session.")
            except TypeError:
                # Fallback if your ml_helpers doesn't yet expose time_grouped/intake_only kwargs
                results = ML.predictive_models_comparison(
                    df=df,
                    target=target_col,
                    test_size=test_size_pct / 100.0,
                    split_strategy="time",
                    time_col="incident_datetime",
                    leak_corr_threshold=leak_corr_threshold,
                )
                st.session_state.trained_models = results
                st.info(
                    "Trained with basic time split (fallback). "
                    "Update ml_helpers.predictive_models_comparison to use time_grouped + intake_only."
                )
            except Exception as e:
                st.error("Training failed:")
                st.exception(e)

        # Inline preview of the selected model’s performance
        if st.session_state.trained_models:
            st.subheader("Intake-only model results (preview)")
            names = list(st.session_state.trained_models.keys())
            chosen = st.selectbox("Select model", names, index=0)
            blob = st.session_state.trained_models[chosen]

            c1, c2, c3 = st.columns(3)
            c1.metric("Test accuracy", f"{blob['accuracy']:.3f}")
            if blob.get("cv_scores"):
                cv_mean = sum(blob["cv_scores"]) / len(blob["cv_scores"])
                c2.metric("Train CV accuracy (mean)", f"{cv_mean:.3f}")
            c3.metric("Features used", len(blob.get("feature_names", [])))

            # Try to render the enhanced confusion matrix from ml_helpers
            try:
                fig = ML.enhanced_confusion_matrix_analysis(
                    y_test=blob["y_test"],
                    y_pred=blob["predictions"],
                    y_proba=blob["probabilities"],
                    target_names=(["No", "Yes"] if pd.Series(blob["y_test"]).nunique() == 2
                                  else [str(c) for c in sorted(pd.Series(blob["y_test"]).unique())]),
                    model_name=chosen,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render confusion matrix: {e}")

            with st.expander("Show kept features"):
                st.write(blob.get("feature_names", []))

    # === Executive Summary (drop-in) ===
def display_executive_summary_section(df):
    """
    Render an executive summary for the current (filtered) dataframe.
    Safe with missing columns. Uses Streamlit metrics & simple charts.
    """
    import streamlit as st
    import pandas as pd
    import numpy as np

    if df is None or len(df) == 0:
        st.info("No data to summarise.")
        return df

    d = df.copy()

    # ---------- Dates ----------
    # Prefer incident_datetime, else incident_date
    if "incident_datetime" in d.columns:
        dt = pd.to_datetime(d["incident_datetime"], errors="coerce")
    else:
        dt = pd.to_datetime(d.get("incident_date", pd.NaT), errors="coerce")
    now = pd.Timestamp.utcnow().tz_localize(None)
    last_30_start = now - pd.Timedelta(days=30)
    prev_30_start = now - pd.Timedelta(days=60)

    # Masks (handle NaT gracefully)
    m_last30 = dt >= last_30_start
    m_prev30 = (dt >= prev_30_start) & (dt < last_30_start)

    # ---------- Totals & deltas ----------
    total_incidents = int(len(d))
    incidents_last_30 = int(m_last30.fillna(False).sum())
    incidents_prev_30 = int(m_prev30.fillna(False).sum())
    delta_30 = incidents_last_30 - incidents_prev_30

    # ---------- Investigations ----------
    # We respect your new column; if absent we compute a cautious default = 0
    inv_col = d.get("investigation_required", None)
    if inv_col is not None:
        inv_flag = (
            inv_col.astype(str).str.strip().str.lower()
            .isin({"1", "true", "yes", "y", "t"})
            if inv_col.dtype.kind not in "biu"
            else inv_col.astype(bool)
        )
        inv_rate = float(inv_flag.mean()) if len(inv_flag) else 0.0
    else:
        inv_rate = 0.0

    # ---------- Medical attention ----------
    if "medical_attention_required_bin" in d.columns:
        med_rate = float(pd.to_numeric(d["medical_attention_required_bin"], errors="coerce").fillna(0).clip(0,1).mean())
    elif "medical_attention_required" in d.columns:
        med_rate = float(
            d["medical_attention_required"].astype(str).str.strip().str.lower()
            .isin({"1", "true", "yes", "y", "t"}).mean()
        )
    else:
        med_rate = 0.0

    # ---------- Severity ----------
    if "severity" in d.columns:
        sev_series = d["severity"].astype(str).str.title()
    elif "severity_numeric" in d.columns:
        m = {1:"Low", 2:"Medium", 3:"High", 4:"Critical"}
        sev_series = d["severity_numeric"].map(m).fillna("Medium").astype(str)
    else:
        sev_series = pd.Series(["Unknown"]*len(d))
    sev_counts = sev_series.value_counts(dropna=False)

    # ---------- Incident types ----------
    itype = d.get("incident_type")
    if itype is not None:
        top_types = itype.astype(str).str.strip().replace({"": "Unknown"}).value_counts().head(7)
    else:
        top_types = pd.Series(dtype=int)

    # ---------- Layout ----------
    st.markdown("## Executive Summary")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total incidents", f"{total_incidents:,}")
    with k2:
        st.metric("Last 30 days", f"{incidents_last_30:,}", delta=f"{delta_30:+,}")
    with k3:
        st.metric("Investigation rate", f"{inv_rate:.1%}")
    with k4:
        st.metric("Medical attention rate", f"{med_rate:.1%}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Incident types (Top)")
        if len(top_types):
            st.bar_chart(top_types.sort_values(ascending=True))
        else:
            st.caption("No `incident_type` column found.")

    with c2:
        st.markdown("#### Severity mix")
        if len(sev_counts):
            # Streamlit has no native pie; use Plotly if available
            try:
                import plotly.express as px
                fig = px.pie(sev_counts.reset_index(), names="index", values="severity",
                             title=None)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(sev_counts.sort_values(ascending=True))
        else:
            st.caption("No severity data found.")

    # Optional: show investigations by reason if present
    if "investigation_reason" in d.columns:
        with st.expander("Investigation reasons (sample)"):
            sample = (
                d.loc[d.get("investigation_required", False) == True, ["investigation_reason"]]
                .dropna()
                .head(25)
            )
            if len(sample):
                st.dataframe(sample, use_container_width=True)
            else:
                st.caption("No investigation reasons available.")

    return df


    # ------ PAGE DISPATCH ------
    if page == "📊 Executive Summary":
        display_executive_summary_section(filtered_df)
    elif page == "📈 Operational Performance & Risk Analysis":
        display_operational_performance_section(filtered_df)
    elif page == "📋 Compliance & Investigation":
        display_compliance_investigation_section(filtered_df)
    elif page == "🤖 ML Insights":
        display_ml_insights_section(filtered_df)
    elif page == "🗺️ Incident Map":
        render_incident_mapping(df, filtered_df)


if __name__ == "__main__":
    main()
