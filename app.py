# app.py
import os
import sys
from datetime import datetime
import pandas as pd
import streamlit as st

# Ensure local imports resolve
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import dashboard_pages as dp  # our pages & helpers


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="NDIS Incident Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Data loading
# ----------------------------
def load_data():
    """
    Load and lightly normalize the NDIS incidents dataset.
    Uses the loader defined in dashboard_pages.py for consistency.
    """
    df = dp.load_ndis_data()
    if df.empty:
        return df

    # Parse common date/time fields defensively
    for col in ["incident_date", "notification_date", "incident_time", "dob"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convenience fields
    if "incident_date" in df.columns:
        df["incident_weekday"] = df["incident_date"].dt.day_name()
        df["year_month"] = df["incident_date"].dt.to_period("M").astype(str)

    # Booleans as proper dtype if present
    for bcol in ["reportable", "medical_attention_required", "treatment_required", "actions_documented"]:
        if bcol in df.columns:
            # Cast carefully (values may be 0/1 or strings)
            df[bcol] = df[bcol].apply(lambda x: bool(int(x)) if str(x).isdigit() else bool(x))

    return df


# ----------------------------
# Sidebar Filters
# ----------------------------
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    if df.empty:
        st.sidebar.info("No data loaded.")
        return df

    # Date range
    if "incident_date" in df.columns and df["incident_date"].notna().any():
        dmin = df["incident_date"].min().date()
        dmax = df["incident_date"].max().date()
        start_date, end_date = st.sidebar.date_input(
            "Incident date range", (dmin, dmax), min_value=dmin, max_value=dmax
        )
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df["incident_date"] >= start_date) & (df["incident_date"] <= end_date)]

    # Severity
    if "severity" in df.columns:
        severities = sorted(df["severity"].dropna().astype(str).unique().tolist())
        selected_sev = st.sidebar.multiselect("Severity", ["All"] + severities, default=["All"])
        if "All" not in selected_sev:
            df = df[df["severity"].astype(str).isin(selected_sev)]

    # Location
    if "location" in df.columns:
        locs = df["location"].dropna().astype(str).value_counts().index.tolist()
        selected_locs = st.sidebar.multiselect("Location", ["All"] + locs, default=["All"])
        if "All" not in selected_locs and selected_locs:
            df = df[df["location"].astype(str).isin(selected_locs)]

    # Reportable
    if "reportable" in df.columns:
        rep_choice = st.sidebar.selectbox("Reportable", ["All", "Reportable only", "Not reportable"])
        if rep_choice == "Reportable only":
            df = df[df["reportable"] == True]
        elif rep_choice == "Not reportable":
            df = df[df["reportable"] == False]

    # Keep an easy copy for ML page
    st.session_state["df"] = df.copy()

    return df


# ----------------------------
# Main
# ----------------------------
def main():
    st.title("NDIS Incident Dashboard")

    # Load
    df = load_data()

    # Sidebar navigation
    page = st.sidebar.radio("Page", dp.PAGE_ORDER, index=0)

    # Apply filters (shared across pages)
    filtered_df = sidebar_filters(df)

    # Optional raw data view
    with st.sidebar.expander("Data preview"):
        st.write(f"Rows: {len(filtered_df)}")
        st.dataframe(filtered_df.head(200), use_container_width=True)

    # ------ PAGE DISPATCH ------
    try:
        dp.render_page(page, filtered_df)
    except Exception as e:
        st.error("Something went wrong while rendering the page.")
        st.exception(e)


if __name__ == "__main__":
    main()
