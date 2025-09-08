# ---- BEGIN: robust imports (top of dashboard_pages.py) ----
import os, sys, importlib
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Optional utils helper; fall back if not present
try:
    from utils.factor_labels import shorten_factor
except Exception:
    def shorten_factor(x):
        if x is None: return ""
        s = str(x)
        return (s.split(";")[0] or s)[:30]

# Import ml_helpers as a module and verify required symbols
try:
    ML = importlib.import_module("ml_helpers")
except Exception as e:
    st.error("Could not import ml_helpers from dashboard_pages:")
    st.exception(e)
    st.stop()

REQUIRED = [
    "incident_volume_forecasting",
    "seasonal_temporal_patterns",
    "plot_time_with_causes",
    "plot_carer_performance_scatter",
    "create_comprehensive_features",
    "correlation_analysis",
    "clustering_analysis",
    "predictive_models_comparison",
    "incident_type_risk_profiling",
    "create_predictive_risk_scoring",  # if you call it here
]
_missing = [name for name in REQUIRED if not hasattr(ML, name)]
if _missing:
    st.error(f"ml_helpers is missing: {_missing}. Check function names in ml_helpers.py.")
    st.stop()

# Bind names used below
incident_volume_forecasting      = ML.incident_volume_forecasting
seasonal_temporal_patterns       = ML.seasonal_temporal_patterns
plot_time_with_causes            = ML.plot_time_with_causes
plot_carer_performance_scatter   = ML.plot_carer_performance_scatter
create_comprehensive_features    = ML.create_comprehensive_features
correlation_analysis             = ML.correlation_analysis
clustering_analysis              = ML.clustering_analysis
predictive_models_comparison     = ML.predictive_models_comparison
profile_incident_type_risk       = getattr(ML, "profile_incident_type_risk", ML.incident_type_risk_profiling)
create_predictive_risk_scoring   = getattr(ML, "create_predictive_risk_scoring", None)
# ---- END: robust imports ----


# ----------------------------
# Imports
# ----------------------------
import os
import re
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


from ml_helpers import (
    # Charts/utilities you already used
    incident_volume_forecasting,
    seasonal_temporal_patterns,
    plot_time_with_causes,
    create_comprehensive_features,
    correlation_analysis,
    clustering_analysis,
    predictive_models_comparison,
    incident_type_risk_profiling as profile_incident_type_risk,
    # 🔽 New ML bits for the Insights page
    enhanced_confusion_matrix_analysis,
    create_predictive_risk_scoring,
    incident_similarity_analysis,
    ensure_incident_datetime,
    plot_3d_clusters,
)


# Compatibility wrapper so we don't care which signature ml_helpers currently exposes
def _call_incident_forecast(df, horizon):
    try:
        # preferred alias that accepts `periods`
        from ml_helpers import forecast_incident_volume
        return forecast_incident_volume(df, periods=int(horizon))
    except Exception:
        # fall back to the direct function, try several kwargs
        from ml_helpers import incident_volume_forecasting as _ivf
        for kwargs in (
            {"horizon_months": int(horizon)},
            {"months": int(horizon)},
            {"n_periods": int(horizon)},
            {"horizon": int(horizon)},
        ):
            try:
                return _ivf(df, **kwargs)
            except TypeError:
                continue
        # last resort: positional
        return _ivf(df, int(horizon))

# ----------------------------
# Utility
# ----------------------------
def calculate_trend(current_value, previous_value):
    if previous_value == 0:
        return 0, "→"
    change_pct = ((current_value - previous_value) / previous_value) * 100
    if change_pct > 0:
        return change_pct, "↗️"
    elif change_pct < 0:
        return abs(change_pct), "↘️"
    else:
        return 0, "→"

def load_ndis_data():
    """Load the NDIS incident CSV from GitHub and return a pandas DataFrame."""
    url = "https://raw.githubusercontent.com/darolin8/NDIS_dashboard/main/text%20data/ndis_incident_1000.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Could not load data from GitHub: {e}")
        return pd.DataFrame()

# ----------------------------
# Metric/Gauge cards
# ----------------------------
def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph="rgba(0,104,201,0.2)"):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        value=value,
        gauge={"axis": {"visible": False}},
        number={"prefix": prefix, "suffix": suffix, "font.size": 28},
        title={"text": label, "font": {"size": 24}},
    ))
    if show_graph:
        if "High Severity" in label:
            trend_data = [random.randint(5, 25) for _ in range(30)]
        elif "Total" in label:
            trend_data = [random.randint(20, 80) for _ in range(30)]
        else:
            trend_data = [random.randint(0, 50) for _ in range(30)]
        fig.add_trace(go.Scatter(
            y=trend_data,
            hoverinfo="skip",
            fill="tozeroy",
            fillcolor=color_graph,
            line={"color": color_graph},
        ))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )
    st.plotly_chart(fig, use_container_width=True, key=label.replace(" ", "_")+"_metric")

def plot_gauge(indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound):
    fig = go.Figure(go.Indicator(
        value=indicator_number,
        mode="gauge+number",
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": indicator_suffix, "font.size": 26},
        gauge={
            "axis": {"range": [0, max_bound], "tickwidth": 1},
            "bar": {"color": indicator_color},
            "steps": [
                {"range": [0, max_bound*0.5], "color": "lightgray"},
                {"range": [max_bound*0.5, max_bound*0.8], "color": "gray"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": max_bound*0.9
            }
        },
        title={"text": indicator_title, "font": {"size": 20}},
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10, pad=8))
    st.plotly_chart(fig, use_container_width=True, key=indicator_title.replace(" ", "_")+"_gauge")

# ----------------------------
# Core plots
# ----------------------------
def plot_time_analysis(df):
    if df.empty or 'incident_time' not in df.columns:
        st.warning("No time data available for analysis")
        return

    df = df.copy()
    df['incident_hour'] = pd.to_datetime(df['incident_time'], errors='coerce').dt.hour
    df = df.dropna(subset=['incident_hour'])
    df['incident_hour'] = df['incident_hour'].astype(int)

    # Select the first present cause column
    cause_col = None
    for c in ["incident_type", "contributing_factors", "cause", "root_cause"]:
        if c in df.columns:
            cause_col = c
            break
    if cause_col is None:
        st.warning("No cause column found. Expected one of: incident_type / contributing_factors / cause / root_cause.")
        return

    df[cause_col] = df[cause_col].astype(str).fillna("Unknown")
    top_n = 5
    top_causes = df[cause_col].value_counts().head(top_n).index
    df['cause_group'] = np.where(df[cause_col].isin(top_causes), df[cause_col], 'Other')

    hours = list(range(24))
    grouped = df.groupby(['incident_hour', 'cause_group']).size().reset_index(name='count')
    pivot = (grouped.pivot_table(index='incident_hour', columns='cause_group',
                                 values='count', aggfunc='sum', fill_value=0)
                   .reindex(hours, fill_value=0))
    totals = pivot.sum(axis=1)

    fig = go.Figure()

    # Stacked bars for causes
    for cause in sorted(pivot.columns):
        fig.add_trace(go.Bar(
            x=hours,
            y=pivot[cause].values,
            name=str(cause),
            opacity=0.7,
            hovertemplate="Hour %{x}: %{y} incidents<br>Cause: " + str(cause) + "<extra></extra>",
        ))

    # Line for total incidents
    fig.add_trace(go.Scatter(
        x=hours,
        y=totals.values,
        mode="lines+markers",
        name="Total incidents",
        line=dict(width=3, color="orange"),
        yaxis="y2",
        hovertemplate="Hour %{x}: %{y} total<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=2, range=[-0.5, 23.5]),
        yaxis=dict(title="Number of Incidents", side="left"),
        yaxis2=dict(title="Total Incidents", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, r=20, l=60, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="time_analysis")

def plot_weekday_analysis(df):
    if 'incident_weekday' not in df.columns:
        if 'incident_date' in df.columns:
            df = df.copy()
            df['incident_weekday'] = pd.to_datetime(df['incident_date'], errors='coerce').dt.day_name()
        elif 'incident_time' in df.columns:
            df = df.copy()
            df['incident_weekday'] = pd.to_datetime(df['incident_time'], errors='coerce').dt.day_name()
        else:
            st.warning("No data available for weekday analysis")
            return

    weekday_counts = df['incident_weekday'].value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = weekday_counts.reindex(day_order, fill_value=0)

    fig = px.bar(
        x=weekday_counts.index,
        y=weekday_counts.values,
        labels={'x': 'Day of Week', 'y': 'Number of Incidents'},
        color=weekday_counts.values,
        color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig, use_container_width=True, key="weekday_analysis")

def plot_reportable_analysis(df):
    if df.empty or 'reportable' not in df.columns:
        st.warning("No data available for reportable analysis")
        return
    reportable_counts = df['reportable'].value_counts()
    if set(reportable_counts.index) <= {0, 1}:
        reportable_counts.index = ['Not Reportable' if i == 0 else 'Reportable' for i in reportable_counts.index]
    elif set(reportable_counts.index) <= {False, True}:
        reportable_counts.index = ['Not Reportable' if not i else 'Reportable' for i in reportable_counts.index]

    fig = px.pie(
        values=reportable_counts.values,
        names=reportable_counts.index,
        color_discrete_sequence=['#90EE90', '#FFB6C1']
    )
    st.plotly_chart(fig, use_container_width=True, key="reportable_analysis")

def plot_medical_outcomes(df):
    need = {'treatment_required', 'medical_attention_required'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for medical outcomes")
        return
    medical_summary = {
        'Treatment Required': int(df['treatment_required'].sum()),
        'Medical Attention Required': int(df['medical_attention_required'].sum()),
        'No Medical Intervention': int(len(df) - df[['treatment_required', 'medical_attention_required']].any(axis=1).sum())
    }
    fig = px.bar(
        x=list(medical_summary.keys()),
        y=list(medical_summary.values()),
        title="Medical Intervention Requirements",
        labels={'x': 'Medical Outcome', 'y': 'Number of Cases'},
        color=list(medical_summary.values()),
        color_continuous_scale='RdYlBu_r',
        height=400
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True, key="medical_outcomes")

def plot_reporter_type_metrics(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'reported_by' in df.columns:
            value = df['reported_by'].nunique()
            plot_metric("Reporter Types", value, color_graph="#5B8FF9")
    with col2:
        if 'medical_attention_required' in df.columns:
            value = int(df['medical_attention_required'].sum())
            plot_metric("Medical Attention Required", value, color_graph="#F6BD16")
    with col3:
        if 'participant_age' in df.columns:
            avg_age = df['participant_age'].mean()
            plot_metric("Avg Participant Age", avg_age, suffix=" yrs", color_graph="#5AD8A6")

def plot_severity_distribution(df):
    if df.empty or 'severity' not in df.columns:
        st.warning("No data available for severity distribution")
        return
    severity_counts = df['severity'].value_counts()
    colors = {'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'}
    fig = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        color=severity_counts.index,
        color_discrete_map=colors,
        height=400
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
    fig.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5))
    st.plotly_chart(fig, use_container_width=True, key="severity_dist")

def plot_monthly_incidents_by_severity(df):
    need = {'incident_date', 'severity'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for monthly trends")
        return
    df = df.copy()
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['year_month'] = df['incident_date'].dt.to_period('M').astype(str)
    monthly_severity = df.groupby(['year_month', 'severity']).size().reset_index(name='count')
    fig = px.bar(
        monthly_severity,
        x='year_month',
        y='count',
        color='severity',
        labels={'year_month': 'Month', 'count': 'Number of Incidents'},
        color_discrete_map={'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'},
        height=400
    )
    fig.update_layout(xaxis_tickangle=-45, legend_title="Severity", showlegend=True,
                      xaxis_title="Month", yaxis_title="Number of Incidents")
    st.plotly_chart(fig, use_container_width=True, key="monthly_incidents_severity")

def plot_incident_types_bar(df):
    if df.empty or 'incident_type' not in df.columns:
        st.warning("No data available for incident types")
        return
    incident_counts = df['incident_type'].value_counts().head(10)
    fig = px.bar(
        x=incident_counts.values,
        y=incident_counts.index,
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Incident Type'},
        color=incident_counts.values,
        color_continuous_scale='Viridis',
        height=400
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="incident_types_bar")

def plot_location_analysis(df):
    if df.empty or 'location' not in df.columns:
        st.warning("No data available for location analysis")
        return

    location_counts = df['location'].value_counts().head(8)
    vals = location_counts.values

    # Black -> Red continuous scale
    black_red_scale = [
        [0.00, "#000000"], [0.15, "#1a0000"], [0.30, "#330000"], [0.45, "#4d0000"],
        [0.60, "#660000"], [0.75, "#800000"], [0.90, "#b30000"], [1.00, "#ff0000"],
    ]

    fig = px.bar(
        x=location_counts.index,
        y=vals,
        labels={'x': 'Location', 'y': 'Number of Incidents'},
        color=vals,
        color_continuous_scale=black_red_scale,
        range_color=[vals.min(), vals.max()],
        height=400
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False,
                      coloraxis_colorbar=dict(title="Color", len=1.5))
    st.plotly_chart(fig, use_container_width=True, key="location_analysis")

def plot_incident_trends(df):
    if df.empty or 'incident_date' not in df.columns:
        st.warning("No data available for incident trends")
        return
    daily_counts = df.groupby(pd.to_datetime(df['incident_date'], errors='coerce').dt.date).size().reset_index(name='count')
    daily_counts.columns = ['date', 'incidents']
    fig = px.line(daily_counts, x='date', y='incidents', title="Daily Incident Trends", markers=True)
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Incidents", height=300)
    st.plotly_chart(fig, use_container_width=True, key="incident_trends")

def plot_serious_injury_age_severity(df):
    need = {'severity', 'participant_age'}
    if df.empty or not need.issubset(df.columns):
        st.info("No high severity incidents found for age analysis")
        return
    serious_df = df[df['severity'].astype(str).str.lower() == 'high']
    if not serious_df.empty:
        fig = px.histogram(
            serious_df,
            x='participant_age',
            color='severity',
            nbins=20,
            title="High Severity Incidents: Age Distribution"
        )
        st.plotly_chart(fig, use_container_width=True, key="serious_injury_age_severity")
    else:
        st.info("No high severity incidents found for age analysis")

def add_age_and_age_range_columns(df):
    df = df.copy()
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        today = pd.to_datetime('today').normalize()
        df['participant_age'] = ((today - df['dob']).dt.days // 365).astype('float')
        df['participant_age'] = df['participant_age'].where(df['dob'].notnull())
        df['participant_age'] = df['participant_age'].astype('Int64')

    def get_age_range(age):
        if pd.isnull(age):
            return "Unknown"
        if age < 18:
            return "Under 18"
        elif age < 30:
            return "18-29"
        elif age < 45:
            return "30-44"
        elif age < 60:
            return "45-59"
        else:
            return "60+"

    if 'participant_age' in df.columns:
        df['age_range'] = df['participant_age'].apply(get_age_range)
    return df

# NOTE: renamed to avoid clashing with ml_helpers.plot_carer_performance_scatter
def plot_carer_performance_scatter_local(df):
    need = {'carer_id', 'notification_date', 'incident_date'}
    if df.empty or not need.issubset(df.columns):
        st.warning(f"Missing columns for carer performance analysis: {need}")
        return

    data = df.copy()
    data['incident_date'] = pd.to_datetime(data['incident_date'], errors='coerce')
    data['notification_date'] = pd.to_datetime(data['notification_date'], errors='coerce')
    data = data.dropna(subset=['incident_date', 'notification_date', 'carer_id'])
    data['delay_days'] = (data['notification_date'] - data['incident_date']).dt.days
    data['carer_id'] = data['carer_id'].astype(str)

    if data.empty:
        st.info("No valid rows after parsing dates.")
        return

    # ---- Filters (with 'All' options) ----
    with st.expander("Filters & Display Options", expanded=True):
        # Dates
        min_d = data['incident_date'].min().date()
        max_d = data['incident_date'].max().date()
        use_all_dates = st.checkbox("Use all dates", value=True)
        if not use_all_dates:
            start_d, end_d = st.date_input("Incident date range", (min_d, max_d))
            start_d, end_d = pd.to_datetime(start_d), pd.to_datetime(end_d)
            data = data[(data['incident_date'] >= start_d) & (data['incident_date'] <= end_d)]

        # Carers
        carer_counts = data['carer_id'].value_counts()
        carer_options = ["All"] + list(carer_counts.index)
        selected_carers = st.multiselect(
            "Carer(s) to include",
            options=carer_options,
            default=["All"],
            help="Choose 'All' to include every carer"
        )
        if "All" not in selected_carers and selected_carers:
            data = data[data['carer_id'].isin(selected_carers)]

        # Recompute for slider bounds
        carer_counts = data['carer_id'].value_counts()
        max_inc = int(max(1, (carer_counts.max() if not carer_counts.empty else 1)))
        min_inc = st.slider("Minimum incidents per carer", 1, max_inc, 1)

        size_max = st.slider("Bubble max size", 20, 100, 60)

    if data.empty:
        st.info("No rows after filters.")
        return

    # ---- Aggregate ----
    perf = (
        data.groupby('carer_id', as_index=False)
            .agg(
                avg_delay=('delay_days', 'mean'),
                total_incidents=('incident_date', 'count')
            )
    )
    perf = perf[perf['total_incidents'] >= min_inc]
    if perf.empty:
        st.info("No carers meet the selected filters/minimum incident count.")
        return

    # ---- Plot (linear axes, no scale toggles) ----
    fig = px.scatter(
        perf,
        x='avg_delay',
        y='total_incidents',
        color='carer_id',
        size='total_incidents',
        size_max=size_max,
        labels={
            'avg_delay': 'Average Notification Delay (days)',
            'total_incidents': 'Total Incidents',
            'carer_id': 'Carer'
        },
        title='Carer Performance Analysis',
        opacity=0.8
    )

    fig.update_traces(
        marker=dict(line=dict(width=1.5, color='rgba(0,0,0,0.25)')),
        hovertemplate="Carer: %{marker.color}<br>Avg delay: %{x:.2f} days<br>Total incidents: %{y}<extra></extra>"
    )

    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='lightblue',
        zeroline=True, zerolinecolor='lightblue', rangemode='tozero',
        rangeslider=dict(visible=True)
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='lightblue',
        zeroline=True, zerolinecolor='lightblue', rangemode='tozero'
    )

    fig.update_layout(
        legend_title_text='Carer',
        plot_bgcolor='white',
        hovermode='closest',
        dragmode='zoom',
        margin=dict(t=60, r=20, l=60, b=60),
        uirevision="keep"
    )

    config = {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["lasso2d", "select2d", "resetScale2d"]
    }
    st.plotly_chart(fig, use_container_width=True, key="carer_performance_scatter", config=config)

# ----------------------------
# Investigation rules
# ----------------------------
def apply_investigation_rules(df):
    def requires_investigation(row):
        if str(row.get('severity', '')).lower() == 'high':
            return True
        if row.get('reportable', False):
            return True
        serious_types = ['unethical behavior', 'assault', 'unauthorized restraints']
        if str(row.get('incident_type', '')).strip().lower() in serious_types:
            return True
        if row.get('medical_attention_required', False) or row.get('treatment_required', False):
            return True
        return False

    def action_completed(row):
        if str(row.get('medical_outcome', '')).strip().lower() == 'recovered':
            return True
        if str(row.get('severity', '').lower()) == 'low' and not (row.get('medical_attention_required', False) or row.get('treatment_required', False)):
            return True
        if row.get('actions_documented', False):
            return True
        return False

    df = df.copy()
    df['investigation_required'] = df.apply(requires_investigation, axis=1)
    df['action_complete'] = df.apply(action_completed, axis=1)
    return df

# ----------------------------
# Executive Summary
# ----------------------------
def display_executive_summary_section(df):
    st.markdown("""
    <style>
    .main-container {
        max-width: 1100px;
        margin: 0 auto;
        padding: 18px 24px;
    }
    .card-container {
        display: flex;
        gap: 2rem;
        margin-bottom: 2.5rem;
        justify-content: flex-start;
        flex-wrap: wrap;
    }
    .dashboard-card {
        background: #fff;
        border: 1px solid #e3e3e3;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        padding: 1.6rem 1.2rem 1.2rem 1.2rem;
        min-width: 170px;
        max-width: 220px;
        text-align: center;
        flex: 1;
    }
    .dashboard-card-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.6rem; color: #222; }
    .dashboard-card-value { font-size: 2.1rem; font-weight: 700; color: #1769aa; margin-bottom: 0.3rem; }
    .dashboard-card-desc  { font-size: 0.97rem; color: #444; margin-bottom: 0.1rem; }
    .section-title { font-size: 1.35rem; font-weight: 700; margin: 2rem 0 1rem 0; }
    .divider { margin: 2rem 0 2rem 0; border-top: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

    # ---- CARD DATA ----
    top_type = df['incident_type'].value_counts().idxmax() if 'incident_type' in df.columns and not df.empty else "N/A"

    latest_month_str = "N/A"
    latest_month_count = 0
    prev_month_str = "N/A"
    prev_month_count = 0

    if 'incident_date' in df.columns and not df.empty:
        d = pd.to_datetime(df['incident_date'], errors='coerce')
        latest_month = d.max().to_period('M')
        latest_month_str = latest_month.strftime('%B %Y')
        latest_month_count = (d.dt.to_period('M') == latest_month).sum()

        prev_month = latest_month - 1
        prev_month_str = prev_month.strftime('%B %Y')
        prev_month_count = (d.dt.to_period('M') == prev_month).sum()

    high_severity_count = int((df['severity'].astype(str).str.lower() == 'high').sum()) if 'severity' in df.columns else 0
    reportable_count = int(df['reportable'].sum()) if 'reportable' in df.columns else 0

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card-container">
      <div class="dashboard-card">
        <div class="dashboard-card-title">Top Incident Type</div>
        <div class="dashboard-card-value">{top_type}</div>
        <div class="dashboard-card-desc">Most frequent</div>
      </div>
      <div class="dashboard-card">
        <div class="dashboard-card-title">Latest Month Incidents</div>
        <div class="dashboard-card-value">{latest_month_count}</div>
        <div class="dashboard-card-desc">{latest_month_str}</div>
      </div>
      <div class="dashboard-card">
        <div class="dashboard-card-title">Previous Month Incidents</div>
        <div class="dashboard-card-value">{prev_month_count}</div>
        <div class="dashboard-card-desc">{prev_month_str}</div>
      </div>
      <div class="dashboard-card">
        <div class="dashboard-card-title">High Severity Incidents</div>
        <div class="dashboard-card-value">{high_severity_count}</div>
        <div class="dashboard-card-desc">Critical cases</div>
      </div>
      <div class="dashboard-card">
        <div class="dashboard-card-title">Reportable Incidents</div>
        <div class="dashboard-card-value">{reportable_count}</div>
        <div class="dashboard-card-desc">Regulatory events</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Summary visuals ---
    st.markdown('<div class="section-title">Severity Distribution</div>', unsafe_allow_html=True)
    plot_severity_distribution(df)

    st.markdown('<div class="section-title">Top 10 Incident Types</div>', unsafe_allow_html=True)
    plot_incident_types_bar(df)

    st.markdown('<div class="section-title">Location Analysis</div>', unsafe_allow_html=True)
    plot_location_analysis(df)

    st.markdown('<div class="section-title">Monthly Trends</div>', unsafe_allow_html=True)
    plot_monthly_incidents_by_severity(df)

    st.markdown('<div class="section-title">Daily Incident Trends</div>', unsafe_allow_html=True)
    plot_incident_trends(df)

    st.markdown('<div class="section-title">Incidents by Day of Week</div>', unsafe_allow_html=True)
    plot_weekday_analysis(df)

    st.markdown('<div class="section-title">Incidents by Hour of Day</div>', unsafe_allow_html=True)
    plot_time_analysis(df)

    st.markdown('<div class="section-title">Reportable Analysis</div>', unsafe_allow_html=True)
    plot_reportable_analysis(df)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Operational Performance
# ----------------------------
def display_operational_performance_section(df):
    st.header(" Operational Performance & Risk Analysis Metrics")
    st.markdown("---")

    # Average Participant Age
    if 'dob' in df.columns:
        df = df.copy()
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        today = pd.to_datetime('today')
        df['participant_age'] = ((today - df['dob']).dt.days // 365).astype('float')
        avg_age = df['participant_age'].mean()
        avg_age_txt = f"{avg_age:.1f} yrs" if pd.notnull(avg_age) else "N/A"
    else:
        avg_age_txt = "N/A"
    
    # Location Reportable Rate
    location_reportable_rate = (100 * df['reportable'].sum() / len(df)) if 'reportable' in df.columns and len(df) > 0 else 0.0

    # Medical Attention Rate / Count
    if 'medical_attention_required' in df.columns and len(df) > 0:
        medical_attention_rate = 100 * df['medical_attention_required'].sum() / len(df)
        medical_attention_required = int(df['medical_attention_required'].sum())
    else:
        medical_attention_rate = 0.0
        medical_attention_required = 0

    # Cards row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Location Reportable Rate</span><br>
                <span style="font-size:2rem;font-weight:700;color:#1769aa;">{location_reportable_rate:.1f}%</span>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Medical Attention Rate</span><br>
                <span style="font-size:2rem;font-weight:700;color:#d9534f;">{medical_attention_rate:.1f}%</span>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Medical Attention Required</span><br>
                <span style="font-size:2rem;font-weight:700;color:#f0ad4e;">{medical_attention_required}</span>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Average Participant Age</span><br>
                <span style="font-size:2rem;font-weight:700;color:#5ad8a6;">{avg_age_txt}</span>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        plot_incident_types_bar(df)
    with col2:
        plot_medical_outcomes(df)
    plot_carer_performance_scatter_local(df)
    plot_serious_injury_age_severity(df)

# ----------------------------
# Compliance / Investigation
# ----------------------------
def display_compliance_investigation_cards(df):
    need = {'incident_date', 'reportable'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for compliance cards")
        return
    df = apply_investigation_rules(df)

    current_date = pd.to_datetime(df['incident_date'], errors='coerce').max()
    current_month = current_date.to_period('M')
    previous_month = current_month - 1

    df = df.copy()
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    current_df = df[df['incident_date'].dt.to_period('M') == current_month]
    previous_df = df[df['incident_date'].dt.to_period('M') == previous_month]

    current_df = current_df.copy()
    previous_df = previous_df.copy()
    current_df['report_delay_hours'] = (current_df['notification_date'] - current_df['incident_date']).dt.total_seconds() / 3600
    previous_df['report_delay_hours'] = (previous_df['notification_date'] - previous_df['incident_date']).dt.total_seconds() / 3600

    st.markdown("### 📋 Compliance & Investigation Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_reportable = int(current_df['reportable'].sum()) if len(current_df) > 0 else 0
        previous_reportable = int(previous_df['reportable'].sum()) if len(previous_df) > 0 else 0
        change = current_reportable - previous_reportable
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric("📊 Reportable Incidents", current_reportable,
                  delta=f"{trend_arrow} {abs(change)}",
                  delta_color="inverse" if change > 0 else "normal")

    with col2:
        current_compliance = int((current_df['report_delay_hours'] <= 24).sum()) if len(current_df) > 0 else 0
        previous_compliance = int((previous_df['report_delay_hours'] <= 24).sum()) if len(previous_df) > 0 else 0
        change = current_compliance - previous_compliance
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric("⏱️ 24hr Compliance", current_compliance,
                  delta=f"{trend_arrow} {abs(change)}",
                  delta_color="normal" if change > 0 else "inverse")

    with col3:
        current_overdue = int((current_df['report_delay_hours'] > 24).sum()) if len(current_df) > 0 else 0
        previous_overdue = int((previous_df['report_delay_hours'] > 24).sum()) if len(previous_df) > 0 else 0
        change = current_overdue - previous_overdue
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric("⚠️ Overdue Reports", current_overdue,
                  delta=f"{trend_arrow} {abs(change)}",
                  delta_color="inverse" if change > 0 else "normal")

    with col4:
        current_total = len(current_df)
        previous_total = len(previous_df)
        current_investigation_rate = (current_df['investigation_required'].sum() / current_total * 100) if current_total > 0 else 0
        previous_investigation_rate = (previous_df['investigation_required'].sum() / previous_total * 100) if previous_total > 0 else 0
        trend_pct, trend_arrow = calculate_trend(current_investigation_rate, previous_investigation_rate)
        st.metric("🔍 Investigation Rate",
                  f"{current_investigation_rate:.1f}%",
                  delta=f"{trend_arrow} {trend_pct:.1f}%",
                  delta_color="inverse")

def plot_compliance_metrics_poly(df):
    need = {'reportable', 'incident_date', 'notification_date'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for compliance metrics")
        return
    total = len(df)
    if total == 0:
        st.warning("No data available for compliance metrics")
        return

    df = apply_investigation_rules(df)
    df = df.copy()
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')

    reportable_count = int(df['reportable'].sum())
    df['report_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
    compliance_24h_count = int((df['report_delay_hours'] <= 24).sum())
    overdue_count = int((df['report_delay_hours'] > 24).sum())
    inv_required = int(df['investigation_required'].sum())
    inv_rate = inv_required / total * 100 if total > 0 else 0
    action_complete = int(df['action_complete'].sum())
    breach_count = overdue_count
    inv_status_pct = (action_complete / inv_required * 100) if inv_required > 0 else 0
    inv_status_suffix = f" ({action_complete}/{inv_required})"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        plot_metric("Reportable Incidents", reportable_count, color_graph="#5B8FF9")
    with col2:
        plot_metric("24hr Compliance", compliance_24h_count,
                    suffix=f" ({compliance_24h_count/total*100:.1f}%)" if total else "",
                    color_graph="#5AD8A6")
    with col3:
        plot_metric("Overdue Reports", overdue_count, color_graph="#F6BD16")
    with col4:
        plot_metric("Investigation Rate", inv_rate, suffix="%", color_graph="#E86452")
    with col5:
        plot_metric("Investigation Status", inv_status_pct, suffix=inv_status_suffix, color_graph="#6DC8EC")
    with col6:
        plot_metric("Compliance Breach", breach_count, color_graph="#FF2B2B")

def plot_reporting_delay_by_date(df):
    need = {'incident_date', 'notification_date'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for reporting delay analysis")
        return
    df = df.copy()
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
    df['report_delay'] = (df['notification_date'] - df['incident_date']).dt.days
    agg = df.groupby('incident_date').agg(avg_delay=('report_delay', 'mean')).reset_index()
    fig = px.line(agg, x='incident_date', y='avg_delay',
                  title="Average Reporting Delay by Incident Date",
                  labels={'incident_date': 'Incident Date', 'avg_delay': 'Average Delay (Days)'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="reporting_delay_by_date")

def plot_24h_compliance_rate_by_location(df):
    need = {'location', 'notification_date', 'incident_date'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for compliance rate by location")
        return
    df = df.copy()
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
    df['within_24h'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() <= 24*3600
    compliance = df.groupby('location')['within_24h'].mean().reset_index()
    compliance['within_24h'] = compliance['within_24h'] * 100
    fig = px.bar(
        compliance,
        x='location',
        y='within_24h',
        labels={'within_24h': '% Within 24hr', 'location': 'Location'},
        title="24 Hour Compliance Rate by Location",
        color='within_24h',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True, key="compliance_location")

def plot_investigation_pipeline(df):
    need = {'investigation_required', 'action_complete'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for investigation pipeline")
        return
    all_incidents = len(df)
    required = int(df['investigation_required'].sum())
    complete = int(df['action_complete'].sum())
    values = [all_incidents, required, complete]
    names = ['All Incidents', 'Required Investigation', 'Action Complete']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig = px.bar(x=names, y=values, title="Investigation Pipeline",
                 labels={'x': 'Stage', 'y': 'Count'},
                 color=names, color_discrete_sequence=colors)
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True, key="investigation_pipeline")

def plot_contributing_factors_by_month(df, top_k=25):
    need = {'contributing_factors', 'incident_date'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for Contributing Factors heatmap")
        return

    d = df.copy()
    d['incident_date'] = pd.to_datetime(d['incident_date'], errors='coerce')
    d = d.dropna(subset=['incident_date'])

    # 1–2 word labels
    d['factor_short'] = d['contributing_factors'].apply(shorten_factor)

    # focus on the most common to keep the y-axis readable
    top_factors = d['factor_short'].value_counts().head(top_k).index
    d = d[d['factor_short'].isin(top_factors)]

    # chronological months
    d['month_period'] = d['incident_date'].dt.to_period('M')
    d['Count'] = 1

    heatmap = d.pivot_table(
        index='factor_short',
        columns='month_period',     # PeriodIndex keeps true order
        values='Count',
        aggfunc='sum',
        fill_value=0
    ).sort_index(axis=1)

    # don’t annotate zeros
    mask = (heatmap == 0)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        heatmap, annot=True, fmt="d", cmap="Blues",
        mask=mask, cbar_kws={'label': 'Count'}, ax=ax
    )
    ax.set_xlabel("Month-Year")
    ax.set_ylabel("Incident Type")
    ax.set_xticklabels([p.strftime('%b %Y') for p in heatmap.columns], rotation=45, ha='right')
    ax.set_title("Contributing Factors by Month-Year")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def display_compliance_investigation_section(df):
    st.header("Compliance & Investigation Metrics")
    st.markdown("---")

    # Ensure investigation columns for downstream visuals
    df = apply_investigation_rules(df)

    # Compute reporting delay in hours (if possible)
    if 'incident_date' in df.columns and 'notification_date' in df.columns:
        df = df.copy()
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
        df['report_delay_hours'] = ((df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600)
        compliance_24hr = int((df['report_delay_hours'] <= 24).sum())
        overdue_reports = int((df['report_delay_hours'] > 24).sum())
    else:
        compliance_24hr = 0
        overdue_reports = 0

    reportable_incidents = int(df['reportable'].sum()) if 'reportable' in df.columns else 0
    investigation_rate = (100 * df['investigation_required'].sum() / len(df)) if len(df) > 0 else 0.0

    # Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Reportable Incidents</span><br>
                <span style="font-size:2rem;font-weight:700;color:#1769aa;">{reportable_incidents}</span>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">24hr Compliance</span><br>
                <span style="font-size:2rem;font-weight:700;color:#5ad8a6;">{compliance_24hr}</span>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Overdue Reports</span><br>
                <span style="font-size:2rem;font-weight:700;color:#d9534f;">{overdue_reports}</span>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Investigation Rate</span><br>
                <span style="font-size:2rem;font-weight:700;color:#f0ad4e;">{investigation_rate:.1f}%</span>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        plot_reporting_delay_by_date(df)
    with col2:
        plot_24h_compliance_rate_by_location(df)
    plot_investigation_pipeline(df)
    plot_contributing_factors_by_month(df)


# ----------------------------
# ML Insights (robust, minimal dependencies)
# ----------------------------
def display_ml_insights_section(filtered_df):
    """
    ML-focused page:
      1) Model evaluation (confusion matrix + ROC/PR)
      2) Predictive risk scoring sandbox
      3) Similar-incident finder
      4) Forecasting & seasonality
      5) Clustering & risk profiles
      6) Correlations
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    from ml_helpers import (
        ensure_incident_datetime,
        create_comprehensive_features,
        predictive_models_comparison,
        enhanced_confusion_matrix_analysis,
        create_predictive_risk_scoring,
        incident_similarity_analysis,
        incident_volume_forecasting as _ivf,
        seasonal_temporal_patterns,
        clustering_analysis,
        plot_3d_clusters,
        correlation_analysis,
    )

    st.header("🤖 ML Insights")

    # --------- Scope selection ---------
    use_filtered = st.toggle("Use filtered data for ML widgets", value=True,
                             help="Turn off to use the full dataset.")
    df_full = getattr(st.session_state, "df", None)
    if df_full is None:
        st.error("No dataframe in session. Make sure data was loaded successfully.")
        return
    df_used = filtered_df if use_filtered else df_full

    # Build features for the current scope
    try:
        X, feature_names, features_df = create_comprehensive_features(df_used)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        return

    # ---------------------------------
    # Optional: quick trainer
    # ---------------------------------
    with st.expander("🔧 Train baseline models (optional)"):
        if st.button("Train / Refresh", use_container_width=True):
            try:
                st.session_state['trained_models'] = predictive_models_comparison(
                    df_used,
                    split_strategy="time",
                    time_col="incident_datetime",
                )
                st.success("✅ Models trained and stored in session.")
            except Exception as e:
                st.warning(f"Training failed: {e}")

    # ---------------------------------
    # 1) Model evaluation
    # ---------------------------------
    st.subheader("📊 Model Evaluation")
    models = st.session_state.get("trained_models", {})
    if not models:
        st.info("No trained models found. Use the expander above to train baseline models.")
    else:
        for model_name, md in models.items():
            try:
                y_test = md.get("y_test")
                y_pred = md.get("predictions")
                y_proba = md.get("probabilities")
                classes = ['No', 'Yes'] if len(np.unique(y_test)) == 2 else [str(c) for c in sorted(np.unique(y_test))]
                fig = enhanced_confusion_matrix_analysis(y_test, y_pred, y_proba, classes, model_name)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render metrics for {model_name}: {e}")

        # Feature importance preview (RF only)
        st.markdown("#### 🔍 Feature Importance (if available)")
        try:
            best_name, best_blob = max(models.items(), key=lambda kv: kv[1].get("accuracy", 0))
            best_model = best_blob["model"]
            if hasattr(best_model, "named_steps") and "model" in best_model.named_steps:
                mdl = best_model.named_steps["model"]
            else:
                mdl = best_model
            if hasattr(mdl, "feature_importances_"):
                importances = np.array(mdl.feature_importances_)
                trained_feats = best_blob.get("feature_names", feature_names)
                order = np.argsort(importances)[::-1][:20]
                fi_df = pd.DataFrame({
                    "feature": [trained_feats[i] for i in order if i < len(trained_feats)],
                    "importance": [float(importances[i]) for i in order if i < len(importances)],
                })
                if len(fi_df):
                    fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                 title=f"Top Features — {best_name}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No matching feature importances to show.")
            else:
                st.caption(f"{best_name} does not expose feature importances.")
        except Exception as e:
            st.caption(f"Feature-importance preview unavailable: {e}")

    st.divider()

    # ---------------------------------
    # 2) Predictive Risk Scoring (Sandbox)
    # ---------------------------------
    st.subheader("🎯 Predictive Risk Scoring (Sandbox)")
    if not models:
        st.info("Train a model to enable risk scoring.")
    else:
        try:
            best_key = max(models, key=lambda k: models[k].get('accuracy', 0))
            trained_feature_names = models[best_key].get('feature_names', [])
        except Exception:
            best_key = None
        if not trained_feature_names:
            st.warning("No training feature list stored with the model; risk scorer may mismatch.")

        risk_scorer = None
        if best_key and "model" in models.get(best_key, {}):
            risk_scorer = create_predictive_risk_scoring(df_used, models, trained_feature_names)

        if risk_scorer is None:
            st.warning("Risk scorer could not be created from the current models.")
        else:
            try:
                n_expected = getattr(models[best_key]['model'], 'n_features_in_', '?')
                st.caption(f"Model expects {n_expected} features; scorer using {len(trained_feature_names)}.")
            except Exception:
                pass

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                hour = st.slider("Hour of day", 0, 23, 8)
            with c2:
                loc_options = (
                    df_used["location"].dropna().astype(str).value_counts().index.tolist()
                    if "location" in df_used.columns else []
                ) or ["kitchen", "bathroom", "living room", "activity room"]
                location = st.selectbox("Location", options=loc_options[:25])
            with c3:
                max_p_hist = (
                    int(df_used.groupby("participant_id")["incident_id"].count().max())
                    if {"participant_id", "incident_id"}.issubset(df_used.columns) else 20
                )
                p_hist = st.slider("Participant prior incidents", 0, max_p_hist, min(3, max_p_hist))
            with c4:
                max_c_hist = (
                    int(df_used.groupby("carer_id")["incident_id"].count().max())
                    if {"carer_id", "incident_id"}.issubset(df_used.columns) else 20
                )
                c_hist = st.slider("Carer prior incidents", 0, max_c_hist, min(5, max_c_hist))

            # Estimate a coarse location risk proxy from your data (fallback to 2.0)
            try:
                if "severity_numeric" in df_used.columns and "location" in df_used.columns:
                    loc_risk = float(df_used.loc[df_used["location"] == location, "severity_numeric"].mean())
                    if not np.isfinite(loc_risk):
                        loc_risk = 2.0
                else:
                    loc_risk = 2.0
            except Exception:
                loc_risk = 2.0

            scenario = {
                "hour": int(hour),
                "location": str(location),
                "participant_history": int(p_hist),
                "carer_history": int(c_hist),
                "day_type": "weekend" if st.checkbox("Weekend?", value=False, key="sandbox_weekend") else "weekday",
                "location_risk": float(loc_risk),
            }

            try:
                res = risk_scorer(scenario)
                if res["risk_level"] == "HIGH":
                    st.error(f"🚨 HIGH RISK — confidence {res['confidence']:.1%} (model: {res['model_used']})")
                elif res["risk_level"] == "MEDIUM":
                    st.warning(f"⚠️  MEDIUM RISK — confidence {res['confidence']:.1%} (model: {res['model_used']})")
                else:
                    st.success(f"✅ LOW RISK — confidence {res['confidence']:.1%} (model: {res['model_used']})")
            except Exception as e:
                st.exception(e)

    st.divider()


    # Add this in your ML Insights section, perhaps after the existing model comparison
if st.button("Debug Data Leakage"):
    st.subheader("Data Leakage Diagnosis")
    
    with st.expander("Quick Leakage Test"):
        perfect_predictors = quick_leakage_test(df, target="reportable_bin")
    
    with st.expander("Full Diagnostic Report"):
        diagnosis = diagnose_data_leakage(df, target="reportable_bin")
    
    with st.expander("Ultra-Safe Model Test"):
        safe_results = safe_predictive_models_comparison(
            df, 
            target="reportable_bin",
            force_simple_features=True,
            min_correlation_threshold=0.3
        )
    # ---------------------------------
    # 3) Similar Incident Finder
    # ---------------------------------
    st.subheader("🧭 Similar Incident Finder")
    if len(df_used) >= 3:
        try:
            finder, sim = incident_similarity_analysis(df_used, X, feature_names)
            idx = st.number_input("Incident index (0-based)", min_value=0, max_value=max(0, len(df_used)-1), value=0, step=1)
            topk = st.slider("Top-K similar", min_value=3, max_value=10, value=5, step=1)
            if st.button("Find similar"):
                results = finder(int(idx), top_k=int(topk))
                if not results:
                    st.info("No similar incidents found.")
                else:
                    for r in results:
                        st.write(f"• idx {r['index']}  |  similarity {r['similarity_score']:.3f}")
                        with st.expander("View incident details"):
                            st.json(r["incident_data"].to_dict() if r["incident_data"] is not None else {})
        except Exception as e:
            st.warning(f"Similarity search unavailable: {e}")
    else:
        st.info("Not enough rows to compute similarity.")

    st.divider()

    # ---------------------------------
    # 4) Forecasting & Seasonality
    # ---------------------------------
    st.subheader("📈 Forecasting & Seasonality")
    df_used = ensure_incident_datetime(df_used)
    horizon = int(st.session_state.get("ml_forecast_months", 6))
    try:
        fig, forecast_df = _ivf(df_used, months=horizon)
        st.caption(f"Forecast horizon: {horizon} months")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show forecast table"):
            st.dataframe(
                forecast_df.reset_index().rename(columns={"index": "date"}),
                use_container_width=True
            )
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")

    try:
        valid_dt = int(df_used["incident_datetime"].notna().sum())
        if valid_dt >= 3:
            season_fig = seasonal_temporal_patterns(df_used, date_col="incident_datetime")
            st.plotly_chart(season_fig, use_container_width=True)
        else:
            st.info("Not enough valid datetimes to render seasonal heatmap.")
    except Exception as e:
        st.warning(f"Seasonality plot failed: {e}")

    st.divider()

    # ---------------------------------
    # 5) Clustering & Risk Profiles
    # ---------------------------------
    st.subheader("🧩 Clustering & Risk Profiles")
    with st.expander("Clustering controls", expanded=True):
        k = st.slider("k (number of clusters)", 2, 12, 4, step=1, key="ml_k_clusters_insights")
        sample3d = st.slider("Max points in 3D plot", 500, 10000, 2000, step=500, key="ml_k_clusters_3d_sample")

    # Build a safe feature frame for clustering (drop outcomes/proxies)
    def _safe_feats_for_clustering(df_feats: pd.DataFrame) -> pd.DataFrame:
        drop_like = {
            "severity", "severity_numeric", "reportable", "reportable_bin",
            "medical_attention_required", "treatment_required",
            "investigation_required", "incident_resolved",
            "notified_to_commission", "reporting_timeframe",
            "actions_documented", "medical_outcome", "report_delay_hours", "within_24h",
        }
        keep = [c for c in df_feats.columns
                if c not in drop_like and not any(p in c.lower() for p in
                    ["report", "severity", "medical", "investigat", "outcome", "timeframe", "delay", "compliance"])]
        return (df_feats[keep]
                .select_dtypes(include=[np.number])
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0))

    safe_feats = _safe_feats_for_clustering(features_df)

    # 2D Clustering
    color_map = {}
    try:
        fig2d, labels2d = clustering_analysis(safe_feats, k=k)
        st.plotly_chart(fig2d, use_container_width=True)
        if getattr(fig2d.layout, "meta", None) and "cluster_color_map" in fig2d.layout.meta:
            color_map = fig2d.layout.meta["cluster_color_map"]
    except Exception as e:
        st.warning(f"2D clustering failed: {e}")

    # 3D Clustering
    try:
        fig3d, labels3d, df3d = plot_3d_clusters(safe_feats, k=k, sample=sample3d, color_map=color_map)
        st.plotly_chart(fig3d, use_container_width=True)
    except Exception as e:
        st.warning(f"3D clustering failed: {e}")

    st.divider()

    # ---------------------------------
    # 6) Correlations
    # ---------------------------------
    st.subheader("🔗 Correlations")
    try:
        corr_fig = correlation_analysis(safe_feats, height=900)
        st.plotly_chart(corr_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Correlation analysis failed: {e}")


# ---- Page registry expected by app.py ----
PAGE_TO_RENDERER = {
    "Executive Summary":           display_executive_summary_section,
    "Operational Performance":     display_operational_performance_section,
    "Compliance & Investigation":  display_compliance_investigation_section,
    "ML Insights":                 display_ml_insights_section,
}

PAGE_ORDER = [
    "Executive Summary",
    "Operational Performance",
    "Compliance & Investigation",
    "ML Insights",
]

def render_page(page_name: str, df):
    fn = PAGE_TO_RENDERER.get(page_name)
    if fn is None:
        st.error(f"Unknown page: {page_name}")
        return
    return fn(df)

__all__ = [
    "display_executive_summary_section",
    "display_operational_performance_section",
    "display_compliance_investigation_section",
    "display_ml_insights_section",
    "apply_investigation_rules",         # ← add this line
    "PAGE_TO_RENDERER",
    "PAGE_ORDER",
    "render_page",
]


