# dashboard_pages.py
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
    "create_predictive_risk_scoring",
    "enhanced_confusion_matrix_analysis",
    "incident_similarity_analysis",
    "ensure_incident_datetime",
    "plot_3d_clusters",
]
_missing = [name for name in REQUIRED if not hasattr(ML, name)]
if _missing:
    st.error(f"ml_helpers is missing: {_missing}. Check function names in ml_helpers.py.")
    st.stop()

# Bind names used below
incident_volume_forecasting      = ML.incident_volume_forecasting
seasonal_temporal_patterns       = ML.seasonal_temporal_patterns
plot_time_with_causes            = ML.plot_time_with_causes
plot_carer_performance_scatter_ML= ML.plot_carer_performance_scatter
create_comprehensive_features    = ML.create_comprehensive_features
correlation_analysis             = ML.correlation_analysis
clustering_analysis              = ML.clustering_analysis
predictive_models_comparison     = ML.predictive_models_comparison
profile_incident_type_risk       = ML.incident_type_risk_profiling
create_predictive_risk_scoring   = ML.create_predictive_risk_scoring
enhanced_confusion_matrix_analysis = ML.enhanced_confusion_matrix_analysis
incident_similarity_analysis     = ML.incident_similarity_analysis
ensure_incident_datetime         = ML.ensure_incident_datetime
plot_3d_clusters                 = ML.plot_3d_clusters
# ---- END: robust imports ----

# ----------------------------
# Imports
# ----------------------------
import re, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

def clean_label(label: str) -> str:
    """Remove standard prefixes from feature names for nicer display."""
    for prefix in ("loc_", "type_", "is_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label

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
# Core plots (Exec/Operational)
# ----------------------------
def plot_time_analysis(df):
    if df.empty or 'incident_time' not in df.columns:
        st.warning("No time data available for analysis")
        return
    data = df.copy()
    data['incident_hour'] = pd.to_datetime(data['incident_time'], errors='coerce').dt.hour
    data = data.dropna(subset=['incident_hour'])
    data['incident_hour'] = data['incident_hour'].astype(int)
    cause_col = next((c for c in ["incident_type", "contributing_factors", "cause", "root_cause"] if c in data.columns), None)
    if cause_col is None:
        st.warning("No cause column found. Expected one of: incident_type / contributing_factors / cause / root_cause.")
        return
    data[cause_col] = data[cause_col].astype(str).fillna("Unknown")
    top_n = int(st.session_state.get("ml_top_n_causes", 5))
    top_causes = data[cause_col].value_counts().head(top_n).index
    data['cause_group'] = np.where(data[cause_col].isin(top_causes), data[cause_col], 'Other')
    hours = list(range(24))
    grouped = data.groupby(['incident_hour', 'cause_group']).size().reset_index(name='count')
    pivot = (grouped.pivot_table(index='incident_hour', columns='cause_group',
                                 values='count', aggfunc='sum', fill_value=0)
                   .reindex(hours, fill_value=0))
    totals = pivot.sum(axis=1)
    fig = go.Figure()
    for cause in sorted(pivot.columns):
        fig.add_trace(go.Bar(x=hours, y=pivot[cause].values, name=str(cause), opacity=0.7))
    fig.add_trace(go.Scatter(x=hours, y=totals.values, mode="lines+markers",
                             name="Total incidents", line=dict(width=3, color="orange"), yaxis="y2"))
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
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_counts = weekday_counts.reindex(order, fill_value=0)
    fig = px.bar(x=weekday_counts.index, y=weekday_counts.values,
                 labels={'x': 'Day of Week', 'y': 'Number of Incidents'},
                 color=weekday_counts.values, color_continuous_scale='Plasma')
    st.plotly_chart(fig, use_container_width=True, key="weekday_analysis")

def plot_reportable_analysis(df):
    if df.empty or 'reportable' not in df.columns:
        st.warning("No data available for reportable analysis")
        return
    counts = df['reportable'].value_counts()
    if set(counts.index) <= {0, 1}:
        counts.index = ['Not Reportable' if i == 0 else 'Reportable' for i in counts.index]
    elif set(counts.index) <= {False, True}:
        counts.index = ['Not Reportable' if not i else 'Reportable' for i in counts.index]
    fig = px.pie(values=counts.values, names=counts.index,
                 color_discrete_sequence=['#90EE90', '#FFB6C1'])
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
    fig = px.bar(x=list(medical_summary.keys()), y=list(medical_summary.values()),
                 title="Medical Intervention Requirements",
                 labels={'x': 'Medical Outcome', 'y': 'Number of Cases'},
                 color=list(medical_summary.values()), color_continuous_scale='RdYlBu_r', height=400)
    fig.update_layout(showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True, key="medical_outcomes")

def plot_severity_distribution(df):
    if df.empty or 'severity' not in df.columns:
        st.warning("No data available for severity distribution")
        return
    counts = df['severity'].value_counts()
    colors = {'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'}
    fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                 color_discrete_map=colors, height=400)
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
    fig.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5))
    st.plotly_chart(fig, use_container_width=True, key="severity_dist")

def plot_monthly_incidents_by_severity(df):
    need = {'incident_date', 'severity'}
    if df.empty or not need.issubset(df.columns):
        st.warning("No data available for monthly trends")
        return
    d = df.copy()
    d['incident_date'] = pd.to_datetime(d['incident_date'], errors='coerce')
    d['year_month'] = d['incident_date'].dt.to_period('M').astype(str)
    monthly = d.groupby(['year_month', 'severity']).size().reset_index(name='count')
    fig = px.bar(monthly, x='year_month', y='count', color='severity',
                 labels={'year_month': 'Month', 'count': 'Number of Incidents'},
                 color_discrete_map={'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'}, height=400)
    fig.update_layout(xaxis_tickangle=-45, legend_title="Severity", showlegend=True,
                      xaxis_title="Month", yaxis_title="Number of Incidents")
    st.plotly_chart(fig, use_container_width=True, key="monthly_incidents_severity")

def plot_incident_types_bar(df):
    if df.empty or 'incident_type' not in df.columns:
        st.warning("No data available for incident types")
        return
    counts = df['incident_type'].value_counts().head(10)
    fig = px.bar(x=counts.values, y=counts.index, orientation='h',
                 labels={'x': 'Number of Incidents', 'y': 'Incident Type'},
                 color=counts.values, color_continuous_scale='Viridis', height=400)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="incident_types_bar")

def plot_location_analysis(df):
    if df.empty or 'location' not in df.columns:
        st.warning("No data available for location analysis")
        return
    counts = df['location'].value_counts().head(8)
    vals = counts.values
    black_red = [
        [0.00, "#000000"], [0.15, "#1a0000"], [0.30, "#330000"], [0.45, "#4d0000"],
        [0.60, "#660000"], [0.75, "#800000"], [0.90, "#b30000"], [1.00, "#ff0000"],
    ]
    fig = px.bar(x=counts.index, y=vals, labels={'x': 'Location', 'y': 'Number of Incidents'},
                 color=vals, color_continuous_scale=black_red, range_color=[vals.min(), vals.max()], height=400)
    fig.update_layout(xaxis_tickangle=-45, showlegend=False, coloraxis_colorbar=dict(title="Color", len=1.5))
    st.plotly_chart(fig, use_container_width=True, key="location_analysis")

def plot_incident_trends(df):
    if df.empty or 'incident_date' not in df.columns:
        st.warning("No data available for incident trends")
        return
    daily = df.groupby(pd.to_datetime(df['incident_date'], errors='coerce').dt.date).size().reset_index(name='count')
    daily.columns = ['date', 'incidents']
    fig = px.line(daily, x='date', y='incidents', title="Daily Incident Trends", markers=True)
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Incidents", height=300)
    st.plotly_chart(fig, use_container_width=True, key="incident_trends")

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

    d = df.copy()
    d['investigation_required'] = d.apply(requires_investigation, axis=1)
    d['action_complete'] = d.apply(action_completed, axis=1)
    return d

# ----------------------------
# Executive Summary
# ----------------------------
def display_executive_summary_section(df):
    st.markdown("""
    <style>
    .main-container {max-width: 1100px; margin: 0 auto; padding: 18px 24px;}
    .card-container {display: flex; gap: 2rem; margin-bottom: 2.5rem; justify-content: flex-start; flex-wrap: wrap;}
    .dashboard-card {background: #fff; border: 1px solid #e3e3e3; border-radius: 14px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
                     padding: 1.6rem 1.2rem 1.2rem 1.2rem; min-width: 170px; max-width: 220px; text-align: center; flex: 1;}
    .dashboard-card-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.6rem; color: #222; }
    .dashboard-card-value { font-size: 2.1rem; font-weight: 700; color: #1769aa; margin-bottom: 0.3rem; }
    .dashboard-card-desc  { font-size: 0.97rem; color: #444; margin-bottom: 0.1rem; }
    .section-title { font-size: 1.35rem; font-weight: 700; margin: 2rem 0 1rem 0; }
    .divider { margin: 2rem 0 2rem 0; border-top: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

    top_type = df['incident_type'].value_counts().idxmax() if 'incident_type' in df.columns and not df.empty else "N/A"

    latest_month_str = "N/A"; latest_month_count = 0
    prev_month_str   = "N/A"; prev_month_count   = 0
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
      <div class="dashboard-card"><div class="dashboard-card-title">Top Incident Type</div>
        <div class="dashboard-card-value">{top_type}</div><div class="dashboard-card-desc">Most frequent</div></div>
      <div class="dashboard-card"><div class="dashboard-card-title">Latest Month Incidents</div>
        <div class="dashboard-card-value">{latest_month_count}</div><div class="dashboard-card-desc">{latest_month_str}</div></div>
      <div class="dashboard-card"><div class="dashboard-card-title">Previous Month Incidents</div>
        <div class="dashboard-card-value">{prev_month_count}</div><div class="dashboard-card-desc">{prev_month_str}</div></div>
      <div class="dashboard-card"><div class="dashboard-card-title">High Severity Incidents</div>
        <div class="dashboard-card-value">{high_severity_count}</div><div class="dashboard-card-desc">Critical cases</div></div>
      <div class="dashboard-card"><div class="dashboard-card-title">Reportable Incidents</div>
        <div class="dashboard-card-value">{reportable_count}</div><div class="dashboard-card-desc">Regulatory events</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Severity Distribution</div>', unsafe_allow_html=True)
    plot_severity_distribution(df)
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
def add_age_and_age_range_columns(df):
    d = df.copy()
    if 'dob' in d.columns:
        d['dob'] = pd.to_datetime(d['dob'], errors='coerce')
        today = pd.to_datetime('today').normalize()
        d['participant_age'] = ((today - d['dob']).dt.days // 365).astype('float')
        d['participant_age'] = d['participant_age'].where(d['dob'].notnull())
        d['participant_age'] = d['participant_age'].astype('Int64')

    def get_age_range(age):
        if pd.isnull(age): return "Unknown"
        if age < 18: return "Under 18"
        elif age < 30: return "18-29"
        elif age < 45: return "30-44"
        elif age < 60: return "45-59"
        else: return "60+"

    if 'participant_age' in d.columns:
        d['age_range'] = d['participant_age'].apply(get_age_range)
    return d

def plot_carer_performance_scatter(df):
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
    carer_counts = data['carer_id'].value_counts()
    perf = data.groupby('carer_id').agg(
        avg_delay=('delay_days', 'mean'),
        total_incidents=('incident_date', 'count'),
        high_severity_count=('severity', lambda x: (x.astype(str).str.lower() == 'high').sum() if 'severity' in data.columns else 0),
        reportable_count=('reportable', lambda x: x.sum() if 'reportable' in data.columns else 0),
        date_range=('incident_date', lambda x: (x.max() - x.min()).days),
    ).reset_index()
    perf['incidents_per_month'] = perf.apply(
        lambda row: row['total_incidents'] / (max(row['date_range'], 1) / 30.44) if row['date_range'] > 0 else row['total_incidents'],
        axis=1
    )
    perf['high_severity_count'] = perf['high_severity_count'].fillna(0).astype(int)
    perf['reportable_count'] = perf['reportable_count'].fillna(0).astype(int)
    def categorize_workload(ipm):
        if ipm >= 15: return "High Volume (15+/month)"
        elif ipm >= 8: return "Medium Volume (8-14/month)"
        elif ipm >= 3: return "Low Volume (3-7/month)"
        return "Very Low Volume (<3/month)"
    perf['workload_category'] = perf['incidents_per_month'].apply(categorize_workload)
    perf = perf[perf['total_incidents'] >= 3]
    if perf.empty:
        st.info("No carers meet the selected filters/minimum incident count.")
        return
    fig = px.scatter(perf, x='avg_delay', y='total_incidents', color='workload_category',
                     size='incidents_per_month', size_max=60,
                     labels={'avg_delay': 'Average Notification Delay (days)', 'total_incidents': 'Total Incidents'},
                     title='Workload vs Performance Analysis - Capacity Planning View', opacity=0.85)
    fig.add_vline(x=2, line_dash="dash", line_color="orange", opacity=0.7,
                  annotation_text="2-Day Threshold", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True, key="carer_performance_scatter")

def display_operational_performance_section(df):
    st.header(" Operational Performance & Risk Analysis Metrics")
    st.markdown("---")
    # Simple KPI cards
    location_rate = (100 * df['reportable'].sum() / len(df)) if 'reportable' in df.columns and len(df) > 0 else 0.0
    med_rate = 100 * df['medical_attention_required'].sum() / len(df) if 'medical_attention_required' in df.columns and len(df)>0 else 0.0
    med_required = int(df['medical_attention_required'].sum()) if 'medical_attention_required' in df.columns else 0
    avg_age_txt = "N/A"
    if 'dob' in df.columns:
        d = df.copy()
        d['dob'] = pd.to_datetime(d['dob'], errors='coerce')
        today = pd.to_datetime('today')
        d['participant_age'] = ((today - d['dob']).dt.days // 365).astype('float')
        avg_age = d['participant_age'].mean()
        avg_age_txt = f"{avg_age:.1f} yrs" if pd.notnull(avg_age) else "N/A"
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Location Reportable Rate</span><br>
                <span style="font-size:2rem;font-weight:700;color:#1769aa;">{location_rate:.1f}%</span></div>""", unsafe_allow_html=True)
    with col2: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Medical Attention Rate</span><br>
                <span style="font-size:2rem;font-weight:700;color:#d9534f;">{med_rate:.1f}%</span></div>""", unsafe_allow_html=True)
    with col3: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Medical Attention Required</span><br>
                <span style="font-size:2rem;font-weight:700;color:#f0ad4e;">{med_required}</span></div>""", unsafe_allow_html=True)
    with col4: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Average Participant Age</span><br>
                <span style="font-size:2rem;font-weight:700;color:#5ad8a6;">{avg_age_txt}</span></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: plot_incident_types_bar(df)
    with col2: plot_medical_outcomes(df)
    plot_carer_performance_scatter(df)
    # Optional distribution
    if {'severity','participant_age'}.issubset(df.columns):
        serious_df = df[df['severity'].astype(str).str.lower() == 'high']
        if not serious_df.empty:
            fig = px.histogram(serious_df, x='participant_age', color='severity',
                               nbins=20, title="High Severity Incidents: Age Distribution")
            st.plotly_chart(fig, use_container_width=True, key="serious_injury_age_severity")

# ----------------------------
# Compliance / Investigation
# ----------------------------
def plot_investigation_pipeline(df, group_by: str = "carer_id"):
    """Bar group stages by selected grouping. No sidebar; group_by comes from app session."""
    d = df.copy()
    if "incident_date" in d.columns and "notification_date" in d.columns:
        d['incident_date'] = pd.to_datetime(d['incident_date'], errors='coerce')
        d['notification_date'] = pd.to_datetime(d['notification_date'], errors='coerce')
        d['report_delay_hours'] = (d['notification_date'] - d['incident_date']).dt.total_seconds() / 3600
        d['within_24h'] = d['report_delay_hours'] <= 24
        d['overdue_action'] = (~d.get('action_complete', pd.Series(False)).astype(bool)) & (d['report_delay_hours'] > 24)
        d['under_investigation'] = d.get('investigation_required', pd.Series(False)).astype(bool) & (~d.get('action_complete', pd.Series(False)).astype(bool))
    else:
        d['report_delay_hours'] = None
        d['within_24h'] = None
        d['overdue_action'] = None
        d['under_investigation'] = None

    agg = d.groupby(group_by).agg(
        All_Incidents=('report_delay_hours', 'count'),
        Required_Investigation=('investigation_required', 'sum') if 'investigation_required' in d.columns else (group_by, 'count'),
        Under_Investigation=('under_investigation', 'sum') if 'under_investigation' in d.columns else (group_by, 'count'),
        Action_Complete=('action_complete', 'sum') if 'action_complete' in d.columns else (group_by, 'count'),
        Overdue_Actions=('overdue_action', 'sum') if 'overdue_action' in d.columns else (group_by, 'count'),
        Compliance_Breach=('within_24h', lambda x: (~x).sum()) if 'within_24h' in d.columns else (group_by, 'count')
    ).reset_index()

    stages = ['All_Incidents', 'Required_Investigation', 'Under_Investigation',
              'Action_Complete', 'Overdue_Actions', 'Compliance_Breach']
    melted = agg.melt(id_vars=group_by, value_vars=stages, var_name="Stage", value_name="Count")

    fig = px.bar(melted, x=group_by, y="Count", color="Stage", barmode="group",
                 title=f"Enhanced Investigation Pipeline by {group_by.replace('_',' ').title()}",
                 labels={group_by: group_by.replace('_',' ').title(), "Count": "Number of Incidents"})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show pipeline data table"):
        st.dataframe(agg, use_container_width=True)

def display_compliance_investigation_section(df):
    """Reads filtered data + group_by from app-level sidebar (session state)."""
    filtered_df = st.session_state.get("APP_FILTERED_DF", df)
    group_by = st.session_state.get("APP_GROUP_BY", "carer_id")

    st.header("Compliance & Investigation")
    st.write("Compliance rates and investigation pipeline overview.")

    if filtered_df.empty:
        st.info("No data available.")
        return

    # Compute reporting delay in hours (if possible)
    d = apply_investigation_rules(filtered_df)
    if {'incident_date','notification_date'}.issubset(d.columns):
        d = d.copy()
        d['incident_date'] = pd.to_datetime(d['incident_date'], errors='coerce')
        d['notification_date'] = pd.to_datetime(d['notification_date'], errors='coerce')
        d['report_delay_hours'] = ((d['notification_date'] - d['incident_date']).dt.total_seconds() / 3600)
        compliance_24hr = int((d['report_delay_hours'] <= 24).sum())
        overdue_reports = int((d['report_delay_hours'] > 24).sum())
    else:
        compliance_24hr = 0
        overdue_reports = 0

    reportable_incidents = int(d['reportable'].sum()) if 'reportable' in d.columns else 0
    investigation_rate = (100 * d['investigation_required'].sum() / len(d)) if len(d) > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Reportable Incidents</span><br>
                <span style="font-size:2rem;font-weight:700;color:#1769aa;">{reportable_incidents}</span></div>""", unsafe_allow_html=True)
    with col2: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">24hr Compliance</span><br>
                <span style="font-size:2rem;font-weight:700;color:#5ad8a6;">{compliance_24hr}</span></div>""", unsafe_allow_html=True)
    with col3: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Overdue Reports</span><br>
                <span style="font-size:2rem;font-weight:700;color:#d9534f;">{overdue_reports}</span></div>""", unsafe_allow_html=True)
    with col4: st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">Investigation Rate</span><br>
                <span style="font-size:2rem;font-weight:700;color:#f0ad4e;">{investigation_rate:.1f}%</span></div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Use the app-provided group_by
    plot_investigation_pipeline(d, group_by=group_by)

# ----------------------------
# ML Insights (with correlation + cleaned labels)
# ----------------------------
def display_ml_insights_section(filtered_df):
    """
    ML-focused page:
      1) Model evaluation
      2) Predictive risk scoring sandbox
      3) Similar-incident finder
      4) Forecasting & seasonality
      5) Clustering & risk profiles
      6) Correlations  ← integrated here
    """
    import numpy as np
    import pandas as pd

    st.header("🤖 ML Insights")

    # --------- Scope selection ---------
    use_filtered = st.toggle("Use filtered data for ML widgets", value=True,
                             help="Turn off to use the full dataset.")
    df_full = getattr(st.session_state, "df", None)
    if df_full is None:
        st.error("No dataframe in session. Make sure data was loaded successfully.")
        return
    df_used = filtered_df if use_filtered else df_full

    # Build features for the current scope (with safe fallback)
    try:
        X, feature_names, features_df = create_comprehensive_features(df_used)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}. Falling back to numeric-only features.")
        safe_num = (
            df_used.select_dtypes(include=[np.number])
                   .replace([np.inf, -np.inf], np.nan)
                   .fillna(0.0)
        )
        if safe_num.empty:
            st.info("No numeric columns available for fallback. Try widening filters.")
            return
        X = safe_num.to_numpy(dtype=float)
        feature_names = list(safe_num.columns)
        features_df = safe_num.copy()

    # ---------------------------------
    # 1) Model evaluation
    # ---------------------------------
    st.subheader("📊 Model Evaluation")
    models = st.session_state.get("trained_models", {})
    if not models:
        st.info("No trained models found. Use the sidebar on the left to train baseline models.")
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

        # Feature importance / coefficient magnitudes (clean labels)
        st.markdown("#### 🔍 Feature Importance / Coefficients")
        try:
            best_name, best_blob = max(models.items(), key=lambda kv: kv[1].get("accuracy", 0))
            best_model = best_blob["model"]
            trained_feats = best_blob.get("feature_names", feature_names)

            # If pipeline, extract inner model
            if hasattr(best_model, "named_steps") and "model" in best_model.named_steps:
                mdl = best_model.named_steps["model"]
            else:
                mdl = best_model

            # Tree/forest/boosting with feature_importances_
            if hasattr(mdl, "feature_importances_"):
                importances = np.array(mdl.feature_importances_)
                order = np.argsort(importances)[::-1][:20]
                fi_df = pd.DataFrame({
                    "feature": [clean_label(trained_feats[i]) for i in order if i < len(trained_feats)],
                    "importance": [float(importances[i]) for i in order if i < len(importances)],
                })
                if len(fi_df):
                    fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                 title=f"Top Features — {best_name}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No matching feature importances to show.")

            # Linear models with coef_
            elif hasattr(mdl, "coef_"):
                coef = mdl.coef_.ravel()
                mags = np.abs(coef)
                order = np.argsort(mags)[::-1][:20]
                df_coef = pd.DataFrame({
                    "feature": [clean_label(trained_feats[i]) for i in order if i < len(trained_feats)],
                    "magnitude": [float(mags[i]) for i in order if i < len(mags)],
                })
                title = f"Top Coefficient Magnitudes — {best_name}"
                fig = px.bar(df_coef, x="magnitude", y="feature", orientation="h", title=title)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.caption(f"{best_name} does not expose feature importances; showing none.")
        except Exception as e:
            st.caption(f"Importance/coefficients preview unavailable: {e}")

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
            trained_feature_names = []
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
            with c1: hour = st.slider("Hour of day", 0, 23, 8)
            with c2:
                loc_options = (df_used["location"].dropna().astype(str).value_counts().index.tolist()
                               if "location" in df_used.columns else []) or ["kitchen","bathroom","living room","activity room"]
                location = st.selectbox("Location", options=loc_options[:25])
            with c3:
                max_p_hist = (int(df_used.groupby("participant_id")["incident_id"].count().max())
                              if {"participant_id","incident_id"}.issubset(df_used.columns) else 20)
                p_hist = st.slider("Participant prior incidents", 0, max_p_hist, min(3, max_p_hist))
            with c4:
                max_c_hist = (int(df_used.groupby("carer_id")["incident_id"].count().max())
                              if {"carer_id","incident_id"}.issubset(df_used.columns) else 20)
                c_hist = st.slider("Carer prior incidents", 0, max_c_hist, min(5, max_c_hist))

            try:
                loc_risk = float(df_used.loc[df_used["location"] == location, "severity_numeric"].mean()) \
                           if "severity_numeric" in df_used.columns and "location" in df_used.columns else 2.0
                if not np.isfinite(loc_risk): loc_risk = 2.0
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
        fig, forecast_df = incident_volume_forecasting(df_used, months=horizon)
        st.caption(f"Forecast horizon: {horizon} months")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show forecast table"):
            st.dataframe(forecast_df.reset_index().rename(columns={"index": "date"}), use_container_width=True)
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
    # 6) Correlations (IN-PAGE)
    # ---------------------------------
    st.subheader("🔗 Correlations")
    try:
        corr_matrix = safe_feats.corr()
        cleaned_columns = [clean_label(str(col)) for col in corr_matrix.columns]
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=cleaned_columns,
            y=cleaned_columns,
            height=900,
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
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
    "apply_investigation_rules",
    "PAGE_TO_RENDERER",
    "PAGE_ORDER",
    "render_page",
]
