import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
st.warning("EXECUTIVE SUMMARY UPDATED")

import os
st.info(f"Loaded from: {os.path.abspath(__file__)}")


# ================= UTILITY FUNCTIONS =================

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

# ================= EXECUTIVE SUMMARY PLOTS =================

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
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True, key=indicator_title.replace(" ", "_")+"_gauge")
  

def load_ndis_data():
    """Load the NDIS incident CSV from GitHub and return a pandas DataFrame."""
    url = "https://raw.githubusercontent.com/darolin8/NDIS_dashboard/main/text%20data/ndis_incident_1000.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Could not load data from GitHub: {e}")
        return pd.DataFrame()

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
    grouped = (
        df.groupby(['incident_hour', 'cause_group'])
          .size()
          .reset_index(name='count')
    )
    pivot = (
        grouped.pivot_table(index='incident_hour', columns='cause_group',
                            values='count', aggfunc='sum', fill_value=0)
              .reindex(hours, fill_value=0)
    )
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
        title="Incidents by Hour of Day",
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
        # Auto-create weekday column if missing
        if 'incident_time' in df.columns:
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
        title="Incidents by Day of Week",
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
    reportable_labels = ['Not Reportable', 'Reportable']
    if len(reportable_counts) == 2 and 0 in reportable_counts.index and 1 in reportable_counts.index:
        reportable_counts.index = reportable_labels
    elif True in reportable_counts.index or False in reportable_counts.index:
        reportable_counts.index = ['Reportable' if x else 'Not Reportable' for x in reportable_counts.index]
    fig = px.pie(
        values=reportable_counts.values,
        names=reportable_counts.index,
        title="Reportable Incidents Distribution",
        color_discrete_sequence=['#90EE90', '#FFB6C1']
    )
    st.plotly_chart(fig, use_container_width=True, key="reportable_analysis")

def plot_medical_outcomes(df):
    if df.empty or 'treatment_required' not in df.columns or 'medical_attention_required' not in df.columns:
        st.warning("No data available for medical outcomes")
        return
    medical_summary = {
        'Treatment Required': df['treatment_required'].sum(),
        'Medical Attention Required': df['medical_attention_required'].sum(),
        'No Medical Intervention': len(df) - df[['treatment_required', 'medical_attention_required']].any(axis=1).sum()
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
        title="Incident Severity Distribution",
        color=severity_counts.index,
        color_discrete_map=colors,
        height=400
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=12
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    st.plotly_chart(fig, use_container_width=True, key="severity_dist")

def plot_top_incidents_by_volume_severity(df):
    if df.empty or 'incident_type' not in df.columns or 'severity' not in df.columns:
        st.warning("No data available for top incidents analysis")
        return
    top_incidents = df['incident_type'].value_counts().head(5).index
    filtered_df = df[df['incident_type'].isin(top_incidents)]
    severity_counts = filtered_df.groupby(['incident_type', 'severity']).size().reset_index(name='count')
    fig = px.bar(
        severity_counts,
        x='incident_type',
        y='count',
        color='severity',
        title="Top 5 Incident Types by Volume & Severity",
        labels={'incident_type': 'Incident Type', 'count': 'Number of Incidents'},
        color_discrete_map={'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'},
        height=400
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title="Severity",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key="top_incidents_volume_severity")

def plot_monthly_incidents_by_severity(df):
    if df.empty or 'incident_date' not in df.columns or 'severity' not in df.columns:
        st.warning("No data available for monthly trends")
        return
    df = df.copy()
    df['year_month'] = df['incident_date'].dt.to_period('M').astype(str)
    monthly_severity = df.groupby(['year_month', 'severity']).size().reset_index(name='count')
    fig = px.bar(
        monthly_severity,
        x='year_month',
        y='count',
        color='severity',
        title="Monthly Incidents by Severity",
        labels={'year_month': 'Month', 'count': 'Number of Incidents'},
        color_discrete_map={'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'},
        height=400
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title="Severity",
        showlegend=True,
        xaxis_title="Month",
        yaxis_title="Number of Incidents"
    )
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
        title="Top 10 Incident Types",
        labels={'x': 'Number of Incidents', 'y': 'Incident Type'},
        color=incident_counts.values,
        color_continuous_scale='Viridis',
        height=400
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, key="incident_types_bar")

def plot_location_analysis(df):
    if df.empty or 'location' not in df.columns:
        st.warning("No data available for location analysis")
        return

    location_counts = df['location'].value_counts().head(8)
    vals = location_counts.values

    # Black -> Red continuous scale
    black_red_scale = [
        [0.00, "#000000"],
        [0.15, "#1a0000"],
        [0.30, "#330000"],
        [0.45, "#4d0000"],
        [0.60, "#660000"],
        [0.75, "#800000"],
        [0.90, "#b30000"],
        [1.00, "#ff0000"],
    ]

    fig = px.bar(
        x=location_counts.index,
        y=vals,
        title="Incidents by Location",
        labels={'x': 'Location', 'y': 'Number of Incidents'},
        color=vals,
        color_continuous_scale=black_red_scale,
        range_color=[vals.min(), vals.max()],
        height=400
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        coloraxis_colorbar=dict(title="Color", len=1.5)
    )

    st.plotly_chart(fig, use_container_width=True, key="location_analysis")


def plot_incident_trends(df):
    if df.empty or 'incident_date' not in df.columns:
        st.warning("No data available for incident trends")
        return
    daily_counts = df.groupby(df['incident_date'].dt.date).size().reset_index(name='count')
    daily_counts.columns = ['date', 'incidents']
    fig = px.line(
        daily_counts,
        x='date',
        y='incidents',
        title="Daily Incident Trends",
        markers=True
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Incidents",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True, key="incident_trends")



"""def plot_reporter_performance_scatter(df):
    if df.empty or not {'reported_by','notification_date','incident_date'}.issubset(df.columns):
        st.warning("No data available for reporter performance analysis")
        return
    df = df.copy()
    perf = (
        df.groupby('reported_by')
        .agg(
            avg_delay=('notification_date', lambda x: (x - df.loc[x.index, 'incident_date']).dt.days.mean()),
            total_incidents=('incident_date', 'count')
        ).reset_index()
    )
    fig = px.scatter(
        perf,
        x='avg_delay',
        y='total_incidents',
        color='reported_by',
        size='total_incidents',
        size_max=60,
        labels={
            'avg_delay': 'Average Notification Delay (Days)',
            'total_incidents': 'Total Incidents',
            'reported_by': 'Reporter Type'
        },
        title='Reporter Performance Analysis',
        opacity=0.7
    )
    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='lightblue', griddash='dash')
    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='lightblue', griddash='dash')
    fig.update_traces(marker=dict(line=dict(width=2, color='rgba(0,0,0,0.3)')))
    fig.update_layout(
        legend_title_text='Reporter Type',
        xaxis=dict(zeroline=True, zerolinecolor='lightblue', zerolinewidth=2),
        yaxis=dict(zeroline=True, zerolinecolor='lightblue', zerolinewidth=2),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True, key="reporter_performance_scatter")"""

def plot_serious_injury_age_severity(df):
    if df.empty or 'severity' not in df.columns or 'participant_age' not in df.columns:
        st.info("No high severity incidents found for age analysis")
        return
    serious_df = df[df['severity'].str.lower() == 'high']
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


import streamlit as st
import pandas as pd
import plotly.express as px

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
        rangeslider=dict(visible=True)   # range slider keeps easy zoom on X
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
        uirevision="keep"  # keeps zoom when tweaking filters
    )

    config = {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["lasso2d", "select2d", "resetScale2d"]
    }
    st.plotly_chart(fig, use_container_width=True, key="carer_performance_scatter", config=config)



def plot_serious_injury_age_severity(df):
    if df.empty or 'severity' not in df.columns or 'participant_age' not in df.columns:
        st.info("No high severity incidents found for age analysis")
        return
    serious_df = df[df['severity'].str.lower() == 'high']
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

def display_executive_summary_section(df):
    import calendar

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
    .dashboard-card-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: #222;
    }
    .dashboard-card-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1769aa;
        margin-bottom: 0.3rem;
    }
    .dashboard-card-desc {
        font-size: 0.97rem;
        color: #444;
        margin-bottom: 0.1rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
    }
    .divider {
        margin: 2rem 0 2rem 0;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- CARD DATA ----
    # Top Incident Type
    top_type = df['incident_type'].value_counts().idxmax() if 'incident_type' in df.columns and not df.empty else "N/A"
    # Latest Month Incident
    if 'incident_date' in df.columns and not df.empty:
        latest_month = df['incident_date'].max().to_period('M')
        latest_month_str = latest_month.strftime('%B %Y')
        latest_month_count = df[df['incident_date'].dt.to_period('M') == latest_month].shape[0]
    else:
        latest_month_str = "N/A"
        latest_month_count = 0
    # Previous Month Incident
    if 'incident_date' in df.columns and not df.empty:
        prev_month = latest_month - 1
        prev_month_str = prev_month.strftime('%B %Y')
        prev_month_count = df[df['incident_date'].dt.to_period('M') == prev_month].shape[0]
    else:
        prev_month_str = "N/A"
        prev_month_count = 0
    # High Severity
    high_severity_count = int((df['severity'].str.lower() == 'high').sum()) if 'severity' in df.columns else 0
    # Reportable Incidents
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



# ========== OPERATIONAL PERFORMANCE FUNCTIONS ==========

def display_operational_performance_cards(df):
    """Display operational performance cards with trend indicators"""
    if df.empty or 'incident_date' not in df.columns:
        st.warning("No data available for operational performance cards")
        return
    
    # Calculate current month and previous month data
    current_date = df['incident_date'].max()
    current_month = current_date.to_period('M')
    previous_month = current_month - 1
    
    current_df = df[df['incident_date'].dt.to_period('M') == current_month]
    previous_df = df[df['incident_date'].dt.to_period('M') == previous_month]
    
    st.markdown("### 📈 Operational Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Location Reportable Rate
        if 'location' in df.columns and 'reportable' in df.columns:
            current_reportable_rate = (current_df['reportable'].sum() / len(current_df) * 100) if len(current_df) > 0 else 0
            previous_reportable_rate = (previous_df['reportable'].sum() / len(previous_df) * 100) if len(previous_df) > 0 else 0
            
            trend_pct, trend_arrow = calculate_trend(current_reportable_rate, previous_reportable_rate)
            
            st.metric(
                label="🏢 Location Reportable Rate",
                value=f"{current_reportable_rate:.1f}%",
                delta=f"{trend_arrow} {trend_pct:.1f}%",
                delta_color="inverse",
                help="Percentage of incidents that are reportable by location"
            )
    
    with col2:
        # Average Participant Age
        if 'participant_age' in df.columns:
            current_avg_age = current_df['participant_age'].mean() if len(current_df) > 0 else 0
            previous_avg_age = previous_df['participant_age'].mean() if len(previous_df) > 0 else 0
            
            trend_pct, trend_arrow = calculate_trend(current_avg_age, previous_avg_age)
            
            st.metric(
                label="👥 Average Participant Age",
                value=f"{current_avg_age:.1f} yrs",
                delta=f"{trend_arrow} {trend_pct:.1f}%",
                delta_color="normal",
                help="Average age of participants involved in incidents"
            )
    
    with col3:
        # Medical Attention Rate
        if 'medical_attention_required' in df.columns:
            current_medical_rate = (current_df['medical_attention_required'].sum() / len(current_df) * 100) if len(current_df) > 0 else 0
            previous_medical_rate = (previous_df['medical_attention_required'].sum() / len(previous_df) * 100) if len(previous_df) > 0 else 0
            
            trend_pct, trend_arrow = calculate_trend(current_medical_rate, previous_medical_rate)
            
            st.metric(
                label="🏥 Medical Attention Rate",
                value=f"{current_medical_rate:.1f}%",
                delta=f"{trend_arrow} {trend_pct:.1f}%",
                delta_color="inverse",
                help="Percentage of incidents requiring medical attention"
            )
def display_operational_performance_section(df):
    st.header("📈 Operational Performance & Risk Analysis")
    display_operational_performance_cards(df)
    st.markdown("---")
    plot_reporter_type_metrics(df)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        plot_incident_types_bar(df)
    with col2:
        plot_medical_outcomes(df)
    plot_monthly_incidents_by_severity(df)  # <--- FIXED HERE
    plot_carer_performance_scatter(df)
    plot_serious_injury_age_severity(df)
    
def plot_reporter_type_metrics(df):
    """Display reporter type related metrics"""
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



def plot_serious_injury_age_severity(df):
    """Create serious injury age and severity analysis"""
    if df.empty or 'severity' not in df.columns or 'participant_age' not in df.columns:
        st.info("No high severity incidents found for age analysis")
        return
    serious_df = df[df['severity'].str.lower() == 'high']
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

# ================= INVESTIGATION/COMPLIANCE FUNCTIONS =================

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

def display_compliance_investigation_cards(df):
    if df.empty or 'incident_date' not in df.columns or 'reportable' not in df.columns:
        st.warning("No data available for compliance cards")
        return
    if 'investigation_required' not in df.columns:
        df = apply_investigation_rules(df)
    current_date = df['incident_date'].max()
    current_month = current_date.to_period('M')
    previous_month = current_month - 1
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
        st.metric(
            label="📊 Reportable Incidents",
            value=current_reportable,
            delta=f"{trend_arrow} {abs(change)}",
            delta_color="inverse" if change > 0 else "normal",
            help="Number of incidents that require regulatory reporting"
        )
    with col2:
        current_compliance = int((current_df['report_delay_hours'] <= 24).sum()) if len(current_df) > 0 else 0
        previous_compliance = int((previous_df['report_delay_hours'] <= 24).sum()) if len(previous_df) > 0 else 0
        change = current_compliance - previous_compliance
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric(
            label="⏱️ 24hr Compliance",
            value=current_compliance,
            delta=f"{trend_arrow} {abs(change)}",
            delta_color="normal" if change > 0 else "inverse",
            help="Number of incidents reported within 24 hours"
        )
    with col3:
        current_overdue = int((current_df['report_delay_hours'] > 24).sum()) if len(current_df) > 0 else 0
        previous_overdue = int((previous_df['report_delay_hours'] > 24).sum()) if len(previous_df) > 0 else 0
        change = current_overdue - previous_overdue
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric(
            label="⚠️ Overdue Reports",
            value=current_overdue,
            delta=f"{trend_arrow} {abs(change)}",
            delta_color="inverse" if change > 0 else "normal",
            help="Number of incidents with reporting delays > 24 hours"
        )
    with col4:
        current_total = len(current_df)
        previous_total = len(previous_df)
        current_investigation_rate = (current_df['investigation_required'].sum() / current_total * 100) if current_total > 0 else 0
        previous_investigation_rate = (previous_df['investigation_required'].sum() / previous_total * 100) if previous_total > 0 else 0
        trend_pct, trend_arrow = calculate_trend(current_investigation_rate, previous_investigation_rate)
        st.metric(
            label="🔍 Investigation Rate",
            value=f"{current_investigation_rate:.1f}%",
            delta=f"{trend_arrow} {trend_pct:.1f}%",
            delta_color="inverse",
            help="Percentage of incidents requiring formal investigation"
        )

def plot_compliance_metrics_poly(df):
    if df.empty or 'reportable' not in df.columns or 'incident_date' not in df.columns:
        st.warning("No data available for compliance metrics")
        return
    total = len(df)
    if total == 0:
        st.warning("No data available for compliance metrics")
        return
    if 'investigation_required' not in df.columns:
        df = apply_investigation_rules(df)
    df = df.copy()
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
        plot_metric("24hr Compliance", compliance_24h_count, suffix=f" ({compliance_24h_count/total*100:.1f}%)" if total else "", color_graph="#5AD8A6")
    with col3:
        plot_metric("Overdue Reports", overdue_count, color_graph="#F6BD16")
    with col4:
        plot_metric("Investigation Rate", inv_rate, suffix="%", color_graph="#E86452")
    with col5:
        plot_metric("Investigation Status", inv_status_pct, suffix=inv_status_suffix, color_graph="#6DC8EC")
    with col6:
        plot_metric("Compliance Breach", breach_count, color_graph="#FF2B2B")

def plot_reporting_delay_by_date(df):
    if df.empty or not {'incident_date','notification_date'}.issubset(df.columns):
        st.warning("No data available for reporting delay analysis")
        return
    df = df.copy()
    df['report_delay'] = (df['notification_date'] - df['incident_date']).dt.days
    agg = df.groupby('incident_date').agg(avg_delay=('report_delay', 'mean')).reset_index()
    fig = px.line(
        agg,
        x='incident_date',
        y='avg_delay',
        title="Average Reporting Delay by Incident Date",
        labels={'incident_date': 'Incident Date', 'avg_delay': 'Average Delay (Days)'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="reporting_delay_by_date")

def plot_24h_compliance_rate_by_location(df):
    if df.empty or not {'location','notification_date','incident_date'}.issubset(df.columns):
        st.warning("No data available for compliance rate by location")
        return
    df = df.copy()
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
    if df.empty or 'investigation_required' not in df.columns or 'action_complete' not in df.columns:
        st.warning("No data available for investigation pipeline")
        return
    all_incidents = len(df)
    required = df['investigation_required'].sum()
    complete = df['action_complete'].sum()
    values = [all_incidents, required, complete]
    names = ['All Incidents', 'Required Investigation', 'Action Complete']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig = px.bar(
        x=names,
        y=values,
        title="Investigation Pipeline",
        labels={'x': 'Stage', 'y': 'Count'},
        color=names,
        color_discrete_sequence=colors
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True, key="investigation_pipeline")


def plot_contributing_factors_by_month(df):
    # --- Create necessary columns ---
    # Incident Type Keyword
    if 'Incident Type Keyword' not in df.columns:
        def extract_keyword(s):
            m = re.search(r'for the (.+?) incident', s, re.IGNORECASE)
            if m:
                return m.group(1).strip()
            m2 = re.search(r'for the (.+?) Incident', s, re.IGNORECASE)
            if m2:
                return m2.group(1).strip()
            m3 = re.search(r'for the (.+?) event', s, re.IGNORECASE)
            if m3:
                return m3.group(1).strip()
            words = s.split()
            if len(words) > 2:
                return " ".join(words[-3:-1])
            return s
        df['Incident Type Keyword'] = df['contributing_factors'].apply(extract_keyword)
    # Month-Year
    if 'Month-Year' not in df.columns:
        df['Month-Year'] = pd.to_datetime(df['incident_date'], errors='coerce').dt.strftime('%b %Y')
    # Count
    if 'Count' not in df.columns:
        df['Count'] = 1

    # --- Pivot and plot ---
    heatmap_data = df.pivot_table(
        index='Incident Type Keyword',
        columns='Month-Year',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_ylabel("Incident Type")
    ax.set_xlabel("Month-Year")
    ax.set_title("Contributing Factors by Month-Year")
    st.pyplot(fig)
# ================= PAGE SECTIONS =================

def display_executive_summary_section(df):
    import calendar

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
    .dashboard-card-title {
        font-size: 1 rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: #222;
    }
    .dashboard-card-value {
        font-size: 1.7 rem;
        font-weight: 700;
        color: #1769aa;
        margin-bottom: 0.3rem;
    }
    .dashboard-card-desc {
        font-size: 0.75rem;
        color: #444;
        margin-bottom: 0.1rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
    }
    .divider {
        margin: 2rem 0 2rem 0;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- CARD DATA ----
    # Top Incident Type
    top_type = df['incident_type'].value_counts().idxmax() if 'incident_type' in df.columns and not df.empty else "N/A"
    # Latest Month Incident
    if 'incident_date' in df.columns and not df.empty:
        latest_month = df['incident_date'].max().to_period('M')
        latest_month_str = latest_month.strftime('%B %Y')
        latest_month_count = df[df['incident_date'].dt.to_period('M') == latest_month].shape[0]
    else:
        latest_month_str = "N/A"
        latest_month_count = 0
    # Previous Month Incident
    if 'incident_date' in df.columns and not df.empty:
        prev_month = latest_month - 1
        prev_month_str = prev_month.strftime('%B %Y')
        prev_month_count = df[df['incident_date'].dt.to_period('M') == prev_month].shape[0]
    else:
        prev_month_str = "N/A"
        prev_month_count = 0
    # High Severity
    high_severity_count = int((df['severity'].str.lower() == 'high').sum()) if 'severity' in df.columns else 0
    # Reportable Incidents
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

    # --- Your other summary plots/sections below as needed ---
    st.markdown('<div class="section-title">Severity Distribution</div>', unsafe_allow_html=True)
    plot_severity_distribution(df)

    st.markdown('<div class="section-title">Top 10 Incident Types</div>', unsafe_allow_html=True)
    plot_incident_types_bar(df)

    st.markdown('<div class="section-title">Location Analysis</div>', unsafe_allow_html=True)
    plot_location_analysis(df)

    st.markdown('<div class="section-title">Monthly Trends</div>', unsafe_allow_html=True)
    plot_monthly_incidents_by_severity(df)

    st.markdown('<div class="section-title">Medical Outcomes</div>', unsafe_allow_html=True)
    plot_medical_outcomes(df)

    st.markdown('<div class="section-title">Daily Incident Trends</div>', unsafe_allow_html=True)
    plot_incident_trends(df)

    st.markdown('<div class="section-title">Incidents by Day of Week</div>', unsafe_allow_html=True)
    plot_weekday_analysis(df)

    st.markdown('<div class="section-title">Incidents by Hour of Day</div>', unsafe_allow_html=True)
    plot_time_analysis(df)

    st.markdown('<div class="section-title">Reportable Analysis</div>', unsafe_allow_html=True)
    plot_reportable_analysis(df)

    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

def display_operational_performance_section(df):
    st.header(" Operational Performance & Risk Analysis Metrics")
    st.markdown("---")

    # Calculate Average Participant Age
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        today = pd.to_datetime('today')
        df['participant_age'] = ((today - df['dob']).dt.days // 365).astype('float')
        avg_age = df['participant_age'].mean()
        avg_age_txt = f"{avg_age:.1f} yrs" if pd.notnull(avg_age) else "N/A"
    else:
        avg_age_txt = "N/A"
    
    # Calculate Location Reportable Rate
    if 'reportable' in df.columns and len(df) > 0:
        location_reportable_rate = 100 * df['reportable'].sum() / len(df)
    else:
        location_reportable_rate = 0.0

    # Calculate Medical Attention Rate
    if 'medical_attention_required' in df.columns and len(df) > 0:
        medical_attention_rate = 100 * df['medical_attention_required'].sum() / len(df)
    else:
        medical_attention_rate = 0.0

    # Medical Attention Required Count
    medical_attention_required = (
        int(df['medical_attention_required'].sum())
        if 'medical_attention_required' in df.columns
        else 0
    )

    # Display cards in one row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  Location Reportable Rate
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#1769aa;">
                  {location_reportable_rate:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  Medical Attention Rate
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#d9534f;">
                  {medical_attention_rate:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  Medical Attention Required
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#f0ad4e;">
                  {medical_attention_required}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px;
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  Average Participant Age
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#5ad8a6;">
                  {avg_age_txt}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        plot_incident_types_bar(df)
    with col2:
        plot_medical_outcomes(df)
    plot_monthly_incidents_by_severity(df)  # <--- FIXED HERE!
    plot_carer_performance_scatter(df)
    plot_serious_injury_age_severity(df)

def display_compliance_investigation_section(df):
    st.header("Compliance & Investigation Metrics")
    st.markdown("---")

    # Compute reporting delay in hours
    if 'incident_date' in df.columns and 'notification_date' in df.columns:
        df = df.copy()
        df['report_delay_hours'] = (
            (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
        )
        compliance_24hr = (df['report_delay_hours'] <= 24).sum()
        overdue_reports = (df['report_delay_hours'] > 24).sum()
    else:
        compliance_24hr = 0
        overdue_reports = 0

    # Reportable incidents
    reportable_incidents = int(df['reportable'].sum()) if 'reportable' in df.columns else 0

    # Investigation rate: percent of incidents completed for investigation, if available
    if 'investigation_completed' in df.columns and len(df) > 0:
        investigation_rate = 100 * df['investigation_completed'].sum() / len(df)
    else:
        investigation_rate = 0.0

    # Display all cards in one row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  Reportable Incidents
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#1769aa;">
                  {reportable_incidents}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  24hr Compliance
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#5ad8a6;">
                  {compliance_24hr}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  Overdue Reports
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#d9534f;">
                  {overdue_reports}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e3e3e3;border-radius:14px; 
                        padding:1.2rem 0.5rem;text-align:center;min-height:120px;">
                <span style="font-size:1rem;font-weight:600;color:#222;">
                  Investigation Rate
                </span><br>
                <span style="font-size:2rem;font-weight:700;color:#f0ad4e;">
                  {investigation_rate:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
 
    col1, col2 = st.columns(2)
    with col1:
        plot_reporting_delay_by_date(df)
    with col2:
        plot_24h_compliance_rate_by_location(df)
    plot_investigation_pipeline(df)
    plot_contributing_factors_by_month(df)


from ml_helpers import (
    compare_models,                 # Returns (metrics_df, roc_fig)
    forecast_incident_volume,       # Returns (actual, forecast)
    profile_location_risk,          # Returns (loc_df, loc_fig)
    profile_incident_type_risk,     # Returns (type_df, type_fig)
    detect_seasonal_patterns,       # Returns pattern_fig
    perform_clustering_analysis,    # Returns (clustered, features, sil_score, pca)
    plot_3d_clusters,               # Returns fig3d
    plot_correlation_heatmap,       # Returns corr_fig
    train_severity_prediction_model,
    perform_anomaly_detection,
    analyze_cluster_characteristics
)

def display_ml_insights_section(df):
    st.header("🧠 Advanced ML Insights Dashboard")
    tabs = st.tabs([
        "Predictive Models",
        "Forecasting",
        "Risk Analysis",
        "Pattern Detection",
        "Clustering Analysis",
        "Correlations"
    ])

    with tabs[0]:
        st.subheader("Predictive Models Comparison")
        metrics_df, roc_fig = compare_models(df)
        st.dataframe(metrics_df)
        st.plotly_chart(roc_fig)
        st.subheader("Severity Prediction Model")
        model, acc, features = train_severity_prediction_model(df)
        if model is not None and features is not None:
            st.write(f"Model accuracy: {acc:.2%}")
            st.write(f"Features used: {features}")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)
                fig = px.bar(
                    importance_df, x="Feature", y="Importance",
                    title="Feature Importances"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to train severity prediction model.")

    with tabs[1]:
        st.subheader("Incident Volume Forecasting")
        actual, forecast = forecast_incident_volume(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual.index.astype(str), y=actual.values, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast.index.astype(str), y=forecast.values, mode='lines', name='Forecast'))
        st.plotly_chart(fig)

    with tabs[2]:
        st.subheader("Location Risk Profile")
        loc_df, loc_fig = profile_location_risk(df)
        st.dataframe(loc_df)
        st.plotly_chart(loc_fig)
        st.subheader("Incident Type Risk Profile")
        type_df, type_fig = profile_incident_type_risk(df)
        st.dataframe(type_df)
        st.plotly_chart(type_fig)

    with tabs[3]:
        st.subheader("Seasonal & Temporal Pattern Detection")
        pattern_fig = detect_seasonal_patterns(df)
        st.plotly_chart(pattern_fig)
        st.subheader("Anomaly Detection (Isolation Forest & SVM)")
        out, features = perform_anomaly_detection(df)
        if out is not None and features is not None:
            try:
                if "pca_x" not in out.columns or "pca_y" not in out.columns:
                    from sklearn.decomposition import PCA
                    X = out[features]
                    if X.shape[1] >= 2:
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X)
                        out['pca_x'], out['pca_y'] = X_pca[:, 0], X_pca[:, 1]
            except Exception as e:
                st.info("Skipping PCA visualization: " + str(e))

            st.dataframe(out[['incident_date', 'location', 'incident_type', 'isolation_forest_anomaly', 'svm_anomaly', 'anomaly_score']].head(20))

            if "pca_x" in out.columns and "pca_y" in out.columns:
                fig = px.scatter(
                    out,
                    x='pca_x',
                    y='pca_y',
                    color=out['isolation_forest_anomaly'].map({True: "Anomaly", False: "Normal"}),
                    symbol=out['svm_anomaly'].map({True: "Anomaly", False: "Normal"}),
                    title="Isolation Forest & SVM Anomalies (PCA View)",
                    hover_data=["incident_date", "location", "incident_type", "severity"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for anomaly detection.")

    with tabs[4]:
        st.subheader("Clustering Analysis (2D)")
        clustered, features, sil_score, pca = perform_clustering_analysis(df)
        if clustered is not None and features is not None:
            fig2d = px.scatter(
                clustered, x="pca_x", y="pca_y", color=clustered['cluster'].astype(str),
                hover_data=["incident_date", "location", "incident_type", "severity"],
                title="Incident Clusters (2D PCA View)"
            )
            st.plotly_chart(fig2d)
            st.write(f"Silhouette Score: {sil_score}")
            st.subheader("Clustering Analysis (3D)")
            fig3d = plot_3d_clusters(clustered)
            st.plotly_chart(fig3d)
            st.subheader("Cluster Characteristics")
            cluster_info = analyze_cluster_characteristics(clustered)
            if cluster_info:
                st.write(pd.DataFrame(cluster_info).T)
        else:
            st.warning("Not enough data for clustering.")

    with tabs[5]:
        st.subheader("Feature Correlation Analysis")
        corr_fig = plot_correlation_heatmap(df)
        st.pyplot(corr_fig)  # Use st.plotly_chart if you return a plotly figure




# --- Executive Summary ---
def display_executive_summary_section(df):
    st.header("📊 Executive Summary")
    st.markdown("#### Quick Data Overview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Total incidents: {len(df)}")
    if "incident_date" in df.columns:
        st.write(f"Date range: {df['incident_date'].min().date()} to {df['incident_date'].max().date()}")
    if "location" in df.columns:
        st.write(f"Unique locations: {df['location'].nunique()}")
    if "incident_type" in df.columns:
        st.write(f"Incident types: {df['incident_type'].nunique()}")
    if "severity" in df.columns:
        st.write(f"Severity levels: {df['severity'].unique()}")

# --- Operational Performance ---
def display_operational_performance_section(df):
    st.header("📈 Operational Performance & Risk Analysis")
    st.markdown("#### Incident Volume by Location")
    if "location" in df.columns:
        location_counts = df["location"].value_counts().head(15)
        st.bar_chart(location_counts)
    st.markdown("#### Incident Volume by Type")
    if "incident_type" in df.columns:
        type_counts = df["incident_type"].value_counts().head(15)
        st.bar_chart(type_counts)
    st.markdown("#### Severity Distribution")
    if "severity" in df.columns:
        severity_counts = df["severity"].value_counts()
        st.bar_chart(severity_counts)
    st.markdown("#### Monthly Trends")
    if "incident_date" in df.columns:
        monthly_counts = df.groupby(df["incident_date"].dt.to_period("M")).size()
        monthly_counts.index = monthly_counts.index.to_timestamp()
        st.line_chart(monthly_counts)

# --- Compliance & Investigation ---
def display_compliance_investigation_section(df):
    st.header("📋 Compliance & Investigation")
    st.markdown("#### Reportable Incidents")
    if "reportable" in df.columns:
        reportable_counts = df["reportable"].value_counts()
        st.bar_chart(reportable_counts)
        st.write(f"Total reportable incidents: {df['reportable'].sum()}")
    st.markdown("#### Incidents Requiring Medical Attention")
    if "medical_attention_required" in df.columns:
        med_counts = df["medical_attention_required"].value_counts()
        st.bar_chart(med_counts)
        st.write(f"Total needing medical attention: {df['medical_attention_required'].sum()}")
    st.markdown("#### High Severity Investigations")
    if "severity" in df.columns:
        high_sev = df[df["severity"].str.lower() == "high"]
        st.write(f"High severity incidents: {len(high_sev)}")
        st.dataframe(high_sev.head(10), use_container_width=True)

# --- ML Insights ---
def display_ml_insights_section(df):
    st.header("🤖 ML Insights")
    st.markdown("#### ML-Driven Insights")
    st.info("For advanced ML analytics, use the 🧠 Advanced ML Analytics tab in the dashboard sidebar.")
    st.markdown("You can explore:")
    st.markdown("- Location risk profiling")
    st.markdown("- Incident type risk")
    st.markdown("- Temporal and seasonal patterns")
    st.markdown("- Severity prediction models")
    st.markdown("- Anomaly detection")
    st.markdown("- Clustering analysis")
    st.markdown("- Correlation matrix")
    st.markdown("- Incident forecasting")
    st.markdown("Use available filters to customize your ML analysis.")

# --- Investigation Rules ---
def apply_investigation_rules(df):
    # Example: Flag incidents for investigation if high severity or reportable or medical attention required
    df = df.copy()
    df["needs_investigation"] = False
    if "severity" in df.columns:
        df.loc[df["severity"].str.lower() == "high", "needs_investigation"] = True
    if "reportable" in df.columns:
        df.loc[df["reportable"] == 1, "needs_investigation"] = True
    if "medical_attention_required" in df.columns:
        df.loc[df["medical_attention_required"] == 1, "needs_investigation"] = True
    return df
