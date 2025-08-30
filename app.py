"""
Incident Management Dashboard
Run with `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dashboard_pages import (
    plot_metric,
    plot_gauge,
    plot_incident_trends,
    plot_severity_distribution,
    plot_location_analysis,
    plot_incident_types_bar,
    plot_monthly_trends,
    plot_medical_outcomes
)
#######################################
# PAGE SETUP
#######################################

st.set_page_config(
    page_title="Incident Management Dashboard", 
    page_icon="🚨", 
    layout="wide"
)

st.title("🚨 Incident Management Dashboard")
st.markdown("_Real-time incident tracking and analysis_")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    # Date range filter
    st.subheader("Filters")

#######################################
# DATA LOADING
#######################################

RAW_CSV_URL = "https://raw.githubusercontent.com/darolin8/NDIS_dashboard/main/text%20data/ndis_incidents_1000.csv"

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess incident data"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(RAW_CSV_URL)
    
    # Convert date columns
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
    
    # Extract additional date features
    df['incident_year'] = df['incident_date'].dt.year
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_month_name'] = df['incident_date'].dt.strftime('%B')
    df['incident_weekday'] = df['incident_date'].dt.day_name()
    
    # Clean boolean columns
    for col in ['reportable', 'treatment_required', 'medical_attention_required']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df[col] = False
    
    return df

df = load_data(uploaded_file)

if df is None or df.empty or df['incident_date'].isnull().all():
    st.info("No data available. Please upload a valid CSV.", icon="ℹ️")
    st.stop()

# Sidebar filters
with st.sidebar:
    # Date range
    min_date = df['incident_date'].min().date()
    max_date = df['incident_date'].max().date()
    
    start_date = st.date_input("Start Date", min_date)
    end_date = st.date_input("End Date", max_date)
    
    # Severity filter
    severities = ['All'] + sorted([s for s in df['severity'].dropna().unique()])
    selected_severity = st.selectbox("Severity Level", severities)
    
    # Location filter
    locations = ['All'] + sorted([l for l in df['location'].dropna().unique()])
    selected_location = st.selectbox("Location", locations)
    
    # Incident type filter
    incident_types = ['All'] + sorted([i for i in df['incident_type'].dropna().unique()])
    selected_type = st.selectbox("Incident Type", incident_types)

# Apply filters
filtered_df = df[
    (df['incident_date'].dt.date >= start_date) & 
    (df['incident_date'].dt.date <= end_date)
]

if selected_severity != 'All':
    filtered_df = filtered_df[filtered_df['severity'] == selected_severity]
    
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
    
if selected_type != 'All':
    filtered_df = filtered_df[filtered_df['incident_type'] == selected_type]

with st.expander("📊 Data Preview"):
    st.dataframe(
        filtered_df.head(100),
        column_config={
            "incident_date": st.column_config.DateColumn("Incident Date"),
            "notification_date": st.column_config.DateColumn("Notification Date"),
            "reportable": st.column_config.CheckboxColumn("Reportable"),
            "treatment_required": st.column_config.CheckboxColumn("Treatment Required"),
            "medical_attention_required": st.column_config.CheckboxColumn("Medical Attention Required"),
        },
    )

#######################################
# METRICS CALCULATIONS
#######################################

total_incidents = len(filtered_df)
high_severity_count = len(filtered_df[filtered_df['severity'] == 'High'])
reportable_count = len(filtered_df[filtered_df['reportable'] == True])
medical_attention_count = len(filtered_df[filtered_df['medical_attention_required'] == True])

# Calculate percentages
high_severity_pct = (high_severity_count / total_incidents * 100) if total_incidents > 0 else 0
reportable_pct = (reportable_count / total_incidents * 100) if total_incidents > 0 else 0
medical_pct = (medical_attention_count / total_incidents * 100) if total_incidents > 0 else 0

# Recent trends (last 30 days)
thirty_days_ago = datetime.now() - timedelta(days=30)
recent_incidents = filtered_df[filtered_df['incident_date'] >= thirty_days_ago]
recent_count = len(recent_incidents)

#######################################
# DASHBOARD LAYOUT
#######################################

# Top metrics row
top_left_column, top_right_column = st.columns((3, 1))

with top_left_column:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        plot_metric(
            "Total Incidents",
            total_incidents,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="rgba(0, 104, 201, 0.2)",
        )
        plot_gauge(high_severity_pct, "#FF2B2B", "%", "High Severity", 100)
    
    with col2:
        plot_metric(
            "High Severity",
            high_severity_count,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="rgba(255, 43, 43, 0.2)",
        )
        plot_gauge(reportable_pct, "#FF8700", "%", "Reportable", 100)
    
    with col3:
        plot_metric(
            "Reportable Cases",
            reportable_count,
            prefix="",
            suffix="",
            show_graph=False
        )
        plot_gauge(medical_pct, "#29B09D", "%", "Medical Attention", 100)
    
    with col4:
        plot_metric(
            "Recent (30 days)",
            recent_count,
            prefix="",
            suffix="",
            show_graph=False
        )
        # Average response time gauge (mock data for now)
        avg_response_hours = 2.4
        plot_gauge(avg_response_hours, "#0068C9", "hrs", "Avg Response", 24)

with top_right_column:
    plot_severity_distribution(filtered_df)

# Middle row - charts
middle_left_column, middle_right_column = st.columns(2)

with middle_left_column:
    plot_incident_types_bar(filtered_df)

with middle_right_column:
    plot_location_analysis(filtered_df)

# Bottom row - trends and medical outcomes
bottom_left_column, bottom_right_column = st.columns(2)

with bottom_left_column:
    plot_monthly_trends(filtered_df)

with bottom_right_column:
    plot_medical_outcomes(filtered_df)

#######################################
# ADDITIONAL INSIGHTS
#######################################

st.markdown("---")
st.subheader("📈 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Most Common Incident Type",
        value=filtered_df['incident_type'].mode().iloc[0] if len(filtered_df) > 0 else "N/A",
        delta=f"{len(filtered_df[filtered_df['incident_type'] == filtered_df['incident_type'].mode().iloc[0]])} cases" if len(filtered_df) > 0 else "0 cases"
    )

with col2:
    st.metric(
        label="Highest Risk Location",
        value=filtered_df['location'].mode().iloc[0] if len(filtered_df) > 0 else "N/A",
        delta=f"{len(filtered_df[filtered_df['location'] == filtered_df['location'].mode().iloc[0]])} incidents" if len(filtered_df) > 0 else "0 incidents"
    )

with col3:
    # Calculate average days between incident and notification
    if len(filtered_df) > 0:
        avg_notification_delay = (filtered_df['notification_date'] - filtered_df['incident_date']).dt.days.mean()
        st.metric(
            label="Avg Notification Delay",
            value=f"{avg_notification_delay:.1f} days",
            delta="Target: Same day"
        )
    else:
        st.metric(label="Avg Notification Delay", value="N/A", delta="No data")

#######################################
# EXPORT FUNCTIONALITY
#######################################

st.markdown("---")
st.subheader("📤 Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Generate Summary Report"):
        summary_stats = {
            'Total Incidents': total_incidents,
            'High Severity': high_severity_count,
            'Reportable Cases': reportable_count,
            'Medical Attention Required': medical_attention_count,
            'Date Range': f"{start_date} to {end_date}"
        }
        st.json(summary_stats)

with col2:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name=f'filtered_incidents_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )
