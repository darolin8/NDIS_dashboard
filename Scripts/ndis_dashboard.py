
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# Page config
st.set_page_config(
    page_title="Advanced NDIS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .insight-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: #1a1a1a;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .alert-box {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-info {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left-color: #2196f3;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def create_sample_ndis_data():
    """Create sample NDIS incident data matching your CSV structure"""
    np.random.seed(42)
    n_records = 100
    
    # Sample participants
    participants = [f'Participant_{i:03d}' for i in range(1, 31)]
    
    # NDIS numbers
    ndis_numbers = [np.random.randint(10000000, 99999999) for _ in range(n_records)]
    
    # Date ranges
    dobs = pd.date_range('1960-01-01', '2005-12-31', periods=30)
    incident_dates = pd.date_range('2023-01-01', '2024-12-31', periods=n_records)
    
    # Notification dates
    notification_delays = np.random.choice([0, 1, 2, 3, 4, 5], n_records, p=[0.4, 0.3, 0.15, 0.08, 0.05, 0.02])
    notification_dates = incident_dates + pd.to_timedelta(notification_delays, unit='days')
    
    # Times with proper probability distribution
    hour_probs = [0.02]*6 + [0.06]*12 + [0.04]*6
    hour_probs = np.array(hour_probs) / np.sum(hour_probs)
    
    incident_times = [f"{h:02d}:{m:02d}" for h, m in zip(
        np.random.choice(24, n_records, p=hour_probs), 
        np.random.randint(0, 60, n_records)
    )]
    
    # Locations and types
    locations = ['Main Office', 'Community Center', 'Residential Care', 'Day Program', 'Supported Living']
    incident_types = ['Fall', 'Medication Error', 'Behavioral Incident', 'Property Damage', 'Injury', 'Verbal Aggression']
    severity_levels = ['Low', 'Medium', 'High', 'Critical']
    severity_weights = [0.4, 0.35, 0.2, 0.05]
    
    # Create the dataset
    data = []
    for i in range(n_records):
        participant = np.random.choice(participants)
        incident_type = np.random.choice(incident_types)
        severity = np.random.choice(severity_levels, p=severity_weights)
        
        # Determine injury and medical requirements
        injury_likely = incident_type in ['Fall', 'Injury'] or np.random.random() < 0.3
        if injury_likely:
            injury_type = np.random.choice(['Bruise', 'Cut', 'Sprain', 'Fracture'], p=[0.5, 0.3, 0.15, 0.05])
            injury_severity = np.random.choice(['Minor', 'Moderate', 'Major'], p=[0.7, 0.25, 0.05])
            medical_required = 'Yes' if injury_severity in ['Moderate', 'Major'] else np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
        else:
            injury_type = 'None'
            injury_severity = 'None'
            medical_required = 'No'
        
        # Reportable incidents
        reportable = 'Yes' if (severity == 'Critical' or injury_severity == 'Major' or 
                             incident_type == 'Medication Error') else np.random.choice(['Yes', 'No'], p=[0.2, 0.8])
        
        record = {
            'incident_id': f'INC{i+1:06d}',
            'participant_name': participant,
            'ndis_number': ndis_numbers[i],
            'dob': dobs[i % len(dobs)].strftime('%d/%m/%Y'),
            'incident_date': incident_dates[i].strftime('%d/%m/%Y'),
            'incident_time': incident_times[i],
            'notification_date': notification_dates[i].strftime('%d/%m/%Y'),
            'location': np.random.choice(locations),
            'incident_type': incident_type,
            'subcategory': f'{incident_type} - Type {np.random.randint(1,4)}',
            'severity': severity,
            'reportable': reportable,
            'description': f'{incident_type} incident involving {participant}.',
            'immediate_action': f'{np.random.choice(["First aid provided", "Supervisor notified", "Medical consultation"])}.',
            'actions_taken': f'{np.random.choice(["Staff debriefing", "Risk assessment", "Training scheduled"])}.',
            'contributing_factors': np.random.choice([
                'Environmental factors, Staff shortage',
                'Equipment failure',
                'Communication breakdown',
                'Participant condition'
            ]),
            'reported_by': np.random.choice(['Support Worker', 'Team Leader', 'Nurse', 'Manager']),
            'injury_type': injury_type,
            'injury_severity': injury_severity,
            'treatment_required': 'Yes' if medical_required == 'Yes' else 'No',
            'medical_attention_required': medical_required,
            'medical_treatment_type': 'First Aid' if medical_required == 'Yes' else 'None',
            'medical_outcome': 'Recovery' if medical_required == 'Yes' else 'No Treatment Required'
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    return process_data(df)

def process_data(df):
    """Process and enhance the NDIS incident data"""
    try:
        df = df.copy()
        
        # Process dates
        date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
        
        for date_col in ['incident_date', 'notification_date', 'dob']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
                
                if df[date_col].isna().any():
                    for fmt in date_formats:
                        try:
                            mask = df[date_col].isna()
                            df.loc[mask, date_col] = pd.to_datetime(df.loc[mask, date_col], format=fmt, errors='coerce')
                            if not df[date_col].isna().all():
                                break
                        except:
                            continue
        
        # Calculate age if DOB available
        if 'dob' in df.columns:
            df['age'] = ((df['incident_date'] - df['dob']).dt.days / 365.25).round().clip(lower=0, upper=120)
        else:
            df['age'] = 40
        
        # Calculate notification delay
        if 'notification_date' in df.columns:
            df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.days.fillna(0).clip(lower=0)
        else:
            df['notification_delay'] = 0
        
        # Time-based features
        df['month'] = df['incident_date'].dt.month_name()
        df['day_of_week'] = df['incident_date'].dt.day_name()
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
        df['year'] = df['incident_date'].dt.year
        
        # Extract hour from time
        if 'incident_time' in df.columns:
            df['hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour.fillna(12).astype(int)
        else:
            df['hour'] = 12
        
        # Risk scoring
        severity_weights = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df['severity_score'] = df['severity'].map(severity_weights).fillna(1)
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 50, 65, 100], 
                                labels=['18-25', '26-35', '36-50', '51-65', '65+'],
                                include_lowest=True)
        
        # Participant analysis
        if 'participant_name' in df.columns:
            participant_history = df.groupby('participant_name').agg({
                'incident_id': 'count',
                'severity_score': 'mean',
                'incident_date': ['min', 'max']
            }).round(2)
            
            participant_history.columns = ['incident_count', 'avg_severity', 'first_incident', 'last_incident']
            df = df.merge(participant_history, left_on='participant_name', right_index=True, how='left')
            
            df['participant_risk_level'] = pd.cut(
                df['incident_count'], 
                bins=[0, 1, 3, 5, float('inf')], 
                labels=['New', 'Low', 'Medium', 'High'],
                include_lowest=True
            )
        else:
            df['incident_count'] = 1
            df['avg_severity'] = df['severity_score']
            df['participant_risk_level'] = 'New'
        
        # Medical attention
        if 'medical_attention_required' in df.columns:
            df['medical_attention'] = df['medical_attention_required']
        else:
            df['medical_attention'] = 'No'
        
        # Time period
        df['time_period'] = df['hour'].apply(lambda x: 
            'Morning' if 6 <= x < 12 else
            'Afternoon' if 12 <= x < 18 else
            'Evening' if 18 <= x < 22 else
            'Night'
        )
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def load_data_from_file(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format")
            return None
        
        if df.empty:
            st.error("❌ File is empty")
            return None
        
        # Validate required columns
        required_cols = ['incident_date', 'incident_type', 'severity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            return None
        
        return process_data(df)
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

# Enhanced sidebar
st.sidebar.image("https://via.placeholder.com/300x100/667eea/ffffff?text=NDIS+Analytics", use_container_width=True)
st.sidebar.markdown("---")

# Data Source Selection
st.sidebar.subheader("📁 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["📊 Use Sample NDIS Data", "📤 Upload Your File"]
)

# Load data
df = None

if data_source == "📊 Use Sample NDIS Data":
    st.sidebar.success("✅ Using sample NDIS data")
    try:
        df = create_sample_ndis_data()
        if df is not None:
            st.session_state.data_loaded = True
            st.sidebar.info(f"📊 Sample Data: {len(df)} incidents")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

else:  # Upload file
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", 
        type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            df = load_data_from_file(uploaded_file)
            if df is not None:
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ File loaded: {len(df)} incidents")

# Only proceed if data is loaded
if df is None or len(df) == 0:
    st.title("🏥 Advanced NDIS Incident Analytics Dashboard")
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;">
        <h2>📊 Welcome to NDIS Analytics</h2>
        <p style="font-size: 1.2em; margin: 1rem 0;">Select sample data or upload your incident data to begin</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Sample Data Features")
        st.markdown("""
        - 100 realistic NDIS incidents
        - All 23 columns from your CSV structure
        - 2023-2024 date range
        - Proper NDIS numbers and medical data
        - Ready for immediate analysis
        """)
    
    with col2:
        st.markdown("### 📤 Upload Requirements")
        st.markdown("""
        **Required columns:**
        - incident_date, incident_type, severity
        
        **Supported formats:**
        - CSV (.csv) files
        - Excel (.xlsx, .xls) files
        """)
    
    st.stop()

# Time controls with All Time as default
st.sidebar.subheader("📅 Time Controls")
time_options = {
    "📅 All Time": None,
    "📅 Last 30 Days": 30,
    "📅 Last Quarter": 90, 
    "📅 Last 6 Months": 180,
    "📅 Last Year": 365
}

selected_range = st.sidebar.selectbox("Time Period", list(time_options.keys()))

# Apply time filter
if time_options[selected_range] is not None:
    end_date = df['incident_date'].max()
    start_date = end_date - timedelta(days=time_options[selected_range])
    df_filtered = df[(df['incident_date'] >= start_date) & (df['incident_date'] <= end_date)]
else:
    df_filtered = df.copy()

st.sidebar.write(f"Showing {len(df_filtered)} of {len(df)} records")

# Risk focus filter
st.sidebar.subheader("🎯 Focus")
risk_options = {
    "🔍 All Incidents": "all",
    "🚨 Critical Only": "critical",
    "⚠️ High Risk": "high_risk"
}

risk_focus = st.sidebar.selectbox("Risk Focus", list(risk_options.keys()))

if risk_options[risk_focus] == "critical":
    df_filtered = df_filtered[df_filtered['severity'] == 'Critical']
elif risk_options[risk_focus] == "high_risk":
    df_filtered = df_filtered[df_filtered['severity'].isin(['High', 'Critical'])]

# Severity and location filters
severity_options = sorted(df['severity'].unique())
location_options = sorted(df['location'].unique())

col1, col2 = st.sidebar.columns(2)
with col1:
    severity_filter = st.multiselect("⚠️ Severity", severity_options, default=severity_options)
with col2:
    location_filter = st.multiselect("📍 Location", location_options, default=location_options)

# Apply filters
df_filtered = df_filtered[
    (df_filtered['severity'].isin(severity_filter)) &
    (df_filtered['location'].isin(location_filter))
]

# Reset button
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

# Main dashboard
st.title("🏥 Advanced NDIS Incident Analytics Dashboard")

# Status bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success(f"✅ {len(df_filtered)} incidents")
with col2:
    if len(df_filtered) > 0:
        date_range = f"{df_filtered['incident_date'].min().strftime('%b %Y')} - {df_filtered['incident_date'].max().strftime('%b %Y')}"
        st.info(f"📅 {date_range}")
with col3:
    compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
    st.info(f"📋 {compliance_rate:.1f}% compliant")
with col4:
    critical_count = len(df_filtered[df_filtered['severity'] == 'Critical'])
    if critical_count == 0:
        st.success("✅ No critical")
    else:
        st.error(f"🚨 {critical_count} critical")

# KPIs
st.subheader("📊 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_incidents = len(df_filtered)
    st.metric("📊 Total Incidents", total_incidents)

with col2:
    critical_count = len(df_filtered[df_filtered['severity'] == 'Critical'])
    st.metric("🚨 Critical", critical_count)

with col3:
    avg_delay = df_filtered['notification_delay'].mean()
    st.metric("⏱️ Avg Delay (days)", f"{avg_delay:.1f}")

with col4:
    medical_count = len(df_filtered[df_filtered['medical_attention'] == 'Yes'])
    st.metric("🏥 Medical Attention", medical_count)

with col5:
    reportable_count = len(df_filtered[df_filtered['reportable'] == 'Yes'])
    st.metric("📋 Reportable", reportable_count)

# Visualizations
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Monthly Trend")
    if len(df_filtered) > 0:
        monthly_data = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).size()
        monthly_data.index = monthly_data.index.astype(str)
        
        if len(monthly_data) > 0:
            fig_trend = px.line(
                x=monthly_data.index,
                y=monthly_data.values,
                title="Incidents Over Time",
                markers=True
            )
            fig_trend.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Incidents"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("⚠️ Severity Distribution")
    severity_counts = df_filtered['severity'].value_counts()
    
    colors = {'Critical': '#DC143C', 'High': '#FF6347', 'Medium': '#FFA500', 'Low': '#32CD32'}
    fig_severity = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Severity Breakdown",
        color=severity_counts.index,
        color_discrete_map=colors
    )
    st.plotly_chart(fig_severity, use_container_width=True)

# Location analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Incidents by Location")
    location_counts = df_filtered['location'].value_counts()
    
    fig_location = px.bar(
        x=location_counts.values,
        y=location_counts.index,
        orientation='h',
        title="Incidents by Location"
    )
    st.plotly_chart(fig_location, use_container_width=True)

with col2:
    st.subheader("🕐 Hourly Pattern")
    hourly_data = df_filtered.groupby('hour').size()
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    hourly_counts = [hourly_data.get(h, 0) for h in range(24)]
    
    fig_hourly = px.bar(
        x=hour_labels,
        y=hourly_counts,
        title="Incidents by Hour"
    )
    fig_hourly.update_xaxes(tickangle=45)
    st.plotly_chart(fig_hourly, use_container_width=True)

# Data table
st.subheader("📋 Incident Data")

# Column selection
available_columns = ['incident_id', 'participant_name', 'incident_date', 'incident_type', 
                    'severity', 'location', 'reportable', 'medical_attention_required']
display_columns = [col for col in available_columns if col in df_filtered.columns]

selected_columns = st.multiselect(
    "Select columns to display",
    options=display_columns,
    default=display_columns[:6]
)

if selected_columns:
    display_df = df_filtered[selected_columns].copy()
    
    # Sort by date if available
    if 'incident_date' in selected_columns:
        display_df = display_df.sort_values('incident_date', ascending=False)
    
    st.dataframe(display_df, use_container_width=True, height=400)

# Export
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export CSV"):
        csv_data = df_filtered.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_data,
            f"ndis_incidents_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

with col2:
    if st.button("📊 Generate Report"):
        st.subheader("📈 Executive Summary")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Incidents", len(df_filtered))
            st.metric("Critical Incidents", len(df_filtered[df_filtered['severity'] == 'Critical']))
        with col_b:
            st.metric("Avg Delay", f"{df_filtered['notification_delay'].mean():.1f} days")
            st.metric("Compliance Rate", f"{(df_filtered['notification_delay'] <= 1).mean() * 100:.1f}%")
        with col_c:
            st.metric("Medical Attention", len(df_filtered[df_filtered['medical_attention'] == 'Yes']))
            st.metric("Reportable", len(df_filtered[df_filtered['reportable'] == 'Yes']))

with col3:
    if st.button("🔄 Refresh"):
        st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown(f"**Records displayed:** {len(df_filtered)} of {len(df)}")
with col3:
    st.markdown("**Status:** ✅ Dashboard Active")
