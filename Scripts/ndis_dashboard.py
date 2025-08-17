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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .correlation-card {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
        animation: fadeIn 0.5s;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .alert-warning {
        background-color: #fff8e1;
        border-left-color: #ff9800;
        color: #ef6c00;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #1565c0;
    }
    .alert-success {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
        color: #2e7d32;
    }
    .insight-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
    }
    .risk-low { background-color: #4caf50; }
    .risk-medium { background-color: #ff9800; }
    .risk-high { background-color: #f44336; }
    .risk-critical { background-color: #9c27b0; }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Advanced NDIS Incident Analytics Dashboard")

@st.cache_data
def load_data():
    """Load and preprocess NDIS incidents data with enhanced features"""
    
    # Add file upload option for web deployment
    st.sidebar.subheader("📁 Data Source")
    data_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Use Demo Data"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
                return create_demo_data()
        else:
            st.sidebar.info("Please upload a CSV file to continue")
            return create_demo_data()
    else:
        # Try to load local file first, fallback to demo data
        try:
            df = pd.read_csv("/Users/darolinvinisha/PycharmProjects/MD651/Using Ollama/ndis_incidents_synthetic.csv")
        except FileNotFoundError:
            df = create_demo_data()
    
    # Process the data
    return process_data(df)

def create_demo_data():
    """Create demo data for testing"""
    np.random.seed(42)
    n_records = 100
    
    data = {
        'incident_id': [f'INC{i:06d}' for i in range(1, n_records + 1)],
        'participant_name': [f'Participant_{i:03d}' for i in np.random.randint(1, 50, n_records)],
        'incident_date': pd.date_range('2024-01-01', '2024-12-31', periods=n_records),
        'incident_time': [f"{h:02d}:{m:02d}" for h, m in zip(np.random.randint(0, 24, n_records), np.random.randint(0, 60, n_records))],
        'incident_type': np.random.choice(['Medication Error', 'Fall', 'Behavioral Incident'], n_records),
        'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_records),
        'location': np.random.choice(['Main Office', 'Community Center', 'Residential Care'], n_records),
        'reportable': np.random.choice(['Yes', 'No'], n_records),
        'description': [f'Demo incident {i}' for i in range(n_records)],
        'immediate_action': [f'Demo action {i}' for i in range(n_records)]
    }
    
    return pd.DataFrame(data)

def process_data(df):
    """Process and enhance the loaded data"""
    try:
        # Convert date columns
        if 'incident_date' in df.columns:
            df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
            if df['incident_date'].isna().all():
                df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        
        if 'notification_date' in df.columns:
            df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
            if df['notification_date'].isna().all():
                df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
        else:
            # Create notification dates with some delay
            df['notification_date'] = df['incident_date'] + pd.to_timedelta(np.random.randint(0, 3, len(df)), unit='days')
        
        # Calculate notification delay
        df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.days
        
        # Add time-based columns
        df['month'] = df['incident_date'].dt.month_name()
        df['day_of_week'] = df['incident_date'].dt.day_name()
        
        if 'incident_time' in df.columns:
            df['hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour
        else:
            df['hour'] = np.random.randint(0, 24, len(df))
            
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
        
        # Risk scoring
        severity_weights = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df['severity_score'] = df['severity'].map(severity_weights)
        
        # Create age groups if age column exists
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        else:
            # Create a dummy age column and age groups
            df['age'] = np.random.normal(35, 15, len(df)).astype(int).clip(18, 85)
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return create_demo_data()

def calculate_correlations(df):
    """Calculate key correlations for analysis"""
    numeric_df = df.copy()
    
    # Convert categorical to numeric (only if columns exist)
    if 'severity' in df.columns:
        numeric_df['severity_numeric'] = numeric_df['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4})
    else:
        numeric_df['severity_numeric'] = 1
        
    if 'reportable' in df.columns:
        numeric_df['reportable_numeric'] = numeric_df['reportable'].map({'No': 0, 'Yes': 1})
    else:
        numeric_df['reportable_numeric'] = 0
        
    if 'medical_attention' in df.columns:
        numeric_df['medical_attention_numeric'] = numeric_df['medical_attention'].map({'No': 0, 'Yes': 1})
    else:
        numeric_df['medical_attention_numeric'] = 0
        
    if 'is_weekend' in df.columns:
        numeric_df['is_weekend_numeric'] = numeric_df['is_weekend'].astype(int)
    else:
        numeric_df['is_weekend_numeric'] = 0
    
    # Select only available columns for correlation
    correlation_vars = []
    possible_vars = ['age', 'severity_numeric', 'notification_delay', 'reportable_numeric', 
                    'medical_attention_numeric', 'is_weekend_numeric', 'hour']
    
    for var in possible_vars:
        if var in numeric_df.columns:
            correlation_vars.append(var)
    
    if len(correlation_vars) < 2:
        # Create minimal correlation matrix if not enough variables
        correlation_vars = ['severity_numeric', 'notification_delay']
        for var in correlation_vars:
            if var not in numeric_df.columns:
                numeric_df[var] = 1
    
    corr_matrix = numeric_df[correlation_vars].corr()
    
    return corr_matrix, numeric_df

def generate_insights(df):
    """Generate automated insights from the data"""
    insights = []
    
    try:
        # Age-related insights
        if 'age_group' in df.columns and 'severity_score' in df.columns:
            age_severity = df.groupby('age_group', observed=True)['severity_score'].mean()
            if len(age_severity) > 0:
                high_risk_age = age_severity.idxmax()
                insights.append(f"🎯 Age group '{high_risk_age}' has the highest average incident severity")
        
        # Temporal insights
        if 'is_weekend' in df.columns and 'severity_score' in df.columns:
            weekend_incidents = df[df['is_weekend']]['severity_score'].mean()
            weekday_incidents = df[~df['is_weekend']]['severity_score'].mean()
            if pd.notna(weekend_incidents) and pd.notna(weekday_incidents) and weekday_incidents > 0:
                if weekend_incidents > weekday_incidents:
                    insights.append(f"⏰ Weekend incidents are {((weekend_incidents/weekday_incidents - 1) * 100):.1f}% more severe on average")
        
        # Location insights
        if 'location' in df.columns and 'severity_score' in df.columns:
            location_risk = df.groupby('location').agg({
                'severity_score': 'mean',
                'incident_id': 'count'
            })
            if len(location_risk[location_risk['incident_id'] > 50]) > 0:
                high_risk_location = location_risk.loc[location_risk['incident_id'] > 50, 'severity_score'].idxmax()
                insights.append(f"🏢 '{high_risk_location}' shows highest severity among high-volume locations")
        
        # Reporter insights
        if 'reporter_type' in df.columns and 'notification_delay' in df.columns:
            reporter_performance = df.groupby('reporter_type').agg({
                'notification_delay': 'mean',
                'incident_id': 'count'
            })
            if len(reporter_performance) > 0:
                fastest_reporters = reporter_performance['notification_delay'].idxmin()
                insights.append(f"📞 {fastest_reporters} are the fastest at reporting incidents")
        
        # Medical attention patterns
        if 'severity' in df.columns and 'medical_attention' in df.columns:
            medical_by_severity = df.groupby('severity')['medical_attention'].apply(lambda x: (x == 'Yes').mean())
            if 'Critical' in medical_by_severity.index:
                insights.append(f"🏥 {medical_by_severity['Critical']*100:.1f}% of critical incidents require medical attention")
        
        # Default insights if no data available
        if len(insights) == 0:
            insights = [
                "📊 Data analysis in progress",
                "🔍 Exploring incident patterns",
                "📈 Building risk assessments"
            ]
            
    except Exception as e:
        insights = [f"⚠️ Analysis temporarily unavailable: {str(e)[:50]}..."]
    
    return insights

# Load data with better error handling
df = None
try:
    df = load_data()
    if df is not None and len(df) > 0:
        corr_matrix, numeric_df = calculate_correlations(df)
        insights = generate_insights(df)
        
        st.success(f"✅ Successfully loaded {len(df)} incidents from {df['incident_date'].min().strftime('%B %Y')} to {df['incident_date'].max().strftime('%B %Y')}")
    else:
        st.warning("⚠️ No data loaded. Please upload a CSV file or use demo data.")
        df = create_demo_data()
        df = process_data(df)
        corr_matrix, numeric_df = calculate_correlations(df)
        insights = generate_insights(df)
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("📁 Please use the file upload option in the sidebar or try demo data.")
    df = create_demo_data()
    df = process_data(df)
    corr_matrix, numeric_df = calculate_correlations(df)
    insights = ["📊 Using demo data for analysis"]

# Enhanced Sidebar with Analysis Mode
st.sidebar.header("🎛️ Advanced Controls")

# Analysis Mode Selection
analysis_mode = st.sidebar.selectbox(
    "🔬 Analysis Mode",
    ["Executive Overview", "Risk Analysis", "Correlation Explorer", "Predictive Insights", "Performance Analytics"]
)

# Interactive Date Range with Presets
st.sidebar.subheader("📅 Time Period")
preset_ranges = {
    "Last 30 Days": 30,
    "Last 90 Days": 90,
    "Last 6 Months": 180,
    "Last Year": 365,
    "All Time": None
}

time_preset = st.sidebar.selectbox("Quick Select", list(preset_ranges.keys()), index=4)

if preset_ranges[time_preset]:
    end_date = df['incident_date'].max()
    start_date = end_date - timedelta(days=preset_ranges[time_preset])
    df_filtered = df[(df['incident_date'] >= start_date) & (df['incident_date'] <= end_date)]
else:
    df_filtered = df.copy()

# Dynamic Filters
st.sidebar.subheader("🎯 Smart Filters")

# Risk-based filtering
risk_level = st.sidebar.selectbox(
    "Risk Focus",
    ["All Incidents", "High Risk Only", "Critical Only", "Repeat Participants", "High-Volume Locations"]
)

if risk_level == "High Risk Only":
    df_filtered = df_filtered[df_filtered['severity'].isin(['High', 'Critical'])]
elif risk_level == "Critical Only":
    df_filtered = df_filtered[df_filtered['severity'] == 'Critical']
elif risk_level == "Repeat Participants":
    repeat_participants = df_filtered['participant_name'].value_counts()
    repeat_names = repeat_participants[repeat_participants > 1].index
    df_filtered = df_filtered[df_filtered['participant_name'].isin(repeat_names)]
elif risk_level == "High-Volume Locations":
    location_counts = df_filtered['location'].value_counts()
    high_volume_locations = location_counts[location_counts > location_counts.quantile(0.75)].index
    df_filtered = df_filtered[df_filtered['location'].isin(high_volume_locations)]

# Multi-select filters
severity_options = df['severity'].unique()
severity_filter = st.sidebar.multiselect(
    "⚠️ Severity Level",
    options=severity_options,
    default=severity_options
)

incident_type_options = sorted(df['incident_type'].unique())
incident_type_filter = st.sidebar.multiselect(
    "📋 Incident Type",
    options=incident_type_options,
    default=incident_type_options
)

# Apply filters
df_filtered = df_filtered[
    (df_filtered['severity'].isin(severity_filter)) &
    (df_filtered['incident_type'].isin(incident_type_filter))
]

# Real-time Insights Panel
with st.sidebar:
    st.subheader("💡 Live Insights")
    for insight in insights[:3]:  # Show top 3 insights
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

# Main Dashboard Content based on Analysis Mode
if analysis_mode == "Executive Overview":
    # Enhanced KPIs with trend indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_incidents = len(df_filtered)
        prev_period_data = df[(df['incident_date'] >= df['incident_date'].max() - timedelta(days=60)) & 
                            (df['incident_date'] < df['incident_date'].max() - timedelta(days=30))]
        prev_period = len(prev_period_data)
        trend = ((total_incidents - prev_period) / prev_period * 100) if prev_period > 0 else 0
        st.metric("📊 Total Incidents", total_incidents, delta=f"{trend:+.1f}%")
    
    with col2:
        critical_count = len(df_filtered[df_filtered['severity'] == 'Critical'])
        st.metric("🚨 Critical", critical_count, delta=f"{critical_count/total_incidents*100:.1f}%" if total_incidents > 0 else "0%")
    
    with col3:
        avg_delay = df_filtered['notification_delay'].mean()
        target_delay = 1.0  # Target: 1 day
        delay_status = "🟢" if avg_delay <= target_delay else "🔴"
        st.metric("⏱️ Avg Delay", f"{avg_delay:.1f}d", delta=f"{delay_status}")
    
    with col4:
        repeat_participants = df_filtered['participant_name'].value_counts()
        repeat_count = len(repeat_participants[repeat_participants > 1])
        st.metric("🔄 Repeat Participants", repeat_count)
    
    with col5:
        compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
        st.metric("✅ Compliance Rate", f"{compliance_rate:.1f}%")
    
    # Interactive Incident Heatmap
    st.subheader("🔥 Incident Heatmap: Location vs Time")
    
    # Create heatmap data
    heatmap_data = df_filtered.pivot_table(
        values='incident_id', 
        index='location', 
        columns=df_filtered['incident_date'].dt.hour, 
        aggfunc='count', 
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Incident Frequency by Location and Hour of Day",
        labels=dict(x="Hour of Day", y="Location", color="Incidents"),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Trend Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly trends
        monthly_trends = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).size()
        monthly_trends.index = monthly_trends.index.astype(str)
        
        fig_trend = px.line(
            x=monthly_trends.index,
            y=monthly_trends.values,
            title="📊 Monthly Incident Trends",
            markers=True
        )
        fig_trend.update_xaxes(tickangle=45)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Severity distribution
        severity_counts = df_filtered['severity'].value_counts()
        colors = {'Critical': '#ff4444', 'High': '#ff8800', 'Medium': '#ffcc00', 'Low': '#44ff44'}
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="⚠️ Severity Distribution",
            color=severity_counts.index,
            color_discrete_map=colors
        )
        fig_severity.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_severity, use_container_width=True)

elif analysis_mode == "Risk Analysis":
    st.subheader("🎯 Advanced Risk Analysis")
    
    # Risk Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity vs Frequency Risk Matrix
        risk_data = []
        for location in df_filtered['location'].unique():
            location_data = df_filtered[df_filtered['location'] == location]
            total_incidents = len(location_data)
            avg_severity = location_data['severity_score'].mean()
            
            risk_data.append({
                'location': location,
                'total_incidents': total_incidents,
                'avg_severity': avg_severity,
                'risk_score': total_incidents * avg_severity
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        fig_risk = px.scatter(
            risk_df, 
            x='total_incidents', 
            y='avg_severity',
            size='risk_score',
            color='risk_score',
            hover_name='location',
            title="Risk Matrix: Volume vs Severity by Location",
            labels={'total_incidents': 'Incident Volume', 'avg_severity': 'Average Severity Score'},
            color_continuous_scale="Reds"
        )
        
        # Add quadrant lines
        median_volume = risk_df['total_incidents'].median()
        median_severity = risk_df['avg_severity'].median()
        
        fig_risk.add_hline(y=median_severity, line_dash="dash", line_color="gray", opacity=0.5)
        fig_risk.add_vline(x=median_volume, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Age vs Incident Type Risk Analysis
        age_incident_matrix = pd.crosstab(df_filtered['age_group'], df_filtered['incident_type'])
        age_incident_pct = age_incident_matrix.div(age_incident_matrix.sum(axis=1), axis=0) * 100
        
        fig_age_risk = px.imshow(
            age_incident_pct,
            title="Incident Type Risk by Age Group (%)",
            labels=dict(x="Incident Type", y="Age Group", color="Percentage"),
            color_continuous_scale="YlOrRd"
        )
        fig_age_risk.update_xaxes(tickangle=45)
        st.plotly_chart(fig_age_risk, use_container_width=True)
    
    # Risk Factors Analysis
    st.subheader("📊 Risk Factor Analysis")
    
    risk_factors = {}
    
    # Weekend vs weekday risk
    if len(df_filtered[df_filtered['is_weekend']]) > 0:
        risk_factors['Weekend Incidents'] = df_filtered[df_filtered['is_weekend']]['severity_score'].mean()
    
    # Night hours risk
    night_hours = list(range(22, 24)) + list(range(0, 7))
    night_incidents = df_filtered[df_filtered['hour'].isin(night_hours)]
    if len(night_incidents) > 0:
        risk_factors['Night Hours (22-06)'] = night_incidents['severity_score'].mean()
    
    # Repeat participants risk
    repeat_participants = df_filtered['participant_name'].value_counts()
    repeat_names = repeat_participants[repeat_participants > 1].index
    if len(repeat_names) > 0:
        risk_factors['Repeat Participants'] = df_filtered[df_filtered['participant_name'].isin(repeat_names)]['severity_score'].mean()
    
    # Delayed reporting risk
    delayed_reports = df_filtered[df_filtered['notification_delay'] > 1]
    if len(delayed_reports) > 0:
        risk_factors['Delayed Reporting'] = delayed_reports['severity_score'].mean()
    
    if risk_factors:
        risk_factor_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Risk Score'])
        risk_factor_df = risk_factor_df.sort_values('Risk Score', ascending=True)
        
        fig_factors = px.bar(
            risk_factor_df,
            x='Risk Score',
            y='Factor',
            orientation='h',
            title="Risk Factor Impact on Incident Severity",
            color='Risk Score',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_factors, use_container_width=True)

elif analysis_mode == "Correlation Explorer":
    st.subheader("🔗 Interactive Correlation Analysis")
    
    # Correlation matrix heatmap
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Key Variables",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Key Correlations")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.1:  # Only show meaningful correlations
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if corr_pairs:
            corr_pairs_df = pd.DataFrame(corr_pairs)
            corr_pairs_df = corr_pairs_df.reindex(corr_pairs_df.correlation.abs().sort_values(ascending=False).index)
            
            for _, row in corr_pairs_df.head(5).iterrows():
                strength = "Strong" if abs(row['correlation']) > 0.5 else "Moderate" if abs(row['correlation']) > 0.3 else "Weak"
                direction = "Positive" if row['correlation'] > 0 else "Negative"
                
                st.markdown(f"""
                <div class="correlation-card">
                    <strong>{row['var1']} ↔ {row['var2']}</strong><br>
                    {direction} {strength}<br>
                    r = {row['correlation']:.3f}
                </div>
                """, unsafe_allow_html=True)
    
    # Interactive scatter plots
    st.subheader("🔍 Relationship Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X-axis Variable", corr_matrix.columns, index=0)
    with col2:
        y_var = st.selectbox("Y-axis Variable", corr_matrix.columns, index=1)
    with col3:
        color_var = st.selectbox("Color by", ['severity', 'location', 'incident_type', 'age_group'])
    
    if x_var != y_var:
        fig_scatter = px.scatter(
            numeric_df,
            x=x_var,
            y=y_var,
            color=color_var,
            title=f"Relationship between {x_var} and {y_var}",
            trendline="ols" if st.checkbox("Show trend line") else None,
            hover_data=['participant_name', 'incident_date']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Statistical significance test
        valid_data = numeric_df[[x_var, y_var]].dropna()
        if len(valid_data) > 2:
            correlation, p_value = stats.pearsonr(valid_data[x_var], valid_data[y_var])
            significance = "Significant" if p_value < 0.05 else "Not significant"
            
            st.info(f"**Statistical Analysis:** Correlation = {correlation:.3f}, p-value = {p_value:.3f} ({significance})")

elif analysis_mode == "Predictive Insights":
    st.subheader("🔮 Predictive Analytics & Forecasting")
    
    # Time series analysis
    monthly_incidents = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).size()
    monthly_incidents.index = monthly_incidents.index.to_timestamp()
    
    if len(monthly_incidents) > 3:
        # Simple trend analysis
        trend_data = pd.DataFrame({
            'month': range(len(monthly_incidents)),
            'incidents': monthly_incidents.values
        })
        
        # Calculate linear trend
        z = np.polyfit(trend_data['month'], trend_data['incidents'], 1)
        trend_line = np.poly1d(z)
        
        # Create forecast
        future_months = range(len(monthly_incidents), len(monthly_incidents) + 6)
        forecast = [trend_line(m) for m in future_months]
        
        # Plot
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=monthly_incidents.index,
            y=monthly_incidents.values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Trend line
        fig_forecast.add_trace(go.Scatter(
            x=monthly_incidents.index,
            y=[trend_line(i) for i in range(len(monthly_incidents))],
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        # Forecast
        future_dates = pd.date_range(monthly_incidents.index[-1], periods=7, freq='M')[1:]
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange')
        ))
        
        fig_forecast.update_layout(
            title="Incident Trend Forecast (6 Months Ahead)",
            xaxis_title="Date",
            yaxis_title="Number of Incidents"
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Risk Prediction Calculator
    st.subheader("⚠️ Risk Prediction Engine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Scenario Planning")
        
        # Interactive risk calculator
        selected_age = st.slider("Participant Age", 18, 85, 35)
        selected_location = st.selectbox("Location", df['location'].unique())
        selected_time = st.selectbox("Time of Day", ["Morning (6-12)", "Afternoon (12-18)", "Evening (18-22)", "Night (22-6)"])
        is_weekend_scenario = st.checkbox("Weekend?")
        
        # Calculate risk based on historical data
        time_mapping = {
            "Morning (6-12)": list(range(6, 12)), 
            "Afternoon (12-18)": list(range(12, 18)), 
            "Evening (18-22)": list(range(18, 22)), 
            "Night (22-6)": list(range(22, 24)) + list(range(0, 6))
        }
        
        scenario_filter = (
            (df['age'] >= selected_age - 5) & (df['age'] <= selected_age + 5) &
            (df['location'] == selected_location) &
            (df['hour'].isin(time_mapping[selected_time])) &
            (df['is_weekend'] == is_weekend_scenario)
        )
        
        scenario_incidents = df[scenario_filter]
        
        if len(scenario_incidents) > 0:
            avg_severity = scenario_incidents['severity_score'].mean()
            incident_probability = len(scenario_incidents) / len(df) * 100
            
            risk_level = "Low" if avg_severity < 1.5 else "Medium" if avg_severity < 2.5 else "High" if avg_severity < 3.5 else "Critical"
            risk_color = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high", "Critical": "risk-critical"}[risk_level]
            
            st.markdown(f"""
            <div class="risk-card {risk_color}">
                <h3>Risk Assessment</h3>
                <p><strong>Risk Level: {risk_level}</strong></p>
                <p>Average Severity: {avg_severity:.2f}/4.0</p>
                <p>Historical Probability: {incident_probability:.2f}%</p>
                <p>Sample Size: {len(scenario_incidents)} incidents</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No historical data available for this scenario combination.")
    
    with col2:
        st.markdown("### 📈 Risk Trends")
        
        # Risk trend over time
        risk_over_time = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M'))['severity_score'].mean()
        risk_over_time.index = risk_over_time.index.astype(str)
        
        fig_risk_trend = px.line(
            x=risk_over_time.index,
            y=risk_over_time.values,
            title="Average Risk Score Over Time",
            labels={'x': 'Month', 'y': 'Average Risk Score'}
        )
        fig_risk_trend.update_xaxes(tickangle=45)
        st.plotly_chart(fig_risk_trend, use_container_width=True)

elif analysis_mode == "Performance Analytics":
    st.subheader("📊 Performance Analytics Dashboard")
    
    # Check which columns are available
    has_reporter_type = 'reporter_type' in df_filtered.columns
    has_medical_attention = 'medical_attention' in df_filtered.columns
    
    if has_reporter_type:
        # Reporter Performance Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 Reporter Performance")
            
            reporter_stats = df_filtered.groupby('reporter_type').agg({
                'incident_id': 'count',
                'notification_delay': 'mean',
                'severity_score': 'mean'
            }).round(2)
            
            reporter_stats.columns = ['Total Reports', 'Avg Delay (days)', 'Avg Severity']
            
            fig_reporter = px.scatter(
                reporter_stats.reset_index(),
                x='Avg Delay (days)',
                y='Avg Severity',
                size='Total Reports',
                color='Total Reports',
                hover_name='reporter_type',
                title="Reporter Performance: Speed vs Quality",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_reporter, use_container_width=True)
        
        with col2:
            st.markdown("### 🏢 Location Performance")
            
            # Calculate medical rate only if column exists
            agg_dict = {
                'incident_id': 'count',
                'notification_delay': 'mean',
                'severity_score': 'mean'
            }
            
            if has_medical_attention:
                agg_dict['medical_attention'] = lambda x: (x == 'Yes').mean()
                
            location_stats = df_filtered.groupby('location').agg(agg_dict).round(2)
            
            if has_medical_attention:
                location_stats.columns = ['Total Incidents', 'Avg Delay', 'Avg Severity', 'Medical Rate']
                # Performance score (lower is better)
                location_stats['Performance Score'] = (
                    location_stats['Avg Delay'] * 0.3 + 
                    location_stats['Avg Severity'] * 0.4 + 
                    location_stats['Medical Rate'] * 0.3
                )
            else:
                location_stats.columns = ['Total Incidents', 'Avg Delay', 'Avg Severity']
                # Performance score without medical rate
                location_stats['Performance Score'] = (
                    location_stats['Avg Delay'] * 0.5 + 
                    location_stats['Avg Severity'] * 0.5
                )
            
            fig_location = px.bar(
                location_stats.reset_index().sort_values('Performance Score'),
                x='location',
                y='Performance Score',
                title="Location Performance Score (Lower = Better)",
                color='Performance Score',
                color_continuous_scale="RdYlGn_r"
            )
            fig_location.update_xaxes(tickangle=45)
            st.plotly_chart(fig_location, use_container_width=True)
    else:
        st.info("👥 Reporter performance analysis requires 'reporter_type' column in your data")
        
        # Show basic location analysis instead
        st.markdown("### 🏢 Location Analysis")
        location_stats = df_filtered.groupby('location').agg({
            'incident_id': 'count',
            'severity_score': 'mean'
        }).round(2)
        location_stats.columns = ['Total Incidents', 'Avg Severity']
        
        fig_location = px.bar(
            location_stats.reset_index(),
            x='location',
            y='Total Incidents',
            color='Avg Severity',
            title="Incidents by Location",
            color_continuous_scale="Reds"
        )
        fig_location.update_xaxes(tickangle=45)
        st.plotly_chart(fig_location, use_container_width=True)
    
    # Compliance Dashboard
    st.subheader("✅ Compliance Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Notification compliance
        compliance_by_severity = df_filtered.groupby('severity').apply(
            lambda x: (x['notification_delay'] <= 1).mean() * 100
        )
        
        fig_compliance = px.bar(
            x=compliance_by_severity.index,
            y=compliance_by_severity.values,
            title="Notification Compliance by Severity (%)",
            color=compliance_by_severity.values,
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_compliance, use_container_width=True)
    
    with col2:
        if has_medical_attention:
            # Medical attention compliance
            medical_compliance = df_filtered[df_filtered['severity'].isin(['High', 'Critical'])]
            medical_rate = (medical_compliance['medical_attention'] == 'Yes').mean() * 100
            
            fig_medical = go.Figure(go.Indicator(
                mode="gauge+number",
                value=medical_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Medical Attention Rate for High/Critical Incidents (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig_medical, use_container_width=True)
        else:
            st.info("🏥 Medical attention analysis requires 'medical_attention' column")
    
    with col3:
        # Reportable incident compliance
        if 'reportable' in df_filtered.columns:
            reportable_incidents = df_filtered[df_filtered['reportable'] == 'Yes']
            if len(reportable_incidents) > 0:
                reportable_compliance = (reportable_incidents['notification_delay'] <= 0.5).mean() * 100
            else:
                reportable_compliance = 0
            
            fig_reportable = go.Figure(go.Indicator(
                mode="gauge+number",
                value=reportable_compliance,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Reportable Incident Compliance (< 12 hours)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            st.plotly_chart(fig_reportable, use_container_width=True)
        else:
            st.info("📋 Reportable compliance analysis requires 'reportable' column")

# Interactive Data Table with Advanced Filtering
st.subheader("📋 Interactive Incident Explorer")

# Advanced search and filter options
col1, col2, col3 = st.columns(3)

with col1:
    search_term = st.text_input("🔍 Search descriptions/actions")

with col2:
    date_range_filter = st.date_input(
        "📅 Specific Date Range",
        value=[],
        min_value=df['incident_date'].min().date(),
        max_value=df['incident_date'].max().date()
    )

with col3:
    export_format = st.selectbox("📥 Export Format", ["CSV", "Excel", "JSON"])

# Apply additional filters
display_df = df_filtered.copy()

if search_term:
    search_cols = ['description', 'immediate_action', 'actions_taken']
    search_mask = pd.Series(False, index=display_df.index)
    for col in search_cols:
        if col in display_df.columns:
            search_mask |= display_df[col].str.contains(search_term, case=False, na=False)
    display_df = display_df[search_mask]

if len(date_range_filter) == 2:
    start_date, end_date = date_range_filter
    display_df = display_df[
        (display_df['incident_date'].dt.date >= start_date) & 
        (display_df['incident_date'].dt.date <= end_date)
    ]

# Column selection for display
available_columns = ['incident_id', 'participant_name', 'age', 'incident_date', 'incident_type',
                     'severity', 'location', 'reportable', 'notification_delay', 'medical_attention']
display_columns = st.multiselect(
    "Select columns to display",
    options=[col for col in available_columns if col in display_df.columns],
    default=[col for col in available_columns[:8] if col in display_df.columns]
)

# Display the data
if display_columns:
    # Add risk scoring to display
    if 'severity_score' not in display_columns and 'severity_score' in display_df.columns:
        display_df_show = display_df[display_columns + ['severity_score']].copy()
        display_df_show = display_df_show.sort_values(['severity_score', 'incident_date'], ascending=[False, False])
    else:
        display_df_show = display_df[display_columns].copy()
        if 'incident_date' in display_columns:
            display_df_show = display_df_show.sort_values('incident_date', ascending=False)
    
    # Style the dataframe
    if 'severity' in display_columns:
        def highlight_severity(val):
            if val == 'Critical':
                return 'background-color: #ffebee; color: #c62828; font-weight: bold'
            elif val == 'High':
                return 'background-color: #fff3e0; color: #ef6c00; font-weight: bold'
            elif val == 'Medium':
                return 'background-color: #fffde7; color: #f57f17'
            return ''
        
        styled_df = display_df_show.style.map(highlight_severity, subset=['severity'])
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        st.dataframe(display_df_show, use_container_width=True, height=400)

# Export functionality
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Data"):
        if export_format == "CSV":
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                f"ndis_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        elif export_format == "JSON":
            json_data = display_df.to_json(orient='records', date_format='iso')
            st.download_button(
                "Download JSON",
                json_data,
                f"ndis_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

with col2:
    if st.button("📊 Generate Summary Report"):
        st.subheader("📈 Executive Summary")
        
        summary_stats = {
            "Total Incidents": len(display_df),
            "Critical Incidents": len(display_df[display_df['severity'] == 'Critical']),
            "Average Notification Delay": f"{display_df['notification_delay'].mean():.1f} days",
            "Compliance Rate": f"{(display_df['notification_delay'] <= 1).mean() * 100:.1f}%",
            "Most Common Incident Type": display_df['incident_type'].mode().iloc[0] if len(display_df) > 0 else 'N/A',
            "Highest Risk Location": display_df.groupby('location')['severity_score'].mean().idxmax() if len(display_df) > 0 else 'N/A'
        }
        
        for key, value in summary_stats.items():
            st.metric(key, value)

with col3:
    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

# Footer with metadata
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown(f"**Records displayed:** {len(display_df)} of {len(df)}")
with col3:
    st.markdown(f"**Analysis mode:** {analysis_mode}")
