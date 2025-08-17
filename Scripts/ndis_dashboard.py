import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# ML Libraries for advanced analytics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Association rules libraries
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    st.warning("⚠️ mlxtend not installed. Association rules analysis will be disabled. Install with: pip install mlxtend")

import warnings
warnings.filterwarnings('ignore')

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

# Data loading functions (without cache and widgets)
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

@st.cache_data
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

def load_data_from_file(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        return process_data(df)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def load_local_data():
    """Try to load local data file"""
    try:
        df = pd.read_csv("/Users/darolinvinisha/PycharmProjects/MD651/Using Ollama/ndis_incidents_synthetic.csv")
        return process_data(df)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading local file: {str(e)}")
        return None

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

@st.cache_data
def prepare_ml_features(df):
    """Prepare features for ML analysis"""
    try:
        ml_df = df.copy()
        
        # Create derived features
        if 'incident_date' in ml_df.columns:
            ml_df['incident_year'] = ml_df['incident_date'].dt.year
            ml_df['incident_month'] = ml_df['incident_date'].dt.month
            ml_df['incident_weekday'] = ml_df['incident_date'].dt.dayofweek
        
        # Calculate age if available
        if 'age' in ml_df.columns:
            ml_df['age_at_incident'] = ml_df['age']
        else:
            ml_df['age_at_incident'] = np.random.normal(35, 15, len(ml_df)).clip(18, 85)
        
        # Encode categorical variables
        categorical_cols = ['incident_type', 'severity', 'location']
        if 'reportable' in ml_df.columns:
            categorical_cols.append('reportable')
        
        label_encoders = {}
        for col in categorical_cols:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col] = ml_df[col].fillna('Unknown')
                ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col].astype(str))
                label_encoders[col] = le
        
        return ml_df, label_encoders
        
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return df, {}

def perform_clustering_analysis(df, method='kmeans', n_clusters=5):
    """Perform clustering analysis"""
    try:
        # Prepare features
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age_at_incident' in df.columns:
            feature_cols.append('age_at_incident')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'incident_month' in df.columns:
            feature_cols.append('incident_month')
        if 'incident_weekday' in df.columns:
            feature_cols.append('incident_weekday')
        
        if not feature_cols:
            st.warning("No suitable features found for clustering")
            return None, None
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Calculate metrics
        metrics = {}
        if len(set(cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
            metrics['calinski_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
            metrics['n_clusters'] = len(set(cluster_labels))
        
        return cluster_labels, metrics
        
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")
        return None, None

def detect_anomalies(df, method='isolation_forest', contamination=0.1):
    """Detect anomalies in the data"""
    try:
        # Prepare features
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age_at_incident' in df.columns:
            feature_cols.append('age_at_incident')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        
        if not feature_cols:
            st.warning("No suitable features found for anomaly detection")
            return None
        
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination)
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
        
        if method == 'local_outlier_factor':
            anomaly_labels = detector.fit_predict(X_scaled)
        else:
            anomaly_labels = detector.fit_predict(X_scaled)
        
        return anomaly_labels
        
    except Exception as e:
        st.error(f"Anomaly detection error: {str(e)}")
        return None

def find_association_rules(df, min_support=0.1, min_confidence=0.6):
    """Find association rules in the data"""
    if not MLXTEND_AVAILABLE:
        return None, None
        
    try:
        # Prepare transaction data
        categorical_cols = ['incident_type', 'severity', 'location']
        if 'reportable' in df.columns:
            categorical_cols.append('reportable')
        
        # Create transactions
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in categorical_cols:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{col}_{row[col]}")
            if transaction:  # Only add non-empty transactions
                transactions.append(transaction)
        
        if not transactions:
            return None, None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return None, None
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return frequent_itemsets, rules
        
    except Exception as e:
        st.error(f"Association rules error: {str(e)}")
        return None, None

# Data loading UI (moved outside cached functions)
st.sidebar.subheader("📁 Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Use Demo Data", "Upload CSV"])

# Load data based on user selection
df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)
        if df is not None:
            st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.info("Please upload a CSV file to continue")
        df = process_data(create_demo_data())
else:
    # Try local file first, then demo data
    df = load_local_data()
    if df is None:
        df = process_data(create_demo_data())

# Load data with better error handling
if df is not None and len(df) > 0:
    corr_matrix, numeric_df = calculate_correlations(df)
    insights = generate_insights(df)
    
    st.success(f"✅ Successfully loaded {len(df)} incidents from {df['incident_date'].min().strftime('%B %Y')} to {df['incident_date'].max().strftime('%B %Y')}")
else:
    st.error("❌ Failed to load data")
    st.stop()

# Enhanced Sidebar with Analysis Mode
st.sidebar.header("🎛️ Advanced Controls")

# Analysis Mode Selection
analysis_mode = st.sidebar.selectbox(
    "🔬 Analysis Mode",
    ["Executive Overview", "Risk Analysis", "Correlation Explorer", "Predictive Insights", "Performance Analytics", "🤖 ML Analytics"]
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
        # Check if notification_delay column exists
        if 'notification_delay' in df_filtered.columns:
            avg_delay = df_filtered['notification_delay'].mean()
            target_delay = 1.0  # Target: 1 day
            delay_status = "🟢" if avg_delay <= target_delay else "🔴"
            st.metric("⏱️ Avg Delay", f"{avg_delay:.1f}d", delta=f"{delay_status}")
        else:
            st.metric("⏱️ Avg Delay", "N/A", delta="No data")
    
    with col4:
        repeat_participants = df_filtered['participant_name'].value_counts()
        repeat_count = len(repeat_participants[repeat_participants > 1])
        st.metric("🔄 Repeat Participants", repeat_count)
    
    with col5:
        # Check if notification_delay column exists for compliance calculation
        if 'notification_delay' in df_filtered.columns:
            compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
            st.metric("✅ Compliance Rate", f"{compliance_rate:.1f}%")
        else:
            st.metric("✅ Compliance Rate", "N/A", delta="No data")
    
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
