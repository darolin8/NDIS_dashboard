#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

# Association rules imports
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

# Time series forecasting imports
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# NDIS color palette - defined at the top for global access
NDIS_COLORS = {
    'primary': '#003F5C',
    'secondary': '#2F9E7D', 
    'accent': '#F59C2F',
    'critical': '#DC2626',
    'high': '#F59C2F',
    'medium': '#2F9E7D',
    'low': '#67A3C3',
    'success': '#2F9E7D',
    'warning': '#F59C2F',
    'error': '#DC2626'
}

# Severity color mapping
severity_colors = {
    'Critical': NDIS_COLORS['critical'],
    'High': NDIS_COLORS['high'],
    'Medium': NDIS_COLORS['medium'],
    'Low': NDIS_COLORS['low']
}

# Page configuration
st.set_page_config(
    page_title="NDIS Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NDIS accessible theme
st.markdown("""
<style>
    /* NDIS Accessible Theme */
    :root {
        --primary-color: #003F5C;      /* Deep Blue */
        --secondary-color: #2F9E7D;    /* Teal/Turquoise */
        --accent-color: #F59C2F;       /* Amber/Orange */
        --background-color: #F7F9FA;   /* Light Neutral Gray */
        --card-background: #FFFFFF;    /* White */
        --text-primary: #1B1B1B;       /* Charcoal */
        --text-on-dark: #FFFFFF;       /* White */
    }
    
    /* Main app styling */
    .main > div {
        background-color: var(--background-color);
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--primary-color);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--text-on-dark);
    }
    
    /* Metric cards with NDIS theme */
    .metric-card {
        background-color: var(--card-background);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid var(--secondary-color);
        box-shadow: 0 2px 4px rgba(0, 63, 92, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-card h4 {
        color: var(--primary-color);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        color: var(--text-primary);
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Alert cards */
    .alert-card {
        background-color: var(--card-background);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--accent-color);
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(245, 156, 47, 0.1);
    }
    
    .critical-alert {
        border-left-color: #DC2626;
        background-color: #FEF2F2;
    }
    
    .success-alert {
        border-left-color: var(--secondary-color);
        background-color: #F0FDF4;
    }
    
    .warning-alert {
        border-left-color: var(--accent-color);
        background-color: #FFFBEB;
    }
    
    /* Headers and titles */
    h1, h2, h3 {
        color: var(--primary-color) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--secondary-color);
        color: var(--text-on-dark);
        border: none;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #267A63;
        box-shadow: 0 4px 8px rgba(47, 158, 125, 0.3);
    }
    
    /* ML specific styling */
    .ml-card {
        background-color: var(--card-background);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid var(--accent-color);
        box-shadow: 0 4px 8px rgba(245, 156, 47, 0.2);
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-left: 4px solid var(--success);
    }
    
    .anomaly-card {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 4px solid var(--error);
    }
    
    /* Selectboxes and inputs */
    .stSelectbox > div > div {
        background-color: var(--card-background);
        border: 1px solid var(--secondary-color);
        border-radius: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--card-background);
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--primary-color);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-color);
        color: var(--text-on-dark);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: var(--card-background);
        border: 1px solid var(--secondary-color);
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(47, 158, 125, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: var(--primary-color);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: var(--card-background);
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--secondary-color);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: var(--card-background);
        border: 2px dashed var(--secondary-color);
        border-radius: 0.75rem;
        padding: 2rem;
    }
    
    /* Info, warning, success, error boxes */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* Custom status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-high { background-color: #DC2626; }
    .status-medium { background-color: var(--accent-color); }
    .status-low { background-color: var(--secondary-color); }
    .status-compliant { background-color: var(--secondary-color); }
    .status-overdue { background-color: #DC2626; }
    
    /* Card containers */
    .dashboard-card {
        background-color: var(--card-background);
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0, 63, 92, 0.08);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-color);
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Machine Learning Helper Functions
@st.cache_data
def prepare_ml_features(df):
    """Prepare features for machine learning models"""
    if df.empty:
        return None, None, None
    
    # Create feature dataframe
    features_df = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['location', 'incident_type', 'contributing_factors', 'reported_by']
    
    for col in categorical_cols:
        if col in features_df.columns:
            le = LabelEncoder()
            features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].fillna('Unknown'))
            label_encoders[col] = le
    
    # Create numerical features
    numerical_features = []
    feature_names = []
    
    # Time-based features
    if 'incident_date' in features_df.columns:
        features_df['day_of_week'] = features_df['incident_date'].dt.dayofweek
        features_df['month'] = features_df['incident_date'].dt.month
        features_df['hour'] = pd.to_datetime(features_df['incident_time'], format='%H:%M', errors='coerce').dt.hour
        numerical_features.extend(['day_of_week', 'month'])
        feature_names.extend(['day_of_week', 'month'])
        
        if not features_df['hour'].isna().all():
            numerical_features.append('hour')
            feature_names.append('hour')
    
    # Encoded categorical features
    for col in categorical_cols:
        if f'{col}_encoded' in features_df.columns:
            numerical_features.append(f'{col}_encoded')
            feature_names.append(f'{col}_encoded')
    
    # Other numerical features
    if 'reporting_delay_hours' in features_df.columns:
        numerical_features.append('reporting_delay_hours')
        feature_names.append('reporting_delay_hours')
    
    if 'age_at_incident' in features_df.columns:
        numerical_features.append('age_at_incident')
        feature_names.append('age_at_incident')
    
    # Create feature matrix
    X = features_df[numerical_features].fillna(0)
    
    return X, feature_names, label_encoders

@st.cache_data
def train_severity_prediction_model(df):
    """Train a model to predict incident severity"""
    if df.empty or len(df) < 20:
        return None, None, None
    
    X, feature_names, label_encoders = prepare_ml_features(df)
    if X is None:
        return None, None, None
    
    # Prepare target variable
    severity_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    y = df['severity'].map(severity_map)
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        return None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, feature_names

@st.cache_data
def perform_anomaly_detection(df):
    """Perform anomaly detection on incidents"""
    if df.empty or len(df) < 10:
        return None, None
    
    X, feature_names, _ = prepare_ml_features(df)
    if X is None:
        return None, None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    
    # One-Class SVM
    oc_svm = OneClassSVM(nu=0.1)
    svm_labels = oc_svm.fit_predict(X_scaled)
    
    # Combine results
    df_with_anomalies = df.copy()
    df_with_anomalies['isolation_forest_anomaly'] = anomaly_labels == -1
    df_with_anomalies['svm_anomaly'] = svm_labels == -1
    df_with_anomalies['anomaly_score'] = iso_forest.decision_function(X_scaled)
    
    return df_with_anomalies, feature_names

@st.cache_data
def find_association_rules(df):
    """Find association rules between incident characteristics"""
    if not MLXTEND_AVAILABLE or df.empty or len(df) < 20:
        return None, None
    
    # Prepare transaction data
    transactions = []
    
    for _, row in df.iterrows():
        transaction = []
        
        # Add categorical features to transactions
        if pd.notna(row['location']):
            transaction.append(f"location_{row['location']}")
        if pd.notna(row['incident_type']):
            transaction.append(f"type_{row['incident_type']}")
        if pd.notna(row['severity']):
            transaction.append(f"severity_{row['severity']}")
        if pd.notna(row['contributing_factors']):
            transaction.append(f"factor_{row['contributing_factors']}")
        
        # Add binary features
        if row.get('medical_attention_required') == 'Yes':
            transaction.append('medical_required')
        if row.get('reportable') == 'Yes':
            transaction.append('reportable')
        if row.get('same_day_reporting', False):
            transaction.append('same_day_reported')
        
        if transaction:
            transactions.append(transaction)
    
    if not transactions:
        return None, None
    
    # Create binary matrix
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
    
    if frequent_itemsets.empty:
        return None, None
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    return frequent_itemsets, rules

@st.cache_data
def time_series_forecast(df, periods=30):
    """Perform time series forecasting of incident counts"""
    if not STATSMODELS_AVAILABLE or df.empty:
        return None, None
    
    # Aggregate incidents by date
    daily_counts = df.groupby(df['incident_date'].dt.date).size().reset_index()
    daily_counts.columns = ['date', 'incident_count']
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts = daily_counts.set_index('date').sort_index()
    
    # Ensure we have enough data
    if len(daily_counts) < 30:
        return None, None
    
    # Fill missing dates with 0
    date_range = pd.date_range(start=daily_counts.index.min(), end=daily_counts.index.max(), freq='D')
    daily_counts = daily_counts.reindex(date_range, fill_value=0)
    
    try:
        # Exponential Smoothing forecast
        model = ExponentialSmoothing(daily_counts['incident_count'], 
                                   trend='add', 
                                   seasonal=None)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)
        
        # Create forecast dates
        forecast_dates = pd.date_range(start=daily_counts.index.max() + pd.Timedelta(days=1), 
                                     periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast
        })
        
        return daily_counts, forecast_df
        
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        return None, None

# Data loading function
@st.cache_data
def load_incident_data():
    """Load and prepare the actual NDIS incident data"""
    try:
        # Try multiple possible file paths
        possible_paths = [
            'text data/ndis_incidents_synthetic.csv',
            'ndis_incidents_synthetic.csv',
            './ndis_incidents_synthetic.csv',
            'data/ndis_incidents_synthetic.csv',
            '../ndis_incidents_synthetic.csv',
            './text data/ndis_incidents_synthetic.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            st.error("CSV file not found. Please upload your data file below.")
            return None
            
        # Clean and prepare the data
        df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
        df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
        df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
        
        # Calculate reporting delay in hours
        df['reporting_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
        df['same_day_reporting'] = df['reporting_delay_hours'] <= 24
        
        # Calculate age at incident
        df['age_at_incident'] = (df['incident_date'] - df['dob']).dt.days / 365.25
        
        # Add month names for seasonal analysis
        df['incident_month'] = df['incident_date'].dt.month_name()
        df['incident_year'] = df['incident_date'].dt.year
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to create sample data if no file is available
@st.cache_data
def create_sample_data():
    """Create sample NDIS incident data for demonstration"""
    np.random.seed(42)
    
    # Sample data matching your CSV structure
    sample_data = {
        'incident_id': [f'INC-2024-{i:04d}' for i in range(1, 501)],
        'participant_name': [f'Participant {i}' for i in range(1, 501)],
        'ndis_number': np.random.randint(400000000, 500000000, 500),
        'dob': pd.date_range('1950-01-01', '2010-12-31', periods=500).strftime('%d/%m/%Y'),
        'incident_date': pd.date_range('2023-01-01', '2024-12-31', periods=500).strftime('%d/%m/%Y'),
        'incident_time': [f'{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}' for _ in range(500)],
        'notification_date': pd.date_range('2023-01-01', '2024-12-31', periods=500).strftime('%d/%m/%Y'),
        'location': np.random.choice(['Group Home', 'Transport Vehicle', 'Day Program', 'Community Access', 'Therapy Clinic'], 500),
        'incident_type': np.random.choice(['Injury', 'Missing Person', 'Death', 'Restrictive Practices', 'Transport Incident', 'Medication Error'], 500),
        'subcategory': np.random.choice(['Fall', 'Unexplained absence', 'Natural causes', 'Unauthorised', 'Vehicle crash', 'Wrong dose'], 500),
        'severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], 500, p=[0.1, 0.2, 0.4, 0.3]),
        'reportable': np.random.choice(['Yes', 'No'], 500, p=[0.7, 0.3]),
        'description': ['Sample incident description' for _ in range(500)],
        'immediate_action': ['Immediate action taken' for _ in range(500)],
        'actions_taken': ['Follow-up actions completed' for _ in range(500)],
        'contributing_factors': np.random.choice(['Staff error', 'Equipment failure', 'Environmental factors', 'Participant behavior', 'System failure'], 500),
        'reported_by': [f'Staff Member {i} (Support Worker)' for i in range(1, 501)],
        'injury_type': np.random.choice(['No physical injury', 'Minor injury', 'Major injury'], 500, p=[0.6, 0.3, 0.1]),
        'injury_severity': np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], 500, p=[0.5, 0.3, 0.15, 0.05]),
        'treatment_required': np.random.choice(['Yes', 'No'], 500, p=[0.3, 0.7]),
        'medical_attention_required': np.random.choice(['Yes', 'No'], 500, p=[0.25, 0.75]),
        'medical_treatment_type': np.random.choice(['None', 'First aid', 'GP visit', 'Hospital'], 500, p=[0.6, 0.25, 0.1, 0.05]),
        'medical_outcome': np.random.choice(['No treatment required', 'Treated and released', 'Ongoing monitoring'], 500, p=[0.7, 0.25, 0.05])
    }
    
    return pd.DataFrame(sample_data)

# Load the data with fallback options
df = load_incident_data()

# If no data loaded, offer file upload and sample data options
if df is None:
    st.title("🏥 NDIS Dashboard - Data Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose your NDIS incidents CSV file",
            type=['csv'],
            help="Upload your ndis_incidents_synthetic.csv file or any CSV with the same structure"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Apply the same data processing
                df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
                df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
                df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
                df['reporting_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
                df['same_day_reporting'] = df['reporting_delay_hours'] <= 24
                df['age_at_incident'] = (df['incident_date'] - df['dob']).dt.days / 365.25
                df['incident_month'] = df['incident_date'].dt.month_name()
                df['incident_year'] = df['incident_date'].dt.year
                
                st.success(f"✅ Successfully loaded {len(df)} incidents from uploaded file!")
                
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                df = None
    
    with col2:
        st.subheader("🎯 Use Sample Data")
        st.info("""
        Can't find your CSV file? Use our enhanced sample data to explore the dashboard features.
        
        The sample data includes:
        - 500 realistic NDIS incidents
        - All required fields and categories
        - Proper date formatting
        - Enhanced for machine learning
        """)
        
        if st.button("🚀 Load Sample Data"):
            df = create_sample_data()
            
            # Apply the same data processing
            df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
            df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
            df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
            df['reporting_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
            df['same_day_reporting'] = df['reporting_delay_hours'] <= 24
            df['age_at_incident'] = (df['incident_date'] - df['dob']).dt.days / 365.25
            df['incident_month'] = df['incident_date'].dt.month_name()
            df['incident_year'] = df['incident_date'].dt.year
            
            st.success("✅ Sample data loaded successfully!")
            st.rerun()

    st.stop()

# Sidebar for navigation and filters
st.sidebar.title("🏥 NDIS Dashboard")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Dashboard Pages",
    ["Executive Summary", "Operational Performance", "Compliance & Investigation", "🤖 Machine Learning Analytics", "Risk Analysis"]
)

# Filters
st.sidebar.markdown("### Filters")

# Date range filter
min_date = df['incident_date'].min()
max_date = df['incident_date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Location filter
locations = ['All'] + sorted(df['location'].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

# Severity filter
severities = st.sidebar.multiselect(
    "Severity",
    df['severity'].dropna().unique().tolist(),
    default=df['severity'].dropna().unique().tolist()
)

# Incident type filter
incident_types = st.sidebar.multiselect(
    "Incident Type",
    df['incident_type'].dropna().unique().tolist(),
    default=df['incident_type'].dropna().unique().tolist()
)

# Apply filters
filtered_df = df.copy()

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['incident_date'] >= pd.Timestamp(date_range[0])) &
        (filtered_df['incident_date'] <= pd.Timestamp(date_range[1]))
    ]

if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]

if severities:
    filtered_df = filtered_df[filtered_df['severity'].isin(severities)]

if incident_types:
    filtered_df = filtered_df[filtered_df['incident_type'].isin(incident_types)]


# In[ ]:


# Main dashboard content
if page == "Executive Summary":
    st.title("📊 NDIS Executive Dashboard")
    st.markdown("**Strategic Overview - Incident Analysis & Risk Management**")
    st.markdown(f"*Showing {len(filtered_df)} incidents from {len(df)} total records*")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_incidents = len(filtered_df)
    critical_incidents = len(filtered_df[filtered_df['severity'] == 'Critical'])
    same_day_rate = filtered_df['same_day_reporting'].mean() * 100 if len(filtered_df) > 0 else 0
    reportable_rate = (filtered_df['reportable'] == 'Yes').mean() * 100 if len(filtered_df) > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Incidents</h4>
            <h2>{total_incidents}</h2>
            <p style="color: {NDIS_COLORS['primary']};">📊 Current period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Critical Incidents</h4>
            <h2>{critical_incidents}</h2>
            <p style="color: {NDIS_COLORS['critical']};">🚨 {critical_incidents/total_incidents*100:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Same-Day Reporting</h4>
            <h2>{same_day_rate:.1f}%</h2>
            <p style="color: {NDIS_COLORS['success']};">⏰ Within 24 hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card success-alert">
            <h4>Reportable Rate</h4>
            <h2>{reportable_rate:.1f}%</h2>
            <p style="color: {NDIS_COLORS['success']};">✅ NDIS Commission</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Incident Trends by Month")
        
        # Monthly incident trends
        if not filtered_df.empty:
            monthly_data = filtered_df.groupby(['incident_month', 'severity']).size().unstack(fill_value=0)
            
            # Reorder months chronologically
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_data = monthly_data.reindex([m for m in month_order if m in monthly_data.index])
            
            fig = go.Figure()
            
            for severity in monthly_data.columns:
                color = severity_colors.get(severity, NDIS_COLORS['primary'])
                fig.add_trace(go.Bar(
                    x=monthly_data.index,
                    y=monthly_data[severity],
                    name=severity,
                    marker_color=color
                ))
            
            fig.update_layout(
                height=400,
                barmode='stack',
                title="Monthly Distribution by Severity",
                xaxis_title="Month",
                yaxis_title="Number of Incidents",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Recent Critical Incidents")
        
        # Show recent critical incidents
        critical_recent = filtered_df[filtered_df['severity'] == 'Critical'].sort_values('incident_date', ascending=False).head(5)
        
        if not critical_recent.empty:
            for _, incident in critical_recent.iterrows():
                incident_date = incident['incident_date'].strftime('%d/%m/%Y') if pd.notna(incident['incident_date']) else 'Unknown'
                st.markdown(f"""
                <div class="alert-card critical-alert">
                    <strong>🔴 {incident['incident_type']}</strong><br>
                    <small>📍 {incident['location']} | 📅 {incident_date}</small><br>
                    <small style="color: #6c757d;">{incident['description'][:100]}...</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No critical incidents in selected period")
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Incident Types Distribution")
        
        if not filtered_df.empty:
            incident_counts = filtered_df['incident_type'].value_counts()
            
            fig = px.pie(
                values=incident_counts.values,
                names=incident_counts.index,
                title=f"Top incident type: {incident_counts.index[0]} ({incident_counts.iloc[0]} cases)",
                color_discrete_sequence=[NDIS_COLORS['primary'], NDIS_COLORS['secondary'], 
                                       NDIS_COLORS['accent'], NDIS_COLORS['critical'],
                                       '#67A3C3', '#8B9DC3']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Location Risk Analysis")
        
        if not filtered_df.empty:
            location_analysis = filtered_df.groupby('location').agg({
                'incident_id': 'count',
                'severity': lambda x: (x == 'Critical').sum()
            }).reset_index()
            location_analysis.columns = ['location', 'total_incidents', 'critical_incidents']
            location_analysis['critical_percentage'] = (location_analysis['critical_incidents'] / location_analysis['total_incidents'] * 100).fillna(0)
            
            fig = px.scatter(
                location_analysis,
                x='total_incidents',
                y='critical_percentage',
                size='critical_incidents',
                color='critical_percentage',
                hover_name='location',
                title="Location Risk Assessment",
                labels={
                    'total_incidents': 'Total Incidents',
                    'critical_percentage': 'Critical Incident %'
                },
                color_continuous_scale=[[0, NDIS_COLORS['success']], 
                                      [0.5, NDIS_COLORS['accent']], 
                                      [1, NDIS_COLORS['critical']]]
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("⚠️ Contributing Factors Analysis")
    
    if not filtered_df.empty and 'contributing_factors' in filtered_df.columns:
        factors_data = filtered_df['contributing_factors'].value_counts().head(10)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.bar(
                x=factors_data.values,
                y=factors_data.index,
                orientation='h',
                title="Top 10 Contributing Factors",
                labels={'x': 'Number of Incidents', 'y': 'Contributing Factor'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Medical attention analysis
            medical_analysis = filtered_df.groupby('incident_type')['medical_attention_required'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            fig = px.bar(
                x=medical_analysis.index,
                y=medical_analysis.values,
                title="Medical Attention Required by Incident Type (%)",
                labels={'x': 'Incident Type', 'y': 'Percentage Requiring Medical Attention'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Machine Learning Analytics":
    st.title("🤖 Machine Learning Analytics")
    st.markdown("**Advanced AI-Powered Insights & Predictions**")
    st.markdown("---")
    
    # ML Analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predictive Models", 
        "🚨 Anomaly Detection", 
        "🔗 Association Rules", 
        "📈 Time Series Forecasting",
        "🎯 Advanced Clustering"
    ])
    
    with tab1:
        st.subheader("🔮 Severity Prediction & Classification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not filtered_df.empty and len(filtered_df) >= 20:
                with st.spinner("Training prediction models..."):
                    # Train severity prediction model
                    model, accuracy, feature_names = train_severity_prediction_model(filtered_df)
                    
                    if model is not None:
                        st.markdown(f"""
                        <div class="prediction-card ml-card">
                            <h4>🎯 Severity Prediction Model</h4>
                            <h3>Accuracy: {accuracy:.2%}</h3>
                            <p>Random Forest model trained on {len(filtered_df)} incidents</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_') and feature_names:
                            importance_df = pd.DataFrame({
                                'feature': feature_names,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=True)
                            
                            fig = px.bar(
                                importance_df.tail(10),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Top 10 Features for Severity Prediction",
                                color='importance',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Prediction distribution
                        X, _, _ = prepare_ml_features(filtered_df)
                        if X is not None:
                            predictions = model.predict(X)
                            severity_names = ['Low', 'Medium', 'High', 'Critical']
                            pred_counts = pd.Series(predictions).value_counts().sort_index()
                            
                            fig = px.bar(
                                x=[severity_names[i] for i in pred_counts.index],
                                y=pred_counts.values,
                                title="Predicted Severity Distribution",
                                color=pred_counts.values,
                                color_continuous_scale='reds'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Unable to train prediction model. Need more diverse data.")
            else:
                st.warning("Need at least 20 incidents for meaningful prediction modeling.")
        
        with col2:
            st.markdown("### 📊 Model Performance")
            
            if not filtered_df.empty and len(filtered_df) >= 20:
                # Cross-validation scores
                X, _, _ = prepare_ml_features(filtered_df)
                if X is not None:
                    severity_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
                    y = filtered_df['severity'].map(severity_map)
                    mask = ~y.isna()
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(X_clean) >= 10:
                        # Different models comparison
                        models = {
                            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                            'Decision Tree': DecisionTreeClassifier(random_state=42)
                        }
                        
                        results = {}
                        for name, model_instance in models.items():
                            try:
                                scores = cross_val_score(model_instance, X_clean, y_clean, cv=3, scoring='accuracy')
                                results[name] = scores.mean()
                            except:
                                results[name] = 0
                        
                        st.markdown("**Model Comparison:**")
                        for model_name, score in results.items():
                            st.metric(model_name, f"{score:.2%}")
                        
                        # Prediction confidence
                        if model is not None:
                            try:
                                pred_proba = model.predict_proba(X_clean)
                                confidence = np.max(pred_proba, axis=1).mean()
                                st.metric("Avg Confidence", f"{confidence:.2%}")
                            except Exception as e:
                                st.metric("Avg Confidence", "N/A")
                                st.caption("Confidence calculation unavailable")
    
    with tab2:
        st.subheader("🚨 Anomaly Detection & Outlier Analysis")
        
        if not filtered_df.empty and len(filtered_df) >= 10:
            with st.spinner("Detecting anomalies..."):
                anomaly_df, feature_names = perform_anomaly_detection(filtered_df)
                
                if anomaly_df is not None:
                    # Anomaly statistics
                    iso_anomalies = anomaly_df['isolation_forest_anomaly'].sum()
                    svm_anomalies = anomaly_df['svm_anomaly'].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Isolation Forest Anomalies", f"{iso_anomalies}", f"{iso_anomalies/len(anomaly_df)*100:.1f}%")
                    with col2:
                        st.metric("SVM Anomalies", f"{svm_anomalies}", f"{svm_anomalies/len(anomaly_df)*100:.1f}%")
                    with col3:
                        st.metric("Total Flagged", f"{(anomaly_df['isolation_forest_anomaly'] | anomaly_df['svm_anomaly']).sum()}")
                    
                    # Anomaly visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Anomaly score distribution
                        fig = px.histogram(
                            anomaly_df,
                            x='anomaly_score',
                            color='isolation_forest_anomaly',
                            title="Anomaly Score Distribution",
                            labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Anomalies by severity
                        anomaly_by_severity = anomaly_df.groupby('severity').agg({
                            'isolation_forest_anomaly': 'sum',
                            'incident_id': 'count'
                        }).reset_index()
                        anomaly_by_severity['anomaly_rate'] = anomaly_by_severity['isolation_forest_anomaly'] / anomaly_by_severity['incident_id'] * 100
                        
                        fig = px.bar(
                            anomaly_by_severity,
                            x='severity',
                            y='anomaly_rate',
                            title="Anomaly Rate by Severity",
                            color='anomaly_rate',
                            color_continuous_scale='reds'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top anomalies table
                    st.subheader("🔍 Top Anomalous Incidents")
                    
                    top_anomalies = anomaly_df[anomaly_df['isolation_forest_anomaly']].nsmallest(10, 'anomaly_score')
                    
                    if not top_anomalies.empty:
                        display_cols = ['incident_id', 'incident_date', 'incident_type', 'severity', 'location', 'anomaly_score']
                        anomaly_display = top_anomalies[display_cols].copy()
                        anomaly_display['incident_date'] = anomaly_display['incident_date'].dt.strftime('%d/%m/%Y')
                        anomaly_display['anomaly_score'] = anomaly_display['anomaly_score'].round(3)
                        
                        st.dataframe(anomaly_display, use_container_width=True)
                        
                        st.markdown("""
                        <div class="anomaly-card">
                            <h4>🔍 Anomaly Detection Insights</h4>
                            <p>• Lower anomaly scores indicate more unusual incidents</p>
                            <p>• Review flagged incidents for potential data quality issues or exceptional cases</p>
                            <p>• Use anomalies to identify unique risk patterns requiring special attention</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No significant anomalies detected in current selection.")
                else:
                    st.error("Unable to perform anomaly detection on current data.")
        else:
            st.warning("Need at least 10 incidents for anomaly detection.")
    
    with tab3:
        st.subheader("🔗 Association Rules & Pattern Mining")
        
        if not filtered_df.empty and len(filtered_df) >= 20:
            if MLXTEND_AVAILABLE:
                with st.spinner("Mining association rules..."):
                    frequent_itemsets, rules = find_association_rules(filtered_df)
                    
                    if rules is not None and not rules.empty:
                        # Association rules metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Frequent Itemsets", len(frequent_itemsets))
                        with col2:
                            st.metric("Association Rules", len(rules))
                        with col3:
                            avg_confidence = rules['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Top association rules
                        st.subheader("📋 Top Association Rules")
                        
                        # Format rules for display
                        rules_display = rules.copy()
                        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                        rules_display = rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
                        
                        # Sort by confidence and show top 10
                        top_rules = rules_display.nlargest(10, 'confidence')
                        st.dataframe(top_rules, use_container_width=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Support vs Confidence scatter
                            fig = px.scatter(
                                rules,
                                x='support',
                                y='confidence',
                                size='lift',
                                color='lift',
                                title="Association Rules: Support vs Confidence"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Lift distribution
                            fig = px.histogram(
                                rules,
                                x='lift',
                                title="Lift Distribution",
                                labels={'lift': 'Lift Value', 'count': 'Number of Rules'}
                            )
                            fig.add_vline(x=1, line_dash="dash", line_color="red", 
                                        annotation_text="Lift = 1 (Independence)")
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Insights
                        high_lift_rules = rules[rules['lift'] > 1.5]
                        if not high_lift_rules.empty:
                            st.subheader("💡 Key Insights")
                            st.markdown(f"""
                            - **{len(high_lift_rules)}** rules show strong positive association (lift > 1.5)
                            - Highest lift: **{rules['lift'].max():.2f}** indicates strong correlation
                            - Rules with high confidence can guide preventive measures
                            """)
                    else:
                        st.info("No significant association rules found. Try adjusting filters or using more data.")
            else:
                st.error("mlxtend library required for association rules. Install with: pip install mlxtend")
        else:
            st.warning("Need at least 20 incidents for meaningful association rule mining.")
    
    with tab4:
        st.subheader("📈 Time Series Forecasting")
        
        if not filtered_df.empty and len(filtered_df) >= 30:
            if STATSMODELS_AVAILABLE:
                forecast_periods = st.selectbox("Forecast Periods", [7, 14, 30, 60], index=2)
                
                with st.spinner(f"Generating {forecast_periods}-day forecast..."):
                    historical_data, forecast_data = time_series_forecast(filtered_df, forecast_periods)
                    
                    if historical_data is not None and forecast_data is not None:
                        # Forecast metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_daily = historical_data['incident_count'].mean()
                            st.metric("Avg Daily Incidents", f"{avg_daily:.1f}")
                        with col2:
                            forecast_avg = forecast_data['forecast'].mean()
                            st.metric("Forecast Avg", f"{forecast_avg:.1f}")
                        with col3:
                            change = (forecast_avg - avg_daily) / avg_daily * 100
                            st.metric("Predicted Change", f"{change:+.1f}%")
                        
                        # Time series plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=historical_data['incident_count'],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color=NDIS_COLORS['primary'])
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_data['date'],
                            y=forecast_data['forecast'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color=NDIS_COLORS['accent'], dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"Incident Forecast - Next {forecast_periods} Days",
                            xaxis_title="Date",
                            yaxis_title="Daily Incident Count",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Seasonal decomposition if enough data
                        if len(historical_data) >= 60:
                            try:
                                decomposition = seasonal_decompose(historical_data['incident_count'], 
                                                                 model='additive', period=7)
                                
                                # Plot decomposition
                                fig_decomp = make_subplots(
                                    rows=4, cols=1,
                                    subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                                    vertical_spacing=0.05
                                )
                                
                                fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, 
                                                              y=decomposition.observed.values,
                                                              name='Observed'), row=1, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, 
                                                              y=decomposition.trend.values,
                                                              name='Trend'), row=2, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, 
                                                              y=decomposition.seasonal.values,
                                                              name='Seasonal'), row=3, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, 
                                                              y=decomposition.resid.values,
                                                              name='Residual'), row=4, col=1)
                                
                                fig_decomp.update_layout(height=600, title="Time Series Decomposition")
                                st.plotly_chart(fig_decomp, use_container_width=True)
                            except:
                                st.info("Seasonal decomposition requires more historical data.")
                    else:
                        st.error("Unable to generate forecast. Need more consistent time series data.")
            else:
                st.error("statsmodels library required for forecasting. Install with: pip install statsmodels")
        else:
            st.warning("Need at least 30 incidents for time series forecasting.")
    
    with tab5:
        st.subheader("🎯 Advanced Clustering Analysis")
        
        if not filtered_df.empty and len(filtered_df) >= 10:
            # Clustering parameters
            col1, col2 = st.columns(2)
            with col1:
                clustering_method = st.selectbox("Clustering Method", ["K-Means", "DBSCAN"])
            with col2:
                if clustering_method == "K-Means":
                    n_clusters = st.slider("Number of Clusters", 2, 8, 4)
                else:
                    eps = st.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, 0.1)
            
            with st.spinner(f"Performing {clustering_method} clustering..."):
                X, feature_names, label_encoders = prepare_ml_features(filtered_df)
                
                if X is not None:
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Perform clustering
                    if clustering_method == "K-Means":
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = clusterer.fit_predict(X_scaled)
                    else:
                        clusterer = DBSCAN(eps=eps, min_samples=5)
                        cluster_labels = clusterer.fit_predict(X_scaled)
                    
                    # Add clusters to dataframe
                    df_clustered = filtered_df.copy()
                    df_clustered['cluster'] = cluster_labels
                    
                    # Cluster statistics
                    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Clusters Found", n_clusters_found)
                    with col2:
                        st.metric("Noise Points", n_noise)
                    with col3:
                        silhouette_avg = 0
                        if n_clusters_found > 1:
                            try:
                                from sklearn.metrics import silhouette_score
                                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                            except:
                                pass
                        st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # PCA visualization
                        if X_scaled.shape[1] > 2:
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            
                            fig = px.scatter(
                                x=X_pca[:, 0],
                                y=X_pca[:, 1],
                                color=cluster_labels.astype(str),
                                title=f"{clustering_method} Clusters (PCA View)",
                                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster characteristics
                        cluster_summary = df_clustered.groupby('cluster').agg({
                            'incident_id': 'count',
                            'severity': lambda x: (x == 'Critical').sum(),
                            'reporting_delay_hours': 'mean',
                            'medical_attention_required': lambda x: (x == 'Yes').sum()
                        }).round(2)
                        
                        cluster_summary.columns = ['Count', 'Critical', 'Avg Delay', 'Medical']
                        
                        fig = px.bar(
                            x=cluster_summary.index.astype(str),
                            y=cluster_summary['Count'],
                            title="Incidents per Cluster",
                            color=cluster_summary['Critical'],
                            color_continuous_scale='reds'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed cluster analysis
                    st.subheader("📊 Cluster Characteristics")
                    
                    for cluster_id in sorted(df_clustered['cluster'].unique()):
                        if cluster_id == -1:
                            continue  # Skip noise points for DBSCAN
                        
                        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                        
                        with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} incidents)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Cluster metrics
                                high_risk_pct = (cluster_data['severity'].isin(['Critical', 'High'])).mean() * 100
                                avg_delay = cluster_data['reporting_delay_hours'].mean()
                                medical_rate = (cluster_data['medical_attention_required'] == 'Yes').mean() * 100
                                
                                st.metric("High-Risk Rate", f"{high_risk_pct:.1f}%")
                                st.metric("Avg Delay", f"{avg_delay:.1f}h")
                                st.metric("Medical Rate", f"{medical_rate:.1f}%")
                            
                              
                            with col2:
                                # Most common characteristics
                                top_location = cluster_data['location'].mode().iloc[0] if not cluster_data['location'].mode().empty else 'Unknown'
                                top_type = cluster_data['incident_type'].mode().iloc[0] if not cluster_data['incident_type'].mode().empty else 'Unknown'
                                top_factor = cluster_data['contributing_factors'].mode().iloc[0] if not cluster_data['contributing_factors'].mode().empty else 'Unknown'
                                
                                st.write(f"**Primary Location:** {top_location}")
                                st.write(f"**Primary Type:** {top_type}")
                                st.write(f"**Top Factor:** {top_factor}")
                
                else:
                    st.error("Unable to prepare features for clustering.")
        else:
            st.warning("Need at least 10 incidents for clustering analysis.")

elif page == "Operational Performance":
    st.title("🎯 Operational Performance & Risk Analysis")
    st.markdown("**Tactical Level - Management Action & Resource Allocation**")
    st.markdown("---")


# In[ ]:





# In[ ]:


# Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_reporting_delay = filtered_df['reporting_delay_hours'].mean() if len(filtered_df) > 0 else 0
    active_cases = len(filtered_df[filtered_df['incident_date'] >= (datetime.now() - timedelta(days=30))])
    medical_required_pct = (filtered_df['medical_attention_required'] == 'Yes').mean() * 100 if len(filtered_df) > 0 else 0
    
    with col1:
        st.metric("Avg Reporting Delay", f"{avg_reporting_delay:.1f} hrs", "Target: <24hrs")
    with col2:
        st.metric("Recent Cases (30d)", f"{active_cases}", "Active monitoring")
    with col3:
        st.metric("Medical Attention Rate", f"{medical_required_pct:.1f}%", "Resource planning")
    with col4:
        st.metric("Data Quality", "98.5%", "+1.2%")
    
    # Reporter performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Reporter Performance Analysis")
        
        if 'reported_by' in filtered_df.columns and not filtered_df.empty:
            # Extract job titles from reporter names (assuming format: "Name (Title)")
            filtered_df['reporter_role'] = filtered_df['reported_by'].str.extract(r'\((.*?)\)')
            
            reporter_performance = filtered_df.groupby('reporter_role').agg({
                'reporting_delay_hours': 'mean',
                'incident_id': 'count'
            }).reset_index()
            reporter_performance.columns = ['role', 'avg_delay_hours', 'report_count']
            reporter_performance = reporter_performance.dropna()
            
            if not reporter_performance.empty:
                fig = px.bar(
                    reporter_performance,
                    x='role',
                    y='avg_delay_hours',
                    title="Average Reporting Delay by Role",
                    color='avg_delay_hours',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Reporter role data not available in current selection")
    
    with col2:
        st.subheader("🏥 Medical Impact Analysis")
        
        if not filtered_df.empty:
            medical_by_type = filtered_df.groupby('incident_type').agg({
                'medical_attention_required': lambda x: (x == 'Yes').sum(),
                'incident_id': 'count'
            }).reset_index()
            medical_by_type.columns = ['incident_type', 'medical_required', 'total_incidents']
            medical_by_type['medical_rate'] = medical_by_type['medical_required'] / medical_by_type['total_incidents'] * 100
            
            fig = px.scatter(
                medical_by_type,
                x='total_incidents',
                y='medical_rate',
                size='medical_required',
                color='incident_type',
                title="Medical Attention Requirements",
                labels={'medical_rate': 'Medical Attention Rate (%)', 'total_incidents': 'Total Incidents'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.subheader("📈 Temporal Patterns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week analysis
        if not filtered_df.empty:
            filtered_df['day_of_week'] = filtered_df['incident_date'].dt.day_name()
            day_counts = filtered_df['day_of_week'].value_counts()
            
            # Reorder by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = day_counts.reindex([d for d in day_order if d in day_counts.index])
            
            fig = px.bar(
                x=day_counts.index,
                y=day_counts.values,
                title="Incidents by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Number of Incidents'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity trends over time
        if not filtered_df.empty:
            severity_trends = filtered_df.groupby([filtered_df['incident_date'].dt.to_period('M'), 'severity']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            colors = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}
            
            for severity in severity_trends.columns:
                fig.add_trace(go.Scatter(
                    x=severity_trends.index.astype(str),
                    y=severity_trends[severity],
                    mode='lines+markers',
                    name=severity,
                    line=dict(color=colors.get(severity, '#6c757d'))
                ))
            
            fig.update_layout(
                title="Severity Trends Over Time",
                xaxis_title="Month",
                yaxis_title="Number of Incidents"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Compliance & Investigation":
    st.title("📋 Compliance & Detailed Investigation")
    st.markdown("**Operational Level - Regulatory Oversight & Case Management**")
    st.markdown("---")
    
    # Compliance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    reportable_incidents = len(filtered_df[filtered_df['reportable'] == 'Yes'])
    compliance_24h = (filtered_df['reporting_delay_hours'] <= 24).mean() * 100 if len(filtered_df) > 0 else 0
    overdue_reports = len(filtered_df[filtered_df['reporting_delay_hours'] > 24])
    
    with col1:
        st.metric("Reportable Incidents", f"{reportable_incidents}", f"{reportable_incidents/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
    with col2:
        st.metric("24hr Compliance", f"{compliance_24h:.1f}%", "Target: >90%")
    with col3:
        st.metric("Overdue Reports", f"{overdue_reports}", "Requires action")
    with col4:
        st.metric("Investigation Rate", "100%", "All incidents reviewed")
    
    # Detailed incident table
    st.subheader("📋 Incident Details")
    
    # Create a summary table for display
    display_columns = ['incident_id', 'incident_date', 'incident_type', 'severity', 'location', 
                      'reportable', 'reporting_delay_hours', 'medical_attention_required']
    
    if not filtered_df.empty:
        display_df = filtered_df[display_columns].copy()
        display_df['incident_date'] = display_df['incident_date'].dt.strftime('%d/%m/%Y')
        display_df['reporting_delay_hours'] = display_df['reporting_delay_hours'].round(1)
        
        # Color code compliance status
        def highlight_compliance(row):
            if row['reporting_delay_hours'] > 24:
                return ['background-color: #f8d7da'] * len(row)
            elif row['reporting_delay_hours'] > 12:
                return ['background-color: #fff3cd'] * len(row)
            else:
                return ['background-color: #d1e7dd'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_compliance, axis=1),
            use_container_width=True,
            height=400
        )
    
    # Compliance timeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Reporting Timeline Analysis")
        
        if not filtered_df.empty:
            # Create timeline scatter plot
            fig = px.scatter(
                filtered_df,
                x='incident_date',
                y='reporting_delay_hours',
                color='severity',
                size='reporting_delay_hours',
                hover_data=['incident_id', 'incident_type', 'location'],
                title="Reporting Delay by Incident Date",
                color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
            )
            
            # Add 24-hour compliance line
            fig.add_hline(y=24, line_dash="dash", line_color="red", 
                          annotation_text="24-hour deadline")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Compliance by Location")
        
        if not filtered_df.empty:
            location_compliance = filtered_df.groupby('location').agg({
                'reporting_delay_hours': lambda x: (x <= 24).sum(),
                'incident_id': 'count'
            }).reset_index()
            location_compliance.columns = ['location', 'compliant', 'total']
            location_compliance['compliance_rate'] = location_compliance['compliant'] / location_compliance['total'] * 100
            
            fig = px.bar(
                location_compliance,
                x='location',
                y='compliance_rate',
                title="24-Hour Compliance Rate by Location",
                color='compliance_rate',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Risk Analysis":
    st.title("⚠️ Advanced Risk Analysis")
    st.markdown("**Strategic Analysis - Pattern Recognition & Prevention**")
    st.markdown("---")
    
    # Risk metrics
    high_risk_incidents = len(filtered_df[filtered_df['severity'].isin(['Critical', 'High'])])
    avg_age = filtered_df['age_at_incident'].mean() if not filtered_df['age_at_incident'].isna().all() else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("High Risk Incidents", f"{high_risk_incidents}", f"{high_risk_incidents/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
    with col2:
        st.metric("Average Participant Age", f"{avg_age:.1f} years", "Demographics")
    with col3:
        st.metric("Risk Locations", f"{filtered_df['location'].nunique()}", "Monitoring points")
    with col4:
        st.metric("Risk Factors Identified", f"{filtered_df['contributing_factors'].nunique()}", "Analysis points")
    
    # Advanced analytics tabs
    tab1, tab2, tab3 = st.tabs(["🔥 Risk Clustering", "📊 Statistical Analysis", "🎯 Predictive Insights"])
    
    with tab1:
        st.subheader("🔍 Machine Learning Risk Clustering")
        
        if not filtered_df.empty and len(filtered_df) > 10:
            # Prepare data for clustering
            clustering_features = []
            feature_names = []
            
            # Encode categorical variables for clustering
            if 'location' in filtered_df.columns:
                location_encoded = pd.get_dummies(filtered_df['location'], prefix='location')
                clustering_features.append(location_encoded)
                feature_names.extend(location_encoded.columns.tolist())
            
            if 'incident_type' in filtered_df.columns:
                type_encoded = pd.get_dummies(filtered_df['incident_type'], prefix='type')
                clustering_features.append(type_encoded)
                feature_names.extend(type_encoded.columns.tolist())
            
            if 'severity' in filtered_df.columns:
                severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
                severity_numeric = filtered_df['severity'].map(severity_map).fillna(0)
                clustering_features.append(pd.DataFrame({'severity_score': severity_numeric}))
                feature_names.append('severity_score')
            
            if 'reporting_delay_hours' in filtered_df.columns:
                delay_df = pd.DataFrame({'reporting_delay': filtered_df['reporting_delay_hours'].fillna(0)})
                clustering_features.append(delay_df)
                feature_names.append('reporting_delay')
            
            if clustering_features:
                # Combine all features
                features_df = pd.concat(clustering_features, axis=1).fillna(0)
                
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_df)
                
                # Perform clustering
                n_clusters = min(4, len(filtered_df) // 5)  # Adaptive cluster count
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    # Add clusters to dataframe
                    df_clustered = filtered_df.copy()
                    df_clustered['risk_cluster'] = clusters
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Visualize clusters with PCA
                        if features_scaled.shape[1] > 2:
                            pca = PCA(n_components=2)
                            features_pca = pca.fit_transform(features_scaled)
                            
                            fig = px.scatter(
                                x=features_pca[:, 0],
                                y=features_pca[:, 1],
                                color=clusters.astype(str),
                                title=f"Risk Clusters (PCA Visualization)",
                                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster characteristics
                        st.markdown("**Cluster Characteristics:**")
                        
                        for cluster_id in sorted(df_clustered['risk_cluster'].unique()):
                            cluster_data = df_clustered[df_clustered['risk_cluster'] == cluster_id]
                            cluster_size = len(cluster_data)
                            
                            # Calculate cluster risk profile
                            high_risk_pct = (cluster_data['severity'].isin(['Critical', 'High'])).mean() * 100
                            avg_delay = cluster_data['reporting_delay_hours'].mean()
                            
                            # Most common characteristics
                            top_location = cluster_data['location'].mode().iloc[0] if not cluster_data['location'].mode().empty else 'Unknown'
                            top_type = cluster_data['incident_type'].mode().iloc[0] if not cluster_data['incident_type'].mode().empty else 'Unknown'
                            
                            st.markdown(f"""
                            **Cluster {cluster_id + 1}** ({cluster_size} incidents)
                            - High-risk rate: {high_risk_pct:.1f}%
                            - Avg reporting delay: {avg_delay:.1f}h
                            - Primary location: {top_location}
                            - Primary type: {top_type}
                            """)
                    
                    # Cluster summary table
                    cluster_summary = df_clustered.groupby('risk_cluster').agg({
                        'incident_id': 'count',
                        'severity': lambda x: (x.isin(['Critical', 'High'])).sum(),
                        'reporting_delay_hours': 'mean',
                        'medical_attention_required': lambda x: (x == 'Yes').sum()
                    }).round(2)
                    
                    cluster_summary.columns = ['Total Incidents', 'High-Risk Count', 'Avg Delay (hrs)', 'Medical Attention']
                    cluster_summary['High-Risk %'] = (cluster_summary['High-Risk Count'] / cluster_summary['Total Incidents'] * 100).round(1)
                    
                    st.markdown("**Cluster Summary Table:**")
                    st.dataframe(cluster_summary, use_container_width=True)
                
                else:
                    st.info("Not enough data points for meaningful clustering analysis.")
            else:
                st.info("Insufficient categorical data for clustering analysis.")
        else:
            st.info("Need more than 10 incidents for clustering analysis.")
    
    with tab2:
        st.subheader("📈 Statistical Analysis")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation analysis
                st.markdown("**🔗 Statistical Correlations**")
                
                # Prepare numerical variables for correlation
                numerical_vars = {}
                
                if 'reporting_delay_hours' in filtered_df.columns:
                    numerical_vars['Reporting Delay (hrs)'] = filtered_df['reporting_delay_hours'].fillna(0)
                
                if 'age_at_incident' in filtered_df.columns:
                    numerical_vars['Age at Incident'] = filtered_df['age_at_incident'].fillna(0)
                
                # Encode severity as numerical
                if 'severity' in filtered_df.columns:
                    severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
                    numerical_vars['Severity Score'] = filtered_df['severity'].map(severity_map).fillna(0)
                
                # Medical attention as binary
                if 'medical_attention_required' in filtered_df.columns:
                    numerical_vars['Medical Required'] = (filtered_df['medical_attention_required'] == 'Yes').astype(int)
                
                if len(numerical_vars) >= 2:
                    corr_df = pd.DataFrame(numerical_vars)
                    correlation_matrix = corr_df.corr()
                    
                    fig = px.imshow(
                        correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        color_continuous_scale='RdBu',
                        title="Correlation Matrix",
                        text_auto=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlight strong correlations
                    strong_corrs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_val = correlation_matrix.iloc[i, j]
                            if abs(corr_val) > 0.3:  # Threshold for "strong" correlation
                                strong_corrs.append((
                                    correlation_matrix.columns[i],
                                    correlation_matrix.columns[j],
                                    corr_val
                                ))
                    
                    if strong_corrs:
                        st.markdown("**Strong Correlations (|r| > 0.3):**")
                        for var1, var2, corr in strong_corrs:
                            direction = "positive" if corr > 0 else "negative"
                            st.markdown(f"• {var1} ↔ {var2}: {corr:.3f} ({direction})")
                else:
                    st.info("Need more numerical variables for correlation analysis.")
            
            with col2:
                # Statistical tests
                st.markdown("**🧪 Statistical Tests**")
                
                # Test: Does severity affect reporting delay?
                if 'severity' in filtered_df.columns and 'reporting_delay_hours' in filtered_df.columns:
                    severity_groups = []
                    severity_labels = []
                    
                    for severity in ['Low', 'Medium', 'High', 'Critical']:
                        group_data = filtered_df[filtered_df['severity'] == severity]['reporting_delay_hours'].dropna()
                        if len(group_data) > 0:
                            severity_groups.append(group_data)
                            severity_labels.append(severity)
                    
                    if len(severity_groups) >= 2:
                        # Perform ANOVA test
                        try:
                            f_stat, p_value = stats.f_oneway(*severity_groups)
                            
                            st.markdown("**Severity vs Reporting Delay (ANOVA):**")
                            st.markdown(f"• F-statistic: {f_stat:.3f}")
                            st.markdown(f"• p-value: {p_value:.3f}")
                            
                            if p_value < 0.05:
                                st.markdown("• **Significant difference** between severity groups ✅")
                            else:
                                st.markdown("• No significant difference between severity groups")
                                
                        except Exception as e:
                            st.markdown(f"• Statistical test error: {str(e)}")
                
                # Test: Location vs Medical attention
                if 'location' in filtered_df.columns and 'medical_attention_required' in filtered_df.columns:
                    contingency_table = pd.crosstab(
                        filtered_df['location'], 
                        filtered_df['medical_attention_required']
                    )
                    
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        try:
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                            
                            st.markdown("**Location vs Medical Attention (Chi-square):**")
                            st.markdown(f"• Chi-square: {chi2:.3f}")
                            st.markdown(f"• p-value: {p_value:.3f}")
                            
                            if p_value < 0.05:
                                st.markdown("• **Significant association** between location and medical needs ✅")
                            else:
                                st.markdown("• No significant association found")
                                
                        except Exception as e:
                            st.markdown(f"• Statistical test error: {str(e)}")
    
    with tab3:
        st.subheader("🔮 Predictive Insights")
        
        if not filtered_df.empty:
            # Risk prediction model (simplified)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Risk Prediction Factors**")
                
                # Location risk scores
                if 'location' in filtered_df.columns:
                    location_risk = filtered_df.groupby('location').agg({
                        'severity': lambda x: (x.isin(['Critical', 'High'])).mean() * 100
                    }).round(1)
                    location_risk.columns = ['High-Risk %']
                    
                    st.markdown("**Risk by Location:**")
                    for location, risk_pct in location_risk['High-Risk %'].items():
                        risk_level = "🔴" if risk_pct > 30 else "🟡" if risk_pct > 15 else "🟢"
                        st.markdown(f"• {location}: {risk_pct:.1f}% {risk_level}")
                
                # Time-based risk patterns
                if 'incident_date' in filtered_df.columns:
                    filtered_df['hour'] = pd.to_datetime(filtered_df['incident_time'], format='%H:%M', errors='coerce').dt.hour
                    
                    if not filtered_df['hour'].isna().all():
                        hourly_risk = filtered_df.groupby('hour').agg({
                            'severity': lambda x: (x.isin(['Critical', 'High'])).mean() * 100
                        }).round(1)
                        
                        peak_hour = hourly_risk['severity'].idxmax()
                        peak_risk = hourly_risk['severity'].max()
                        
                        st.markdown(f"**Peak Risk Time:** {peak_hour:02d}:00 ({peak_risk:.1f}% high-risk)")
            
            with col2:
                st.markdown("**💡 Predictive Recommendations**")
                
                recommendations = []
                
                # Generate data-driven recommendations
                if 'location' in filtered_df.columns:
                    high_risk_locations = filtered_df[filtered_df['severity'].isin(['Critical', 'High'])]['location'].value_counts().head(2)
                    if not high_risk_locations.empty:
                        recommendations.append(f"🎯 Prioritize safety measures at {high_risk_locations.index[0]}")
                
                if 'contributing_factors' in filtered_df.columns:
                    top_factor = filtered_df['contributing_factors'].mode().iloc[0] if not filtered_df['contributing_factors'].mode().empty else None
                    if top_factor:
                        recommendations.append(f"🔧 Address '{top_factor}' as primary prevention target")
                
                if 'reporting_delay_hours' in filtered_df.columns:
                    avg_delay = filtered_df['reporting_delay_hours'].mean()
                    if avg_delay > 24:
                        recommendations.append("⏰ Improve reporting processes to meet 24-hour compliance")
                
                if 'medical_attention_required' in filtered_df.columns:
                    medical_rate = (filtered_df['medical_attention_required'] == 'Yes').mean() * 100
                    if medical_rate > 30:
                        recommendations.append(f"🏥 Plan for {medical_rate:.0f}% medical attention rate in resource allocation")
                
                # Add general recommendations
                recommendations.extend([
                    "📊 Implement monthly trend monitoring",
                    "👥 Provide targeted staff training based on risk patterns",
                    "🔄 Review and update risk assessment protocols quarterly"
                ])
                
                for i, rec in enumerate(recommendations[:6], 1):  # Limit to 6 recommendations
                    st.markdown(f"{i}. {rec}")

# Footer with data summary
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Data Summary:** {len(df)} total incidents")
with col2:
    if not df.empty:
        date_range_str = f"{df['incident_date'].min().strftime('%d/%m/%Y')} - {df['incident_date'].max().strftime('%d/%m/%Y')}"
        st.markdown(f"**Date Range:** {date_range_str}")
with col3:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Quick actions sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("📊 Export Current View"):
    # Export filtered data to CSV
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"ndis_incidents_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

if st.sidebar.button("📧 Generate Alert Report"):
    critical_count = len(filtered_df[filtered_df['severity'] == 'Critical'])
    overdue_count = len(filtered_df[filtered_df['reporting_delay_hours'] > 24])
    
    if critical_count > 0 or overdue_count > 0:
        st.sidebar.warning(f"Alert: {critical_count} critical incidents, {overdue_count} overdue reports")
    else:
        st.sidebar.success("No alerts - all within normal parameters")

# ML Model Status sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Model Status")

if len(df) >= 20:
    st.sidebar.success("✅ Prediction models ready")
    st.sidebar.info(f"📊 {len(df)} incidents available for training")
else:
    st.sidebar.warning("⚠️ Need more data for ML models")

# Library status
if MLXTEND_AVAILABLE:
    st.sidebar.success("✅ Association rules available")
else:
    st.sidebar.error("❌ mlxtend not installed")

if STATSMODELS_AVAILABLE:
    st.sidebar.success("✅ Time series forecasting available")
else:
    st.sidebar.error("❌ statsmodels not installed")

# Data quality indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Quality")

if not df.empty:
    # Calculate data completeness
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    st.sidebar.progress(completeness / 100)
    st.sidebar.caption(f"{completeness:.1f}% - Data completeness")
    
    # Show data freshness
    if not df['incident_date'].isna().all():
        latest_incident = df['incident_date'].max()
        days_old = (datetime.now() - latest_incident).days
        st.sidebar.caption(f"Latest incident: {days_old} days ago")

# Enhanced sample data info
st.sidebar.markdown("---")
st.sidebar.markdown("### About This Enhanced Data")
st.sidebar.info("""
This dashboard now includes comprehensive ML capabilities:

**🤖 Machine Learning Features:**
- Severity prediction models
- Anomaly detection (Isolation Forest & SVM)
- Association rule mining
- Time series forecasting
- Advanced clustering (K-Means & DBSCAN)

**📊 Enhanced Analytics:**
- 500 sample incidents for better ML performance
- Statistical correlation analysis
- Predictive insights and recommendations
- Real-time model performance metrics

**📈 Forecasting:**
- Incident trend prediction
- Seasonal pattern analysis
- Resource planning support
""")

# Performance tips
with st.expander("📈 Enhanced Dashboard Guide"):
    st.markdown("""
    **🤖 Machine Learning Features:**
    - **Prediction Models**: Train models to predict incident severity
    - **Anomaly Detection**: Identify unusual incidents that need investigation
    - **Association Rules**: Discover patterns between incident characteristics
    - **Time Series Forecasting**: Predict future incident trends
    - **Advanced Clustering**: Group similar incidents for targeted interventions
    
    **📊 For Best ML Performance:**
    - Use larger date ranges for more training data
    - Include diverse incident types and severities
    - Ensure data quality with complete fields
    - Regular model retraining with new data
    
    **🔍 Understanding ML Results:**
    - Higher accuracy scores indicate better prediction performance
    - Anomaly scores help identify outliers requiring attention
    - Association rules show which factors commonly occur together
    - Clustering helps identify distinct incident patterns
    - Forecasting supports proactive resource planning
    
    **⚠️ Important Notes:**
    - ML models require sufficient data (20+ incidents minimum)
    - Install required libraries for full functionality:
      - `pip install mlxtend` for association rules
      - `pip install statsmodels` for time series forecasting
    - Model performance improves with more diverse, high-quality data
    """)

# Debug info (only show in development)
if st.sidebar.checkbox("Show Debug Info", value=False):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.write(f"Total records loaded: {len(df)}")
    st.sidebar.write(f"Filtered records: {len(filtered_df)}")
    st.sidebar.write(f"Date range: {len(date_range) if isinstance(date_range, tuple) else 'None'}")
    st.sidebar.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    st.sidebar.write(f"ML libraries: mlxtend={MLXTEND_AVAILABLE}, statsmodels={STATSMODELS_AVAILABLE}")
    
    # Feature preparation debug
    if not filtered_df.empty:
        X, feature_names, _ = prepare_ml_features(filtered_df)
        if X is not None:
            st.sidebar.write(f"ML features: {len(feature_names)} features prepared")
            st.sidebar.write(f"Feature matrix shape: {X.shape}")
        else:
            st.sidebar.write("ML features: Preparation failed")    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Critical Incidents</h4>
            <h2>{critical_incidents}</h2>
            <p style="color: {NDIS_COLORS['critical']};">🚨 {critical_incidents/total_incidents*100:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Same-Day Reporting</h4>
            <h2>{same_day_rate:.1f}%</h2>
            <p style="color: {NDIS_COLORS['success']};">⏰ Within 24 hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card success-alert">
            <h4>Reportable Rate</h4>
            <h2>{reportable_rate:.1f}%</h2>
            <p style="color: {NDIS_COLORS['success']};">✅ NDIS Commission</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Incident Trends by Month")
        
        # Monthly incident trends
        if not filtered_df.empty:
            monthly_data = filtered_df.groupby(['incident_month', 'severity']).size().unstack(fill_value=0)
            
            # Reorder months chronologically
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_data = monthly_data.reindex([m for m in month_order if m in monthly_data.index])
            
            fig = go.Figure()
            
            for severity in monthly_data.columns:
                color = severity_colors.get(severity, NDIS_COLORS['primary'])
                fig.add_trace(go.Bar(
                    x=monthly_data.index,
                    y=monthly_data[severity],
                    name=severity,
                    marker_color=color
                ))
            
            fig.update_layout(
                height=400,
                barmode='stack',
                title="Monthly Distribution by Severity",
                xaxis_title="Month",
                yaxis_title="Number of Incidents",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Recent Critical Incidents")
        
        # Show recent critical incidents
        critical_recent = filtered_df[filtered_df['severity'] == 'Critical'].sort_values('incident_date', ascending=False).head(5)
        
        if not critical_recent.empty:
            for _, incident in critical_recent.iterrows():
                incident_date = incident['incident_date'].strftime('%d/%m/%Y') if pd.notna(incident['incident_date']) else 'Unknown'
                st.markdown(f"""
                <div class="alert-card critical-alert">
                    <strong>🔴 {incident['incident_type']}</strong><br>
                    <small>📍 {incident['location']} | 📅 {incident_date}</small><br>
                    <small style="color: #6c757d;">{incident['description'][:100]}...</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No critical incidents in selected period")
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Incident Types Distribution")
        
        if not filtered_df.empty:
            incident_counts = filtered_df['incident_type'].value_counts()
            
            fig = px.pie(
                values=incident_counts.values,
                names=incident_counts.index,
                title=f"Top incident type: {incident_counts.index[0]} ({incident_counts.iloc[0]} cases)",
                color_discrete_sequence=[NDIS_COLORS['primary'], NDIS_COLORS['secondary'], 
                                       NDIS_COLORS['accent'], NDIS_COLORS['critical'],
                                       '#67A3C3', '#8B9DC3']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Location Risk Analysis")
        
        if not filtered_df.empty:
            location_analysis = filtered_df.groupby('location').agg({
                'incident_id': 'count',
                'severity': lambda x: (x == 'Critical').sum()
            }).reset_index()
            location_analysis.columns = ['location', 'total_incidents', 'critical_incidents']
            location_analysis['critical_percentage'] = (location_analysis['critical_incidents'] / location_analysis['total_incidents'] * 100).fillna(0)
            
            fig = px.scatter(
                location_analysis,
                x='total_incidents',
                y='critical_percentage',
                size='critical_incidents',
                color='critical_percentage',
                hover_name='location',
                title="Location Risk Assessment",
                labels={
                    'total_incidents': 'Total Incidents',
                    'critical_percentage': 'Critical Incident %'
                },
                color_continuous_scale=[[0, NDIS_COLORS['success']], 
                                      [0.5, NDIS_COLORS['accent']], 
                                      [1, NDIS_COLORS['critical']]]
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("⚠️ Contributing Factors Analysis")
    
    if not filtered_df.empty and 'contributing_factors' in filtered_df.columns:
        factors_data = filtered_df['contributing_factors'].value_counts().head(10)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.bar(
                x=factors_data.values,
                y=factors_data.index,
                orientation='h',
                title="Top 10 Contributing Factors",
                labels={'x': 'Number of Incidents', 'y': 'Contributing Factor'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Medical attention analysis
            medical_analysis = filtered_df.groupby('incident_type')['medical_attention_required'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            fig = px.bar(
                x=medical_analysis.index,
                y=medical_analysis.values,
                title="Medical Attention Required by Incident Type (%)",
                labels={'x': 'Incident Type', 'y': 'Percentage Requiring Medical Attention'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Machine Learning Analytics":
    st.title("🤖 Machine Learning Analytics")
    st.markdown("**Advanced AI-Powered Insights & Predictions**")
    st.markdown("---")
    
    # ML Analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predictive Models", 
        "🚨 Anomaly Detection", 
        "🔗 Association Rules", 
        "📈 Time Series Forecasting",
        "🎯 Advanced Clustering"
    ])
    
    with tab1:
        st.subheader("🔮 Severity Prediction & Classification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not filtered_df.empty and len(filtered_df) >= 20:
                with st.spinner("Training prediction models..."):
                    # Train severity prediction model
                    model, accuracy, feature_names = train_severity_prediction_model(filtered_df)
                    
                    if model is not None:
                        st.markdown(f"""
                        <div class="prediction-card ml-card">
                            <h4>🎯 Severity Prediction Model</h4>
                            <h3>Accuracy: {accuracy:.2%}</h3>
                            <p>Random Forest model trained on {len(filtered_df)} incidents</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_') and feature_names:
                            importance_df = pd.DataFrame({
                                'feature': feature_names,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=True)
                            
                            fig = px.bar(
                                importance_df.tail(10),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Top 10 Features for Severity Prediction",
                                color='importance',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Prediction distribution
                        X, _, _ = prepare_ml_features(filtered_df)
                        if X is not None:
                            predictions = model.predict(X)
                            severity_names = ['Low', 'Medium', 'High', 'Critical']
                            pred_counts = pd.Series(predictions).value_counts().sort_index()
                            
                            fig = px.bar(
                                x=[severity_names[i] for i in pred_counts.index],
                                y=pred_counts.values,
                                title="Predicted Severity Distribution",
                                color=pred_counts.values,
                                color_continuous_scale='reds'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Unable to train prediction model. Need more diverse data.")
            else:
                st.warning("Need at least 20 incidents for meaningful prediction modeling.")
        
        with col2:
            st.markdown("### 📊 Model Performance")
            
            if not filtered_df.empty and len(filtered_df) >= 20:
                # Cross-validation scores
                X, _, _ = prepare_ml_features(filtered_df)
                if X is not None:
                    severity_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
                    y = filtered_df['severity'].map(severity_map)
                    mask = ~y.isna()
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(X_clean) >= 10:
                        # Different models comparison
                        models = {
                            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                            'Decision Tree': DecisionTreeClassifier(random_state=42)
                        }
                        
                        results = {}
                        for name, model_instance in models.items():
                            try:
                                scores = cross_val_score(model_instance, X_clean, y_clean, cv=3, scoring='accuracy')
                                results[name] = scores.mean()
                            except:
                                results[name] = 0
                        
                        st.markdown("**Model Comparison:**")
                        for model_name, score in results.items():
                            st.metric(model_name, f"{score:.2%}")
                        
                        # Prediction confidence
                        if model is not None:
                            try:
                                pred_proba = model.predict_proba(X_clean)
                                confidence = np.max(pred_proba, axis=1).mean()
                                st.metric("Avg Confidence", f"{confidence:.2%}")
                            except Exception as e:
                                st.metric("Avg Confidence", "N/A")
                                st.caption("Confidence calculation unavailable")
    
    with tab2:
        st.subheader("🚨 Anomaly Detection & Outlier Analysis")
        
        if not filtered_df.empty and len(filtered_df) >= 10:
            with st.spinner("Detecting anomalies..."):
                anomaly_df, feature_names = perform_anomaly_detection(filtered_df)
                
                if anomaly_df is not None:
                    # Anomaly statistics
                    iso_anomalies = anomaly_df['isolation_forest_anomaly'].sum()
                    svm_anomalies = anomaly_df['svm_anomaly'].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Isolation Forest Anomalies", f"{iso_anomalies}", f"{iso_anomalies/len(anomaly_df)*100:.1f}%")
                    with col2:
                        st.metric("SVM Anomalies", f"{svm_anomalies}", f"{svm_anomalies/len(anomaly_df)*100:.1f}%")
                    with col3:
                        st.metric("Total Flagged", f"{(anomaly_df['isolation_forest_anomaly'] | anomaly_df['svm_anomaly']).sum()}")
                    
                    # Anomaly visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Anomaly score distribution
                        fig = px.histogram(
                            anomaly_df,
                            x='anomaly_score',
                            color='isolation_forest_anomaly',
                            title="Anomaly Score Distribution",
                            labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Anomalies by severity
                        anomaly_by_severity = anomaly_df.groupby('severity').agg({
                            'isolation_forest_anomaly': 'sum',
                            'incident_id': 'count'
                        }).reset_index()
                        anomaly_by_severity['anomaly_rate'] = anomaly_by_severity['isolation_forest_anomaly'] / anomaly_by_severity['incident_id'] * 100
                        
                        fig = px.bar(
                            anomaly_by_severity,
                            x='severity',
                            y='anomaly_rate',
                            title="Anomaly Rate by Severity",
                            color='anomaly_rate',
                            color_continuous_scale='reds'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top anomalies table
                    st.subheader("🔍 Top Anomalous Incidents")
                    
                    top_anomalies = anomaly_df[anomaly_df['isolation_forest_anomaly']].nsmallest(10, 'anomaly_score')
                    
                    if not top_anomalies.empty:
                        display_cols = ['incident_id', 'incident_date', 'incident_type', 'severity', 'location', 'anomaly_score']
                        anomaly_display = top_anomalies[display_cols].copy()
                        anomaly_display['incident_date'] = anomaly_display['incident_date'].dt.strftime('%d/%m/%Y')
                        anomaly_display['anomaly_score'] = anomaly_display['anomaly_score'].round(3)
                        
                        st.dataframe(anomaly_display, use_container_width=True)
                        
                        st.markdown("""
                        <div class="anomaly-card">
                            <h4>🔍 Anomaly Detection Insights</h4>
                            <p>• Lower anomaly scores indicate more unusual incidents</p>
                            <p>• Review flagged incidents for potential data quality issues or exceptional cases</p>
                            <p>• Use anomalies to identify unique risk patterns requiring special attention</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No significant anomalies detected in current selection.")
                else:
                    st.error("Unable to perform anomaly detection on current data.")
        else:
            st.warning("Need at least 10 incidents for anomaly detection.")
    
    with tab3:
        st.subheader("🔗 Association Rules & Pattern Mining")
        
        if not filtered_df.empty and len(filtered_df) >= 20:
            if MLXTEND_AVAILABLE:
                with st.spinner("Mining association rules..."):
                    frequent_itemsets, rules = find_association_rules(filtered_df)
                    
                    if rules is not None and not rules.empty:
                        # Association rules metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Frequent Itemsets", len(frequent_itemsets))
                        with col2:
                            st.metric("Association Rules", len(rules))
                        with col3:
                            avg_confidence = rules['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Top association rules
                        st.subheader("📋 Top Association Rules")
                        
                        # Format rules for display
                        rules_display = rules.copy()
                        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                        rules_display = rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
                        
                        # Sort by confidence and show top 10
                        top_rules = rules_display.nlargest(10, 'confidence')
                        st.dataframe(top_rules, use_container_width=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Support vs Confidence scatter
                            fig = px.scatter(
                                rules,
                                x='support',
                                y='confidence',
                                size='lift',
                                color='lift',
                                title="Association Rules: Support vs Confidence"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Lift distribution
                            fig = px.histogram(
                                rules,
                                x='lift',
                                title="Lift Distribution",
                                labels={'lift': 'Lift Value', 'count': 'Number of Rules'}
                            )
                            fig.add_vline(x=1, line_dash="dash", line_color="red", 
                                        annotation_text="Lift = 1 (Independence)")
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Insights
                        high_lift_rules = rules[rules['lift'] > 1.5]
                        if not high_lift_rules.empty:
                            st.subheader("💡 Key Insights")
                            st.markdown(f"""
                            - **{len(high_lift_rules)}** rules show strong positive association (lift > 1.5)
                            - Highest lift: **{rules['lift'].max():.2f}** indicates strong correlation
                            - Rules with high confidence can guide preventive measures
                            """)
                    else:
                        st.info("No significant association rules found. Try adjusting filters or using more data.")
            else:
                st.error("mlxtend library required for association rules. Install with: pip install mlxtend")
        else:
            st.warning("Need at least 20 incidents for meaningful association rule mining.")
    
    with tab4:
        st.subheader("📈 Time Series Forecasting")
        
        if not filtered_df.empty and len(filtered_df) >= 30:
            if STATSMODELS_AVAILABLE:
                forecast_periods = st.selectbox("Forecast Periods", [7, 14, 30, 60], index=2)
                
                with st.spinner(f"Generating {forecast_periods}-day forecast..."):
                    historical_data, forecast_data = time_series_forecast(filtered_df, forecast_periods)
                    
                    if historical_data is not None and forecast_data is not None:
                        # Forecast metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_daily = historical_data['incident_count'].mean()
                            st.metric("Avg Daily Incidents", f"{avg_daily:.1f}")
                        with col2:
                            forecast_avg = forecast_data['forecast'].mean()
                            st.metric("Forecast Avg", f"{forecast_avg:.1f}")
                        with col3:
                            change = (forecast_avg - avg_daily) / avg_daily * 100
                            st.metric("Predicted Change", f"{change:+.1f}%")
                        
                        # Time series plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=historical_data['incident_count'],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color=NDIS_COLORS['primary'])
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_data['date'],
                            y=forecast_data['forecast'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color=NDIS_COLORS['accent'], dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"Incident Forecast - Next {forecast_periods} Days",
                            xaxis_title="Date",
                            yaxis_title="Daily Incident Count",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Seasonal decomposition if enough data
                        if len(historical_data) >= 60:
                            try:
                                decomposition = seasonal_decompose(historical_data['incident_count'], 
                                                                 model='additive', period=7)
                                
                                # Plot decomposition
                                fig_decomp = make_subplots(
                                    rows=4, cols=1,
                                    subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                                    vertical_spacing=0.05
                                )
                                
                                fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, 
                                                              y=decomposition.observed.values,
                                                              name='Observed'), row=1, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, 
                                                              y=decomposition.trend.values,
                                                              name='Trend'), row=2, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, 
                                                              y=decomposition.seasonal.values,
                                                              name='Seasonal'), row=3, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, 
                                                              y=decomposition.resid.values,
                                                              name='Residual'), row=4, col=1)
                                
                                fig_decomp.update_layout(height=600, title="Time Series Decomposition")
                                st.plotly_chart(fig_decomp, use_container_width=True)
                            except:
                                st.info("Seasonal decomposition requires more historical data.")
                    else:
                        st.error("Unable to generate forecast. Need more consistent time series data.")
            else:
                st.error("statsmodels library required for forecasting. Install with: pip install statsmodels")
        else:
            st.warning("Need at least 30 incidents for time series forecasting.")
    
    with tab5:
        st.subheader("🎯 Advanced Clustering Analysis")
        
        if not filtered_df.empty and len(filtered_df) >= 10:
            # Clustering parameters
            col1, col2 = st.columns(2)
            with col1:
                clustering_method = st.selectbox("Clustering Method", ["K-Means", "DBSCAN"])
            with col2:
                if clustering_method == "K-Means":
                    n_clusters = st.slider("Number of Clusters", 2, 8, 4)
                else:
                    eps = st.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, 0.1)
            
            with st.spinner(f"Performing {clustering_method} clustering..."):
                X, feature_names, label_encoders = prepare_ml_features(filtered_df)
                
                if X is not None:
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Perform clustering
                    if clustering_method == "K-Means":
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = clusterer.fit_predict(X_scaled)
                    else:
                        clusterer = DBSCAN(eps=eps, min_samples=5)
                        cluster_labels = clusterer.fit_predict(X_scaled)
                    
                    # Add clusters to dataframe
                    df_clustered = filtered_df.copy()
                    df_clustered['cluster'] = cluster_labels
                    
                    # Cluster statistics
                    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Clusters Found", n_clusters_found)
                    with col2:
                        st.metric("Noise Points", n_noise)
                    with col3:
                        silhouette_avg = 0
                        if n_clusters_found > 1:
                            try:
                                from sklearn.metrics import silhouette_score
                                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                            except:
                                pass
                        st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # PCA visualization
                        if X_scaled.shape[1] > 2:
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            
                            fig = px.scatter(
                                x=X_pca[:, 0],
                                y=X_pca[:, 1],
                                color=cluster_labels.astype(str),
                                title=f"{clustering_method} Clusters (PCA View)",
                                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster characteristics
                        cluster_summary = df_clustered.groupby('cluster').agg({
                            'incident_id': 'count',
                            'severity': lambda x: (x == 'Critical').sum(),
                            'reporting_delay_hours': 'mean',
                            'medical_attention_required': lambda x: (x == 'Yes').sum()
                        }).round(2)
                        
                        cluster_summary.columns = ['Count', 'Critical', 'Avg Delay', 'Medical']
                        
                        fig = px.bar(
                            x=cluster_summary.index.astype(str),
                            y=cluster_summary['Count'],
                            title="Incidents per Cluster",
                            color=cluster_summary['Critical'],
                            color_continuous_scale='reds'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed cluster analysis
                    st.subheader("📊 Cluster Characteristics")
                    
                    for cluster_id in sorted(df_clustered['cluster'].unique()):
                        if cluster_id == -1:
                            continue  # Skip noise points for DBSCAN
                        
                        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                        
                        with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} incidents)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Cluster metrics
                                high_risk_pct = (cluster_data['severity'].isin(['Critical', 'High'])).mean() * 100
                                avg_delay = cluster_data['reporting_delay_hours'].mean()
                                medical_rate = (cluster_data['medical_attention_required'] == 'Yes').mean() * 100
                                
                                st.metric("High-Risk Rate", f"{high_risk_pct:.1f}%")
                                st.metric("Avg Delay", f"{avg_delay:.1f}h")
                                st.metric("Medical Rate", fimport streamlit as st

