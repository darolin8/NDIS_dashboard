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
    st.warning("mlxtend not available. Install with: pip install mlxtend")

# Time series forecasting imports
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("statsmodels not available. Install with: pip install statsmodels")

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
            'text data/ndis_incidents_synthetic.csv',  # GitHub repo structure
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
            # If no file found, show file upload option
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
        'incident_id': [f'INC-2024-{i:04d}' for i in range(1, 501)],  # Increased to 500 for better ML
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
            <p style="color: {ND}
