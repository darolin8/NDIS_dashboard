import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import json

# ML Libraries for advanced analytics
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Association rules libraries
try:
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="NDIS Analytics Dashboard - Enhanced",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
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
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .insight-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .anomaly-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def process_data(df):
    """Process and enhance the loaded data with additional features for ML"""
    try:
        df = df.copy()
        
        # Ensure we have required columns
        required_columns = ['incident_date', 'incident_type', 'severity', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Convert date columns safely
        if 'incident_date' in df.columns:
            df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce', dayfirst=True)
            if df['incident_date'].isna().all():
                st.error("❌ Could not parse incident_date. Please use DD/MM/YYYY or YYYY-MM-DD format.")
                return None
        
        # Handle notification_date
        if 'notification_date' in df.columns:
            df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce', dayfirst=True)
        else:
            delays = np.random.uniform(0, 2, len(df))
            df['notification_date'] = df['incident_date'] + pd.to_timedelta(delays, unit='days')
        
        # Calculate notification delay safely
        if 'notification_date' in df.columns and not df['notification_date'].isna().all():
            df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / (24 * 3600)
            df['notification_delay'] = df['notification_delay'].fillna(0)
        else:
            df['notification_delay'] = 0
        
        # Add time-based columns
        df['month'] = df['incident_date'].dt.month_name()
        df['month_num'] = df['incident_date'].dt.month
        df['day_of_week'] = df['incident_date'].dt.day_name()
        df['day_of_week_num'] = df['incident_date'].dt.dayofweek
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
        df['day_of_month'] = df['incident_date'].dt.day
        df['week_of_year'] = df['incident_date'].dt.isocalendar().week
        
        # Handle incident_time
        if 'incident_time' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour
            except:
                df['hour'] = pd.to_datetime(df['incident_time'], errors='coerce').dt.hour
        else:
            df['hour'] = np.random.randint(6, 22, len(df))
        
        df['hour'] = df['hour'].fillna(12)
        
        # Add time period categories
        df['time_period'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Risk scoring with more granularity
        severity_mapping = {
            'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4,
            'low': 1, 'medium': 2, 'high': 3, 'critical': 4,
            'L': 1, 'M': 2, 'H': 3, 'C': 4
        }
        df['severity_score'] = df['severity'].map(severity_mapping).fillna(1)
        
        # Create composite risk score
        df['risk_score'] = df['severity_score'] * 2
        if 'notification_delay' in df.columns:
            df['risk_score'] += (df['notification_delay'] > 1).astype(int)
        if 'is_weekend' in df.columns:
            df['risk_score'] += df['is_weekend'].astype(int) * 0.5
        
        # Create age groups
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35)
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        else:
            df['age'] = np.random.normal(40, 20, len(df)).clip(18, 85).astype(int)
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Ensure all participants have names
        if 'participant_name' not in df.columns:
            df['participant_name'] = [f'Participant_{i:03d}' for i in range(1, len(df) + 1)]
        
        # Ensure incident_id exists
        if 'incident_id' not in df.columns:
            df['incident_id'] = [f'INC{i:06d}' for i in range(1, len(df) + 1)]
        
        # Add synthetic features for better ML performance
        df['incident_count_by_participant'] = df.groupby('participant_name')['incident_id'].transform('count')
        df['location_incident_rate'] = df.groupby('location')['incident_id'].transform('count') / len(df)
        df['type_severity_interaction'] = df['incident_type'].astype(str) + '_' + df['severity'].astype(str)
        
        # Clean string columns
        string_columns = ['incident_type', 'severity', 'location']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

@st.cache_data
def prepare_ml_features(df):
    """Prepare enhanced features for ML analysis"""
    try:
        if not SKLEARN_AVAILABLE:
            return df, {}
            
        ml_df = df.copy()
        label_encoders = {}
        
        # Encode categorical variables
        categorical_cols = ['incident_type', 'severity', 'location', 'time_period', 'age_group']
        
        for col in categorical_cols:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col] = ml_df[col].fillna('Unknown').astype(str)
                ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col])
                label_encoders[col] = le
        
        # Add interaction features
        if 'location_encoded' in ml_df.columns and 'incident_type_encoded' in ml_df.columns:
            ml_df['location_type_interaction'] = ml_df['location_encoded'] * ml_df['incident_type_encoded']
        
        return ml_df, label_encoders
        
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return df, {}

def predict_future_incidents(df, prediction_days=30):
    """Predict future incident patterns using multiple models"""
    try:
        if not SKLEARN_AVAILABLE:
            return None, None, None
        
        # Prepare time series data
        daily_counts = df.groupby(df['incident_date'].dt.date).size()
        daily_counts = daily_counts.reindex(pd.date_range(daily_counts.index.min(), 
                                                          daily_counts.index.max()), 
                                          fill_value=0)
        
        # Simple moving average prediction
        ma_window = min(7, len(daily_counts) // 4)
        ma_prediction = daily_counts.rolling(window=ma_window).mean().iloc[-1]
        
        # Trend-based prediction
        days_numeric = np.arange(len(daily_counts))
        z = np.polyfit(days_numeric, daily_counts.values, 1)
        p = np.poly1d(z)
        trend_prediction = p(len(daily_counts) + prediction_days)
        
        # ARIMA prediction if available
        arima_prediction = None
        if STATSMODELS_AVAILABLE and len(daily_counts) > 30:
            try:
                model = ARIMA(daily_counts, order=(1,1,1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=prediction_days)
                arima_prediction = forecast.mean()
            except:
                pass
        
        # Prepare feature-based prediction
        ml_df, label_encoders = prepare_ml_features(df)
        
        # Random Forest prediction for incident severity
        severity_predictor = None
        if 'severity_encoded' in ml_df.columns:
            feature_cols = ['hour', 'day_of_week_num', 'month_num', 'age']
            feature_cols = [col for col in feature_cols if col in ml_df.columns]
            
            if len(feature_cols) >= 2:
                X = ml_df[feature_cols].fillna(0)
                y = ml_df['severity_encoded']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_classifier.fit(X_train, y_train)
                
                severity_predictor = {
                    'model': rf_classifier,
                    'accuracy': rf_classifier.score(X_test, y_test),
                    'features': feature_cols,
                    'label_encoder': label_encoders.get('severity')
                }
        
        predictions = {
            'moving_average': ma_prediction,
            'trend_based': trend_prediction,
            'arima': arima_prediction,
            'severity_predictor': severity_predictor,
            'daily_counts': daily_counts
        }
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def enhanced_anomaly_detection(df):
    """Enhanced multi-method anomaly detection"""
    try:
        if not SKLEARN_AVAILABLE:
            return None
        
        ml_df, _ = prepare_ml_features(df)
        
        # Select features for anomaly detection
        feature_cols = [col for col in ml_df.columns if col.endswith('_encoded')]
        numeric_cols = ['age', 'hour', 'severity_score', 'risk_score', 'notification_delay']
        feature_cols.extend([col for col in numeric_cols if col in ml_df.columns])
        
        if len(feature_cols) < 2:
            return None
        
        X = ml_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply multiple anomaly detection methods
        anomaly_results = {}
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_results['isolation_forest'] = iso_forest.fit_predict(X_scaled)
        
        # One-Class SVM
        try:
            svm_detector = OneClassSVM(nu=0.1)
            anomaly_results['one_class_svm'] = svm_detector.fit_predict(X_scaled)
        except:
            pass
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        anomaly_results['lof'] = lof.fit_predict(X_scaled)
        
        # Ensemble anomaly score
        ensemble_scores = np.zeros(len(X))
        for method, scores in anomaly_results.items():
            ensemble_scores += (scores == -1).astype(int)
        
        # Normalize ensemble scores
        ensemble_scores = ensemble_scores / len(anomaly_results)
        
        # Identify anomalies (threshold at 0.5 - majority vote)
        final_anomalies = ensemble_scores >= 0.5
        
        # Calculate anomaly details
        anomaly_indices = np.where(final_anomalies)[0]
        anomaly_data = ml_df.iloc[anomaly_indices]
        
        # Analyze anomaly patterns
        anomaly_patterns = {}
        if len(anomaly_data) > 0:
            for col in ['incident_type', 'severity', 'location']:
                if col in anomaly_data.columns:
                    anomaly_patterns[col] = anomaly_data[col].value_counts().to_dict()
        
        return {
            'anomaly_scores': ensemble_scores,
            'anomaly_flags': final_anomalies,
            'anomaly_indices': anomaly_indices,
            'anomaly_patterns': anomaly_patterns,
            'methods_used': list(anomaly_results.keys()),
            'feature_cols': feature_cols,
            'X_scaled': X_scaled
        }
        
    except Exception as e:
        st.error(f"Enhanced anomaly detection error: {str(e)}")
        return None

def advanced_association_rules(df):
    """Advanced association rule mining with multiple algorithms"""
    try:
        if not MLXTEND_AVAILABLE:
            return None
        
        # Prepare transaction data with more attributes
        transaction_cols = ['incident_type', 'severity', 'location', 'time_period']
        if 'reportable' in df.columns:
            transaction_cols.append('reportable')
        if 'age_group' in df.columns:
            transaction_cols.append('age_group')
        
        # Create transactions
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in transaction_cols:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    transaction.append(f"{col}={str(row[col]).strip()}")
            
            # Add risk level
            if 'risk_score' in row:
                risk_level = 'high_risk' if row['risk_score'] > 5 else 'low_risk'
                transaction.append(f"risk={risk_level}")
            
            if len(transaction) >= 2:
                transactions.append(transaction)
        
        if len(transactions) < 10:
            return None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Try multiple support levels
        support_levels = [0.01, 0.05, 0.1]
        all_rules = []
        
        for min_support in support_levels:
            try:
                # Use FP-Growth for better performance
                frequent_itemsets = fpgrowth(df_transactions, min_support=min_support, use_colnames=True)
                
                if len(frequent_itemsets) > 0:
                    # Generate rules with multiple metrics
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
                    
                    if len(rules) > 0:
                        # Calculate additional metrics
                        rules['conviction'] = np.where(rules['confidence'] == 1, np.inf,
                                                      (1 - rules['consequent support']) / 
                                                      (1 - rules['confidence']))
                        rules['leverage'] = rules['support'] - (rules['antecedent support'] * 
                                                                rules['consequent support'])
                        
                        all_rules.append(rules)
            except:
                continue
        
        if all_rules:
            # Combine and deduplicate rules
            combined_rules = pd.concat(all_rules, ignore_index=True)
            combined_rules = combined_rules.drop_duplicates(subset=['antecedents', 'consequents'])
            
            # Sort by lift and confidence
            combined_rules = combined_rules.sort_values(['lift', 'confidence'], ascending=False)
            
            return combined_rules
        
        return None
        
    except Exception as e:
        st.error(f"Advanced association rules error: {str(e)}")
        return None

def load_data_from_file(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        if len(df) == 0:
            st.error("❌ The uploaded file is empty.")
            return None
        
        st.info(f"📁 Loaded {len(df)} rows from uploaded file")
        processed_df = process_data(df)
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

# Main Application
st.title("🏥 NDIS Incident Analytics Dashboard - Enhanced ML Edition")

# Data loading UI
st.sidebar.subheader("📁 Data Upload")
st.sidebar.markdown("**Upload your NDIS incidents CSV file to begin analysis**")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file", 
    type="csv",
    help="Upload your NDIS incidents data in CSV format"
)

# Load data
df = None
if uploaded_file is not None:
    with st.spinner("Loading and processing your data..."):
        df = load_data_from_file(uploaded_file)
        if df is not None:
            st.sidebar.success("✅ File uploaded and processed successfully!")
        else:
            st.sidebar.error("❌ Error processing file. Please check the format.")

# Check if we have data to work with
if df is None or len(df) == 0:
    st.markdown("""
    # 🏥 Enhanced NDIS Analytics with Advanced ML
    
    ## 📁 Welcome! Please Upload Your Data
    
    This enhanced dashboard now includes:
    - 🔮 **Predictive Analytics** - Forecast future incident patterns
    - 🚨 **Advanced Anomaly Detection** - Multi-algorithm ensemble approach
    - 🔗 **Sophisticated Association Rules** - Discover complex relationships
    
    ### 📋 Required CSV Columns:
    - `incident_date` - Date of incident (DD/MM/YYYY format)
    - `incident_type` - Type of incident
    - `severity` - Severity level (Low, Medium, High, Critical)
    - `location` - Where the incident occurred
    
    ### 🔧 Optional Columns for Enhanced Analysis:
    - `notification_date` - When incident was reported
    - `participant_name` - Participant involved
    - `age` - Participant age
    - `reportable` - Whether incident is reportable (Yes/No)
    - `incident_time` - Time of incident (HH:MM)
    - `description` - Incident description
    """)
    
    # Show sample data format
    st.subheader("📋 Sample Data Format")
    sample_data = pd.DataFrame({
        'incident_date': ['01/01/2024', '02/01/2024', '03/01/2024'],
        'incident_type': ['Fall', 'Medication Error', 'Behavioral'],
        'severity': ['Medium', 'High', 'Low'],
        'location': ['Day Program', 'Residential', 'Community'],
        'reportable': ['Yes', 'No', 'No']
    })
    st.dataframe(sample_data, use_container_width=True)
    st.stop()

# Process data if loaded successfully
try:
    ml_df, label_encoders = prepare_ml_features(df)
    
    st.success(f"✅ Successfully loaded {len(df)} incidents from {df['incident_date'].min().strftime('%B %Y')} to {df['incident_date'].max().strftime('%B %Y')}")
    
except Exception as e:
    st.error(f"❌ Error processing data: {str(e)}")
    st.stop()

# Sidebar controls
st.sidebar.header("🎛️ Analysis Controls")

# Analysis Mode Selection
analysis_mode = st.sidebar.selectbox(
    "🔬 Analysis Mode",
    ["Executive Overview", "🔮 Predictive Analytics", "🚨 Anomaly Detection", "🔗 Association Rules", "Risk Analysis", "Data Explorer"]
)

# Filters
st.sidebar.subheader("🎯 Filters")

# Date range filter
if 'incident_date' in df.columns:
    min_date = df['incident_date'].min().date()
    max_date = df['incident_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['incident_date'].dt.date >= start_date) & 
            (df['incident_date'].dt.date <= end_date)
        ]
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

# Severity filter
if 'severity' in df.columns:
    severity_options = df['severity'].unique()
    severity_filter = st.sidebar.multiselect(
        "⚠️ Severity Level",
        options=severity_options,
        default=severity_options
    )
    df_filtered = df_filtered[df_filtered['severity'].isin(severity_filter)]

# Location filter
if 'location' in df.columns:
    location_options = df['location'].unique()
    location_filter = st.sidebar.multiselect(
        "📍 Location",
        options=location_options,
        default=location_options
    )
    df_filtered = df_filtered[df_filtered['location'].isin(location_filter)]

# Main dashboard content based on mode
if analysis_mode == "Executive Overview":
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Incidents", len(df_filtered))
    
    with col2:
        if 'severity' in df_filtered.columns:
            critical_count = len(df_filtered[df_filtered['severity'].str.lower().isin(['critical', 'high'])])
            st.metric("🚨 High Severity", critical_count)
        else:
            st.metric("🚨 High Severity", "N/A")
    
    with col3:
        if 'risk_score' in df_filtered.columns:
            avg_risk = df_filtered['risk_score'].mean()
            st.metric("⚡ Avg Risk Score", f"{avg_risk:.1f}")
        else:
            st.metric("⚡ Avg Risk Score", "N/A")
    
    with col4:
        if 'participant_name' in df_filtered.columns:
            unique_participants = df_filtered['participant_name'].nunique()
            st.metric("👥 Participants", unique_participants)
        else:
            st.metric("👥 Participants", "N/A")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Incident types
        if 'incident_type' in df_filtered.columns:
            incident_counts = df_filtered['incident_type'].value_counts().head(10)
            fig1 = px.bar(
                x=incident_counts.values,
                y=incident_counts.index,
                orientation='h',
                title="🔝 Top Incident Types",
                labels={'x': 'Count', 'y': 'Incident Type'},
                color=incident_counts.values,
                color_continuous_scale='viridis'
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Risk distribution
        if 'risk_score' in df_filtered.columns:
            fig2 = px.histogram(
                df_filtered,
                x='risk_score',
                nbins=20,
                title="⚡ Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Number of Incidents'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Time series analysis
    st.subheader("📈 Incident Trends")
    if 'incident_date' in df_filtered.columns:
        daily_incidents = df_filtered.groupby(df_filtered['incident_date'].dt.date).size().reset_index()
        daily_incidents.columns = ['Date', 'Count']
        
        fig3 = px.line(
            daily_incidents,
            x='Date',
            y='Count',
            title="Daily Incident Trends",
            markers=True
        )
        fig3.add_scatter(x=daily_incidents['Date'], 
                        y=daily_incidents['Count'].rolling(7).mean(),
                        mode='lines',
                        name='7-day Moving Average',
                        line=dict(dash='dash'))
        st.plotly_chart(fig3, use_container_width=True)

elif analysis_mode == "🔮 Predictive Analytics":
    st.header("🔮 Predictive Analytics")
    st.markdown("*Forecast future incident patterns using advanced machine learning*")
    
    # Prediction controls
    col1, col2, col3 = st.columns(3)
    with col1:
        prediction_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    with col2:
        prediction_confidence = st.slider("Confidence Level (%)", 80, 99, 95)
    with col3:
        if st.button("🚀 Generate Predictions", type="primary"):
            st.session_state['run_predictions'] = True
    
    if st.session_state.get('run_predictions', False):
        with st.spinner("🔮 Generating predictions..."):
            predictions = predict_future_incidents(df_filtered, prediction_days)
            
            if predictions:
                # Display prediction results
                st.subheader("📊 Incident Volume Predictions")
         
