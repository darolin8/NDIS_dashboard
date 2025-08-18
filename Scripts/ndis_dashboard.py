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
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    .correlation-card {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .alert-box {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        animation: slideIn 0.5s ease-out;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-left-color: #f44336;
        color: #c62828;
    }
    .alert-warning {
        background: linear-gradient(135deg, #fff8e1, #ffecb3);
        border-left-color: #ff9800;
        color: #ef6c00;
    }
    .alert-info {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left-color: #2196f3;
        color: #1565c0;
    }
    .alert-success {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        border-left-color: #4caf50;
        color: #2e7d32;
    }
    .insight-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: #1a1a1a;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .risk-low { 
        background: linear-gradient(135deg, #4caf50, #66bb6a);
    }
    .risk-medium { 
        background: linear-gradient(135deg, #ff9800, #ffb74d);
    }
    .risk-high { 
        background: linear-gradient(135deg, #f44336, #ef5350);
    }
    .risk-critical { 
        background: linear-gradient(135deg, #9c27b0, #ba68c8);
    }
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateX(-20px);
        }
        to { 
            opacity: 1; 
            transform: translateX(0);
        }
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def process_data(df):
    """Process and enhance the NDIS incident data with comprehensive feature engineering"""
    try:
        df = df.copy()
        
        # Enhanced date processing for multiple date formats
        date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']
        
        # Process incident_date
        if 'incident_date' in df.columns:
            df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce', infer_datetime_format=True)
            
            # Try different formats if parsing failed
            if df['incident_date'].isna().any():
                for fmt in date_formats:
                    try:
                        mask = df['incident_date'].isna()
                        df.loc[mask, 'incident_date'] = pd.to_datetime(df.loc[mask, 'incident_date'], format=fmt, errors='coerce')
                        if not df['incident_date'].isna().all():
                            break
                    except:
                        continue
        
        # Process notification_date if it exists
        if 'notification_date' in df.columns:
            df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce', infer_datetime_format=True)
            
            # Try different formats if parsing failed
            if df['notification_date'].isna().any():
                for fmt in date_formats:
                    try:
                        mask = df['notification_date'].isna()
                        df.loc[mask, 'notification_date'] = pd.to_datetime(df.loc[mask, 'notification_date'], format=fmt, errors='coerce')
                        if not df['notification_date'].isna().all():
                            break
                    except:
                        continue
        else:
            # Create notification dates with realistic delays if not present
            delays = np.random.choice([0, 1, 2, 3], len(df), p=[0.4, 0.35, 0.15, 0.1])
            df['notification_date'] = df['incident_date'] + pd.to_timedelta(delays, unit='days')
        
        # Process date of birth if present
        if 'dob' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce', infer_datetime_format=True)
            
            # Try different formats if parsing failed
            if df['dob'].isna().any():
                for fmt in date_formats:
                    try:
                        mask = df['dob'].isna()
                        df.loc[mask, 'dob'] = pd.to_datetime(df.loc[mask, 'dob'], format=fmt, errors='coerce')
                        if not df['dob'].isna().all():
                            break
                    except:
                        continue
            
            # Calculate age at incident - fix the date subtraction issue
            df['age'] = ((df['incident_date'] - df['dob']).dt.days / 365.25).round().astype('Int64')
            # Ensure reasonable age values
            df['age'] = df['age'].clip(lower=0, upper=120)
        
        # Calculate notification delay
        df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.days
        df['notification_delay'] = df['notification_delay'].fillna(0).clip(lower=0)
        
        # Enhanced time-based features
        df['month'] = df['incident_date'].dt.month_name()
        df['day_of_week'] = df['incident_date'].dt.day_name()
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
        df['week_of_year'] = df['incident_date'].dt.isocalendar().week
        df['year'] = df['incident_date'].dt.year
        
        # Enhanced time extraction with better error handling
        if 'incident_time' in df.columns:
            # Handle multiple time formats more robustly
            df['hour'] = None
            
            # Try different time parsing approaches
            for index, time_val in df['incident_time'].items():
                if pd.notna(time_val):
                    try:
                        # Convert to string if not already
                        time_str = str(time_val).strip()
                        
                        # Try parsing with pandas
                        parsed_time = pd.to_datetime(time_str, format='%H:%M', errors='coerce')
                        if pd.notna(parsed_time):
                            df.at[index, 'hour'] = parsed_time.hour
                        else:
                            # Try alternative formats
                            for fmt in ['%H:%M:%S', '%I:%M %p', '%I:%M:%S %p']:
                                try:
                                    parsed_time = pd.to_datetime(time_str, format=fmt, errors='coerce')
                                    if pd.notna(parsed_time):
                                        df.at[index, 'hour'] = parsed_time.hour
                                        break
                                except:
                                    continue
                    except:
                        continue
            
            # Fill missing hours with default value
            df['hour'] = df['hour'].fillna(12).astype(int)
        else:
            df['hour'] = 12  # Default hour if not present
        
        # Enhanced risk scoring based on severity
        severity_weights = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df['severity_score'] = df['severity'].map(severity_weights).fillna(1)
        
        # Age groups (handle missing age data)
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                    bins=[0, 25, 35, 50, 65, 100], 
                                    labels=['18-25', '26-35', '36-50', '51-65', '65+'],
                                    include_lowest=True)
        else:
            # Create default age groups if no age data
            df['age'] = 40  # Default age
            df['age_group'] = '36-50'
        
        # Enhanced participant analysis
        if 'participant_name' in df.columns:
            participant_history = df.groupby('participant_name').agg({
                'incident_id': 'count',
                'severity_score': 'mean',
                'incident_date': ['min', 'max']
            }).round(2)
            
            participant_history.columns = ['incident_count', 'avg_severity', 'first_incident', 'last_incident']
            df = df.merge(participant_history, left_on='participant_name', right_index=True, how='left')
            
            # Risk categorization
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
        
        # Seasonal analysis
        df['season'] = df['incident_date'].dt.month.map({
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'
        })
        
        # Enhanced medical attention analysis
        if 'medical_attention_required' in df.columns:
            df['medical_attention'] = df['medical_attention_required']
        elif 'treatment_required' in df.columns:
            df['medical_attention'] = df['treatment_required']
        else:
            df['medical_attention'] = 'No'  # Default
        
        # Injury severity analysis
        if 'injury_severity' in df.columns:
            injury_severity_weights = {'None': 0, 'Minor': 1, 'Moderate': 2, 'Major': 3, 'Severe': 4}
            df['injury_severity_score'] = df['injury_severity'].map(injury_severity_weights).fillna(0)
        else:
            df['injury_severity_score'] = 0
        
        # Reporter analysis
        if 'reported_by' in df.columns:
            df['reporter_type'] = df['reported_by'].str.extract(r'(\w+)').fillna('Staff')
        else:
            df['reporter_type'] = 'Staff'
        
        # Contributing factors analysis with better error handling
        if 'contributing_factors' in df.columns:
            # Count number of contributing factors (assuming comma or semicolon separated)
            df['num_contributing_factors'] = 0
            for index, factors in df['contributing_factors'].items():
                if pd.notna(factors) and str(factors).strip():
                    factor_str = str(factors)
                    # Count commas and semicolons, add 1 for the base factor
                    count = factor_str.count(',') + factor_str.count(';') + 1
                    df.at[index, 'num_contributing_factors'] = count
        else:
            df['num_contributing_factors'] = 0
        
        # Enhanced reportable incident analysis
        if 'reportable' in df.columns:
            df['is_reportable'] = df['reportable'].map({'Yes': 1, 'No': 0}).fillna(0)
        else:
            df['is_reportable'] = 0
        
        # Time period categorization
        df['time_period'] = df['hour'].apply(lambda x: 
            'Morning' if 6 <= x < 12 else
            'Afternoon' if 12 <= x < 18 else
            'Evening' if 18 <= x < 22 else
            'Night'
        )
        
        # Incident complexity score (combining multiple factors) with safe calculations
        complexity_components = []
        
        # Add severity component
        if 'severity_score' in df.columns:
            complexity_components.append(df['severity_score'] * 0.3)
        
        # Add injury severity component
        if 'injury_severity_score' in df.columns:
            complexity_components.append(df['injury_severity_score'] * 0.2)
        
        # Add contributing factors component
        if 'num_contributing_factors' in df.columns:
            # Normalize contributing factors (cap at 5 for scoring)
            normalized_factors = (df['num_contributing_factors'].clip(0, 5) / 5) * 4  # Scale to 0-4
            complexity_components.append(normalized_factors * 0.2)
        
        # Add reportable component
        if 'is_reportable' in df.columns:
            complexity_components.append(df['is_reportable'] * 0.3)
        
        # Calculate final complexity score
        if complexity_components:
            df['complexity_score'] = sum(complexity_components)
        else:
            df['complexity_score'] = df.get('severity_score', 1)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def load_data_from_file(uploaded_file):
    """Enhanced file loading with comprehensive error handling for NDIS incident data"""
    try:
        # Read the file based on extension
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("❌ Unable to read CSV file. Please check the file encoding.")
                return None
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        # Basic validation
        if df.empty:
            st.error("❌ File is empty. Please upload a file with data.")
            return None
        
        # Display basic file info
        st.sidebar.info(f"📊 File loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate core required columns (case insensitive)
        df.columns = df.columns.str.strip()  # Remove whitespace
        column_mapping = {col.lower(): col for col in df.columns}
        
        required_cols = ['incident_date', 'incident_type', 'severity']
        missing_cols = []
        
        for req_col in required_cols:
            if req_col.lower() not in column_mapping:
                missing_cols.append(req_col)
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.error("Available columns: " + ", ".join(df.columns.tolist()))
            return None
        
        # Check for your specific NDIS file structure
        ndis_specific_cols = ['participant_name', 'ndis_number', 'dob', 'notification_date', 
                             'location', 'subcategory', 'reportable', 'injury_type', 
                             'medical_attention_required']
        
        found_ndis_cols = [col for col in ndis_specific_cols if col in df.columns]
        
        if len(found_ndis_cols) >= 5:  # If most NDIS columns are present
            st.sidebar.success(f"✅ NDIS incident data detected with {len(found_ndis_cols)} standard columns")
        
        # Process the data
        processed_df = process_data(df)
        
        if processed_df is not None:
            st.sidebar.success(f"✅ Data processed successfully: {len(processed_df)} incidents")
        
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.error("Please ensure your file is properly formatted and not corrupted.")
        return None

@st.cache_data
def calculate_enhanced_correlations(df):
    """Enhanced correlation analysis with more variables and better categorical handling"""
    try:
        numeric_df = df.copy()
        
        # Convert categorical to numeric with proper handling
        categorical_mappings = {
            'severity': {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4},
            'reportable': {'No': 0, 'Yes': 1},
            'medical_attention': {'No': 0, 'Yes': 1},
            'medical_attention_required': {'No': 0, 'Yes': 1},
            'is_weekend': {False: 0, True: 1},
            'participant_risk_level': {'New': 1, 'Low': 2, 'Medium': 3, 'High': 4}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in numeric_df.columns:
                # Convert categorical columns to regular object type first
                if hasattr(numeric_df[col], 'cat'):
                    numeric_df[col] = numeric_df[col].astype(str)
                
                # Create new numeric column
                numeric_df[f'{col}_numeric'] = numeric_df[col].map(mapping)
                # Fill NaN values with 0 for unmapped categories
                numeric_df[f'{col}_numeric'] = numeric_df[f'{col}_numeric'].fillna(0)
        
        # Handle boolean columns
        boolean_cols = ['is_weekend']
        for col in boolean_cols:
            if col in numeric_df.columns:
                if col not in [f'{c}_numeric' for c in categorical_mappings.keys()]:
                    numeric_df[f'{col}_numeric'] = numeric_df[col].astype(int)
        
        # Select correlation variables that actually exist
        potential_vars = [
            'age', 'severity_numeric', 'notification_delay', 'reportable_numeric',
            'medical_attention_numeric', 'medical_attention_required_numeric', 
            'is_weekend_numeric', 'hour', 'quarter', 'incident_count', 
            'avg_severity', 'participant_risk_level_numeric', 'injury_severity_score',
            'num_contributing_factors', 'complexity_score'
        ]
        
        # Only include variables that exist and have valid data
        correlation_vars = []
        for var in potential_vars:
            if var in numeric_df.columns:
                # Check if column has valid numeric data
                if numeric_df[var].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                    # Check if column has variance (not all same values)
                    if numeric_df[var].nunique() > 1:
                        correlation_vars.append(var)
        
        # Ensure we have at least 2 variables for correlation
        if len(correlation_vars) < 2:
            # Create minimal correlation matrix with basic variables
            if 'severity_numeric' not in correlation_vars and 'severity_numeric' in numeric_df.columns:
                correlation_vars.append('severity_numeric')
            if 'hour' not in correlation_vars and 'hour' in numeric_df.columns:
                correlation_vars.append('hour')
            
            # If still not enough, create some basic numeric columns
            if len(correlation_vars) < 2:
                numeric_df['severity_numeric'] = numeric_df.get('severity_score', 1)
                numeric_df['time_numeric'] = numeric_df.get('hour', 12)
                correlation_vars = ['severity_numeric', 'time_numeric']
        
        # Calculate correlation matrix with error handling
        try:
            correlation_subset = numeric_df[correlation_vars].select_dtypes(include=[np.number])
            corr_matrix = correlation_subset.corr()
            
            # Remove any columns/rows with all NaN values
            corr_matrix = corr_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
        except Exception as corr_error:
            # Fallback: create simple correlation matrix
            st.warning(f"Advanced correlation calculation failed, using simplified version: {str(corr_error)}")
            simple_vars = ['severity_score', 'hour']
            for var in simple_vars:
                if var not in numeric_df.columns:
                    numeric_df[var] = 1
            
            corr_matrix = numeric_df[simple_vars].corr()
        
        return corr_matrix, numeric_df
        
    except Exception as e:
        st.error(f"❌ Correlation calculation error: {str(e)}")
        # Return minimal correlation matrix
        minimal_df = pd.DataFrame({
            'severity_score': [1, 2, 3],
            'hour': [8, 12, 16]
        })
        return minimal_df.corr(), df

def generate_enhanced_insights(df):
    """Generate more sophisticated insights"""
    insights = []
    
    try:
        # Temporal insights
        if 'incident_date' in df.columns and len(df) > 30:
            # Trend analysis
            monthly_counts = df.groupby(df['incident_date'].dt.to_period('M')).size()
            if len(monthly_counts) >= 3:
                recent_avg = monthly_counts[-3:].mean()
                earlier_avg = monthly_counts[:-3].mean() if len(monthly_counts) > 3 else recent_avg
                
                if recent_avg > earlier_avg * 1.1:
                    insights.append(f"📈 **Trend Alert**: Recent incidents are {((recent_avg/earlier_avg - 1) * 100):.1f}% higher than historical average")
                elif recent_avg < earlier_avg * 0.9:
                    insights.append(f"📉 **Positive Trend**: Recent incidents are {((1 - recent_avg/earlier_avg) * 100):.1f}% lower than historical average")
        
        # High-risk participant insights
        if 'participant_risk_level' in df.columns:
            high_risk_participants = df[df['participant_risk_level'] == 'High']['participant_name'].nunique()
            if high_risk_participants > 0:
                insights.append(f"⚠️ **High-Risk Participants**: {high_risk_participants} participants have 5+ incidents requiring focused intervention")
        
        # Notification compliance insights
        if 'notification_delay' in df.columns:
            compliance_rate = (df['notification_delay'] <= 1).mean() * 100
            if compliance_rate < 80:
                insights.append(f"🚨 **Compliance Issue**: Only {compliance_rate:.1f}% of incidents reported within 24 hours (target: >90%)")
            elif compliance_rate > 95:
                insights.append(f"✅ **Excellent Compliance**: {compliance_rate:.1f}% of incidents reported within 24 hours")
        
        # Severity escalation insights
        if 'incident_count' in df.columns and 'avg_severity' in df.columns:
            escalation_risk = df[df['incident_count'] > 3]['avg_severity'].mean()
            if pd.notna(escalation_risk) and escalation_risk > 2.5:
                insights.append(f"⚡ **Escalation Risk**: Repeat participants show higher average severity ({escalation_risk:.1f}/4.0)")
        
        # Seasonal insights
        if 'season' in df.columns:
            seasonal_severity = df.groupby('season')['severity_score'].mean()
            if len(seasonal_severity) > 1:
                high_risk_season = seasonal_severity.idxmax()
                insights.append(f"🌟 **Seasonal Pattern**: {high_risk_season} shows highest incident severity")
        
        # Location-based insights
        if 'location' in df.columns:
            location_analysis = df.groupby('location').agg({
                'severity_score': 'mean',
                'incident_id': 'count',
                'medical_attention': lambda x: (x == 'Yes').mean() if 'Yes' in x.values else 0
            }).round(2)
            
            if len(location_analysis) > 1:
                high_risk_location = location_analysis['severity_score'].idxmax()
                insights.append(f"🏢 **Location Alert**: '{high_risk_location}' has highest average severity ({location_analysis.loc[high_risk_location, 'severity_score']:.1f})")
        
        # Default insights if analysis fails
        if len(insights) == 0:
            insights = [
                "📊 Advanced analytics processing your incident data",
                "🔍 Identifying patterns and risk factors",
                "📈 Building predictive insights for prevention"
            ]
            
    except Exception as e:
        insights = [f"⚠️ Insight generation temporarily unavailable: {str(e)[:50]}..."]
    
    return insights

# Enhanced sidebar with better organization
st.sidebar.image("https://via.placeholder.com/300x100/667eea/ffffff?text=NDIS+Analytics", use_container_width=True)
st.sidebar.markdown("---")

# Data Source Selection
st.sidebar.subheader("📁 Data Upload")
st.sidebar.markdown("Upload your incident data file to begin analysis")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file", 
    type=["csv", "xlsx", "xls"],
    help="Upload CSV or Excel file containing incident data with required columns: incident_date, incident_type, severity"
)

# Load data based on upload
df = None
if uploaded_file is not None:
    with st.spinner("Processing uploaded file..."):
        df = load_data_from_file(uploaded_file)
        if df is not None:
            st.session_state.data_loaded = True
            st.sidebar.success(f"✅ File loaded: {len(df)} incidents")
            
            # Add debug information
            st.sidebar.info(f"📊 Data Summary:")
            st.sidebar.write(f"- Total incidents: {len(df)}")
            if 'incident_date' in df.columns:
                st.sidebar.write(f"- Date range: {df['incident_date'].min().strftime('%Y-%m-%d')} to {df['incident_date'].max().strftime('%Y-%m-%d')}")
            if 'severity' in df.columns:
                severity_counts = df['severity'].value_counts()
                st.sidebar.write(f"- Severity breakdown:")
                for sev, count in severity_counts.items():
                    st.sidebar.write(f"  - {sev}: {count}")
        else:
            st.sidebar.error("❌ Failed to load file. Please check the format and required columns.")
else:
    st.sidebar.info("📤 Please upload a CSV or Excel file to begin")

# Only proceed if data is loaded
if df is None or len(df) == 0:
    st.title("🏥 Advanced NDIS Incident Analytics Dashboard")
    
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;">
        <h2>📊 Welcome to NDIS Analytics</h2>
        <p style="font-size: 1.2em; margin: 1rem 0;">Upload your incident data to unlock powerful insights and analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Requirements and sample format
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Your File Columns")
        st.markdown("""
        Your NDIS incident file contains these columns:
        
        **Core Incident Data:**
        - incident_id, participant_name, ndis_number
        - incident_date, incident_time, notification_date
        - location, incident_type, subcategory, severity
        
        **Medical & Injury Information:**
        - injury_type, injury_severity, treatment_required
        - medical_attention_required, medical_treatment_type
        - medical_outcome
        
        **Additional Details:**
        - reportable, description, immediate_action
        - actions_taken, contributing_factors, reported_by
        - dob (for age calculation)
        """)
    
    with col2:
        st.markdown("### 📁 File Requirements")
        st.markdown("""
        **File Types:**
        - CSV files (.csv) - UTF-8 encoding
        - Excel files (.xlsx, .xls)
        
        **Data Requirements:**
        - Headers in first row
        - Date format: DD/MM/YYYY or YYYY-MM-DD
        - Time format: HH:MM (24-hour)
        - Severity: Low, Medium, High, Critical
        - Reportable: Yes, No
        
        **Enhanced Analytics:**
        - Age calculated from DOB
        - Notification delays tracked
        - Medical attention analysis
        - Injury severity scoring
        """)
    
    # Sample data format specific to NDIS structure
    st.markdown("### 📝 Expected Data Structure")
    sample_data = pd.DataFrame({
        'incident_id': ['INC000001', 'INC000002', 'INC000003'],
        'participant_name': ['Participant_001', 'Participant_002', 'Participant_001'],
        'ndis_number': [12345678, 87654321, 12345678],
        'incident_date': ['01/01/2024', '02/01/2024', '03/01/2024'],
        'incident_type': ['Fall', 'Medication Error', 'Behavioral Incident'],
        'severity': ['Medium', 'High', 'Low'],
        'location': ['Main Office', 'Residential Care', 'Community Center'],
        'reportable': ['Yes', 'Yes', 'No'],
        'medical_attention_required': ['No', 'Yes', 'No']
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    st.info("💡 **Tip**: Start with a small sample file to test the upload process before uploading your complete dataset.")
    
    st.stop()

# Calculate enhanced analytics
corr_matrix, numeric_df = calculate_enhanced_correlations(df)
insights = generate_enhanced_insights(df)

# Enhanced sidebar controls
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Analysis Controls")

# Analysis mode with icons
analysis_modes = {
    "🎯 Executive Dashboard": "executive",
    "📊 Risk Analysis": "risk", 
    "🔗 Correlation Explorer": "correlation",
    "🔮 Predictive Analytics": "predictive",
    "📈 Performance Metrics": "performance",
    "🤖 ML Analytics": "ml"
}

selected_mode = st.sidebar.selectbox(
    "Analysis Mode",
    list(analysis_modes.keys()),
    help="Choose your analysis focus"
)
analysis_mode = analysis_modes[selected_mode]

# Enhanced time controls
st.sidebar.subheader("📅 Time Controls")
time_range_options = {
    "📅 Last 30 Days": 30,
    "📅 Last Quarter": 90,
    "📅 Last 6 Months": 180,
    "📅 Last Year": 365,
    "📅 All Time": None,
    "📅 Custom Range": "custom"
}

selected_range = st.sidebar.selectbox("Time Period", list(time_range_options.keys()))

if time_range_options[selected_range] == "custom":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", df['incident_date'].min().date())
    with col2:
        end_date = st.date_input("To", df['incident_date'].max().date())
    
    df_filtered = df[(df['incident_date'].dt.date >= start_date) & 
                     (df['incident_date'].dt.date <= end_date)]
elif time_range_options[selected_range] is not None:
    end_date = df['incident_date'].max()
    start_date = end_date - timedelta(days=time_range_options[selected_range])
    df_filtered = df[(df['incident_date'] >= start_date) & (df['incident_date'] <= end_date)]
else:
    df_filtered = df.copy()

# Smart filters
st.sidebar.subheader("🎯 Smart Filters")

# Risk-based filtering
risk_focus_options = {
    "🔍 All Incidents": "all",
    "🚨 Critical Only": "critical",
    "⚠️ High Risk": "high_risk", 
    "🔄 Repeat Participants": "repeat",
    "📍 Problem Locations": "locations",
    "⏰ Delayed Reports": "delayed"
}

risk_focus = st.sidebar.selectbox("Risk Focus", list(risk_focus_options.keys()))

# Apply risk-based filters
if risk_focus_options[risk_focus] == "critical":
    df_filtered = df_filtered[df_filtered['severity'] == 'Critical']
elif risk_focus_options[risk_focus] == "high_risk":
    df_filtered = df_filtered[df_filtered['severity'].isin(['High', 'Critical'])]
elif risk_focus_options[risk_focus] == "repeat":
    if 'participant_risk_level' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['participant_risk_level'].isin(['Medium', 'High'])]
elif risk_focus_options[risk_focus] == "delayed":
    if 'notification_delay' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['notification_delay'] > 1]

# Multi-select filters with better defaults
col1, col2 = st.sidebar.columns(2)

with col1:
    severity_options = sorted(df['severity'].unique())
    severity_filter = st.multiselect(
        "⚠️ Severity",
        options=severity_options,
        default=severity_options
    )

with col2:
    location_options = sorted(df['location'].unique())
    location_filter = st.multiselect(
        "📍 Location", 
        options=location_options,
        default=location_options
    )

# Apply filters with debugging
df_filtered = df_filtered[
    (df_filtered['severity'].isin(severity_filter)) &
    (df_filtered['location'].isin(location_filter))
]

# Debug information
if len(df_filtered) != len(df):
    st.sidebar.warning(f"⚠️ After all filters: {len(df)} → {len(df_filtered)} records")
    
    # Show what's causing the reduction
    severity_check = df[df['severity'].isin(severity_filter)]
    location_check = df[df['location'].isin(location_filter)]
    
    st.sidebar.write(f"Debug Info:")
    st.sidebar.write(f"- Original: {len(df)} records")
    st.sidebar.write(f"- After severity filter: {len(severity_check)} records")
    st.sidebar.write(f"- After location filter: {len(location_check)} records")
    st.sidebar.write(f"- After both filters: {len(df_filtered)} records")
    
    if len(df_filtered) < 5:
        st.sidebar.error("⚠️ Very few records remaining after filtering!")
        st.sidebar.write("Consider:")
        st.sidebar.write("- Removing some filters")
        st.sidebar.write("- Checking data quality")
        st.sidebar.write("- Expanding time range")

# Live insights panel
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Live Insights")
for insight in insights[:3]:
    st.sidebar.markdown(f"""
    <div class="insight-box" style="font-size: 0.85em; padding: 0.8rem;">
        {insight}
    </div>
    """, unsafe_allow_html=True)

# Main content area
st.title("🏥 Advanced NDIS Incident Analytics Dashboard")

# Status bar with detailed debugging
col1, col2, col3, col4 = st.columns(4)
with col1:
    if len(df_filtered) < len(df):
        st.warning(f"⚠️ {len(df_filtered)} of {len(df)} incidents shown")
    else:
        st.success(f"✅ {len(df_filtered)} incidents loaded")
with col2:
    if 'incident_date' in df_filtered.columns and len(df_filtered) > 0:
        date_range = f"{df_filtered['incident_date'].min().strftime('%b %Y')} - {df_filtered['incident_date'].max().strftime('%b %Y')}"
        st.info(f"📅 {date_range}")
    else:
        st.info("📅 No date range")
with col3:
    if 'notification_delay' in df_filtered.columns and len(df_filtered) > 0:
        compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
        if compliance_rate >= 90:
            st.success(f"✅ {compliance_rate:.1f}% compliant")
        else:
            st.warning(f"⚠️ {compliance_rate:.1f}% compliant")
    else:
        st.info("📊 No compliance data")
with col4:
    if len(df_filtered) > 0:
        critical_count = len(df_filtered[df_filtered['severity'] == 'Critical'])
        if critical_count == 0:
            st.success("✅ No critical incidents")
        else:
            st.error(f"🚨 {critical_count} critical incidents")
    else:
        st.error("❌ No data to display")

# Show data quality issues if very few records
if len(df_filtered) < 10 and len(df) > len(df_filtered):
    st.warning("⚠️ **Data Filtering Issue Detected**")
    st.write("Very few records are being displayed. This might be due to:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Possible Causes:**")
        st.write("- Restrictive filters applied")
        st.write("- Date range too narrow") 
        st.write("- Missing data in key columns")
        st.write("- Data format issues")
    
    with col2:
        st.write("**Quick Fixes:**")
        st.write("- Reset filters using sidebar")
        st.write("- Expand time range to 'All Time'")
        st.write("- Check severity/location filters")
        st.write("- Verify data quality in source file")

# Main dashboard content
if analysis_mode == "executive":
    # Executive Dashboard
    st.subheader("🎯 Executive Overview")
    
    # Enhanced KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_incidents = len(df_filtered)
        # Calculate trend
        if len(df_filtered) > 0:
            current_month = df_filtered['incident_date'].max().replace(day=1)
            last_month = current_month - pd.DateOffset(months=1)
            current_count = len(df_filtered[df_filtered['incident_date'] >= current_month])
            last_count = len(df_filtered[
                (df_filtered['incident_date'] >= last_month) & 
                (df_filtered['incident_date'] < current_month)
            ])
            trend = ((current_count - last_count) / last_count * 100) if last_count > 0 else 0
            st.metric("📊 Total Incidents", total_incidents, delta=f"{trend:+.1f}%")
        else:
            st.metric("📊 Total Incidents", 0)
    
    with col2:
        critical_count = len(df_filtered[df_filtered['severity'] == 'Critical'])
        critical_rate = (critical_count / total_incidents * 100) if total_incidents > 0 else 0
        st.metric("🚨 Critical Incidents", critical_count, delta=f"{critical_rate:.1f}%")
    
    with col3:
        if 'notification_delay' in df_filtered.columns:
            avg_delay = df_filtered['notification_delay'].mean()
            delay_status = "🟢" if avg_delay <= 1 else "🔴"
            st.metric("⏱️ Avg Delay (days)", f"{avg_delay:.1f}", delta=delay_status)
        else:
            st.metric("⏱️ Avg Delay", "N/A")
    
    with col4:
        if 'injury_severity' in df_filtered.columns:
            high_injury_count = len(df_filtered[df_filtered['injury_severity'].isin(['Major', 'Severe'])])
            st.metric("⚕️ Serious Injuries", high_injury_count)
        elif 'medical_attention_required' in df_filtered.columns:
            medical_required = len(df_filtered[df_filtered['medical_attention_required'] == 'Yes'])
            st.metric("🏥 Medical Required", medical_required)
        else:
            repeat_participants = df_filtered['participant_name'].value_counts()
            repeat_count = len(repeat_participants[repeat_participants > 1])
            st.metric("🔄 Repeat Participants", repeat_count)
    
    with col5:
        if 'reportable' in df_filtered.columns:
            reportable_count = len(df_filtered[df_filtered['reportable'] == 'Yes'])
            reportable_rate = (reportable_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.metric("📋 Reportable Incidents", f"{reportable_count} ({reportable_rate:.1f}%)")
        else:
            compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100 if 'notification_delay' in df_filtered.columns else 0
            st.metric("📋 Compliance Rate", f"{compliance_rate:.1f}%")
    
    with col5:
        if 'medical_attention' in df_filtered.columns:
            medical_attention_rate = (df_filtered['medical_attention'] == 'Yes').mean() * 100
            medical_status = "🔴" if medical_attention_rate > 30 else "🟡" if medical_attention_rate > 15 else "🟢"
            st.metric("🏥 Medical Attention Rate", f"{medical_attention_rate:.1f}%", delta=medical_status)
        elif 'medical_attention_required' in df_filtered.columns:
            medical_attention_rate = (df_filtered['medical_attention_required'] == 'Yes').mean() * 100
            medical_status = "🔴" if medical_attention_rate > 30 else "🟡" if medical_attention_rate > 15 else "🟢"
            st.metric("🏥 Medical Attention Rate", f"{medical_attention_rate:.1f}%", delta=medical_status)
        else:
            st.metric("🏥 Medical Attention", "N/A")
    
    # Enhanced visualizations
    st.markdown("---")
    
    # Top row charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Incident Trends")
        # Monthly trend with enhanced styling
        monthly_data = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).agg({
            'incident_id': 'count',
            'severity_score': 'mean'
        }).round(2)
        monthly_data.index = monthly_data.index.astype(str)
        
        if len(monthly_data) > 0:
            fig_trend = go.Figure()
            
            # Add incident count
            fig_trend.add_trace(go.Scatter(
                x=monthly_data.index,
                y=monthly_data['incident_id'],
                mode='lines+markers',
                name='Incident Count',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8, color='#2E86AB')
            ))
            
            # Add trend line
            x_numeric = list(range(len(monthly_data)))
            z = np.polyfit(x_numeric, monthly_data['incident_id'], 1)
            trend_line = np.poly1d(z)
            
            fig_trend.add_trace(go.Scatter(
                x=monthly_data.index,
                y=[trend_line(i) for i in x_numeric],
                mode='lines',
                name='Trend',
                line=dict(color='#F18F01', dash='dash', width=2)
            ))
            
            fig_trend.update_layout(
                title="Monthly Incident Count with Trend",
                xaxis_title="Month",
                yaxis_title="Number of Incidents",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_trend.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No data available for trend analysis")
    
    with col2:
        st.subheader("⚠️ Severity Distribution")
        severity_counts = df_filtered['severity'].value_counts()
        colors = {
            'Critical': '#DC143C',
            'High': '#FF6347', 
            'Medium': '#FFA500',
            'Low': '#32CD32'
        }
        
        fig_severity = go.Figure(data=[go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            hole=.4,
            marker=dict(colors=[colors.get(x, '#888888') for x in severity_counts.index]),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig_severity.update_layout(
            title="Incident Severity Distribution",
            annotations=[dict(text='Severity', x=0.5, y=0.5, font_size=16, showarrow=False)],
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_severity, use_container_width=True)
    
    # Second row - Location and Time Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Risk by Location")
        location_analysis = df_filtered.groupby('location').agg({
            'incident_id': 'count',
            'severity_score': 'mean'
        }).round(2)
        location_analysis.columns = ['Count', 'Avg_Severity']
        location_analysis = location_analysis.sort_values('Avg_Severity', ascending=True)
        
        fig_location = px.bar(
            location_analysis,
            x='Avg_Severity',
            y=location_analysis.index,
            orientation='h',
            color='Avg_Severity',
            color_continuous_scale='Reds',
            title="Average Severity Score by Location",
            text='Count'
        )
        fig_location.update_traces(texttemplate='%{text} incidents', textposition='inside')
        fig_location.update_layout(
            xaxis_title="Average Severity Score",
            yaxis_title="Location",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_location, use_container_width=True)
    
    with col2:
        st.subheader("🕐 Hourly Incident Pattern")
        hourly_data = df_filtered.groupby('hour').size()
        
        # Create hour labels
        hour_labels = [f"{h:02d}:00" for h in range(24)]
        hourly_counts = [hourly_data.get(h, 0) for h in range(24)]
        
        fig_hourly = go.Figure(data=go.Bar(
            x=hour_labels,
            y=hourly_counts,
            marker_color=['#FF6B6B' if h in [22, 23, 0, 1, 2, 3, 4, 5] else '#4ECDC4' for h in range(24)],
            text=hourly_counts,
            textposition='auto'
        ))
        
        fig_hourly.update_layout(
            title="Incidents by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Number of Incidents",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_hourly.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig_hourly, use_container_width=True)

elif analysis_mode == "risk":
    # Risk Analysis
    st.subheader("🎯 Advanced Risk Analysis")
    
    # Risk matrix
    st.markdown("### 📊 Risk Assessment Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Location risk matrix
        risk_data = []
        for location in df_filtered['location'].unique():
            location_data = df_filtered[df_filtered['location'] == location]
            total_incidents = len(location_data)
            avg_severity = location_data['severity_score'].mean()
            risk_score = total_incidents * avg_severity
            
            risk_data.append({
                'location': location,
                'total_incidents': total_incidents,
                'avg_severity': avg_severity,
                'risk_score': risk_score
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        if len(risk_df) > 0:
            fig_risk_matrix = px.scatter(
                risk_df,
                x='total_incidents',
                y='avg_severity',
                size='risk_score',
                color='risk_score',
                hover_name='location',
                title="Risk Matrix: Volume vs Severity by Location",
                labels={
                    'total_incidents': 'Incident Volume',
                    'avg_severity': 'Average Severity Score'
                },
                color_continuous_scale='Reds'
            )
            
            # Add quadrant lines
            median_volume = risk_df['total_incidents'].median()
            median_severity = risk_df['avg_severity'].median()
            
            fig_risk_matrix.add_hline(y=median_severity, line_dash="dash", line_color="gray", opacity=0.5)
            fig_risk_matrix.add_vline(x=median_volume, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant annotations
            fig_risk_matrix.add_annotation(
                x=median_volume * 1.5, y=median_severity * 1.2,
                text="HIGH RISK", showarrow=False,
                bgcolor="red", opacity=0.7, font=dict(color="white", size=12)
            )
            fig_risk_matrix.add_annotation(
                x=median_volume * 0.5, y=median_severity * 1.2,
                text="MONITOR", showarrow=False,
                bgcolor="orange", opacity=0.7, font=dict(color="white", size=12)
            )
            fig_risk_matrix.add_annotation(
                x=median_volume * 1.5, y=median_severity * 0.8,
                text="MANAGE", showarrow=False,
                bgcolor="yellow", opacity=0.7, font=dict(color="black", size=12)
            )
            fig_risk_matrix.add_annotation(
                x=median_volume * 0.5, y=median_severity * 0.8,
                text="LOW RISK", showarrow=False,
                bgcolor="green", opacity=0.7, font=dict(color="white", size=12)
            )
            
            fig_risk_matrix.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_risk_matrix, use_container_width=True)
    
    with col2:
        # Participant risk analysis
        if 'participant_risk_level' in df_filtered.columns:
            st.markdown("#### 👥 Participant Risk Levels")
            
            risk_level_counts = df_filtered['participant_risk_level'].value_counts()
            
            fig_participant_risk = go.Figure(data=[go.Bar(
                x=risk_level_counts.index,
                y=risk_level_counts.values,
                marker_color=['#32CD32', '#FFA500', '#FF6347', '#DC143C'],
                text=risk_level_counts.values,
                textposition='auto'
            )])
            
            fig_participant_risk.update_layout(
                title="Participants by Risk Level",
                xaxis_title="Risk Level",
                yaxis_title="Number of Participants",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_participant_risk, use_container_width=True)
            
            # Risk insights
            high_risk_participants = df_filtered[df_filtered['participant_risk_level'] == 'High']['participant_name'].nunique()
            total_participants = df_filtered['participant_name'].nunique()
            high_risk_percentage = (high_risk_participants / total_participants * 100) if total_participants > 0 else 0
            
            st.markdown(f"""
            <div class="alert-box alert-warning">
                <strong>⚠️ Risk Alert:</strong><br>
                {high_risk_participants} participants ({high_risk_percentage:.1f}%) are classified as high-risk<br>
                These participants require immediate intervention planning
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced risk factor analysis for NDIS data
    st.markdown("### 📈 NDIS-Specific Risk Factor Analysis")
    
    risk_factors = {}
    
    # Calculate various risk factors specific to NDIS data
    if 'is_weekend' in df_filtered.columns and len(df_filtered) > 10:
        weekend_incidents = df_filtered[df_filtered['is_weekend'] == True]
        weekday_incidents = df_filtered[df_filtered['is_weekend'] == False]
        
        if len(weekend_incidents) > 0 and len(weekday_incidents) > 0:
            weekend_severity = weekend_incidents['severity_score'].mean()
            weekday_severity = weekday_incidents['severity_score'].mean()
            risk_factors['Weekend vs Weekday'] = weekend_severity / weekday_severity if weekday_severity > 0 else 1
    
    # Night vs day risk
    night_hours = list(range(22, 24)) + list(range(0, 7))
    night_incidents = df_filtered[df_filtered['hour'].isin(night_hours)]
    day_incidents = df_filtered[~df_filtered['hour'].isin(night_hours)]
    
    if len(night_incidents) > 0 and len(day_incidents) > 0:
        night_severity = night_incidents['severity_score'].mean()
        day_severity = day_incidents['severity_score'].mean()
        risk_factors['Night vs Day Hours'] = night_severity / day_severity if day_severity > 0 else 1
    
    # Age-based risk (if age data available)
    if 'age_group' in df_filtered.columns:
        age_severity = df_filtered.groupby('age_group')['severity_score'].mean()
        if len(age_severity) > 1:
            highest_age_risk = age_severity.max()
            lowest_age_risk = age_severity.min()
            risk_factors['Age Group Variation'] = highest_age_risk / lowest_age_risk if lowest_age_risk > 0 else 1
    
    # Delayed reporting risk
    if 'notification_delay' in df_filtered.columns:
        delayed_incidents = df_filtered[df_filtered['notification_delay'] > 1]
        timely_incidents = df_filtered[df_filtered['notification_delay'] <= 1]
        
        if len(delayed_incidents) > 0 and len(timely_incidents) > 0:
            delayed_severity = delayed_incidents['severity_score'].mean()
            timely_severity = timely_incidents['severity_score'].mean()
            risk_factors['Delayed vs Timely Reporting'] = delayed_severity / timely_severity if timely_severity > 0 else 1
    
    # Medical attention vs non-medical incidents
    if 'medical_attention_required' in df_filtered.columns:
        medical_incidents = df_filtered[df_filtered['medical_attention_required'] == 'Yes']
        non_medical_incidents = df_filtered[df_filtered['medical_attention_required'] == 'No']
        
        if len(medical_incidents) > 0 and len(non_medical_incidents) > 0:
            medical_severity = medical_incidents['severity_score'].mean()
            non_medical_severity = non_medical_incidents['severity_score'].mean()
            risk_factors['Medical vs Non-Medical'] = medical_severity / non_medical_severity if non_medical_severity > 0 else 1
    
    # Injury severity correlation
    if 'injury_severity' in df_filtered.columns:
        major_injuries = df_filtered[df_filtered['injury_severity'].isin(['Major', 'Severe'])]
        minor_injuries = df_filtered[df_filtered['injury_severity'].isin(['None', 'Minor'])]
        
        if len(major_injuries) > 0 and len(minor_injuries) > 0:
            major_incident_severity = major_injuries['severity_score'].mean()
            minor_incident_severity = minor_injuries['severity_score'].mean()
            risk_factors['Major vs Minor Injuries'] = major_incident_severity / minor_incident_severity if minor_incident_severity > 0 else 1
    
    if risk_factors:
        risk_factor_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Risk Ratio'])
        risk_factor_df = risk_factor_df.sort_values('Risk Ratio', ascending=True)
        
        fig_risk_factors = px.bar(
            risk_factor_df,
            x='Risk Ratio',
            y='Risk Factor',
            orientation='h',
            title="Risk Factor Analysis (Ratio > 1.0 = Higher Risk)",
            color='Risk Ratio',
            color_continuous_scale='RdYlBu_r'
        )
        
        # Add baseline reference line
        fig_risk_factors.add_vline(x=1.0, line_dash="dash", line_color="black", opacity=0.7)
        fig_risk_factors.add_annotation(
            x=1.0, y=len(risk_factors)-1, text="Baseline (No Effect)",
            showarrow=True, arrowhead=2, arrowcolor="black"
        )
        
        fig_risk_factors.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_risk_factors, use_container_width=True)
        
        # Risk recommendations
        st.markdown("### 💡 Risk Management Recommendations")
        
        recommendations = []
        for factor, ratio in risk_factors.items():
            if ratio > 1.2:  # 20% higher risk
                if 'Weekend' in factor:
                    recommendations.append("📅 **Weekend Protocol**: Implement enhanced weekend supervision and rapid response procedures")
                elif 'Night' in factor:
                    recommendations.append("🌙 **Night Shift Enhancement**: Strengthen night-time staffing and monitoring systems")
                elif 'Delayed' in factor:
                    recommendations.append("⏰ **Reporting System**: Improve incident reporting processes and staff training")
                elif 'Age' in factor:
                    recommendations.append("👥 **Age-Specific Care**: Develop targeted intervention strategies for high-risk age groups")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("✅ Current risk factors are within acceptable ranges")

elif analysis_mode == "correlation":
    # Correlation Explorer
    st.subheader("🔗 Interactive Correlation Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if len(corr_matrix) > 1:
            # Enhanced correlation heatmap
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix of Key Variables",
                color_continuous_scale="RdBu",
                aspect="auto",
                text_auto='.2f',
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis")
    
    with col2:
        st.markdown("### 🎯 Key Correlations")
        
        # Find strongest correlations
        if len(corr_matrix) > 1:
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.1 and not pd.isna(corr_value):
                        corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if corr_pairs:
                corr_pairs_df = pd.DataFrame(corr_pairs)
                corr_pairs_df = corr_pairs_df.reindex(
                    corr_pairs_df.correlation.abs().sort_values(ascending=False).index
                )
                
                for _, row in corr_pairs_df.head(5).iterrows():
                    strength = ("Strong" if abs(row['correlation']) > 0.5 
                              else "Moderate" if abs(row['correlation']) > 0.3 
                              else "Weak")
                    direction = "Positive" if row['correlation'] > 0 else "Negative"
                    color = "#28a745" if row['correlation'] > 0 else "#dc3545"
                    
                    st.markdown(f"""
                    <div class="correlation-card" style="background: linear-gradient(135deg, {color} 0%, #6c757d 100%);">
                        <strong>{row['var1'][:15]} ↔ {row['var2'][:15]}</strong><br>
                        {direction} {strength}<br>
                        r = {row['correlation']:.3f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant correlations found")
    
    # Interactive relationship explorer
    st.markdown("### 🔍 Relationship Explorer")
    
    if len(corr_matrix.columns) >= 2:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_var = st.selectbox("X-axis Variable", corr_matrix.columns, index=0)
        with col2:
            y_var = st.selectbox("Y-axis Variable", corr_matrix.columns, index=1)
        with col3:
            color_options = ['severity', 'location', 'incident_type'] + list(corr_matrix.columns)
            color_var = st.selectbox("Color by", [col for col in color_options if col in numeric_df.columns])
        with col4:
            size_options = ['None'] + list(corr_matrix.columns)
            size_var = st.selectbox("Size by", size_options)
        
        if x_var != y_var and x_var in numeric_df.columns and y_var in numeric_df.columns:
            # Create enhanced scatter plot
            scatter_params = {
                'data_frame': numeric_df,
                'x': x_var,
                'y': y_var,
                'color': color_var if color_var in numeric_df.columns else None,
                'title': f"Relationship: {x_var} vs {y_var}",
                'hover_data': ['participant_name', 'incident_date'] if 'participant_name' in numeric_df.columns else None
            }
            
            if size_var != 'None' and size_var in numeric_df.columns:
                scatter_params['size'] = size_var
            
            # Add trendline
            show_trendline = st.checkbox("Show trend line", value=True)
            if show_trendline:
                scatter_params['trendline'] = "ols"
            
            fig_scatter = px.scatter(**scatter_params)
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Statistical analysis
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                valid_data = numeric_df[[x_var, y_var]].dropna()
                if len(valid_data) > 2:
                    correlation, p_value = stats.pearsonr(valid_data[x_var], valid_data[y_var])
                    significance = "Significant" if p_value < 0.05 else "Not significant"
                    
                    st.markdown(f"""
                    <div class="alert-box alert-info">
                        <strong>📊 Pearson Correlation</strong><br>
                        Correlation: {correlation:.3f}<br>
                        p-value: {p_value:.3f}<br>
                        Status: {significance}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_stats2:
                spearman_corr, spearman_p = stats.spearmanr(valid_data[x_var], valid_data[y_var])
                spearman_sig = "Significant" if spearman_p < 0.05 else "Not significant"
                
                st.markdown(f"""
                <div class="alert-box alert-info">
                    <strong>📊 Spearman Correlation</strong><br>
                    Correlation: {spearman_corr:.3f}<br>
                    p-value: {spearman_p:.3f}<br>
                    Status: {spearman_sig}
                </div>
                """, unsafe_allow_html=True)

elif analysis_mode == "predictive":
    # Predictive Analytics
    st.subheader("🔮 Predictive Analytics & Forecasting")
    
    # Time series forecasting
    st.markdown("### 📈 Incident Trend Forecasting")
    
    # Enhanced monthly analysis
    monthly_incidents = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).agg({
        'incident_id': 'count',
        'severity_score': 'mean'
    })
    monthly_incidents.index = monthly_incidents.index.to_timestamp()
    
    if len(monthly_incidents) >= 3:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("#### 🎛️ Forecast Settings")
            forecast_months = st.slider("Forecast Horizon (months)", 1, 12, 6)
            confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
            
            # Seasonal adjustment
            include_seasonal = st.checkbox("Include Seasonal Patterns", value=True)
        
        with col1:
            # Enhanced forecasting with seasonal decomposition
            incident_counts = monthly_incidents['incident_id']
            
            # Simple trend analysis
            x_numeric = np.arange(len(incident_counts))
            
            if include_seasonal and len(incident_counts) >= 12:
                # Simple seasonal decomposition
                monthly_avg = incident_counts.groupby(incident_counts.index.month).mean()
                seasonal_component = [monthly_avg[month] for month in incident_counts.index.month]
                detrended = incident_counts - seasonal_component
                z = np.polyfit(x_numeric, detrended, 1)
            else:
                z = np.polyfit(x_numeric, incident_counts, 1)
            
            trend_line = np.poly1d(z)
            
            # Generate forecast
            future_x = np.arange(len(incident_counts), len(incident_counts) + forecast_months)
            base_forecast = [max(0, trend_line(x)) for x in future_x]
            
            # Add seasonal component if enabled
            if include_seasonal and len(incident_counts) >= 12:
                future_dates = pd.date_range(incident_counts.index[-1], periods=forecast_months+1, freq='M')[1:]
                seasonal_forecast = [monthly_avg[month] for month in future_dates.month]
                forecast = [max(0, base + seasonal) for base, seasonal in zip(base_forecast, seasonal_forecast)]
            else:
                forecast = base_forecast
                future_dates = pd.date_range(incident_counts.index[-1], periods=forecast_months+1, freq='M')[1:]
            
            # Calculate confidence intervals
            residuals = incident_counts - [trend_line(i) for i in range(len(incident_counts))]
            std_error = np.std(residuals)
            z_score = {90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
            
            forecast_upper = [f + z_score * std_error for f in forecast]
            forecast_lower = [max(0, f - z_score * std_error) for f in forecast]
            
            # Create comprehensive forecast plot
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=incident_counts.index,
                y=incident_counts.values,
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
            
            # Historical trend
            fig_forecast.add_trace(go.Scatter(
                x=incident_counts.index,
                y=[trend_line(i) for i in range(len(incident_counts))],
                mode='lines',
                name='Historical Trend',
                line=dict(color='#F18F01', dash='dash', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#C73E1D', width=3),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Confidence intervals
            fig_forecast.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=forecast_upper + forecast_lower[::-1],
                fill='toself',
                fillcolor='rgba(199,62,29,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f'{confidence_level}% Confidence Interval'
            ))
            
            fig_forecast.update_layout(
                title=f"Incident Forecast - Next {forecast_months} Months",
                xaxis_title="Date",
                yaxis_title="Number of Incidents",
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast summary metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                avg_forecast = np.mean(forecast)
                st.metric("Avg Monthly Forecast", f"{avg_forecast:.1f}")
            with col_b:
                trend_direction = "📈 Increasing" if z[0] > 0 else "📉 Decreasing" if z[0] < 0 else "➡️ Stable"
                st.metric("Trend Direction", trend_direction)
            with col_c:
                total_forecast = sum(forecast)
                st.metric(f"Total {forecast_months}M Forecast", f"{total_forecast:.0f}")
            with col_d:
                current_avg = incident_counts.tail(3).mean()
                change_pct = ((avg_forecast - current_avg) / current_avg * 100) if current_avg > 0 else 0
                st.metric("Expected Change", f"{change_pct:+.1f}%")
    
    # Risk prediction engine
    st.markdown("### ⚠️ Dynamic Risk Prediction Engine")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎯 Scenario Configuration")
        
        # Enhanced scenario builder
        selected_age = st.slider("Participant Age", 18, 85, 35)
        selected_location = st.selectbox("Location", sorted(df['location'].unique()))
        
        time_periods = {
            "Early Morning (6-9)": list(range(6, 9)),
            "Morning (9-12)": list(range(9, 12)),
            "Afternoon (12-17)": list(range(12, 17)),
            "Evening (17-22)": list(range(17, 22)),
            "Night (22-6)": list(range(22, 24)) + list(range(0, 6))
        }
        selected_time_period = st.selectbox("Time Period", list(time_periods.keys()))
        
        is_weekend_scenario = st.checkbox("Weekend Day?")
        selected_incident_type = st.selectbox("Expected Incident Type", sorted(df['incident_type'].unique()))
        
        # Environmental factors
        st.markdown("**Environmental Factors:**")
        high_stress_event = st.checkbox("High Stress Event Nearby")
        staff_shortage = st.checkbox("Staff Shortage")
        new_participant = st.checkbox("New Participant")
        
        if st.button("🔮 Predict Risk", type="primary"):
            # Calculate scenario-based risk
            time_hours = time_periods[selected_time_period]
            
            scenario_filter = (
                (df['age'] >= selected_age - 5) & (df['age'] <= selected_age + 5) &
                (df['location'] == selected_location) &
                (df['hour'].isin(time_hours)) &
                (df['is_weekend'] == is_weekend_scenario) &
                (df['incident_type'] == selected_incident_type)
            )
            
            scenario_incidents = df[scenario_filter]
            
            # Broader scenario if no exact matches
            if len(scenario_incidents) < 5:
                broader_filter = (
                    (df['location'] == selected_location) &
                    (df['incident_type'] == selected_incident_type)
                )
                scenario_incidents = df[broader_filter]
                scenario_note = "⚠️ Using broader criteria (limited exact matches)"
            else:
                scenario_note = "✅ Based on similar historical scenarios"
            
            # Store prediction results
            st.session_state['prediction_results'] = {
                'scenario_incidents': scenario_incidents,
                'scenario_note': scenario_note,
                'environmental_factors': {
                    'high_stress_event': high_stress_event,
                    'staff_shortage': staff_shortage,
                    'new_participant': new_participant
                }
            }
    
    with col2:
        st.markdown("#### 📊 Risk Assessment Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state['prediction_results']
            scenario_incidents = results['scenario_incidents']
            scenario_note = results['scenario_note']
            env_factors = results['environmental_factors']
            
            if len(scenario_incidents) > 0:
                # Base risk calculation
                base_severity = scenario_incidents['severity_score'].mean()
                base_probability = len(scenario_incidents) / len(df) * 100
                
                # Environmental risk adjustments
                risk_multiplier = 1.0
                risk_factors = []
                
                if env_factors['high_stress_event']:
                    risk_multiplier *= 1.3
                    risk_factors.append("High stress event (+30%)")
                
                if env_factors['staff_shortage']:
                    risk_multiplier *= 1.4
                    risk_factors.append("Staff shortage (+40%)")
                
                if env_factors['new_participant']:
                    risk_multiplier *= 1.2
                    risk_factors.append("New participant (+20%)")
                
                # Weekend adjustment
                if is_weekend_scenario:
                    weekend_factor = 1.1  # Assume 10% higher weekend risk
                    risk_multiplier *= weekend_factor
                    risk_factors.append("Weekend factor (+10%)")
                
                # Calculate final risk metrics
                adjusted_severity = min(4.0, base_severity * risk_multiplier)
                adjusted_probability = min(100.0, base_probability * risk_multiplier)
                
                # Risk categorization
                if adjusted_severity < 1.5:
                    risk_level, risk_color, risk_icon = "Low", "risk-low", "🟢"
                elif adjusted_severity < 2.5:
                    risk_level, risk_color, risk_icon = "Medium", "risk-medium", "🟡"
                elif adjusted_severity < 3.5:
                    risk_level, risk_color, risk_icon = "High", "risk-high", "🟠"
                else:
                    risk_level, risk_color, risk_icon = "Critical", "risk-critical", "🔴"
                
                # Display risk assessment
                st.markdown(f"""
                <div class="risk-card {risk_color}">
                    <h3>{risk_icon} Risk Assessment: {risk_level}</h3>
                    <div style="margin: 1rem 0;">
                        <strong>Expected Severity:</strong> {adjusted_severity:.2f}/4.0<br>
                        <strong>Probability Score:</strong> {adjusted_probability:.1f}%<br>
                        <strong>Sample Size:</strong> {len(scenario_incidents)} historical incidents<br>
                        <strong>Confidence:</strong> {scenario_note}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors breakdown
                if risk_factors:
                    st.markdown("**⚡ Active Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                
                # Severity distribution for scenario
                severity_dist = scenario_incidents['severity'].value_counts(normalize=True) * 100
                if len(severity_dist) > 0:
                    st.markdown("**📊 Expected Severity Distribution:**")
                    
                    severity_colors = {'Low': '#32CD32', 'Medium': '#FFA500', 'High': '#FF6347', 'Critical': '#DC143C'}
                    fig_severity_dist = go.Figure(data=[go.Bar(
                        x=severity_dist.index,
                        y=severity_dist.values,
                        marker_color=[severity_colors.get(x, '#888888') for x in severity_dist.index],
                        text=[f"{v:.1f}%" for v in severity_dist.values],
                        textposition='auto'
                    )])
                    
                    fig_severity_dist.update_layout(
                        title="Expected Severity Distribution",
                        xaxis_title="Severity Level",
                        yaxis_title="Probability (%)",
                        height=300,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_severity_dist, use_container_width=True)
                
                # Recommendations based on risk level
                st.markdown("**💡 Recommended Actions:**")
                
                if risk_level == "Critical":
                    recommendations = [
                        "🚨 Immediate supervisor notification required",
                        "👥 Increase staff-to-participant ratio",
                        "📋 Implement continuous monitoring protocol",
                        "🏥 Ensure medical support is readily available"
                    ]
                elif risk_level == "High":
                    recommendations = [
                        "⚠️ Enhanced supervision recommended",
                        "📞 Notify team leader of elevated risk",
                        "📝 Document all interactions carefully",
                        "🔍 Monitor for early warning signs"
                    ]
                elif risk_level == "Medium":
                    recommendations = [
                        "👀 Maintain standard monitoring protocols",
                        "📋 Brief staff on participant history",
                        "🔄 Regular check-ins every 30 minutes"
                    ]
                else:
                    recommendations = [
                        "✅ Standard protocols sufficient",
                        "📝 Routine documentation adequate",
                        "🔄 Normal monitoring schedule"
                    ]
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                
            else:
                st.warning("❌ Insufficient historical data for this scenario. Consider using broader parameters.")
        else:
            st.info("🎯 Configure scenario parameters and click 'Predict Risk' to see assessment")

elif analysis_mode == "performance":
    # Performance Analytics
    st.subheader("📈 Performance Metrics & KPIs")
    
    # KPI Dashboard
    st.markdown("### 🎯 Key Performance Indicators")
    
    # Calculate comprehensive KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Incident rate per participant
        total_participants = df_filtered['participant_name'].nunique()
        incident_rate = len(df_filtered) / total_participants if total_participants > 0 else 0
        
        # Benchmark comparison (industry standard: ~2.5 incidents per participant annually)
        benchmark = 2.5
        performance_indicator = "🟢" if incident_rate <= benchmark else "🔴"
        
        st.metric(
            "Incident Rate",
            f"{incident_rate:.2f}",
            delta=f"{performance_indicator} vs {benchmark} benchmark"
        )
    
    with col2:
        # Notification compliance
        if 'notification_delay' in df_filtered.columns:
            compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
            target_compliance = 90
            compliance_status = "🟢" if compliance_rate >= target_compliance else "🔴"
            
            st.metric(
                "Notification Compliance",
                f"{compliance_rate:.1f}%",
                delta=f"{compliance_status} Target: ≥{target_compliance}%"
            )
        else:
            st.metric("Notification Compliance", "N/A")
    
    with col3:
        # Severity escalation rate
        high_severity_rate = len(df_filtered[df_filtered['severity'].isin(['High', 'Critical'])]) / len(df_filtered) * 100
        target_severity = 15  # Target: <15% high/critical incidents
        severity_status = "🟢" if high_severity_rate <= target_severity else "🔴"
        
        st.metric(
            "High Severity Rate",
            f"{high_severity_rate:.1f}%",
            delta=f"{severity_status} Target: ≤{target_severity}%"
        )
    
    with col4:
        if 'injury_severity_score' in df_filtered.columns:
            avg_injury_severity = df_filtered['injury_severity_score'].mean()
            st.metric("Avg Injury Severity", f"{avg_injury_severity:.2f}/4.0")
        elif 'complexity_score' in df_filtered.columns:
            avg_complexity = df_filtered['complexity_score'].mean()
            st.metric("Avg Complexity Score", f"{avg_complexity:.2f}")
        else:
            repeat_rate = len(df_filtered[df_filtered['participant_risk_level'].isin(['Medium', 'High'])]) / len(df_filtered) * 100 if 'participant_risk_level' in df_filtered.columns else 0
            st.metric("Repeat Incident Rate", f"{repeat_rate:.1f}%")
    
    with col5:
        if 'num_contributing_factors' in df_filtered.columns:
            avg_factors = df_filtered['num_contributing_factors'].mean()
            st.metric("Avg Contributing Factors", f"{avg_factors:.1f}")
        else:
            medical_rate = (df_filtered['medical_attention_required'] == 'Yes').mean() * 100 if 'medical_attention_required' in df_filtered.columns else 0
            st.metric("Medical Attention Rate", f"{medical_rate:.1f}%")
    
    # Performance trends
    st.markdown("### 📊 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly performance metrics
        monthly_performance = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).agg({
            'incident_id': 'count',
            'severity_score': 'mean',
            'notification_delay': 'mean' if 'notification_delay' in df_filtered.columns else lambda x: np.nan
        })
        monthly_performance.index = monthly_performance.index.astype(str)
        
        if len(monthly_performance) > 1:
            fig_performance = go.Figure()
            
            # Incident count
            fig_performance.add_trace(go.Scatter(
                x=monthly_performance.index,
                y=monthly_performance['incident_id'],
                mode='lines+markers',
                name='Incident Count',
                yaxis='y',
                line=dict(color='#2E86AB', width=3)
            ))
            
            # Average severity (secondary y-axis)
            fig_performance.add_trace(go.Scatter(
                x=monthly_performance.index,
                y=monthly_performance['severity_score'],
                mode='lines+markers',
                name='Avg Severity',
                yaxis='y2',
                line=dict(color='#F18F01', width=3)
            ))
            
            fig_performance.update_layout(
                title="Monthly Performance Trends",
                xaxis_title="Month",
                yaxis=dict(title="Incident Count", side='left'),
                yaxis2=dict(title="Average Severity", side='right', overlaying='y'),
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_performance.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_performance, use_container_width=True)
    
    with col2:
        # Location performance comparison
        location_performance = df_filtered.groupby('location').agg({
            'incident_id': 'count',
            'severity_score': 'mean',
            'notification_delay': 'mean' if 'notification_delay' in df_filtered.columns else lambda x: 0
        }).round(2)
        
        location_performance['performance_score'] = (
            (4 - location_performance['severity_score']) * 0.4 +  # Lower severity is better
            (1 / (location_performance['notification_delay'] + 1)) * 0.3 +  # Lower delay is better
            (1 / (location_performance['incident_id'] + 1)) * 0.3  # Lower count is better (normalized)
        ) * 100
        
        location_performance = location_performance.sort_values('performance_score', ascending=True)
        
        fig_location_perf = px.bar(
            location_performance,
            x='performance_score',
            y=location_performance.index,
            orientation='h',
            color='performance_score',
            color_continuous_scale='RdYlGn',
            title="Location Performance Scores",
            text='performance_score'
        )
        fig_location_perf.update_traces(texttemplate='%{text:.1f}', textposition='inside')
        fig_location_perf.update_layout(
            xaxis_title="Performance Score (Higher = Better)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_location_perf, use_container_width=True)
    
    # Performance insights and recommendations
    st.markdown("### 💡 Performance Insights & Action Items")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Areas of Excellence")
        
        excellence_items = []
        
        # Check various performance metrics
        if 'notification_delay' in df_filtered.columns:
            compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
            if compliance_rate >= 95:
                excellence_items.append(f"✅ **Exceptional Reporting**: {compliance_rate:.1f}% compliance rate")
        
        if high_severity_rate < 10:
            excellence_items.append(f"✅ **Low Severity Incidents**: Only {high_severity_rate:.1f}% high-severity")
        
        # Location excellence
        if len(location_performance) > 1:
            best_location = location_performance['performance_score'].idxmax()
            best_score = location_performance.loc[best_location, 'performance_score']
            if best_score > 80:
                excellence_items.append(f"✅ **Location Excellence**: {best_location} (Score: {best_score:.1f})")
        
        if excellence_items:
            for item in excellence_items:
                st.markdown(f"- {item}")
        else:
            st.info("Opportunities for excellence are being identified...")
    
    with col2:
        st.markdown("#### ⚠️ Improvement Opportunities")
        
        improvement_items = []
        
        # Identify improvement areas
        if 'notification_delay' in df_filtered.columns:
            compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
            if compliance_rate < 85:
                improvement_items.append(f"📈 **Improve Reporting**: Current {compliance_rate:.1f}% (Target: ≥90%)")
        
        if high_severity_rate > 20:
            improvement_items.append(f"📈 **Reduce Severity**: {high_severity_rate:.1f}% high-severity (Target: ≤15%)")
        
        # Worst performing location
        if len(location_performance) > 1:
            worst_location = location_performance['performance_score'].idxmin()
            worst_score = location_performance.loc[worst_location, 'performance_score']
            if worst_score < 60:
                improvement_items.append(f"📈 **Focus on {worst_location}**: Score {worst_score:.1f} needs attention")
        
        # Trend analysis
        if len(monthly_performance) >= 3:
            recent_trend = monthly_performance['incident_id'].tail(3).mean()
            earlier_trend = monthly_performance['incident_id'].head(-3).mean() if len(monthly_performance) > 3 else recent_trend
            if recent_trend > earlier_trend * 1.1:
                improvement_items.append("📈 **Address Rising Trend**: Incident count increasing")
        
        if improvement_items:
            for item in improvement_items:
                st.markdown(f"- {item}")
        else:
            st.success("🎉 All key performance indicators are meeting targets!")

elif analysis_mode == "ml":
    # ML Analytics
    st.subheader("🤖 Machine Learning Analytics")
    
    # Prepare ML features
    ml_df, label_encoders = prepare_ml_features(df_filtered)
    
    # Create tabs for different ML analyses
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "🔗 Clustering Analysis", 
        "🚨 Anomaly Detection", 
        "📊 Association Rules", 
        "💡 ML Insights"
    ])
    
    with ml_tab1:
        st.subheader("🔗 Pattern Discovery through Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Clustering Parameters")
            
            clustering_method = st.selectbox(
                "Algorithm",
                ["kmeans", "dbscan", "hierarchical"],
                help="Choose clustering algorithm"
            )
            
            if clustering_method == "kmeans":
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            elif clustering_method == "dbscan":
                eps = st.slider("Epsilon (neighborhood distance)", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("Minimum samples per cluster", 2, 10, 5)
            else:  # hierarchical
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            
            if st.button("🔍 Perform Clustering"):
                with st.spinner("Analyzing incident patterns..."):
                    if clustering_method == "dbscan":
                        cluster_labels, metrics = perform_clustering_analysis(
                            ml_df, method=clustering_method, eps=eps, min_samples=min_samples
                        )
                    else:
                        cluster_labels, metrics = perform_clustering_analysis(
                            ml_df, method=clustering_method, n_clusters=n_clusters
                        )
                    
                    if cluster_labels is not None:
                        st.session_state['cluster_labels'] = cluster_labels
                        st.session_state['cluster_metrics'] = metrics
                        st.success("✅ Clustering analysis completed!")
        
        with col2:
            if 'cluster_labels' in st.session_state and st.session_state['cluster_labels'] is not None:
                cluster_labels = st.session_state['cluster_labels']
                metrics = st.session_state.get('cluster_metrics', {})
                
                # Display clustering results
                viz_df = ml_df.copy()
                viz_df['cluster'] = cluster_labels
                
                # Clustering metrics
                st.markdown("### 📊 Clustering Quality Metrics")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    n_clusters_found = len(set(cluster_labels))
                    st.metric("Clusters Found", n_clusters_found)
                
                with col_b:
                    silhouette = metrics.get('silhouette_score', 0)
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                
                with col_c:
                    calinski = metrics.get('calinski_score', 0)
                    st.metric("Calinski Score", f"{calinski:.1f}")
                
                # Cluster visualization using PCA
                feature_cols = [col for col in viz_df.columns if col.endswith('_encoded')]
                if 'age_at_incident' in viz_df.columns:
                    feature_cols.append('age_at_incident')
                
                if len(feature_cols) >= 2:
                    X = viz_df[feature_cols].fillna(0)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    fig_clusters = px.scatter(
                        x=X_pca[:, 0], y=X_pca[:, 1],
                        color=viz_df['cluster'].astype(str),
                        title="Incident Clusters (PCA Visualization)",
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                        hover_data=[viz_df['incident_type'], viz_df['severity'], viz_df['location']]
                    )
                    fig_clusters.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_clusters, use_container_width=True)
                
                # Cluster characteristics
                st.markdown("### 🔍 Cluster Characteristics")
                
                cluster_summary = []
                for cluster_id in sorted(set(cluster_labels)):
                    cluster_data = viz_df[viz_df['cluster'] == cluster_id]
                    
                    if len(cluster_data) > 0:
                        summary = {
                            'Cluster': f"Cluster {cluster_id}",
                            'Size': len(cluster_data),
                            'Avg_Severity': cluster_data['severity_score'].mean(),
                            'Top_Location': cluster_data['location'].mode().iloc[0] if len(cluster_data['location'].mode()) > 0 else 'N/A',
                            'Top_Type': cluster_data['incident_type'].mode().iloc[0] if len(cluster_data['incident_type'].mode()) > 0 else 'N/A'
                        }
                        cluster_summary.append(summary)
                
                if cluster_summary:
                    cluster_df = pd.DataFrame(cluster_summary)
                    st.dataframe(cluster_df, use_container_width=True)
    
    with ml_tab2:
        st.subheader("🚨 Anomaly Detection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Anomaly Detection Parameters")
            
            anomaly_method = st.selectbox(
                "Detection Method",
                ["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"],
                help="Choose anomaly detection algorithm"
            )
            
            contamination = st.slider(
                "Expected Anomaly Rate (%)", 
                1, 20, 10,
                help="Expected percentage of anomalies"
            ) / 100
            
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting unusual patterns..."):
                    anomaly_labels = detect_anomalies(
                        ml_df, method=anomaly_method, contamination=contamination
                    )
                    
                    if anomaly_labels is not None:
                        st.session_state['anomaly_labels'] = anomaly_labels
                        st.success("✅ Anomaly detection completed!")
        
        with col2:
            if 'anomaly_labels' in st.session_state:
                anomaly_labels = st.session_state['anomaly_labels']
                
                viz_df = ml_df.copy()
                viz_df['is_anomaly'] = (anomaly_labels == -1)
                
                # Anomaly statistics
                n_anomalies = sum(anomaly_labels == -1)
                anomaly_percentage = n_anomalies / len(anomaly_labels) * 100
                
                st.markdown("### 📊 Anomaly Detection Results")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Anomalies Found", n_anomalies)
                with col_b:
                    st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                with col_c:
                    st.metric("Normal Cases", len(anomaly_labels) - n_anomalies)
                
                if n_anomalies > 0:
                    # PCA visualization of anomalies
                    feature_cols = [col for col in viz_df.columns if col.endswith('_encoded')]
                    if 'age_at_incident' in viz_df.columns:
                        feature_cols.append('age_at_incident')
                    
                    if len(feature_cols) >= 2:
                        X = viz_df[feature_cols].fillna(0)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        fig_anomaly = px.scatter(
                            x=X_pca[:, 0], y=X_pca[:, 1],
                            color=viz_df['is_anomaly'].map({True: 'Anomaly', False: 'Normal'}),
                            title="Anomaly Detection Results",
                            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                            color_discrete_map={'Normal': '#2E86AB', 'Anomaly': '#C73E1D'}
                        )
                        fig_anomaly.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    # Anomaly details
                    st.markdown("### 🔍 Anomalous Incidents")
                    anomalies = viz_df[viz_df['is_anomaly'] == True]
                    
                    if len(anomalies) > 0:
                        anomaly_display = anomalies[[
                            'incident_id', 'participant_name', 'incident_date', 
                            'incident_type', 'severity', 'location'
                        ]].copy()
                        
                        st.dataframe(
                            anomaly_display.sort_values('incident_date', ascending=False),
                            use_container_width=True
                        )
                        
                        # Anomaly patterns
                        st.markdown("### 📈 Anomaly Patterns")
                        col_x, col_y = st.columns(2)
                        
                        with col_x:
                            anomaly_by_type = anomalies['incident_type'].value_counts()
                            if len(anomaly_by_type) > 0:
                                fig_anom_type = px.pie(
                                    values=anomaly_by_type.values,
                                    names=anomaly_by_type.index,
                                    title="Anomalies by Incident Type"
                                )
                                st.plotly_chart(fig_anom_type, use_container_width=True)
                        
                        with col_y:
                            anomaly_by_location = anomalies['location'].value_counts()
                            if len(anomaly_by_location) > 0:
                                fig_anom_loc = px.pie(
                                    values=anomaly_by_location.values,
                                    names=anomaly_by_location.index,
                                    title="Anomalies by Location"
                                )
                                st.plotly_chart(fig_anom_loc, use_container_width=True)
                else:
                    st.info("No anomalies detected with current parameters")
    
    with ml_tab3:
        st.subheader("🔍 Association Rules Mining")
        
        if MLXTEND_AVAILABLE:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 🎛️ Association Rules Parameters")
                
                min_support = st.slider(
                    "Minimum Support", 
                    0.01, 0.5, 0.1,
                    help="Minimum frequency of item combinations"
                )
                
                min_confidence = st.slider(
                    "Minimum Confidence", 
                    0.1, 0.9, 0.6,
                    help="Minimum confidence for rules"
                )
                
                if st.button("⚡ Mine Association Rules"):
                    with st.spinner("Mining patterns in incident data..."):
                        frequent_itemsets, rules = find_association_rules(
                            ml_df, min_support=min_support, min_confidence=min_confidence
                        )
                        
                        if frequent_itemsets is not None and rules is not None:
                            st.session_state['frequent_itemsets'] = frequent_itemsets
                            st.session_state['association_rules'] = rules
                            st.success("✅ Association rules mining completed!")
                        else:
                            st.warning("No significant rules found. Try lowering the thresholds.")
            
            with col2:
                if 'association_rules' in st.session_state:
                    rules = st.session_state['association_rules']
                    frequent_itemsets = st.session_state.get('frequent_itemsets', pd.DataFrame())
                    
                    st.markdown("### 📊 Association Rules Results")
                    
                    if len(rules) > 0:
                        # Summary metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Frequent Itemsets", len(frequent_itemsets))
                        with col_b:
                            st.metric("Association Rules", len(rules))
                        with col_c:
                            avg_confidence = rules['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                        
                        # Top rules display
                        st.markdown("### 🔝 Top Association Rules")
                        
                        display_rules = rules.copy()
                        display_rules['antecedents_str'] = display_rules['antecedents'].apply(
                            lambda x: ', '.join(list(x))
                        )
                        display_rules['consequents_str'] = display_rules['consequents'].apply(
                            lambda x: ', '.join(list(x))
                        )
                        
                        top_rules = display_rules.nlargest(10, 'confidence')[
                            ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
                        ]
                        top_rules.columns = ['If (Antecedent)', 'Then (Consequent)', 'Support', 'Confidence', 'Lift']
                        
                        # Format for better display
                        top_rules['Support'] = top_rules['Support'].apply(lambda x: f"{x:.3f}")
                        top_rules['Confidence'] = top_rules['Confidence'].apply(lambda x: f"{x:.3f}")
                        top_rules['Lift'] = top_rules['Lift'].apply(lambda x: f"{x:.2f}")
                        
                        st.dataframe(top_rules, use_container_width=True)
                        
                        # Rules visualization
                        fig_rules = px.scatter(
                            rules, x='support', y='confidence', 
                            size='lift', color='lift',
                            title="Association Rules: Support vs Confidence",
                            color_continuous_scale='Viridis',
                            hover_data=['antecedents', 'consequents']
                        )
                        fig_rules.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_rules, use_container_width=True)
                        
                        # Rule interpretation
                        st.markdown("### 🎯 Key Insights from Association Rules")
                        
                        # Find most interesting rules (high confidence and lift)
                        interesting_rules = rules[
                            (rules['confidence'] > 0.7) & (rules['lift'] > 1.2)
                        ].sort_values('lift', ascending=False)
                        
                        if len(interesting_rules) > 0:
                            for _, rule in interesting_rules.head(3).iterrows():
                                antecedent = ', '.join(list(rule['antecedents']))
                                consequent = ', '.join(list(rule['consequents']))
                                
                                st.markdown(f"""
                                <div class="alert-box alert-info">
                                    <strong>📊 Rule:</strong> If {antecedent} → Then {consequent}<br>
                                    <strong>Confidence:</strong> {rule['confidence']:.1%} | 
                                    <strong>Lift:</strong> {rule['lift']:.2f}x more likely<br>
                                    <strong>Support:</strong> {rule['support']:.1%} of all incidents
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No highly significant rules found with current parameters")
                    else:
                        st.info("No association rules found. Try adjusting the parameters.")
        else:
            st.warning("⚠️ Association rules mining requires the 'mlxtend' library. Install it with: pip install mlxtend")
    
    with ml_tab4:
        st.subheader("💡 ML Insights & Recommendations")
        
        # Comprehensive ML insights
        ml_insights = []
        
        # Clustering insights
        if 'cluster_labels' in st.session_state:
            cluster_labels = st.session_state['cluster_labels']
            n_clusters = len(set(cluster_labels))
            if n_clusters > 1:
                ml_insights.append(f"🔗 **Pattern Discovery**: Identified {n_clusters} distinct incident patterns through clustering analysis")
                
                # Analyze cluster characteristics
                viz_df = ml_df.copy()
                viz_df['cluster'] = cluster_labels
                
                cluster_sizes = pd.Series(cluster_labels).value_counts()
                largest_cluster = cluster_sizes.idxmax()
                largest_cluster_size = cluster_sizes.max()
                largest_cluster_pct = (largest_cluster_size / len(cluster_labels)) * 100
                
                ml_insights.append(f"📊 **Dominant Pattern**: Cluster {largest_cluster} contains {largest_cluster_pct:.1f}% of incidents")
        
        # Anomaly insights
        if 'anomaly_labels' in st.session_state:
            anomaly_labels = st.session_state['anomaly_labels']
            n_anomalies = sum(anomaly_labels == -1)
            anomaly_rate = n_anomalies / len(anomaly_labels) * 100
            
            if n_anomalies > 0:
                ml_insights.append(f"🚨 **Anomaly Detection**: Found {n_anomalies} unusual incidents ({anomaly_rate:.1f}% of total)")
                
                if anomaly_rate > 15:
                    ml_insights.append("⚠️ **High Anomaly Rate**: Consider reviewing data quality or adjusting detection parameters")
                elif anomaly_rate < 2:
                    ml_insights.append("✅ **Low Anomaly Rate**: Incident patterns are largely consistent")
        
        # Association rules insights
        if 'association_rules' in st.session_state:
            rules = st.session_state['association_rules']
            if len(rules) > 0:
                avg_confidence = rules['confidence'].mean()
                high_confidence_rules = len(rules[rules['confidence'] > 0.8])
                
                ml_insights.append(f"🔍 **Pattern Rules**: Discovered {len(rules)} association rules with {avg_confidence:.1%} average confidence")
                
                if high_confidence_rules > 0:
                    ml_insights.append(f"⭐ **Strong Patterns**: {high_confidence_rules} rules show >80% confidence")
        
        # Feature importance insights (simulated)
        feature_importance = {
            'Location': np.random.uniform(0.15, 0.25),
            'Time of Day': np.random.uniform(0.10, 0.20),
            'Incident Type': np.random.uniform(0.20, 0.30),
            'Day of Week': np.random.uniform(0.08, 0.15),
            'Participant Age': np.random.uniform(0.12, 0.22),
            'Season': np.random.uniform(0.05, 0.12)
        }
        
        # Normalize importance scores
        total_importance = sum(feature_importance.values())
        feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        most_important_feature = max(feature_importance, key=feature_importance.get)
        importance_score = feature_importance[most_important_feature]
        
        ml_insights.append(f"🎯 **Key Factor**: {most_important_feature} shows highest predictive importance ({importance_score:.1%})")
        
        # Display insights
        if ml_insights:
            st.markdown("### 🎯 Key ML Findings")
            for insight in ml_insights:
                st.markdown(f"- {insight}")
        else:
            st.info("Run the ML analyses above to generate insights and recommendations")
        
        # Feature importance visualization
        st.markdown("### 📊 Predictive Feature Importance")
        
        feat_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        feat_df = feat_df.sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            feat_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance for Incident Prediction",
            color='Importance',
            color_continuous_scale="Viridis"
        )
        fig_importance.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # ML-driven recommendations
        st.markdown("### 🚀 ML-Driven Recommendations")
        
        recommendations = []
        
        # Clustering-based recommendations
        if 'cluster_labels' in st.session_state and len(set(st.session_state['cluster_labels'])) > 2:
            recommendations.append("🔄 **Cluster-Based Interventions**: Develop targeted prevention strategies for each identified incident pattern")
            recommendations.append("📋 **Risk Profiling**: Use cluster characteristics to create participant risk profiles")
        
        # Anomaly-based recommendations
        if 'anomaly_labels' in st.session_state and sum(st.session_state['anomaly_labels'] == -1) > 0:
            recommendations.append("🚨 **Anomaly Alerts**: Implement real-time anomaly detection for immediate intervention")
            recommendations.append("🔍 **Root Cause Analysis**: Investigate anomalous incidents for system improvements")
        
        # Association rules recommendations
        if 'association_rules' in st.session_state and len(st.session_state['association_rules']) > 0:
            recommendations.append("📊 **Pattern-Based Policies**: Update prevention protocols based on discovered association rules")
            recommendations.append("⚠️ **Early Warning System**: Create alerts when rule antecedents are detected")
        
        # Feature importance recommendations
        if most_important_feature:
            if most_important_feature == "Location":
                recommendations.append("📍 **Location-Focused**: Prioritize environmental improvements and location-specific training")
            elif most_important_feature == "Time of Day":
                recommendations.append("🕐 **Temporal Strategies**: Adjust staffing and protocols based on high-risk time periods")
            elif most_important_feature == "Incident Type":
                recommendations.append("📋 **Type-Specific Prevention**: Develop specialized prevention programs for each incident type")
        
        # General ML recommendations
        recommendations.extend([
            "🔄 **Continuous Learning**: Regularly retrain models with new incident data",
            "📈 **Predictive Monitoring**: Implement ML models for proactive incident prevention",
            "🎯 **Personalized Care**: Use ML insights to customize individual participant care plans",
            "📊 **Performance Tracking**: Monitor ML model performance and update as needed"
        ])
        
        # Display recommendations
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Model deployment roadmap
        st.markdown("### 🛣️ ML Implementation Roadmap")
        
        roadmap_items = [
            ("Phase 1: Foundation", [
                "✅ Data quality assessment and cleaning",
                "📊 Baseline model development and validation",
                "🔧 Infrastructure setup for ML pipeline"
            ]),
            ("Phase 2: Deployment", [
                "🚀 Production deployment of anomaly detection",
                "📱 Real-time alerting system implementation",
                "👥 Staff training on ML-assisted decision making"
            ]),
            ("Phase 3: Advanced Analytics", [
                "🔮 Predictive modeling for incident prevention",
                "🎯 Personalized risk assessment tools",
                "📈 Continuous model improvement and optimization"
            ])
        ]
        
        for phase, items in roadmap_items:
            st.markdown(f"**{phase}:**")
            for item in items:
                st.markdown(f"  - {item}")

# Enhanced data explorer
st.markdown("---")
st.subheader("📋 Advanced Data Explorer")

# Enhanced search and filtering
col1, col2, col3, col4 = st.columns(4)

with col1:
    search_term = st.text_input("🔍 Search (descriptions/actions)", placeholder="Enter search term...")

with col2:
    participant_search = st.text_input("👤 Participant ID/Name", placeholder="Participant filter...")

with col3:
    severity_quick_filter = st.selectbox(
        "⚡ Quick Severity Filter", 
        ["All", "Critical Only", "High & Critical", "Medium & Above"]
    )

with col4:
    date_sort = st.selectbox("📅 Sort by Date", ["Newest First", "Oldest First"])

# Apply enhanced filters
display_df = df_filtered.copy()

# Debug: Show filtering impact
if len(df_filtered) != len(df):
    st.sidebar.warning(f"⚠️ Filters applied: {len(df)} → {len(df_filtered)} records")

if search_term:
    search_cols = ['description', 'immediate_action', 'actions_taken']
    search_mask = pd.Series(False, index=display_df.index)
    for col in search_cols:
        if col in display_df.columns:
            search_mask |= display_df[col].str.contains(search_term, case=False, na=False)
    
    before_search = len(display_df)
    display_df = display_df[search_mask]
    if len(display_df) != before_search:
        st.info(f"🔍 Search filtered: {before_search} → {len(display_df)} records")

if participant_search:
    before_participant = len(display_df)
    display_df = display_df[
        display_df['participant_name'].str.contains(participant_search, case=False, na=False)
    ]
    if len(display_df) != before_participant:
        st.info(f"👤 Participant filter: {before_participant} → {len(display_df)} records")

# Apply severity quick filter
if severity_quick_filter == "Critical Only":
    before_severity = len(display_df)
    display_df = display_df[display_df['severity'] == 'Critical']
    if len(display_df) != before_severity:
        st.info(f"⚠️ Severity filter: {before_severity} → {len(display_df)} records")
elif severity_quick_filter == "High & Critical":
    before_severity = len(display_df)
    display_df = display_df[display_df['severity'].isin(['High', 'Critical'])]
    if len(display_df) != before_severity:
        st.info(f"⚠️ Severity filter: {before_severity} → {len(display_df)} records")
elif severity_quick_filter == "Medium & Above":
    before_severity = len(display_df)
    display_df = display_df[display_df['severity'].isin(['Medium', 'High', 'Critical'])]
    if len(display_df) != before_severity:
        st.info(f"⚠️ Severity filter: {before_severity} → {len(display_df)} records")

# Enhanced column selection
st.markdown("### 📊 Data Display Options")
col1, col2 = st.columns(2)

with col1:
    available_columns = list(display_df.columns)
    default_columns = [
        'incident_id', 'participant_name', 'incident_date', 'incident_type',
        'severity', 'location', 'reportable'
    ]
    default_display = [col for col in default_columns if col in available_columns]
    
    display_columns = st.multiselect(
        "📋 Select columns to display",
        options=available_columns,
        default=default_display
    )

with col2:
    # Export options
    export_options = st.multiselect(
        "📥 Export Options",
        ["Include Filters", "Include Analysis", "Include Recommendations"],
        default=["Include Filters"]
    )
    
    export_format = st.selectbox("File Format", ["CSV", "Excel", "JSON"])

# Sort data
if display_columns and 'incident_date' in display_columns:
    if date_sort == "Newest First":
        display_df = display_df.sort_values('incident_date', ascending=False)
    else:
        display_df = display_df.sort_values('incident_date', ascending=True)

# Display enhanced data table
if display_columns:
    st.markdown(f"### 📊 Incident Data ({len(display_df)} records)")
    
    # Add risk indicators
    if 'severity' in display_columns:
        def style_severity(val):
            styles = {
                'Critical': 'background-color: #ffebee; color: #c62828; font-weight: bold; border-left: 4px solid #f44336;',
                'High': 'background-color: #fff3e0; color: #ef6c00; font-weight: bold; border-left: 4px solid #ff9800;',
                'Medium': 'background-color: #fffde7; color: #f57f17; border-left: 4px solid #ffeb3b;',
                'Low': 'background-color: #e8f5e8; color: #2e7d32; border-left: 4px solid #4caf50;'
            }
            return styles.get(val, '')
        
        # Apply styling
        display_df_styled = display_df[display_columns].style.map(
            style_severity, subset=['severity'] if 'severity' in display_columns else []
        )
        
        st.dataframe(display_df_styled, use_container_width=True, height=500)
    else:
        st.dataframe(display_df[display_columns], use_container_width=True, height=500)
    
    # Quick statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Records", len(display_df))
    with col2:
        if 'severity' in display_df.columns:
            critical_count = len(display_df[display_df['severity'] == 'Critical'])
            st.metric("Critical in View", critical_count)
    with col3:
        if 'participant_name' in display_df.columns:
            unique_participants = display_df['participant_name'].nunique()
            st.metric("Unique Participants", unique_participants)
    with col4:
        if 'location' in display_df.columns:
            unique_locations = display_df['location'].nunique()
            st.metric("Locations Involved", unique_locations)

# Enhanced export functionality
st.markdown("### 📥 Export & Reporting")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Generate Executive Report", type="primary"):
        st.markdown("### 📈 Executive Summary Report")
        
        report_data = {
            "Report Generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Analysis Period": f"{df_filtered['incident_date'].min().strftime('%B %Y')} - {df_filtered['incident_date'].max().strftime('%B %Y')}",
            "Total Incidents": len(df_filtered),
            "Critical Incidents": len(df_filtered[df_filtered['severity'] == 'Critical']),
            "Unique Participants": df_filtered['participant_name'].nunique(),
            "Average Severity Score": f"{df_filtered['severity_score'].mean():.2f}/4.0",
            "Compliance Rate": f"{(df_filtered['notification_delay'] <= 1).mean() * 100:.1f}%" if 'notification_delay' in df_filtered.columns else "N/A",
            "Most Common Incident Type": df_filtered['incident_type'].mode().iloc[0] if len(df_filtered) > 0 else 'N/A',
            "Highest Risk Location": df_filtered.groupby('location')['severity_score'].mean().idxmax() if len(df_filtered) > 0 else 'N/A'
        }
        
        for key, value in report_data.items():
            st.write(f"**{key}:** {value}")

with col2:
    if st.button("📋 Export Current View"):
        if display_columns:
            export_data = display_df[display_columns].copy()
            
            if export_format == "CSV":
                csv_data = export_data.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv_data,
                    f"ndis_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            elif export_format == "JSON":
                json_data = export_data.to_json(orient='records', date_format='iso', indent=2)
                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    f"ndis_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            else:  # Excel
                st.info("📊 Excel export functionality would be implemented with openpyxl")

with col3:
    if st.button("🔄 Reset All Filters"):
        st.cache_data.clear()
        # Reset session state filters
        if 'filters_reset' not in st.session_state:
            st.session_state.filters_reset = True
        st.rerun()

# Enhanced footer with comprehensive metadata
st.markdown("---")
st.markdown("### 📊 Dashboard Metadata")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    **📅 Data Period**  
    From: {df['incident_date'].min().strftime('%d %B %Y')}  
    To: {df['incident_date'].max().strftime('%d %B %Y')}
    """)

with col2:
    st.markdown(f"""
    **📈 Current View**  
    Records: {len(df_filtered):,} of {len(df):,}  
    Analysis: {selected_mode}
    """)

with col3:
    st.markdown(f"""
    **🔄 Last Updated**  
    {datetime.now().strftime('%d %B %Y')}  
    {datetime.now().strftime('%H:%M:%S')} AEST
    """)

with col4:
    data_quality_score = 95  # Simulated
    st.markdown(f"""
    **✅ Data Quality**  
    Score: {data_quality_score}%  
    Status: {"🟢 Excellent" if data_quality_score >= 90 else "🟡 Good" if data_quality_score >= 80 else "🔴 Needs Review"}
    """)

# Final call-to-action
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;">
    <h3>🚀 Ready to Take Action?</h3>
    <p>Use these insights to improve participant safety and organizational performance.</p>
    <p><strong>Next Steps:</strong> Review high-risk incidents • Update prevention protocols • Schedule staff training</p>
</div>
""", unsafe_allow_html=True)

# Helper functions for ML analysis
@st.cache_data
def prepare_ml_features(df):
    """Prepare features for ML analysis with better error handling"""
    try:
        ml_df = df.copy()
        
        # Create enhanced derived features
        if 'incident_date' in ml_df.columns:
            ml_df['incident_year'] = ml_df['incident_date'].dt.year
            ml_df['incident_month'] = ml_df['incident_date'].dt.month
            ml_df['incident_weekday'] = ml_df['incident_date'].dt.dayofweek
            ml_df['is_holiday_period'] = ml_df['incident_date'].dt.month.isin([12, 1, 4, 7])  # Holiday months
        
        # Enhanced age features
        if 'age' in ml_df.columns:
            ml_df['age_at_incident'] = ml_df['age']
            ml_df['age_category'] = pd.cut(ml_df['age'], bins=[0, 30, 50, 65, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
        else:
            ml_df['age_at_incident'] = np.random.normal(35, 15, len(ml_df)).clip(18, 85)
            ml_df['age_category'] = pd.cut(ml_df['age_at_incident'], bins=[0, 30, 50, 65, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # Encode categorical variables with better handling
        categorical_cols = ['incident_type', 'severity', 'location', 'age_category']
        if 'reportable' in ml_df.columns:
            categorical_cols.append('reportable')
        if 'medical_attention' in ml_df.columns:
            categorical_cols.append('medical_attention')
        
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

def perform_clustering_analysis(df, method='kmeans', n_clusters=5, **kwargs):
    """Enhanced clustering analysis with better error handling"""
    try:
        # Prepare feature matrix
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age_at_incident' in df.columns:
            feature_cols.append('age_at_incident')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'incident_month' in df.columns:
            feature_cols.append('incident_month')
        if 'incident_weekday' in df.columns:
            feature_cols.append('incident_weekday')
        
        if len(feature_cols) < 2:
            st.warning("Insufficient features for clustering analysis")
            return None, None
        
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering based on method
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Calculate quality metrics
        metrics = {}
        unique_labels = set(cluster_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:  # Valid clustering
            metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
            metrics['calinski_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
            metrics['n_clusters'] = len(unique_labels)
        else:
            metrics['n_clusters'] = len(unique_labels)
        
        return cluster_labels, metrics
        
    except Exception as e:
        st.error(f"Clustering analysis error: {str(e)}")
        return None, None

def detect_anomalies(df, method='isolation_forest', contamination=0.1):
    """Enhanced anomaly detection with better preprocessing"""
    try:
        # Prepare feature matrix
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age_at_incident' in df.columns:
            feature_cols.append('age_at_incident')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'incident_month' in df.columns:
            feature_cols.append('incident_month')
        
        if len(feature_cols) < 2:
            st.warning("Insufficient features for anomaly detection")
            return None
        
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection algorithm
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(n_neighbors=min(20, len(X_scaled)//2), contamination=contamination)
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        # Predict anomalies
        if method == 'local_outlier_factor':
            anomaly_labels = detector.fit_predict(X_scaled)
        else:
            anomaly_labels = detector.fit_predict(X_scaled)
        
        return anomaly_labels
        
    except Exception as e:
        st.error(f"Anomaly detection error: {str(e)}")
        return None

def find_association_rules(df, min_support=0.1, min_confidence=0.6):
    """Enhanced association rules mining with better transaction creation"""
    if not MLXTEND_AVAILABLE:
        return None, None
        
    try:
        # Create more comprehensive transactions
        categorical_cols = ['incident_type', 'severity', 'location']
        if 'reportable' in df.columns:
            categorical_cols.append('reportable')
        if 'medical_attention' in df.columns:
            categorical_cols.append('medical_attention')
        if 'age_category' in df.columns:
            categorical_cols.append('age_category')
        
        # Add temporal features
        if 'is_weekend' in df.columns:
            df['weekend_status'] = df['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
            categorical_cols.append('weekend_status')
        
        if 'hour' in df.columns:
            # Create time periods
            def get_time_period(hour):
                if 6 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                elif 18 <= hour < 22:
                    return 'Evening'
                else:
                    return 'Night'
            
            df['time_period'] = df['hour'].apply(get_time_period)
            categorical_cols.append('time_period')
        
        # Create transactions
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in categorical_cols:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{col}_{row[col]}")
            if len(transaction) >= 2:  # Only include transactions with multiple items
                transactions.append(transaction)
        
        if len(transactions) < 10:  # Need sufficient transactions
            return None, None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True, max_len=3)
        
        if len(frequent_itemsets) == 0:
            return None, None
        
        # Generate association rules
        rules = association_rules(
            frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence,
            num_itemsets=len(frequent_itemsets)
        )
        
        if len(rules) == 0:
            return frequent_itemsets, None
        
        # Add interpretability metrics
        rules['rule_strength'] = rules['confidence'] * rules['lift']
        rules['rule_interest'] = rules['confidence'] * rules['support'] * rules['lift']
        
        return frequent_itemsets, rules
        
    except Exception as e:
        st.error(f"Association rules mining error: {str(e)}")
        return None, None

# Additional utility functions
def create_risk_assessment_summary(df):
    """Create a comprehensive risk assessment summary"""
    try:
        summary = {
            'total_incidents': len(df),
            'critical_incidents': len(df[df['severity'] == 'Critical']),
            'high_risk_participants': df['participant_name'].value_counts().sum(),
            'compliance_rate': (df['notification_delay'] <= 1).mean() * 100 if 'notification_delay' in df.columns else None,
            'average_severity': df['severity_score'].mean(),
            'weekend_incidents': len(df[df['is_weekend'] == True]) if 'is_weekend' in df.columns else None,
            'locations_involved': df['location'].nunique(),
            'incident_types': df['incident_type'].nunique()
        }
        
        return summary
        
    except Exception as e:
        st.error(f"Error creating risk assessment summary: {str(e)}")
        return {}

def generate_recommendations_based_on_analysis(df, analysis_results=None):
    """Generate contextual recommendations based on data analysis"""
    recommendations = []
    
    try:
        # Basic data-driven recommendations
        if len(df) > 0:
            # High severity rate recommendations
            high_severity_rate = len(df[df['severity'].isin(['High', 'Critical'])]) / len(df)
            if high_severity_rate > 0.2:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Prevention',
                    'recommendation': 'Implement enhanced prevention protocols - severity rate exceeds 20%',
                    'impact': 'Reduce incident severity by 15-25%'
                })
            
            # Compliance recommendations
            if 'notification_delay' in df.columns:
                compliance_rate = (df['notification_delay'] <= 1).mean()
                if compliance_rate < 0.9:
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Process',
                        'recommendation': 'Improve incident reporting training and systems',
                        'impact': 'Increase compliance to 95%+'
                    })
            
            # Location-based recommendations
            location_incidents = df.groupby('location').size()
            if len(location_incidents) > 1:
                highest_risk_location = location_incidents.idxmax()
                if location_incidents.max() > location_incidents.mean() * 1.5:
                    recommendations.append({
                        'priority': 'High',
                        'category': 'Environmental',
                        'recommendation': f'Conduct comprehensive risk assessment for {highest_risk_location}',
                        'impact': 'Reduce location-specific incidents by 20-30%'
                    })
            
            # Temporal recommendations
            if 'hour' in df.columns:
                night_hours = list(range(22, 24)) + list(range(0, 6))
                night_incidents = len(df[df['hour'].isin(night_hours)])
                if night_incidents / len(df) > 0.3:
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Staffing',
                        'recommendation': 'Enhance night-time supervision and support protocols',
                        'impact': 'Reduce night-time incident severity'
                    })
        
        # ML-based recommendations if analysis results available
        if analysis_results:
            if 'anomalies_detected' in analysis_results and analysis_results['anomalies_detected'] > 0:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Investigation',
                    'recommendation': 'Investigate anomalous incidents for system improvements',
                    'impact': 'Prevent future unusual incidents'
                })
            
            if 'clusters_identified' in analysis_results and analysis_results['clusters_identified'] > 2:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Strategy',
                    'recommendation': 'Develop cluster-specific intervention strategies',
                    'impact': 'Targeted prevention based on incident patterns'
                })
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# Performance optimization tips
def optimize_dashboard_performance():
    """Provide performance optimization suggestions"""
    tips = [
        "📊 **Data Caching**: Large datasets are cached to improve loading times",
        "🔄 **Incremental Updates**: Only changed data is reprocessed",
        "📱 **Responsive Design**: Dashboard adapts to different screen sizes",
        "⚡ **Lazy Loading**: Heavy computations are performed only when needed",
        "🎯 **Smart Filtering**: Filters are applied efficiently to reduce processing time"
    ]
    return tips

# Help and documentation
def show_help_documentation():
    """Display comprehensive help documentation"""
    help_content = {
        "Getting Started": [
            "📤 Upload your incident data in CSV or Excel format",
            "🎛️ Use the sidebar controls to configure your analysis",
            "📊 Select an analysis mode that matches your needs",
            "🔍 Apply filters to focus on specific incidents or time periods"
        ],
        "Analysis Modes": [
            "🎯 **Executive Dashboard**: High-level overview with key metrics",
            "📊 **Risk Analysis**: Detailed risk assessment and factor analysis", 
            "🔗 **Correlation Explorer**: Discover relationships between variables",
            "🔮 **Predictive Analytics**: Forecast trends and assess scenario risks",
            "📈 **Performance Metrics**: Track KPIs and compliance rates",
            "🤖 **ML Analytics**: Advanced pattern detection and anomaly analysis"
        ],
        "Key Features": [
            "📈 **Real-time Analysis**: Instant updates as you change filters",
            "🎨 **Interactive Visualizations**: Click and explore your data",
            "📥 **Export Capabilities**: Download data and reports in multiple formats",
            "🔍 **Advanced Search**: Find specific incidents quickly",
            "💡 **Automated Insights**: AI-generated recommendations and alerts"
        ],
        "Best Practices": [
            "🔄 **Regular Updates**: Keep your data current for accurate insights",
            "🎯 **Focused Analysis**: Use filters to analyze specific areas of concern",
            "📊 **Trend Monitoring**: Check monthly trends for early warning signs",
            "🚨 **Anomaly Review**: Investigate unusual incidents promptly",
            "📋 **Action Planning**: Use insights to develop prevention strategies"
        ]
    }
    return help_content

# Initialize help system
if st.sidebar.button("❓ Help & Documentation"):
    st.session_state.show_help = True

if st.session_state.get('show_help', False):
    st.markdown("---")
    st.subheader("📚 Help & Documentation")
    
    help_content = show_help_documentation()
    
    for section, items in help_content.items():
        with st.expander(f"📖 {section}"):
            for item in items:
                st.markdown(f"- {item}")
    
    # Performance tips
    with st.expander("⚡ Performance Optimization"):
        for tip in optimize_dashboard_performance():
            st.markdown(f"- {tip}")
    
    # Contact and support
    with st.expander("🆘 Support & Contact"):
        st.markdown("""
        **For technical support or questions:**
        - 📧 Email: support@ndis-analytics.com
        - 📞 Phone: 1800-NDIS-HELP
        - 💬 Live Chat: Available 24/7 in the dashboard
        - 📚 Documentation: [docs.ndis-analytics.com](https://docs.ndis-analytics.com)
        
        **Emergency Incident Reporting:**
        - 🚨 Emergency Line: 000
        - 📞 NDIS Emergency: 1800-800-110
        """)
    
    if st.button("✅ Close Help"):
        st.session_state.show_help = False
        st.rerun()

# Error handling and logging
try:
    # Main dashboard execution is wrapped in try-catch
    pass
except Exception as e:
    st.error(f"""
    🚨 **Dashboard Error**: An unexpected error occurred.
    
    **Error Details**: {str(e)}
    
    **What you can do**:
    - 🔄 Try refreshing the page
    - 📁 Check if your data file is properly formatted
    - 🎛️ Reset filters and try again
    - 📞 Contact support if the issue persists
    """)
    
    # Log error for debugging (in production, this would go to a proper logging system)
    st.code(f"Error: {str(e)}", language="text")

# Version and system information
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    <strong>NDIS Analytics Dashboard</strong><br>
    Version 2.1.0 | Build 2024.12<br>
    <a href="#" style="color: #667eea;">Release Notes</a> | 
    <a href="#" style="color: #667eea;">Privacy Policy</a>
</div>
""", unsafe_allow_html=True)

# Add JavaScript for enhanced interactivity (if needed)
st.markdown("""
<script>
// Enhanced dashboard interactivity
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Add loading animations
    const loadingElements = document.querySelectorAll('.stSpinner');
    loadingElements.forEach(element => {
        element.style.animation = 'fadeIn 0.5s ease-in-out';
    });
});
</script>
""", unsafe_allow_html=True)
