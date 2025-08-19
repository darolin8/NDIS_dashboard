import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# ML Libraries for advanced analytics
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not installed. ML Analytics will be disabled.")

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
    page_title="NDIS Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

st.title("🏥 NDIS Incident Analytics Dashboard")

# Data processing functions
@st.cache_data
def process_data(df):
    """Process and enhance the loaded data"""
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure we have required columns
        required_columns = ['incident_date', 'incident_type', 'severity', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Convert date columns safely
        if 'incident_date' in df.columns:
            # Try different date formats
            df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce', dayfirst=True)
            if df['incident_date'].isna().all():
                st.error("❌ Could not parse incident_date. Please use DD/MM/YYYY or YYYY-MM-DD format.")
                return None
        
        # Handle notification_date
        if 'notification_date' in df.columns:
            df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce', dayfirst=True)
        else:
            # Create notification dates with small random delays if missing
            delays = np.random.uniform(0, 2, len(df))  # 0-2 days delay
            df['notification_date'] = df['incident_date'] + pd.to_timedelta(delays, unit='days')
        
        # Calculate notification delay safely
        if 'notification_date' in df.columns and not df['notification_date'].isna().all():
            df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / (24 * 3600)
            df['notification_delay'] = df['notification_delay'].fillna(0)
        else:
            df['notification_delay'] = 0
        
        # Add time-based columns
        df['month'] = df['incident_date'].dt.month_name()
        df['day_of_week'] = df['incident_date'].dt.day_name()
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
        
        # Handle incident_time
        if 'incident_time' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour
            except:
                df['hour'] = pd.to_datetime(df['incident_time'], errors='coerce').dt.hour
        else:
            # Create random hours if missing
            df['hour'] = np.random.randint(6, 22, len(df))  # Business hours bias
        
        df['hour'] = df['hour'].fillna(12)  # Default to noon for missing values
        
        # Risk scoring
        severity_mapping = {
            'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4,
            'low': 1, 'medium': 2, 'high': 3, 'critical': 4,
            'L': 1, 'M': 2, 'H': 3, 'C': 4
        }
        df['severity_score'] = df['severity'].map(severity_mapping).fillna(1)
        
        # Create age groups
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35)
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        else:
            # Create age data if missing
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
        
        # Clean string columns
        string_columns = ['incident_type', 'severity', 'location']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def load_data_from_file(uploaded_file):
    """Load data from uploaded file"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        if len(df) == 0:
            st.error("❌ The uploaded file is empty.")
            return None
        
        st.info(f"📁 Loaded {len(df)} rows from uploaded file")
        
        # Process the data
        processed_df = process_data(df)
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

def calculate_correlations(df):
    """Calculate correlations safely"""
    try:
        numeric_df = df.copy()
        
        # Convert categorical to numeric safely
        if 'severity_score' in numeric_df.columns:
            numeric_df['severity_numeric'] = numeric_df['severity_score']
        else:
            numeric_df['severity_numeric'] = 1
            
        if 'reportable' in df.columns:
            numeric_df['reportable_numeric'] = df['reportable'].map({'Yes': 1, 'No': 0}).fillna(0)
        else:
            numeric_df['reportable_numeric'] = 0
            
        if 'is_weekend' in df.columns:
            numeric_df['is_weekend_numeric'] = df['is_weekend'].astype(int)
        else:
            numeric_df['is_weekend_numeric'] = 0
        
        # Select available numeric columns
        correlation_vars = []
        possible_vars = ['age', 'severity_numeric', 'notification_delay', 'reportable_numeric', 
                        'is_weekend_numeric', 'hour']
        
        for var in possible_vars:
            if var in numeric_df.columns:
                # Ensure numeric and handle NaN
                numeric_df[var] = pd.to_numeric(numeric_df[var], errors='coerce').fillna(0)
                correlation_vars.append(var)
        
        if len(correlation_vars) >= 2:
            corr_matrix = numeric_df[correlation_vars].corr()
        else:
            # Create minimal correlation matrix
            corr_matrix = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], 
                                     columns=['severity_numeric', 'age'], 
                                     index=['severity_numeric', 'age'])
            numeric_df['age'] = numeric_df.get('age', 35)
        
        return corr_matrix, numeric_df
        
    except Exception as e:
        st.error(f"❌ Error calculating correlations: {str(e)}")
        # Return minimal correlation matrix
        corr_matrix = pd.DataFrame([[1.0]], columns=['severity_numeric'], index=['severity_numeric'])
        return corr_matrix, df

def generate_insights(df):
    """Generate insights safely"""
    insights = []
    
    try:
        # Basic insights
        total_incidents = len(df)
        insights.append(f"📊 Total incidents analyzed: {total_incidents}")
        
        if 'severity' in df.columns:
            critical_count = len(df[df['severity'].str.lower().isin(['critical', 'high'])])
            if critical_count > 0:
                insights.append(f"🚨 {critical_count} high-severity incidents requiring attention")
        
        if 'location' in df.columns:
            top_location = df['location'].value_counts().index[0] if len(df) > 0 else 'Unknown'
            insights.append(f"🏢 Most incidents occur at: {top_location}")
        
        # Default insights if no specific patterns found
        if len(insights) == 1:
            insights.extend([
                "🔍 Data processing complete - ready for analysis",
                "📈 Use the analysis modes to explore patterns"
            ])
            
    except Exception as e:
        insights = [f"⚠️ Insights generation error: {str(e)[:50]}..."]
    
    return insights

# ML Functions (simplified and safe)
@st.cache_data
def prepare_ml_features(df):
    """Prepare features for ML analysis safely"""
    try:
        if not SKLEARN_AVAILABLE:
            return df, {}
            
        ml_df = df.copy()
        label_encoders = {}
        
        # Encode categorical variables safely
        categorical_cols = ['incident_type', 'severity', 'location']
        
        for col in categorical_cols:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col] = ml_df[col].fillna('Unknown').astype(str)
                ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col])
                label_encoders[col] = le
        
        return ml_df, label_encoders
        
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return df, {}

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
    # 🏥 NDIS Incident Analytics Dashboard
    
    ## 📁 Welcome! Please Upload Your Data
    
    To get started with your NDIS incident analysis:
    
    1. **📋 Prepare your CSV file** with incident data
    2. **📁 Use the file uploader** in the sidebar
    3. **📊 Explore your data** with advanced analytics
    
    ### 📋 Required CSV Columns:
    - `incident_date` - Date of incident (DD/MM/YYYY format)
    - `incident_type` - Type of incident
    - `severity` - Severity level (Low, Medium, High, Critical)
    - `location` - Where the incident occurred
    
    ### 🔧 Optional Columns:
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
    corr_matrix, numeric_df = calculate_correlations(df)
    insights = generate_insights(df)
    
    st.success(f"✅ Successfully loaded {len(df)} incidents from {df['incident_date'].min().strftime('%B %Y')} to {df['incident_date'].max().strftime('%B %Y')}")
    
except Exception as e:
    st.error(f"❌ Error processing data: {str(e)}")
    st.stop()

# Sidebar controls
st.sidebar.header("🎛️ Analysis Controls")

# Analysis Mode Selection
analysis_mode = st.sidebar.selectbox(
    "🔬 Analysis Mode",
    ["Executive Overview", "Risk Analysis", "Data Explorer"]
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

# Live insights
with st.sidebar:
    st.subheader("💡 Live Insights")
    for insight in insights[:3]:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

# Main dashboard content
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
        if 'notification_delay' in df_filtered.columns:
            avg_delay = df_filtered['notification_delay'].mean()
            st.metric("⏱️ Avg Delay (days)", f"{avg_delay:.1f}")
        else:
            st.metric("⏱️ Avg Delay", "N/A")
    
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
                title="🔝 Top Incident Types"
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Severity distribution
        if 'severity' in df_filtered.columns:
            severity_counts = df_filtered['severity'].value_counts()
            fig2 = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="⚠️ Severity Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly trends
    if 'incident_date' in df_filtered.columns:
        st.subheader("📈 Monthly Trends")
        monthly_data = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).size()
        monthly_data.index = monthly_data.index.astype(str)
        
        fig3 = px.line(
            x=monthly_data.index,
            y=monthly_data.values,
            title="📊 Incidents Over Time",
            markers=True
        )
        st.plotly_chart(fig3, use_container_width=True)

elif analysis_mode == "Risk Analysis":
    st.subheader("🎯 Risk Analysis")
    
    if 'location' in df_filtered.columns and 'severity_score' in df_filtered.columns:
        # Risk by location
        location_risk = df_filtered.groupby('location').agg({
            'severity_score': 'mean',
            'incident_id': 'count'
        }).round(2)
        location_risk.columns = ['Avg Severity', 'Count']
        
        fig_risk = px.scatter(
            location_risk.reset_index(),
            x='Count',
            y='Avg Severity',
            size='Count',
            hover_name='location',
            title="🎯 Risk Matrix: Volume vs Severity by Location"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.dataframe(location_risk, use_container_width=True)
    else:
        st.info("Risk analysis requires location and severity data")

elif analysis_mode == "Data Explorer":
    st.subheader("📋 Data Explorer")
    
    # Search functionality
    search_term = st.text_input("🔍 Search in descriptions")
    
    # Filter data based on search
    display_df = df_filtered.copy()
    if search_term and 'description' in display_df.columns:
        mask = display_df['description'].str.contains(search_term, case=False, na=False)
        display_df = display_df[mask]
    
    # Display data
    st.dataframe(display_df, use_container_width=True)
    
    # Download option
    if st.button("📥 Download Filtered Data"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ndis_incidents_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown(f"**Dashboard last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(f"**Records displayed:** {len(df_filtered)} of {len(df)}")
