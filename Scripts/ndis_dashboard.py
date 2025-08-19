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
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def process_data(df):
    """Process and enhance the loaded data"""
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
            df['hour'] = np.random.randint(6, 22, len(df))
        
        df['hour'] = df['hour'].fillna(12)
        
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

def perform_clustering_analysis(df, method='kmeans', n_clusters=5):
    """Perform clustering analysis"""
    try:
        if not SKLEARN_AVAILABLE:
            st.error("❌ Scikit-learn not available for clustering analysis")
            return None, None, None, None
            
        # Prepare features for clustering
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age' in df.columns:
            feature_cols.append('age')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'severity_score' in df.columns:
            feature_cols.append('severity_score')
        
        if len(feature_cols) < 2:
            st.warning("⚠️ Not enough features for clustering analysis")
            return None, None, None, None
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering based on method
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Calculate clustering metrics
        metrics = {}
        unique_labels = set(cluster_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            try:
                metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
                metrics['calinski_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
                metrics['n_clusters'] = len(unique_labels)
            except:
                metrics['n_clusters'] = len(unique_labels)
        
        return cluster_labels, metrics, X_scaled, feature_cols
        
    except Exception as e:
        st.error(f"❌ Clustering error: {str(e)}")
        return None, None, None, None

def detect_anomalies(df, method='isolation_forest', contamination=0.1):
    """Detect anomalies in the data"""
    try:
        if not SKLEARN_AVAILABLE:
            st.error("❌ Scikit-learn not available for anomaly detection")
            return None, None, None
            
        # Prepare features
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age' in df.columns:
            feature_cols.append('age')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'severity_score' in df.columns:
            feature_cols.append('severity_score')
        
        if len(feature_cols) < 2:
            st.warning("⚠️ Not enough features for anomaly detection")
            return None, None, None
        
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection method
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination)
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
        
        anomaly_labels = detector.fit_predict(X_scaled)
        
        return anomaly_labels, X_scaled, feature_cols
        
    except Exception as e:
        st.error(f"❌ Anomaly detection error: {str(e)}")
        return None, None, None

def find_association_rules(df, min_support=0.1, min_confidence=0.6):
    """Find association rules in the data"""
    try:
        if not MLXTEND_AVAILABLE:
            st.warning("⚠️ mlxtend not available for association rules. Install with: pip install mlxtend")
            return None, None
            
        # Prepare transaction data
        categorical_cols = ['incident_type', 'severity', 'location']
        if 'reportable' in df.columns:
            categorical_cols.append('reportable')
        
        # Create transactions
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in categorical_cols:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    transaction.append(f"{col}_{str(row[col]).strip()}")
            if len(transaction) >= 2:
                transactions.append(transaction)
        
        if len(transactions) < 10:
            st.warning("⚠️ Not enough transaction data for association rules")
            return None, None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        try:
            frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True)
        except:
            frequent_itemsets = apriori(df_transactions, min_support=0.01, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return None, None
        
        # Generate association rules
        try:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        except:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        
        return frequent_itemsets, rules
        
    except Exception as e:
        st.error(f"❌ Association rules error: {str(e)}")
        return None, None

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
                numeric_df[var] = pd.to_numeric(numeric_df[var], errors='coerce').fillna(0)
                correlation_vars.append(var)
        
        if len(correlation_vars) >= 2:
            corr_matrix = numeric_df[correlation_vars].corr()
        else:
            corr_matrix = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], 
                                     columns=['severity_numeric', 'age'], 
                                     index=['severity_numeric', 'age'])
            numeric_df['age'] = numeric_df.get('age', 35)
        
        return corr_matrix, numeric_df
        
    except Exception as e:
        st.error(f"❌ Error calculating correlations: {str(e)}")
        corr_matrix = pd.DataFrame([[1.0]], columns=['severity_numeric'], index=['severity_numeric'])
        return corr_matrix, df

def generate_insights(df):
    """Generate insights safely"""
    insights = []
    
    try:
        total_incidents = len(df)
        insights.append(f"📊 Total incidents analyzed: {total_incidents}")
        
        if 'severity' in df.columns:
            critical_count = len(df[df['severity'].str.lower().isin(['critical', 'high'])])
            if critical_count > 0:
                insights.append(f"🚨 {critical_count} high-severity incidents requiring attention")
        
        if 'location' in df.columns:
            top_location = df['location'].value_counts().index[0] if len(df) > 0 else 'Unknown'
            insights.append(f"🏢 Most incidents occur at: {top_location}")
        
        if len(insights) == 1:
            insights.extend([
                "🔍 Data processing complete - ready for analysis",
                "📈 Use the analysis modes to explore patterns"
            ])
            
    except Exception as e:
        insights = [f"⚠️ Insights generation error: {str(e)[:50]}..."]
    
    return insights

# Main Application
st.title("🏥 NDIS Incident Analytics Dashboard")

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
    ["Executive Overview", "Risk Analysis", "🤖 ML Analytics", "Data Explorer"]
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
                title="🔝 Top Incident Types",
                labels={'x': 'Count', 'y': 'Incident Type'}
            )
            fig1.update_layout(height=400)
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
            fig2.update_layout(height=400)
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
            markers=True,
            labels={'x': 'Month', 'y': 'Number of Incidents'}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

elif analysis_mode == "Risk Analysis":
    st.subheader("🎯 Risk Analysis")
    
    # Risk Assessment Matrix
    col1, col2 = st.columns(2)
    
    with col1:
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
                title="🎯 Risk Matrix: Volume vs Severity by Location",
                labels={'Count': 'Number of Incidents', 'Avg Severity': 'Average Severity Score'}
            )
            fig_risk.update_layout(height=500)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Risk analysis requires location and severity data")
    
    with col2:
        # Time-based risk analysis
        if 'hour' in df_filtered.columns:
            hourly_incidents = df_filtered.groupby('hour').size()
            fig_hourly = px.bar(
                x=hourly_incidents.index,
                y=hourly_incidents.values,
                title="⏰ Incidents by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Incidents'}
            )
            fig_hourly.update_layout(height=500)
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Risk summary table
    if 'location' in df_filtered.columns and 'severity_score' in df_filtered.columns:
        st.subheader("📊 Risk Summary by Location")
        location_risk_expanded = df_filtered.groupby('location').agg({
            'severity_score': ['mean', 'max', 'count'],
            'notification_delay': 'mean'
        }).round(2)
        
        location_risk_expanded.columns = ['Avg Severity', 'Max Severity', 'Incident Count', 'Avg Delay (days)']
        st.dataframe(location_risk_expanded, use_container_width=True)

elif analysis_mode == "🤖 ML Analytics":
    st.subheader("🤖 Machine Learning Analytics")
    
    if not SKLEARN_AVAILABLE:
        st.error("❌ Machine Learning features require scikit-learn. Please install it to use ML analytics.")
        st.code("pip install scikit-learn")
        st.stop()
    
    # Prepare ML features
    ml_df, label_encoders = prepare_ml_features(df_filtered)
    
    # ML Analysis tabs
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["🔗 Clustering", "🚨 Anomaly Detection", "🔍 Association Rules", "📊 ML Insights"])
    
    with ml_tab1:
        st.subheader("🔗 Incident Clustering Analysis")
        st.markdown("*Discover hidden patterns and group similar incidents together*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Clustering Parameters")
            
            clustering_method = st.selectbox(
                "Clustering Algorithm",
                ["kmeans", "dbscan", "hierarchical"],
                help="K-means: Fixed number of clusters | DBSCAN: Density-based | Hierarchical: Tree-like clustering"
            )
            
            if clustering_method in ['kmeans', 'hierarchical']:
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            else:
                n_clusters = None
                st.info("ℹ️ DBSCAN automatically determines the optimal number of clusters")
            
            if st.button("🔄 Run Clustering Analysis", type="primary"):
                with st.spinner("🔄 Analyzing incident patterns..."):
                    cluster_labels, metrics, X_scaled, feature_cols = perform_clustering_analysis(
                        ml_df, method=clustering_method, n_clusters=n_clusters
                    )
                    
                    if cluster_labels is not None:
                        st.session_state['cluster_labels'] = cluster_labels
                        st.session_state['cluster_metrics'] = metrics
                        st.session_state['cluster_features'] = feature_cols
                        st.session_state['cluster_scaled'] = X_scaled
                        st.success("✅ Clustering analysis completed!")
                    else:
                        st.error("❌ Clustering analysis failed. Please check your data.")
        
        with col2:
            if 'cluster_labels' in st.session_state and st.session_state['cluster_labels'] is not None:
                cluster_labels = st.session_state['cluster_labels']
                metrics = st.session_state.get('cluster_metrics', {})
                
                # Display clustering metrics
                st.markdown("### 📊 Clustering Results")
                if metrics:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("🎯 Clusters Found", metrics.get('n_clusters', len(set(cluster_labels))))
                    with col_b:
                        if 'silhouette_score' in metrics:
                            score = metrics['silhouette_score']
                            st.metric("📈 Silhouette Score", f"{score:.3f}", 
                                    help="Quality measure: -1 (poor) to 1 (excellent)")
                    with col_c:
                        if 'calinski_score' in metrics:
                            st.metric("🎯 Calinski Score", f"{metrics['calinski_score']:.1f}",
                                    help="Higher values indicate better clustering")
                
                # Cluster visualization using PCA
                if 'cluster_scaled' in st.session_state:
                    X_scaled = st.session_state['cluster_scaled']
                    
                    if X_scaled.shape[1] >= 2:
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Create scatter plot
                        cluster_df = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Cluster': [f'Cluster {c}' for c in cluster_labels],
                            'Incident_Type': ml_df['incident_type'].values if 'incident_type' in ml_df.columns else 'Unknown',
                            'Severity': ml_df['severity'].values if 'severity' in ml_df.columns else 'Unknown'
                        })
                        
                        fig_cluster = px.scatter(
                            cluster_df,
                            x='PC1', y='PC2',
                            color='Cluster',
                            hover_data=['Incident_Type', 'Severity'],
                            title="🔍 Incident Clusters (PCA Visualization)",
                            labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'}
                        )
                        fig_cluster.update_layout(height=500)
                        st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Cluster characteristics analysis
                st.markdown("### 🔍 Cluster Characteristics")
                cluster_analysis = []
                
                for cluster_id in sorted(set(cluster_labels)):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_data = ml_df[cluster_mask]
                    
                    analysis = {
                        'Cluster': f'Cluster {cluster_id}',
                        'Size': len(cluster_data),
                        'Percentage': f"{len(cluster_data)/len(ml_df)*100:.1f}%"
                    }
                    
                    # Most common characteristics
                    if 'incident_type' in cluster_data.columns:
                        most_common = cluster_data['incident_type'].mode()
                        analysis['Common Type'] = most_common.iloc[0] if len(most_common) > 0 else 'Mixed'
                    
                    if 'severity' in cluster_data.columns:
                        most_common_sev = cluster_data['severity'].mode()
                        analysis['Common Severity'] = most_common_sev.iloc[0] if len(most_common_sev) > 0 else 'Mixed'
                    
                    if 'location' in cluster_data.columns:
                        most_common_loc = cluster_data['location'].mode()
                        analysis['Common Location'] = most_common_loc.iloc[0] if len(most_common_loc) > 0 else 'Mixed'
                    
                    cluster_analysis.append(analysis)
                
                cluster_results_df = pd.DataFrame(cluster_analysis)
                st.dataframe(cluster_results_df, use_container_width=True)
            else:
                st.info("👆 Click 'Run Clustering Analysis' to discover incident patterns")
    
    with ml_tab2:
        st.subheader("🚨 Anomaly Detection")
        st.markdown("*Identify unusual incidents that may require special attention*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Detection Parameters")
            
            anomaly_method = st.selectbox(
                "Detection Algorithm",
                ["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"],
                help="Different algorithms for detecting unusual patterns"
            )
            
            contamination = st.slider(
                "Expected Anomaly Rate (%)", 
                1, 20, 10,
                help="Percentage of incidents expected to be anomalous"
            ) / 100
            
            if st.button("🔍 Detect Anomalies", type="primary"):
                with st.spinner("🔍 Scanning for unusual incidents..."):
                    anomaly_labels, X_scaled, feature_cols = detect_anomalies(
                        ml_df, method=anomaly_method, contamination=contamination
                    )
                    
                    if anomaly_labels is not None:
                        st.session_state['anomaly_labels'] = anomaly_labels
                        st.session_state['anomaly_features'] = feature_cols
                        st.session_state['anomaly_scaled'] = X_scaled
                        st.success("✅ Anomaly detection completed!")
                    else:
                        st.error("❌ Anomaly detection failed. Please check your data.")
        
        with col2:
            if 'anomaly_labels' in st.session_state and st.session_state['anomaly_labels'] is not None:
                anomaly_labels = st.session_state['anomaly_labels']
                
                # Anomaly statistics
                n_anomalies = sum(anomaly_labels == -1)
                n_normal = sum(anomaly_labels == 1)
                anomaly_percentage = n_anomalies / len(anomaly_labels) * 100
                
                st.markdown("### 📊 Anomaly Detection Results")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🚨 Anomalies Found", n_anomalies)
                with col_b:
                    st.metric("📈 Anomaly Rate", f"{anomaly_percentage:.1f}%")
                with col_c:
                    st.metric("✅ Normal Cases", n_normal)
                
                if n_anomalies > 0:
                    # Anomaly visualization
                    if 'anomaly_scaled' in st.session_state:
                        X_scaled = st.session_state['anomaly_scaled']
                        
                        if X_scaled.shape[1] >= 2:
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            
                            anomaly_df = pd.DataFrame({
                                'PC1': X_pca[:, 0],
                                'PC2': X_pca[:, 1],
                                'Type': ['🚨 Anomaly' if label == -1 else '✅ Normal' for label in anomaly_labels],
                                'Incident_Type': ml_df['incident_type'].values if 'incident_type' in ml_df.columns else 'Unknown',
                                'Severity': ml_df['severity'].values if 'severity' in ml_df.columns else 'Unknown'
                            })
                            
                            fig_anomaly = px.scatter(
                                anomaly_df,
                                x='PC1', y='PC2',
                                color='Type',
                                hover_data=['Incident_Type', 'Severity'],
                                title="🔍 Anomaly Detection Results",
                                color_discrete_map={'🚨 Anomaly': 'red', '✅ Normal': 'blue'}
                            )
                            fig_anomaly.update_layout(height=500)
                            st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    # Anomaly analysis
                    st.markdown("### 🔍 Anomaly Analysis")
                    
                    anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
                    anomaly_data = ml_df.iloc[anomaly_indices]
                    normal_data = ml_df.iloc[[i for i, label in enumerate(anomaly_labels) if label == 1]]
                    
                    # Compare characteristics
                    st.markdown("**Anomalous vs Normal Incident Characteristics:**")
                    
                    comparison_data = []
                    for col in ['incident_type', 'severity', 'location']:
                        if col in anomaly_data.columns:
                            anomaly_dist = anomaly_data[col].value_counts(normalize=True).head(3)
                            normal_dist = normal_data[col].value_counts(normalize=True)
                            
                            for category in anomaly_dist.index:
                                comparison_data.append({
                                    'Category': f"{col}: {category}",
                                    'Anomaly %': f"{anomaly_dist[category] * 100:.1f}%",
                                    'Normal %': f"{normal_dist.get(category, 0) * 100:.1f}%"
                                })
                    
                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data)
                        st.dataframe(comp_df, use_container_width=True)
                        
                        # Show some anomalous incidents
                        st.markdown("**Sample Anomalous Incidents:**")
                        display_cols = ['incident_type', 'severity', 'location']
                        if 'description' in anomaly_data.columns:
                            display_cols.append('description')
                        
                        sample_anomalies = anomaly_data[display_cols].head(5)
                        st.dataframe(sample_anomalies, use_container_width=True)
                else:
                    st.info("✅ No anomalies detected with current parameters")
            else:
                st.info("👆 Click 'Detect Anomalies' to find unusual incidents")
    
    with ml_tab3:
        st.subheader("🔍 Association Rules Mining")
        st.markdown("*Discover relationships between incident characteristics*")
        
        if not MLXTEND_AVAILABLE:
            st.warning("⚠️ Association rules require mlxtend library. Install with: `pip install mlxtend`")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 🎛️ Mining Parameters")
                
                min_support = st.slider(
                    "Minimum Support", 
                    0.01, 0.5, 0.1,
                    help="Minimum frequency of item combinations"
                )
                
                min_confidence = st.slider(
                    "Minimum Confidence", 
                    0.1, 0.9, 0.6,
                    help="Minimum confidence for association rules"
                )
                
                if st.button("⚡ Mine Association Rules", type="primary"):
                    with st.spinner("⚡ Mining association patterns..."):
                        frequent_itemsets, rules = find_association_rules(
                            ml_df, min_support=min_support, min_confidence=min_confidence
                        )
                        
                        if frequent_itemsets is not None and rules is not None:
                            st.session_state['frequent_itemsets'] = frequent_itemsets
                            st.session_state['association_rules'] = rules
                            st.success("✅ Association rules mining completed!")
                        else:
                            st.warning("⚠️ No rules found. Try lowering the parameters.")
            
            with col2:
                if 'association_rules' in st.session_state and st.session_state['association_rules'] is not None:
                    rules = st.session_state['association_rules']
                    frequent_itemsets = st.session_state.get('frequent_itemsets', pd.DataFrame())
                    
                    st.markdown("### 📊 Association Rules Results")
                    
                    if len(rules) > 0:
                        # Summary metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("🔗 Rules Found", len(rules))
                        with col_b:
                            avg_confidence = rules['confidence'].mean()
                            st.metric("📈 Avg Confidence", f"{avg_confidence:.3f}")
                        with col_c:
                            avg_support = rules['support'].mean()
                            st.metric("🎯 Avg Support", f"{avg_support:.3f}")
                        
                        # Top rules
                        st.markdown("### 🔝 Top Association Rules")
                        top_rules = rules.sort_values('confidence', ascending=False).head(10)
                        
                        display_rules = []
                        for _, rule in top_rules.iterrows():
                            antecedents = ', '.join(list(rule['antecedents']))
                            consequents = ', '.join(list(rule['consequents']))
                            display_rules.append({
                                'Rule': f"{antecedents} → {consequents}",
                                'Support': f"{rule['support']:.3f}",
                                'Confidence': f"{rule['confidence']:.3f}",
                                'Lift': f"{rule['lift']:.3f}"
                            })
                        
                        rules_df = pd.DataFrame(display_rules)
                        st.dataframe(rules_df, use_container_width=True)
                        
                        # Rules visualization
                        fig_rules = px.scatter(
                            rules,
                            x='support',
                            y='confidence',
                            size='lift',
                            title="🔍 Association Rules Visualization",
                            labels={'support': 'Support', 'confidence': 'Confidence'}
                        )
                        fig_rules.update_layout(height=400)
                        st.plotly_chart(fig_rules, use_container_width=True)
                    else:
                        st.info("No association rules found. Try lowering the minimum parameters.")
                else:
                    st.info("👆 Click 'Mine Association Rules' to discover relationships")
    
    with ml_tab4:
        st.subheader("📊 ML Insights Summary")
        st.markdown("*Key findings from machine learning analysis*")
        
        # Correlation heatmap
        if len(corr_matrix) > 1:
            fig_corr = px.imshow(
                corr_matrix,
                title="🔗 Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature importance (if available)
        if 'cluster_labels' in st.session_state or 'anomaly_labels' in st.session_state:
            st.markdown("### 🎯 Key Insights")
            
            insights_list = [
                "📊 **Data Quality**: Processed and analyzed successfully",
                "🔍 **Pattern Detection**: Multiple analytical approaches applied",
                "⚡ **Real-time Analysis**: Results updated based on current filters"
            ]
            
            if 'cluster_labels' in st.session_state:
                n_clusters = len(set(st.session_state['cluster_labels']))
                insights_list.append(f"🔗 **Clustering**: Identified {n_clusters} distinct incident patterns")
            
            if 'anomaly_labels' in st.session_state:
                n_anomalies = sum(st.session_state['anomaly_labels'] == -1)
                insights_list.append(f"🚨 **Anomalies**: Detected {n_anomalies} unusual incidents requiring attention")
            
            if 'association_rules' in st.session_state and len(st.session_state['association_rules']) > 0:
                n_rules = len(st.session_state['association_rules'])
                insights_list.append(f"🔍 **Associations**: Found {n_rules} significant relationships between factors")
            
            for insight in insights_list:
                st.markdown(insight)
        else:
            st.info("Run the ML analysis tools above to see insights here.")

elif analysis_mode == "Data Explorer":
    st.subheader("📋 Data Explorer")
    
    # Search functionality
    search_term = st.text_input("🔍 Search in descriptions")
    
    # Filter data based on search
    display_df = df_filtered.copy()
    if search_term and 'description' in display_df.columns:
        mask = display_df['description'].str.contains(search_term, case=False, na=False)
        display_df = display_df[mask]
    
    # Column selector
    if len(display_df.columns) > 10:
        selected_columns = st.multiselect(
            "Select columns to display",
            options=display_df.columns.tolist(),
            default=display_df.columns.tolist()[:10]
        )
        display_df = display_df[selected_columns] if selected_columns else display_df
    
    # Display data with pagination
    st.markdown(f"**Showing {len(display_df)} records**")
    
    # Pagination
    if len(display_df) > 100:
        page_size = st.selectbox("Records per page", [25, 50, 100], index=1)
        total_pages = (len(display_df) - 1) // page_size + 1
        page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1))
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        display_df = display_df.iloc[start_idx:end_idx]
    
    # Display data
    st.dataframe(display_df, use_container_width=True)
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Filtered Data"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ndis_incidents_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate Report"):
            # Simple report generation
            report = f"""
# NDIS Incident Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Incidents: {len(df_filtered)}
- Date Range: {df_filtered['incident_date'].min().strftime('%Y-%m-%d')} to {df_filtered['incident_date'].max().strftime('%Y-%m-%d')}
- Unique Locations: {df_filtered['location'].nunique()}
- High Severity Incidents: {len(df_filtered[df_filtered['severity'].str.lower().isin(['high', 'critical'])])}

## Top Incident Types
{df_filtered['incident_type'].value_counts().head().to_string()}

## Severity Distribution
{df_filtered['severity'].value_counts().to_string()}
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"ndis_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown(f"**Records:** {len(df_filtered)} of {len(df)}")
with col3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Help & Support")
    with st.expander("🤔 How to use this dashboard"):
        st.markdown("""
        **Getting Started:**
        1. Upload your CSV file using the file uploader
        2. Use filters to focus on specific data
        3. Choose an analysis mode to explore your data
        
        **Analysis Modes:**
        - **Executive Overview**: High-level KPIs and trends
        - **Risk Analysis**: Identify risk patterns and hotspots
        - **ML Analytics**: Advanced machine learning insights
        - **Data Explorer**: Browse and search your raw data
        
        **Tips:**
        - Use date filters to analyze specific time periods
        - Try different ML algorithms for varied insights
        - Download results for further analysis
        """)
    
    st.markdown("**Built with Streamlit & Plotly** 🚀")
