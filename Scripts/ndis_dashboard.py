import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import warnings

# ML Libraries for advanced analytics
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.model_selection import train_test_split
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def process_data(df):
    """Process and enhance the loaded data with additional features"""
    try:
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['incident_date', 'incident_type', 'severity', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Convert date columns
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce', dayfirst=True)
        if df['incident_date'].isna().all():
            st.error("❌ Could not parse incident_date. Please use DD/MM/YYYY or YYYY-MM-DD format.")
            return None
        
        # Handle notification_date
        if 'notification_date' in df.columns:
            df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce', dayfirst=True)
            df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / (24 * 3600)
        else:
            # Generate synthetic notification delays
            delays = np.random.uniform(0, 2, len(df))
            df['notification_date'] = df['incident_date'] + pd.to_timedelta(delays, unit='days')
            df['notification_delay'] = delays
        
        df['notification_delay'] = df['notification_delay'].fillna(0)
        
        # Add time-based features
        df['month'] = df['incident_date'].dt.month_name()
        df['month_num'] = df['incident_date'].dt.month
        df['day_of_week'] = df['incident_date'].dt.day_name()
        df['day_of_week_num'] = df['incident_date'].dt.dayofweek
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
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
        
        # Risk scoring
        severity_mapping = {
            'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4,
            'low': 1, 'medium': 2, 'high': 3, 'critical': 4
        }
        df['severity_score'] = df['severity'].map(severity_mapping).fillna(1)
        
        # Composite risk score
        df['risk_score'] = df['severity_score'] * 2
        df['risk_score'] += (df['notification_delay'] > 1).astype(int)
        df['risk_score'] += df['is_weekend'].astype(int) * 0.5
        
        # Age handling
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35)
        else:
            df['age'] = np.random.normal(40, 20, len(df)).clip(18, 85).astype(int)
        
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 25, 35, 50, 65, 100], 
                               labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Ensure participant names exist
        if 'participant_name' not in df.columns:
            df['participant_name'] = [f'Participant_{i:03d}' for i in range(1, len(df) + 1)]
        
        # Ensure incident_id exists
        if 'incident_id' not in df.columns:
            df['incident_id'] = [f'INC{i:06d}' for i in range(1, len(df) + 1)]
        
        # Additional features
        df['incident_count_by_participant'] = df.groupby('participant_name')['incident_id'].transform('count')
        df['location_incident_rate'] = df.groupby('location')['incident_id'].transform('count') / len(df)
        
        # Clean string columns
        string_columns = ['incident_type', 'severity', 'location']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def prepare_ml_features(df):
    """Prepare features for ML analysis"""
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
                ml_df[col] = ml_df[col].astype(str).fillna('Unknown')
                
                # Handle categorical data from pd.cut
                if hasattr(ml_df[col], 'cat'):
                    ml_df[col] = ml_df[col].astype(str)
                
                ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col])
                label_encoders[col] = le
        
        return ml_df, label_encoders
        
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return df, {}

def predict_future_incidents(df, prediction_days=30):
    """Simple prediction models for future incidents"""
    try:
        if not SKLEARN_AVAILABLE:
            return None
        
        # Prepare time series data
        daily_counts = df.groupby(df['incident_date'].dt.date).size()
        daily_counts = daily_counts.reindex(
            pd.date_range(daily_counts.index.min(), daily_counts.index.max()), 
            fill_value=0
        )
        
        if len(daily_counts) < 7:
            return None
        
        # Simple moving average
        ma_window = min(7, len(daily_counts) // 2)
        ma_prediction = daily_counts.rolling(window=ma_window).mean().iloc[-1]
        
        # Trend-based prediction
        days_numeric = np.arange(len(daily_counts))
        z = np.polyfit(days_numeric, daily_counts.values, 1)
        p = np.poly1d(z)
        trend_prediction = p(len(daily_counts) + prediction_days // 2)
        
        # Random Forest prediction
        rf_prediction = None
        if len(daily_counts) > 14:
            # Create lagged features
            lags = min(7, len(daily_counts) // 3)
            ts_features = []
            for i in range(1, lags + 1):
                ts_features.append(daily_counts.shift(i))
            
            ts_df = pd.DataFrame(ts_features).T
            ts_df.columns = [f'lag_{i}' for i in range(1, lags + 1)]
            ts_df['target'] = daily_counts.values
            ts_df = ts_df.dropna()
            
            if len(ts_df) > 5:
                X = ts_df.drop('target', axis=1)
                y = ts_df['target']
                
                if len(X) > 3:
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(X, y)
                    
                    # Predict next few days
                    last_values = daily_counts.iloc[-lags:].values
                    predictions = []
                    
                    for _ in range(min(7, prediction_days)):
                        pred = rf.predict(last_values.reshape(1, -1))[0]
                        predictions.append(max(0, pred))
                        last_values = np.append(last_values[1:], pred)
                    
                    rf_prediction = np.mean(predictions)
        
        return {
            'moving_average': max(0, ma_prediction),
            'trend_based': max(0, trend_prediction),
            'random_forest': rf_prediction,
            'daily_counts': daily_counts
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def detect_anomalies(df):
    """Detect anomalous incidents using multiple methods"""
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
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_flags = iso_forest.fit_predict(X_scaled) == -1
        anomaly_scores = -iso_forest.score_samples(X_scaled)
        
        # Get anomaly details
        anomaly_indices = np.where(anomaly_flags)[0]
        
        return {
            'anomaly_flags': anomaly_flags,
            'anomaly_scores': anomaly_scores,
            'anomaly_indices': anomaly_indices,
            'n_anomalies': np.sum(anomaly_flags),
            'anomaly_rate': np.sum(anomaly_flags) / len(df) * 100
        }
        
    except Exception as e:
        st.error(f"Anomaly detection error: {str(e)}")
        return None

def perform_clustering(df, method='kmeans', n_clusters=4):
    """Perform clustering analysis"""
    try:
        if not SKLEARN_AVAILABLE:
            return None
        
        ml_df, _ = prepare_ml_features(df)
        
        # Select features for clustering
        feature_cols = [col for col in ml_df.columns if col.endswith('_encoded')]
        numeric_cols = ['age', 'hour', 'severity_score', 'risk_score']
        feature_cols.extend([col for col in numeric_cols if col in ml_df.columns])
        
        if len(feature_cols) < 2:
            return None
        
        X = ml_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Calculate metrics
        unique_labels = np.unique(cluster_labels)
        metrics = {}
        
        if len(unique_labels) > 1 and -1 not in unique_labels:
            try:
                metrics['silhouette'] = silhouette_score(X_scaled, cluster_labels)
            except:
                pass
        
        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        return {
            'labels': cluster_labels,
            'n_clusters': len(unique_labels),
            'X_pca': X_pca,
            'metrics': metrics,
            'method': method
        }
        
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")
        return None

def mine_association_rules(df):
    """Mine association rules from incident data"""
    try:
        if not MLXTEND_AVAILABLE:
            return None
        
        # Prepare transaction data
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            
            # Add categorical attributes
            for col in ['incident_type', 'severity', 'location', 'time_period']:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{col}={str(row[col])}")
            
            # Add risk level
            if 'risk_score' in row:
                risk_level = 'high_risk' if row['risk_score'] > 5 else 'low_risk'
                transaction.append(f"risk={risk_level}")
            
            # Add weekend flag
            if 'is_weekend' in row:
                transaction.append(f"weekend={row['is_weekend']}")
            
            if len(transaction) >= 2:
                transactions.append(transaction)
        
        if len(transactions) < 10:
            return None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df_transactions, min_support=0.1, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return None
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        
        if len(rules) == 0:
            return None
        
        # Add strength score
        rules['strength_score'] = (rules['lift'] * rules['confidence'] * rules['support']) ** (1/3)
        rules = rules.sort_values('strength_score', ascending=False)
        
        return rules
        
    except Exception as e:
        st.error(f"Association rules error: {str(e)}")
        return None

def generate_sample_data(n_records=200):
    """Generate sample NDIS incident data"""
    np.random.seed(42)
    
    incident_types = ['Fall', 'Medication Error', 'Property Damage', 'Injury', 'Behavioral', 
                     'Staff Incident', 'Transportation', 'Medical Emergency', 'Neglect', 'Other']
    severities = ['Low', 'Medium', 'High', 'Critical']
    locations = ['Residential Home A', 'Day Program B', 'Community Center C', 'Workplace D', 
                'Residential Home E', 'Day Program F']
    
    # Generate dates over the last 6 months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    
    data = []
    for i in range(n_records):
        # Generate random date
        days_back = np.random.randint(0, 180)
        incident_date = end_date - timedelta(days=days_back)
        
        # Generate time
        hour = np.random.randint(6, 22) if np.random.random() < 0.8 else np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        incident_time = f"{hour:02d}:{minute:02d}"
        
        # Generate correlated severity and type
        incident_type = np.random.choice(incident_types)
        if incident_type in ['Medical Emergency', 'Injury']:
            severity = np.random.choice(severities, p=[0.1, 0.3, 0.4, 0.2])
        else:
            severity = np.random.choice(severities, p=[0.4, 0.3, 0.2, 0.1])
        
        # Generate notification delay
        notification_delay = np.random.uniform(0, 1) if np.random.random() < 0.8 else np.random.uniform(1, 3)
        notification_date = incident_date + timedelta(days=notification_delay)
        
        record = {
            'incident_id': f'INC{i+1:06d}',
            'participant_name': f'Participant_{np.random.randint(1, n_records//3):03d}',
            'incident_date': incident_date.strftime('%d/%m/%Y'),
            'incident_time': incident_time,
            'notification_date': notification_date.strftime('%d/%m/%Y'),
            'incident_type': incident_type,
            'severity': severity,
            'location': np.random.choice(locations),
            'age': np.random.randint(18, 85),
            'reportable': np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
        }
        data.append(record)
    
    return pd.DataFrame(data)

def main():
    """Main Streamlit application"""
    st.title("🏥 NDIS Analytics Dashboard")
    st.markdown("Advanced analytics for NDIS incident management with ML-powered insights")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Source")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV File", "Use Sample Data"]
        )
        
        df = None
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload NDIS incidents CSV file",
                type=['csv'],
                help="CSV should contain: incident_date, incident_type, severity, location"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded: {len(df)} records")
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        else:  # Use Sample Data
            sample_size = st.slider("Sample data size:", 50, 500, 200)
            if st.button("Generate Sample Data"):
                df = generate_sample_data(sample_size)
                st.success(f"✅ Sample data generated: {len(df)} records")
        
        # Library status
        st.header("🔧 System Status")
        st.write(f"🤖 Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
        st.write(f"🔗 MLxtend: {'✅' if MLXTEND_AVAILABLE else '❌'}")
    
    # Process data if available
    if df is not None:
        df_processed = process_data(df)
        
        if df_processed is None:
            st.error("❌ Failed to process data. Please check your data format.")
            return
        
        # Display basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df_processed)}</h3>
                <p>Total Incidents</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            critical_count = len(df_processed[df_processed['severity'] == 'Critical'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{critical_count}</h3>
                <p>Critical Incidents</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_participants = df_processed['participant_name'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{unique_participants}</h3>
                <p>Participants</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_risk = df_processed['risk_score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_risk:.1f}</h3>
                <p>Avg Risk Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "🔮 Predictions", "⚠️ Anomalies", 
            "🎯 Clustering", "🔗 Rules"
        ])
        
        with tab1:
            st.header("📊 Incident Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily incident trend
                daily_incidents = df_processed.groupby('incident_date').size().reset_index()
                daily_incidents.columns = ['Date', 'Count']
                
                fig_trend = px.line(daily_incidents, x='Date', y='Count',
                                  title='Daily Incident Trend')
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Severity distribution
                severity_dist = df_processed['severity'].value_counts()
                fig_severity = px.pie(values=severity_dist.values, names=severity_dist.index,
                                    title='Incident Severity Distribution')
                st.plotly_chart(fig_severity, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Incident types
                type_dist = df_processed['incident_type'].value_counts().head(10)
                fig_types = px.bar(x=type_dist.index, y=type_dist.values,
                                 title='Top Incident Types')
                fig_types.update_xaxes(tickangle=45)
                st.plotly_chart(fig_types, use_container_width=True)
            
            with col4:
                # Risk score distribution
                fig_risk = px.histogram(df_processed, x='risk_score', nbins=20,
                                      title='Risk Score Distribution')
                st.plotly_chart(fig_risk, use_container_width=True)
        
        with tab2:
            st.header("🔮 Predictive Analytics")
            
            if SKLEARN_AVAILABLE:
                with st.spinner('Generating predictions...'):
                    predictions = predict_future_incidents(df_processed)
                
                if predictions:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        ma_pred = predictions.get('moving_average', 0)
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Moving Average</h3>
                            <h2>{ma_pred:.1f}</h2>
                            <p>incidents/day</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        trend_pred = predictions.get('trend_based', 0)
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Trend Forecast</h3>
                            <h2>{trend_pred:.1f}</h2>
                            <p>incidents/day</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        rf_pred = predictions.get('random_forest')
                        if rf_pred:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>ML Forecast</h3>
                                <h2>{rf_pred:.1f}</h2>
                                <p>incidents/day</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="prediction-card">
                                <h3>ML Forecast</h3>
                                <h2>N/A</h2>
                                <p>Insufficient data</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Prediction comparison chart
                    prediction_methods = []
                    prediction_values = []
                    
                    for method, value in predictions.items():
                        if method != 'daily_counts' and value is not None:
                            prediction_methods.append(method.replace('_', ' ').title())
                            prediction_values.append(max(0, value))
                    
                    if prediction_methods:
                        fig_pred = go.Figure(data=[
                            go.Bar(x=prediction_methods, y=prediction_values, 
                                  marker_color='rgba(55, 128, 191, 0.7)')
                        ])
                        
                        fig_pred.update_layout(
                            title='Prediction Methods Comparison',
                            xaxis_title='Method',
                            yaxis_title='Predicted Daily Incidents'
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    st.warning("⚠️ Unable to generate predictions. Insufficient data.")
            else:
                st.warning("⚠️ Predictive analytics requires scikit-learn installation.")
        
        with tab3:
            st.header("⚠️ Anomaly Detection")
            
            if SKLEARN_AVAILABLE:
                with st.spinner('Detecting anomalies...'):
                    anomaly_results = detect_anomalies(df_processed)
                
                if anomaly_results:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="anomaly-card">
                            <h3>{anomaly_results['n_anomalies']}</h3>
                            <p>Anomalies Found</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="anomaly-card">
                            <h3>{anomaly_results['anomaly_rate']:.1f}%</h3>
                            <p>Anomaly Rate</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="anomaly-card">
                            <h3>Isolation Forest</h3>
                            <p>Detection Method</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Anomaly score distribution
                    fig_dist = px.histogram(x=anomaly_results['anomaly_scores'],
                                          title='Anomaly Score Distribution',
                                          nbins=20)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Show anomalous incidents
                    if anomaly_results['n_anomalies'] > 0:
                        st.subheader("🚨 Anomalous Incidents")
                        anomaly_incidents = df_processed.iloc[anomaly_results['anomaly_indices']]
                        display_columns = ['incident_id', 'incident_date', 'incident_type', 
                                         'severity', 'location', 'risk_score']
                        display_columns = [col for col in display_columns if col in anomaly_incidents.columns]
                        st.dataframe(anomaly_incidents[display_columns], use_container_width=True)
                else:
                    st.warning("⚠️  Unable to perform anomaly detection. Insufficient data.")
            else:
                st.warning("⚠️ Anomaly detection requires scikit-learn installation.")
        
        with tab4:
            st.header("🎯 Clustering Analysis")
            
            if SKLEARN_AVAILABLE:
                # Clustering controls
                col1, col2 = st.columns(2)
                
                with col1:
                    clustering_method = st.selectbox(
                        "Select clustering method:",
                        ['kmeans', 'dbscan', 'hierarchical']
                    )
                
                with col2:
                    if clustering_method in ['kmeans', 'hierarchical']:
                        n_clusters = st.slider("Number of clusters:", 2, 8, 4)
                    else:
                        n_clusters = None
                        st.info("DBSCAN automatically determines clusters")
                
                if st.button("Run Clustering Analysis"):
                    with st.spinner('Performing clustering analysis...'):
                        clustering_results = perform_clustering(df_processed, 
                                                               method=clustering_method,
                                                               n_clusters=n_clusters)
                    
                    if clustering_results:
                        # Clustering metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{clustering_results['n_clusters']}</h3>
                                <p>Clusters Found</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if 'silhouette' in clustering_results['metrics']:
                                silhouette = clustering_results['metrics']['silhouette']
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{silhouette:.3f}</h3>
                                    <p>Silhouette Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>N/A</h3>
                                    <p>Silhouette Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            method_name = clustering_results['method'].title()
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{method_name}</h3>
                                <p>Method Used</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Clustering visualizations
                        col4, col5 = st.columns(2)
                        
                        with col4:
                            # PCA scatter plot
                            fig_pca = go.Figure()
                            
                            unique_clusters = np.unique(clustering_results['labels'])
                            colors = px.colors.qualitative.Set1[:len(unique_clusters)]
                            
                            for i, cluster_id in enumerate(unique_clusters):
                                cluster_mask = clustering_results['labels'] == cluster_id
                                cluster_points = clustering_results['X_pca'][cluster_mask]
                                
                                fig_pca.add_trace(go.Scatter(
                                    x=cluster_points[:, 0],
                                    y=cluster_points[:, 1],
                                    mode='markers',
                                    marker=dict(color=colors[i % len(colors)], size=8, opacity=0.7),
                                    name=f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
                                ))
                            
                            fig_pca.update_layout(
                                title=f'Clustering Results - {clustering_results["method"].title()}',
                                xaxis_title='First Principal Component',
                                yaxis_title='Second Principal Component'
                            )
                            
                            st.plotly_chart(fig_pca, use_container_width=True)
                        
                        with col5:
                            # Cluster size distribution
                            cluster_sizes = []
                            cluster_labels = []
                            
                            for cluster_id in unique_clusters:
                                size = np.sum(clustering_results['labels'] == cluster_id)
                                cluster_sizes.append(size)
                                cluster_labels.append(f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise')
                            
                            fig_sizes = go.Figure(data=[
                                go.Pie(labels=cluster_labels, values=cluster_sizes, hole=0.3)
                            ])
                            
                            fig_sizes.update_layout(title='Cluster Size Distribution')
                            st.plotly_chart(fig_sizes, use_container_width=True)
                    else:
                        st.warning("⚠️ Unable to perform clustering. Insufficient data.")
            else:
                st.warning("⚠️ Clustering analysis requires scikit-learn installation.")
        
        with tab5:
            st.header("🔗 Association Rules")
            
            if MLXTEND_AVAILABLE:
                with st.spinner('Mining association rules...'):
                    association_rules_df = mine_association_rules(df_processed)
                
                if association_rules_df is not None and len(association_rules_df) > 0:
                    st.success(f"✅ Found {len(association_rules_df)} association rules")
                    
                    # Rule strength distribution
                    fig_strength = px.histogram(association_rules_df, x='strength_score',
                                              title='Distribution of Rule Strength Scores',
                                              nbins=20)
                    st.plotly_chart(fig_strength, use_container_width=True)
                    
                    # Top rules
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏆 Top Rules by Confidence")
                        top_confidence = association_rules_df.nlargest(5, 'confidence')
                        
                        for idx, rule in top_confidence.iterrows():
                            antecedents = ', '.join(list(rule['antecedents']))
                            consequents = ', '.join(list(rule['consequents']))
                            
                            st.markdown(f"""
                            <div class="insight-box">
                                <strong>If:</strong> {antecedents}<br>
                                <strong>Then:</strong> {consequents}<br>
                                <small>Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("📈 Top Rules by Lift")
                        top_lift = association_rules_df.nlargest(5, 'lift')
                        
                        for idx, rule in top_lift.iterrows():
                            antecedents = ', '.join(list(rule['antecedents']))
                            consequents = ', '.join(list(rule['consequents']))
                            
                            st.markdown(f"""
                            <div class="insight-box">
                                <strong>If:</strong> {antecedents}<br>
                                <strong>Then:</strong> {consequents}<br>
                                <small>Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Interactive scatter plot
                    fig_rules = px.scatter(association_rules_df, 
                                         x='support', y='confidence', 
                                         size='lift', color='strength_score',
                                         title='Association Rules Visualization',
                                         hover_data=['lift'])
                    st.plotly_chart(fig_rules, use_container_width=True)
                    
                    # Full rules table
                    with st.expander("📋 All Association Rules"):
                        display_rules = association_rules_df.copy()
                        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        display_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'strength_score']
                        st.dataframe(display_rules[display_columns], use_container_width=True)
                else:
                    st.warning("⚠️ No association rules found. Try with more data.")
            else:
                st.warning("⚠️ Association rules require MLxtend installation.")
    
    else:
        st.info("👆 Please upload a CSV file or generate sample data to begin analysis.")
        
        # Show expected data format
        st.subheader("📋 Expected Data Format")
        st.write("Your CSV file should contain the following columns:")
        
        expected_format = pd.DataFrame({
            'Column': ['incident_date', 'incident_type', 'severity', 'location', 'participant_name', 'age'],
            'Description': [
                'Date of incident (DD/MM/YYYY or YYYY-MM-DD)',
                'Type of incident (Fall, Injury, etc.)',
                'Severity level (Low, Medium, High, Critical)',
                'Location where incident occurred',
                'Name or ID of participant (optional)',
                'Age of participant (optional)'
            ],
            'Example': [
                '15/03/2024',
                'Fall',
                'Medium',
                'Residential Home A',
                'Participant_001',
                '45'
            ]
        })
        
        st.table(expected_format)

if __name__ == "__main__":
    main()
