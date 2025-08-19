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
    """Prepare enhanced features for ML analysis - FIXED VERSION"""
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
                
                # Convert to string and handle NaN values FIRST
                ml_df[col] = ml_df[col].astype(str)
                ml_df[col] = ml_df[col].fillna('Unknown').replace('nan', 'Unknown')
                
                # For categorical columns created by pd.cut, convert to string
                if hasattr(ml_df[col], 'cat'):
                    # Add 'Unknown' to categories if it's not already there
                    if 'Unknown' not in ml_df[col].cat.categories:
                        ml_df[col] = ml_df[col].cat.add_categories(['Unknown'])
                    # Convert categorical to string
                    ml_df[col] = ml_df[col].astype(str)
                
                # Now encode
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
    """Enhanced prediction with multiple models and confidence intervals"""
    try:
        if not SKLEARN_AVAILABLE:
            return None
        
        # Prepare time series data
        daily_counts = df.groupby(df['incident_date'].dt.date).size()
        daily_counts = daily_counts.reindex(pd.date_range(daily_counts.index.min(), 
                                                          daily_counts.index.max()), 
                                          fill_value=0)
        
        # Simple moving average prediction
        ma_window = min(7, len(daily_counts) // 4)
        ma_prediction = daily_counts.rolling(window=ma_window).mean().iloc[-1]
        
        # Weighted moving average
        weights = np.exp(np.linspace(-1, 0, ma_window))
        weights /= weights.sum()
        wma_prediction = np.average(daily_counts.iloc[-ma_window:], weights=weights)
        
        # Trend-based prediction
        days_numeric = np.arange(len(daily_counts))
        z = np.polyfit(days_numeric, daily_counts.values, 1)
        p = np.poly1d(z)
        trend_prediction = p(len(daily_counts) + prediction_days)
        
        # Exponential smoothing if available
        exp_smoothing_prediction = None
        if STATSMODELS_AVAILABLE and len(daily_counts) > 10:
            try:
                model = ExponentialSmoothing(daily_counts, seasonal_periods=7, trend='add', seasonal='add')
                model_fit = model.fit()
                exp_smoothing_prediction = model_fit.forecast(steps=prediction_days).mean()
            except:
                pass
        
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
        
        # Random Forest for multi-variate prediction
        ml_df, label_encoders = prepare_ml_features(df)
        
        # Create lagged features for time series
        ts_features = []
        for i in range(1, min(8, len(daily_counts))):
            ts_features.append(daily_counts.shift(i).fillna(0))
        
        if len(ts_features) > 0:
            ts_df = pd.DataFrame(ts_features).T
            ts_df.columns = [f'lag_{i}' for i in range(1, len(ts_features) + 1)]
            ts_df['target'] = daily_counts.values
            ts_df = ts_df.dropna()
            
            if len(ts_df) > 10:
                X_ts = ts_df.drop('target', axis=1)
                y_ts = ts_df['target']
                
                X_train, X_test, y_train, y_test = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)
                
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_train, y_train)
                
                # Generate future predictions
                last_values = daily_counts.iloc[-len(ts_features):].values
                rf_predictions = []
                
                for _ in range(prediction_days):
                    X_pred = last_values.reshape(1, -1)
                    pred = rf_regressor.predict(X_pred)[0]
                    rf_predictions.append(pred)
                    last_values = np.append(last_values[1:], pred)
                
                rf_prediction = np.mean(rf_predictions)
            else:
                rf_prediction = None
        else:
            rf_prediction = None
        
        # Severity predictor
        severity_predictor = None
        if 'severity_encoded' in ml_df.columns:
            feature_cols = ['hour', 'day_of_week_num', 'month_num', 'age', 'risk_score']
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
            'weighted_moving_average': wma_prediction,
            'trend_based': trend_prediction,
            'exponential_smoothing': exp_smoothing_prediction,
            'arima': arima_prediction,
            'random_forest': rf_prediction,
            'severity_predictor': severity_predictor,
            'daily_counts': daily_counts
        }
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def enhanced_anomaly_detection(df):
    """Enhanced multi-method anomaly detection with ensemble approach"""
    try:
        if not SKLEARN_AVAILABLE:
            return None
        
        ml_df, _ = prepare_ml_features(df)
        
        # Select features for anomaly detection
        feature_cols = [col for col in ml_df.columns if col.endswith('_encoded')]
        numeric_cols = ['age', 'hour', 'severity_score', 'risk_score', 'notification_delay', 
                       'incident_count_by_participant', 'location_incident_rate']
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
        anomaly_results['isolation_forest_scores'] = iso_forest.score_samples(X_scaled)
        
        # One-Class SVM
        try:
            svm_detector = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
            anomaly_results['one_class_svm'] = svm_detector.fit_predict(X_scaled)
        except:
            pass
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        anomaly_results['lof'] = lof.fit_predict(X_scaled)
        anomaly_results['lof_scores'] = lof.negative_outlier_factor_
        
        # Elliptic Envelope (Robust Covariance)
        try:
            ee = EllipticEnvelope(contamination=0.1, random_state=42)
            anomaly_results['elliptic_envelope'] = ee.fit_predict(X_scaled)
        except:
            pass
        
        # DBSCAN for density-based anomaly detection
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X_scaled)
            anomaly_results['dbscan'] = np.where(clusters == -1, -1, 1)
        except:
            pass
        
        # Ensemble anomaly score
        ensemble_scores = np.zeros(len(X))
        method_count = 0
        
        for method, scores in anomaly_results.items():
            if not method.endswith('_scores'):
                ensemble_scores += (scores == -1).astype(int)
                method_count += 1
        
        # Normalize ensemble scores
        if method_count > 0:
            ensemble_scores = ensemble_scores / method_count
        
        # Calculate confidence scores
        confidence_scores = np.zeros(len(X))
        if 'isolation_forest_scores' in anomaly_results:
            confidence_scores += -anomaly_results['isolation_forest_scores']
        if 'lof_scores' in anomaly_results:
            confidence_scores += -anomaly_results['lof_scores']
        
        # Normalize confidence scores
        if confidence_scores.max() > confidence_scores.min():
            confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())
        
        # Identify anomalies (threshold at 0.5 - majority vote)
        final_anomalies = ensemble_scores >= 0.5
        
        # Calculate anomaly details
        anomaly_indices = np.where(final_anomalies)[0]
        anomaly_data = ml_df.iloc[anomaly_indices]
        
        # Analyze anomaly patterns
        anomaly_patterns = {}
        if len(anomaly_data) > 0:
            for col in ['incident_type', 'severity', 'location', 'time_period']:
                if col in anomaly_data.columns:
                    anomaly_patterns[col] = anomaly_data[col].value_counts().to_dict()
        
        # Statistical analysis of anomalies
        anomaly_stats = {}
        if len(anomaly_data) > 0:
            for col in ['risk_score', 'age', 'hour', 'notification_delay']:
                if col in anomaly_data.columns:
                    anomaly_stats[col] = {
                        'mean': anomaly_data[col].mean(),
                        'std': anomaly_data[col].std(),
                        'min': anomaly_data[col].min(),
                        'max': anomaly_data[col].max()
                    }
        
        return {
            'anomaly_scores': ensemble_scores,
            'confidence_scores': confidence_scores,
            'anomaly_flags': final_anomalies,
            'anomaly_indices': anomaly_indices,
            'anomaly_patterns': anomaly_patterns,
            'anomaly_stats': anomaly_stats,
            'methods_used': [k for k in anomaly_results.keys() if not k.endswith('_scores')],
            'feature_cols': feature_cols,
            'X_scaled': X_scaled,
            'individual_results': anomaly_results
        }
        
    except Exception as e:
        st.error(f"Enhanced anomaly detection error: {str(e)}")
        return None

def advanced_association_rules(df):
    """Advanced association rule mining with multiple algorithms and metrics"""
    try:
        if not MLXTEND_AVAILABLE:
            return None
        
        # Prepare transaction data with more attributes
        transaction_cols = ['incident_type', 'severity', 'location', 'time_period']
        if 'reportable' in df.columns:
            transaction_cols.append('reportable')
        if 'age_group' in df.columns:
            transaction_cols.append('age_group')
        
        # Create transactions with additional derived attributes
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
            
            # Add time-based attributes
            if 'is_weekend' in row:
                transaction.append(f"weekend={row['is_weekend']}")
            if 'quarter' in row:
                transaction.append(f"quarter=Q{row['quarter']}")
            
            if len(transaction) >= 2:
                transactions.append(transaction)
        
        if len(transactions) < 10:
            return None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Try multiple support levels
        support_levels = [0.01, 0.02, 0.05, 0.1]
        all_rules = []
        
        for min_support in support_levels:
            try:
                # Use FP-Growth for better performance
                frequent_itemsets = fpgrowth(df_transactions, min_support=min_support, use_colnames=True)
                
                if len(frequent_itemsets) > 0:
                    # Generate rules with multiple metrics
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
                    
                    if len(rules) > 0:
                        # Calculate additional metrics
                        rules['conviction'] = np.where(rules['confidence'] == 1, np.inf,
                                                      (1 - rules['consequent support']) / 
                                                      (1 - rules['confidence']))
                        rules['leverage'] = rules['support'] - (rules['antecedent support'] * 
                                                                rules['consequent support'])
                        rules['zhang'] = np.where(rules['leverage'] == 0, 0,
                                                 rules['leverage'] / 
                                                 np.maximum(rules['antecedent support'] * (1 - rules['consequent support']),
                                                           rules['consequent support'] * (1 - rules['antecedent support'])))
                        
                        # Add rule strength score
                        rules['strength_score'] = (rules['lift'] * rules['confidence'] * rules['support']) ** (1/3)
                        
                        all_rules.append(rules)
            except:
                continue
        
        if all_rules:
            # Combine and deduplicate rules
            combined_rules = pd.concat(all_rules, ignore_index=True)
            combined_rules = combined_rules.drop_duplicates(subset=['antecedents', 'consequents'])
            
            # Sort by composite strength score
            combined_rules = combined_rules.sort_values('strength_score', ascending=False)
            
            return combined_rules
        
        return None
        
    except Exception as e:
        st.error(f"Advanced association rules error: {str(e)}")
        return None

def advanced_clustering(df, method='kmeans', n_clusters=4):
    """Advanced clustering with multiple algorithms"""
    try:
        if not SKLEARN_AVAILABLE:
            return None
        
        ml_df, _ = prepare_ml_features(df)
        
        # Select features for clustering
        feature_cols = [col for col in ml_df.columns if col.endswith('_encoded')]
        numeric_cols = ['age', 'hour', 'severity_score', 'risk_score', 'notification_delay']
        feature_cols.extend([col for col in numeric_cols if col in ml_df.columns])
        
        if len(feature_cols) < 2:
            return None
        
        X = ml_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply selected clustering method
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Calculate metrics only if we have valid clusters
        metrics = {}
        unique_labels = np.unique(cluster_labels)
        
        if len(unique_labels) > 1:
            try:
                metrics['silhouette'] = silhouette_score(X_scaled, cluster_labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(X_scaled, cluster_labels)
            except:
                pass
        
        # PCA for visualization
        pca = PCA(n_components=min(3, X_scaled.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Cluster profiles
        ml_df['cluster'] = cluster_labels
        profiles = {}
        
        for cluster_id in unique_labels:
            cluster_data = ml_df[ml_df['cluster'] == cluster_id]
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(ml_df) * 100
            }
            
            # Numeric features
            for col in ['risk_score', 'age', 'severity_score']:
                if col in cluster_data.columns:
                    profile[f'{col}_mean'] = cluster_data[col].mean()
                    profile[f'{col}_std'] = cluster_data[col].std()
            
            # Categorical features
            for col in ['incident_type', 'severity', 'location']:
                if col in cluster_data.columns:
                    top_value = cluster_data[col].mode()[0] if len(cluster_data[col].mode()) > 0 else 'N/A'
                    profile[f'top_{col}'] = top_value
            
            profiles[f'cluster_{cluster_id}'] = profile
        
        return {
            'labels': cluster_labels,
            'n_clusters': len(unique_labels),
            'X_pca': X_pca,
            'pca': pca,
            'metrics': metrics,
            'profiles': profiles,
            'feature_cols': feature_cols,
            'method': method
        }
        
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")
        return None

def generate_sample_data(n_records=100):
    """Generate sample NDIS incident data for demonstration"""
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
        
        # Generate time with higher probability during day hours
        if np.random.random() < 0.8:
            hour = np.random.randint(7, 18)  # Day hours
        else:
            hour = np.random.randint(0, 24)  # Any hour
        minute = np.random.randint(0, 60)
        incident_time = f"{hour:02d}:{minute:02d}"
        
        # Generate correlated severity and type
        incident_type = np.random.choice(incident_types)
        if incident_type in ['Medical Emergency', 'Injury']:
            severity = np.random.choice(severities, p=[0.1, 0.3, 0.4, 0.2])
        else:
            severity = np.random.choice(severities, p=[0.4, 0.3, 0.2, 0.1])
        
        # Generate notification delay (most same day, some delays)
        if np.random.random() < 0.8:
            notification_delay = np.random.uniform(0, 1)  # Same day
        else:
            notification_delay = np.random.uniform(1, 3)  # 1-3 days delay
        
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

def create_prediction_visualizations(predictions, df):
    """Create visualizations for prediction results"""
    if not predictions:
        return None
    
    # Prepare prediction comparison chart
    prediction_methods = []
    prediction_values = []
    
    for method, value in predictions.items():
        if method != 'daily_counts' and method != 'severity_predictor' and value is not None:
            prediction_methods.append(method.replace('_', ' ').title())
            prediction_values.append(max(0, value))
    
    if prediction_methods:
        fig_pred = go.Figure(data=[
            go.Bar(x=prediction_methods, y=prediction_values, 
                  marker_color='rgba(55, 128, 191, 0.7)',
                  text=[f'{v:.1f}' for v in prediction_values],
                  textposition='auto')
        ])
        
        fig_pred.update_layout(
            title='Incident Prediction Comparison (Next 30 Days Average)',
            xaxis_title='Prediction Method',
            yaxis_title='Predicted Daily Incidents',
            height=400
        )
        
        return fig_pred
    
    return None

def create_anomaly_visualizations(anomaly_results, df):
    """Create visualizations for anomaly detection results"""
    if not anomaly_results:
        return None, None
    
    # Anomaly score distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=anomaly_results['anomaly_scores'],
        nbinsx=20,
        marker_color='rgba(255, 99, 132, 0.7)',
        name='Anomaly Scores'
    ))
    
    fig_dist.update_layout(
        title='Distribution of Anomaly Scores',
        xaxis_title='Anomaly Score',
        yaxis_title='Frequency',
        height=400
    )
    
    # Create anomaly scatter plot if PCA data is available
    fig_scatter = None
    if len(anomaly_results['X_scaled']) > 0 and len(anomaly_results['feature_cols']) >= 2:
        # Use first two features for scatter plot
        feature1 = anomaly_results['feature_cols'][0]
        feature2 = anomaly_results['feature_cols'][1] if len(anomaly_results['feature_cols']) > 1 else anomaly_results['feature_cols'][0]
        
        colors = ['red' if flag else 'blue' for flag in anomaly_results['anomaly_flags']]
        
        fig_scatter = go.Figure()
        
        # Normal points
        normal_indices = ~anomaly_results['anomaly_flags']
        if np.any(normal_indices):
            fig_scatter.add_trace(go.Scatter(
                x=anomaly_results['X_scaled'][normal_indices, 0],
                y=anomaly_results['X_scaled'][normal_indices, 1],
                mode='markers',
                marker=dict(color='blue', size=6, opacity=0.6),
                name='Normal',
                hovertemplate='Normal Incident<extra></extra>'
            ))
        
        # Anomalous points
        anomaly_indices = anomaly_results['anomaly_flags']
        if np.any(anomaly_indices):
            fig_scatter.add_trace(go.Scatter(
                x=anomaly_results['X_scaled'][anomaly_indices, 0],
                y=anomaly_results['X_scaled'][anomaly_indices, 1],
                mode='markers',
                marker=dict(color='red', size=8, opacity=0.8),
                name='Anomaly',
                hovertemplate='Anomalous Incident<extra></extra>'
            ))
        
        fig_scatter.update_layout(
            title='Anomaly Detection Results (Feature Space)',
            xaxis_title=f'Scaled {feature1}',
            yaxis_title=f'Scaled {feature2}',
            height=500
        )
    
    return fig_dist, fig_scatter

def create_clustering_visualizations(clustering_results, df):
    """Create visualizations for clustering results"""
    if not clustering_results:
        return None, None
    
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
            name=f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise',
            hovertemplate=f'Cluster {cluster_id}<extra></extra>'
        ))
    
    fig_pca.update_layout(
        title=f'Incident Clustering Results - {clustering_results["method"].title()}',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        height=500
    )
    
    # Cluster size chart
    cluster_sizes = []
    cluster_labels = []
    
    for cluster_id in unique_clusters:
        size = np.sum(clustering_results['labels'] == cluster_id)
        cluster_sizes.append(size)
        cluster_labels.append(f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise')
    
    fig_sizes = go.Figure(data=[
        go.Pie(labels=cluster_labels, values=cluster_sizes, hole=0.3)
    ])
    
    fig_sizes.update_layout(
        title='Cluster Size Distribution',
        height=400
    )
    
    return fig_pca, fig_sizes

def main():
    """Main Streamlit application"""
    st.title("🏥 NDIS Analytics Dashboard - Enhanced")
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
                help="CSV should contain columns: incident_date, incident_type, severity, location"
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
        
        # ML Configuration
        st.header("🤖 ML Configuration")
        
        if not SKLEARN_AVAILABLE:
            st.warning("⚠️ Scikit-learn not available. ML features disabled.")
        
        if not MLXTEND_AVAILABLE:
            st.warning("⚠️ MLxtend not available. Association rules disabled.")
        
        if not STATSMODELS_AVAILABLE:
            st.warning("⚠️ Statsmodels not available. Advanced forecasting disabled.")
    
    # Process data if available
    if df is not None:
        df_processed = process_data(df)
        
        if df_processed is None:
            st.error("❌ Failed to process data. Please check your data format.")
            return
        
        # Display basic info
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
            "📈 Overview", "🔮 Predictions", "⚠️ Anomaly Detection", 
            "🎯 Clustering Analysis", "🔗 Association Rules"
        ])
        
        with tab1:
            st.header("📊 Incident Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Incident trend over time
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
                    # Display prediction cards
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
                            <h3>Trend Analysis</h3>
                            <h2>{trend_pred:.1f}</h2>
                            <p>incidents/day</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        rf_pred = predictions.get('random_forest', 0)
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
                    fig_pred = create_prediction_visualizations(predictions, df_processed)
                    if fig_pred:
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Severity prediction model info
                    if predictions.get('severity_predictor'):
                        sev_pred = predictions['severity_predictor']
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>🎯 Severity Prediction Model</h4>
                            <p>Model Accuracy: <strong>{sev_pred['accuracy']:.2%}</strong></p>
                            <p>Key Features: {', '.join(sev_pred['features'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Unable to generate predictions. Insufficient data.")
            else:
                st.warning("⚠️ Predictive analytics requires scikit-learn installation.")
        
        with tab3:
            st.header("⚠️ Anomaly Detection")
            
            if SKLEARN_AVAILABLE:
                with st.spinner('Detecting anomalies...'):
                    anomaly_results = enhanced_anomaly_detection(df_processed)
                
                if anomaly_results:
                    # Anomaly summary
                    n_anomalies = np.sum(anomaly_results['anomaly_flags'])
                    anomaly_rate = n_anomalies / len(df_processed) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="anomaly-card">
                            <h3>{n_anomalies}</h3>
                            <p>Anomalies Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="anomaly-card">
                            <h3>{anomaly_rate:.1f}%</h3>
                            <p>Anomaly Rate</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        n_methods = len(anomaly_results['methods_used'])
                        st.markdown(f"""
                        <div class="anomaly-card">
                            <h3>{n_methods}</h3>
                            <p>Detection Methods</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Anomaly visualizations
                    fig_dist, fig_scatter = create_anomaly_visualizations(anomaly_results, df_processed)
                    
                    col4, col5 = st.columns(2)
                    
                    with col4:
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col5:
                        if fig_scatter:
                            st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Anomaly patterns
                    if anomaly_results['anomaly_patterns']:
                        st.subheader("🔍 Anomaly Patterns")
                        
                        for pattern_type, patterns in anomaly_results['anomaly_patterns'].items():
                            if patterns:
                                st.write(f"**{pattern_type.title()}:**")
                                for value, count in list(patterns.items())[:5]:
                                    st.write(f"- {value}: {count} incidents")
                    
                    # Show anomalous incidents
                    if n_anomalies > 0:
                        st.subheader("🚨 Anomalous Incidents")
                        anomaly_incidents = df_processed.iloc[anomaly_results['anomaly_indices']]
                        display_columns = ['incident_id', 'incident_date', 'incident_type', 
                                         'severity', 'location', 'risk_score']
                        display_columns = [col for col in display_columns if col in anomaly_incidents.columns]
                        st.dataframe(anomaly_incidents[display_columns], use_container_width=True)
                
                else:
                    st.warning("⚠️ Unable to perform anomaly detection. Insufficient data.")
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
                        n_clusters = st.slider("Number of clusters:", 2, 10, 4)
                    else:
                        n_clusters = None
                        st.info("DBSCAN automatically determines clusters")
                
                if st.button("Run Clustering Analysis"):
                    with st.spinner('Performing clustering analysis...'):
                        clustering_results = advanced_clustering(df_processed, 
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
                        
                        with col3:
                            method_name = clustering_results['method'].title()
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{method_name}</h3>
                                <p>Method Used</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Clustering visualizations
                        fig_pca, fig_sizes = create_clustering_visualizations(clustering_results, df_processed)
                        
                        col4, col5 = st.columns(2)
                        
                        with col4:
                            if fig_pca:
                                st.plotly_chart(fig_pca, use_container_width=True)
                        
                        with col5:
                            if fig_sizes:
                                st.plotly_chart(fig_sizes, use_container_width=True)
                        
                        # Cluster profiles
                        if clustering_results['profiles']:
                            st.subheader("📊 Cluster Profiles")
                            
                            for cluster_name, profile in clustering_results['profiles'].items():
                                with st.expander(f"{cluster_name.replace('_', ' ').title()} ({profile['size']} incidents, {profile['percentage']:.1f}%)"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Numeric Characteristics:**")
                                        for key, value in profile.items():
                                            if '_mean' in key:
                                                feature = key.replace('_mean', '')
                                                std_key = f'{feature}_std'
                                                if std_key in profile:
                                                    st.write(f"- {feature}: {value:.2f} ± {profile[std_key]:.2f}")
                                    
                                    with col2:
                                        st.write("**Categorical Characteristics:**")
                                        for key, value in profile.items():
                                            if key.startswith('top_'):
                                                feature = key.replace('top_', '')
                                                st.write(f"- Most common {feature}: {value}")
                    else:
                        st.warning("⚠️ Unable to perform clustering. Insufficient data.")
            else:
                st.warning("⚠️ Clustering analysis requires scikit-learn installation.")
        
        with tab5:
            st.header("🔗 Association Rules")
            
            if MLXTEND_AVAILABLE:
                with st.spinner('Mining association rules...'):
                    association_rules_df = advanced_association_rules(df_processed)
                
                if association_rules_df is not None and len(association_rules_df) > 0:
                    st.success(f"✅ Found {len(association_rules_df)} association rules")
                    
                    # Rule strength distribution
                    fig_strength = px.histogram(association_rules_df, x='strength_score',
                                              title='Distribution of Rule Strength Scores',
                                              nbins=20)
                    st.plotly_chart(fig_strength, use_container_width=True)
                    
                    # Top rules by different metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏆 Top Rules by Confidence")
                        top_confidence = association_rules_df.nlargest(5, 'confidence')[
                            ['antecedents', 'consequents', 'confidence', 'lift', 'support']
                        ]
                        
                        for idx, rule in top_confidence.iterrows():
                            antecedents = ', '.join(list(rule['antecedents']))
                            consequents = ', '.join(list(rule['consequents']))
                            
                            st.markdown(f"""
                            <div class="insight-box">
                                <strong>If:</strong> {antecedents}<br>
                                <strong>Then:</strong> {consequents}<br>
                                <small>Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}, Support: {rule['support']:.3f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("📈 Top Rules by Lift")
                        top_lift = association_rules_df.nlargest(5, 'lift')[
                            ['antecedents', 'consequents', 'confidence', 'lift', 'support']
                        ]
                        
                        for idx, rule in top_lift.iterrows():
                            antecedents = ', '.join(list(rule['antecedents']))
                            consequents = ', '.join(list(rule['consequents']))
                            
                            st.markdown(f"""
                            <div class="insight-box">
                                <strong>If:</strong> {antecedents}<br>
                                <strong>Then:</strong> {consequents}<br>
                                <small>Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}, Support: {rule['support']:.3f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Interactive scatter plot of rules
                    fig_rules = px.scatter(association_rules_df, 
                                         x='support', y='confidence', 
                                         size='lift', color='strength_score',
                                         title='Association Rules Visualization',
                                         labels={'support': 'Support', 'confidence': 'Confidence'},
                                         hover_data=['lift'])
                    st.plotly_chart(fig_rules, use_container_width=True)
                    
                    # Full rules table
                    with st.expander("📋 All Association Rules"):
                        # Format the rules for better display
                        display_rules = association_rules_df.copy()
                        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        display_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'strength_score']
                        st.dataframe(display_rules[display_columns], use_container_width=True)
                
                else:
                    st.warning("⚠️ No association rules found. Try with more data or adjust parameters.")
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
