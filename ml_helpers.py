import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

@st.cache_data
def prepare_ml_features(df: pd.DataFrame):
    if df.empty:
        return None, None, None
    features_df = df.copy()
    label_encoders = {}
    categorical_cols = ['location', 'incident_type', 'contributing_factors', 'reported_by']
    for col in categorical_cols:
        if col in features_df.columns:
            le = LabelEncoder()
            features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].fillna('Unknown'))
            label_encoders[col] = le
    if 'incident_date' in features_df.columns:
        features_df['day_of_week'] = features_df['incident_date'].dt.dayofweek
        features_df['month'] = features_df['incident_date'].dt.month
        if 'incident_time' in features_df.columns:
            features_df['hour'] = pd.to_datetime(features_df['incident_time'], format='%H:%M', errors='coerce').dt.hour
    num_cols = [c for c in [
        'day_of_week','month','hour',
        'location_encoded','incident_type_encoded','contributing_factors_encoded','reported_by_encoded'
    ] if c in features_df.columns]
    if not num_cols:
        return None, None, None
    X = features_df[num_cols].fillna(0)
    return X, num_cols, label_encoders

@st.cache_data
def train_severity_prediction_model(df: pd.DataFrame):
    if df.empty or len(df) < 20:
        return None, None, None
    X, feature_names, _ = prepare_ml_features(df)
    if X is None:
        return None, None, None
    sev_map = {'Low':0,'Moderate':1,'High':2}
    y = df['severity'].map(sev_map)
    mask = ~y.isna()
    X, y = X[mask], y[mask]
    if len(X) < 10:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, feature_names

@st.cache_data
def perform_anomaly_detection(df: pd.DataFrame):
    if df.empty or len(df) < 10:
        return None, None
    X, feature_names, _ = prepare_ml_features(df)
    if X is None:
        return None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_labels = iso.fit_predict(Xs)
    svm = OneClassSVM(nu=0.1)
    svm_labels = svm.fit_predict(Xs)
    out = df.copy()
    out['isolation_forest_anomaly'] = iso_labels == -1
    out['svm_anomaly'] = svm_labels == -1
    out['anomaly_score'] = iso.decision_function(Xs)
    return out, feature_names

@st.cache_data
def perform_clustering_analysis(df: pd.DataFrame, n_clusters=5, algorithm='kmeans'):
    if df.empty or len(df) < 10:
        return None, None, None, None
    X, feature_names, _ = prepare_ml_features(df)
    if X is None:
        return None, None, None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    if algorithm == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = clusterer.fit_predict(X_scaled)
    sil_score = None
    if len(set(cluster_labels)) > 1:
        try:
            sil_score = silhouette_score(X_scaled, cluster_labels)
        except:
            sil_score = None
    result_df = df.copy()
    result_df['cluster'] = cluster_labels
    result_df['pca_x'] = X_pca[:, 0]
    result_df['pca_y'] = X_pca[:, 1]
    return result_df, feature_names, sil_score, pca

@st.cache_data
def analyze_cluster_characteristics(clustered_df: pd.DataFrame):
    if clustered_df is None or 'cluster' not in clustered_df.columns:
        return None
    cluster_analysis = {}
    for cluster_id in clustered_df['cluster'].unique():
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
        analysis = {
            'size': len(cluster_data),
            'most_common_type': cluster_data['incident_type'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
            'most_common_location': cluster_data['location'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
            'most_common_severity': cluster_data['severity'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
            'avg_medical_attention': cluster_data['medical_attention_required'].mean() if 'medical_attention_required' in cluster_data else 0,
            'avg_reportable': cluster_data['reportable'].mean() if 'reportable' in cluster_data else 0
        }
        cluster_analysis[cluster_id] = analysis
    return cluster_analysis