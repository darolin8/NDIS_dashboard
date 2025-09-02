import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, roc_curve, auc
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

def compare_models(df):
    X, feature_names, _ = prepare_ml_features(df)
    if X is None or 'severity' not in df.columns:
        return pd.DataFrame(), go.Figure()
    sev_map = {'Low':0, 'Moderate':1, 'High':2}
    y = df['severity'].map(sev_map)
    mask = ~y.isna()
    X, y = X[mask], y[mask]
    if len(X) < 10:
        return pd.DataFrame(), go.Figure()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
    }
    metrics = []
    roc_fig = go.Figure()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            # For multiclass, show ROC for each class
            for i in range(y_prob.shape[1]):
                fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} class {i} (AUC={roc_auc:.2f})"))
        metrics.append({'Model': name, 'Accuracy': acc})
    metrics_df = pd.DataFrame(metrics)
    roc_fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return metrics_df, roc_fig

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

def plot_anomaly_scatter(anomaly_df, x_col, y_col, anomaly_column="isolation_forest_anomaly", axis_labels=None):
    if (
        anomaly_df is None
        or x_col not in anomaly_df.columns
        or y_col not in anomaly_df.columns
        or anomaly_column not in anomaly_df.columns
    ):
        raise ValueError("Required columns are missing in the DataFrame.")
    display_x = axis_labels[x_col] if axis_labels and x_col in axis_labels else x_col
    display_y = axis_labels[y_col] if axis_labels and y_col in axis_labels else y_col
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = anomaly_df[anomaly_df[anomaly_column] == False]
    anomaly = anomaly_df[anomaly_df[anomaly_column] == True]
    ax.scatter(normal[x_col], normal[y_col], c='blue', label='Normal', alpha=0.5)
    ax.scatter(anomaly[x_col], anomaly[y_col], c='red', label='Anomaly', alpha=0.7)
    ax.set_xlabel(display_x)
    ax.set_ylabel(display_y)
    ax.set_title("Anomaly Detection Scatter Plot")
    ax.legend()
    fig.tight_layout()
    return fig

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

def plot_3d_clusters(clustered_df):
    # 3D PCA for clustering visualization
    if clustered_df is None or not all(c in clustered_df.columns for c in ['pca_x', 'pca_y']):
        return go.Figure()
    # Optionally compute third component if not present
    if 'pca_z' not in clustered_df.columns:
        features_df = clustered_df.select_dtypes(include=[np.number])
        X = features_df.values
        if X.shape[1] >= 3:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)
            clustered_df['pca_z'] = X_pca[:, 2]
        else:
            clustered_df['pca_z'] = np.zeros(len(clustered_df))
    fig = px.scatter_3d(
        clustered_df, x='pca_x', y='pca_y', z='pca_z',
        color=clustered_df['cluster'].astype(str),
        hover_data=["incident_date", "location", "incident_type", "severity"],
        title="Incident Clusters (3D PCA View)"
    )
    return fig



def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for all numeric columns in the dataframe.
    Warns if there are not enough numeric columns for a meaningful heatmap.
    Returns a matplotlib Figure.
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns for correlation heatmap. Please check your data or add more numeric features.")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation heatmap", 
                fontsize=14, ha='center', va='center')
        ax.axis('off')
        plt.tight_layout()
        return fig

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.tight_layout()
    return fig

st.title("Correlation Heatmap for NDIS Incidents")

csv_url = "https://github.com/darolin8/NDIS_dashboard/raw/main/text%20data/ndis_incidents_1000.csv"

st.write(f"Loading data from: {csv_url}")
df = pd.read_csv(csv_url)

st.write("Data preview:")
st.write(df.head())

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
st.write("Numeric columns detected:", numeric_cols)

fig = plot_correlation_heatmap(df)
st.pyplot(fig)

def forecast_incident_volume(df, periods=6):
    """
    Performs time series forecasting of incident volume using Exponential Smoothing.
    Returns: actual incident counts (series), forecasted counts (series)
    """
    if df.empty or 'incident_date' not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df_sorted = df.sort_values('incident_date')
    df_monthly = df_sorted.groupby(df_sorted['incident_date'].dt.to_period('M')).size()
    df_monthly.index = df_monthly.index.to_timestamp()
    if len(df_monthly) < 3:
        # Not enough data to forecast
        return df_monthly, pd.Series(dtype=float)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(df_monthly, trend='add', seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(periods)
        forecast.index = pd.date_range(start=df_monthly.index[-1]+pd.offsets.MonthBegin(), periods=periods, freq='M')
        return df_monthly, forecast
    except Exception as e:
        print("Forecast error:", e)
        return df_monthly, pd.Series(dtype=float)
        
def profile_location_risk(df):
    """
    Analyzes risk by location: computes incident counts and severity rates,
    returns a DataFrame and a Plotly bar chart.
    """
    if df.empty or 'location' not in df.columns or 'severity' not in df.columns:
        return pd.DataFrame(), None

    # Count incidents per location
    location_counts = df['location'].value_counts().rename('incident_count')
    # Severity rate (proportion high severity)
    sev_map = {'Low':0, 'Moderate':1, 'High':2}
    df['sev_num'] = df['severity'].map(sev_map)
    location_severity = df.groupby('location')['sev_num'].mean().rename('avg_severity')
    risk_df = pd.concat([location_counts, location_severity], axis=1).sort_values('incident_count', ascending=False)

    # Make the plot
    import plotly.express as px
    plot_df = risk_df.reset_index().rename(columns={'index':'location'})
    fig = px.bar(
        plot_df, x='location', y='incident_count', color='avg_severity',
        title='Incident Risk by Location',
        labels={'incident_count': 'Incidents', 'avg_severity': 'Avg Severity'}
    )
    return risk_df, fig

def profile_incident_type_risk(df):
    """
    Analyzes risk by incident type: computes incident counts and severity rates,
    returns a DataFrame and a Plotly bar chart.
    """
    if df.empty or 'incident_type' not in df.columns or 'severity' not in df.columns:
        return pd.DataFrame(), None

    # Count incidents per type
    type_counts = df['incident_type'].value_counts().rename('incident_count')
    # Severity rate (proportion high severity)
    sev_map = {'Low':0, 'Moderate':1, 'High':2}
    df['sev_num'] = df['severity'].map(sev_map)
    type_severity = df.groupby('incident_type')['sev_num'].mean().rename('avg_severity')
    risk_df = pd.concat([type_counts, type_severity], axis=1).sort_values('incident_count', ascending=False)

    # Make the plot
    import plotly.express as px
    plot_df = risk_df.reset_index().rename(columns={'index':'incident_type'})
    fig = px.bar(
        plot_df, x='incident_type', y='incident_count', color='avg_severity',
        title='Incident Risk by Type',
        labels={'incident_count': 'Incidents', 'avg_severity': 'Avg Severity'}
    )
    return risk_df, fig

def detect_seasonal_patterns(df):
    """
    Detects seasonal and temporal patterns in incident data.
    Returns a Plotly figure of incident counts by month.
    """
    if df.empty or 'incident_date' not in df.columns:
        return None

    # Count incidents per month
    monthly_counts = df.groupby(df['incident_date'].dt.to_period('M')).size()
    monthly_counts.index = monthly_counts.index.to_timestamp()

    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_counts.index,
        y=monthly_counts.values,
        mode='lines+markers',
        name='Incidents'
    ))
    fig.update_layout(
        title='Monthly Incident Volume',
        xaxis_title='Month',
        yaxis_title='Incident Count'
    )
    return fig
