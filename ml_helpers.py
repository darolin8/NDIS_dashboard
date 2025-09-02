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


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
import plotly.express as px
import plotly.graph_objs as go

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def phi_coefficient(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape == (2,2):
        a = confusion_matrix.iloc[0,0]
        b = confusion_matrix.iloc[0,1]
        c = confusion_matrix.iloc[1,0]
        d = confusion_matrix.iloc[1,1]
        numerator = a*d - b*c
        denominator = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
        return numerator / denominator if denominator != 0 else np.nan
    else:
        return np.nan

def eta_squared(anova_result, groups, values):
    group_means = [values[groups == g].mean() for g in np.unique(groups)]
    grand_mean = values.mean()
    ssm = sum([len(values[groups == g]) * (group_mean - grand_mean) ** 2 for g, group_mean in zip(np.unique(groups), group_means)])
    sst = sum((values - grand_mean) ** 2)
    return ssm / sst if sst != 0 else np.nan

def perform_correlation_analysis(df):
    """
    Returns: associations_df, association_heatmap
    associations_df columns: feature_1, feature_2, association_type, strength, significant, interpretation
    """
    results = []
    columns = df.columns
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            # Categorical-Categorical
            if df[col1].dtype == "object" and df[col2].dtype == "object":
                try:
                    v = cramers_v(df[col1], df[col2])
                    chi2, p, _, _ = chi2_contingency(pd.crosstab(df[col1], df[col2]))
                    interpretation = (
                        "No association" if v < 0.1 else
                        "Small association" if v < 0.3 else
                        "Medium association" if v < 0.5 else
                        "Large association"
                    )
                    results.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "association_type": "Categorical-Categorical",
                        "strength": v,
                        "significant": p < 0.05,
                        "interpretation": interpretation
                    })
                except Exception:
                    continue
            # Binary-Binary
            elif df[col1].nunique() == 2 and df[col2].nunique() == 2:
                try:
                    v = phi_coefficient(df[col1], df[col2])
                    interpretation = (
                        "No association" if abs(v) < 0.1 else
                        "Small association" if abs(v) < 0.3 else
                        "Medium association" if abs(v) < 0.5 else
                        "Strong association"
                    )
                    results.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "association_type": "Binary-Binary",
                        "strength": v,
                        "significant": abs(v) > 0.1,
                        "interpretation": interpretation
                    })
                except Exception:
                    continue
            # Numerical-Categorical
            elif (df[col1].dtype in ['float64', 'int64'] and df[col2].dtype == 'object') or \
                 (df[col2].dtype in ['float64', 'int64'] and df[col1].dtype == 'object'):
                try:
                    num_col, cat_col = (col1, col2) if df[col1].dtype in ['float64', 'int64'] else (col2, col1)
                    groups = df[cat_col]
                    values = df[num_col]
                    anova = f_oneway(*(values[groups == g] for g in np.unique(groups)))
                    eta2 = eta_squared(anova, groups, values)
                    interpretation = (
                        "Small effect" if eta2 < 0.06 else
                        "Medium effect" if eta2 < 0.14 else
                        "Large effect"
                    )
                    results.append({
                        "feature_1": num_col,
                        "feature_2": cat_col,
                        "association_type": "Numerical-Categorical",
                        "strength": eta2,
                        "significant": anova.pvalue < 0.05,
                        "interpretation": interpretation
                    })
                except Exception:
                    continue
    associations_df = pd.DataFrame(results)
    # Association heatmap
    if not associations_df.empty:
        heatmap_df = associations_df.pivot_table(
            index='feature_1', columns='feature_2', values='strength', fill_value=np.nan
        )
        fig = px.imshow(
            heatmap_df,
            color_continuous_scale='RdBu',
            aspect='auto',
            title='Feature Association Matrix',
            labels=dict(x="Feature 2", y="Feature 1", color="Association Strength")
        )
        return associations_df, fig
    else:
        return associations_df, None

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
