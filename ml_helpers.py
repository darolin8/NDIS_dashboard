import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

import streamlit as st
from scipy.stats import chi2_contingency, f_oneway

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
SEV_MAP_NUM = {
    # treat “Minor/Low” -> 0, “Moderate/Medium” -> 1, “High/Major/Critical” -> 2
    'low': 0, 'minor': 0,
    'moderate': 1, 'medium': 1,
    'high': 2, 'major': 2, 'critical': 2
}

def _norm_severity_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    return s.astype(str).str.strip().str.lower().map(SEV_MAP_NUM)

def _safe_ohe():
    """Handle sklearn <1.2 vs >=1.2 param name change."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def _split_columns(df: pd.DataFrame, target: str = 'severity'):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = [c for c in num_cols if c != target]
    cat_cols = [c for c in cat_cols if c != target]
    return num_cols, cat_cols

def _ensure_datetime(series: pd.Series):
    return pd.to_datetime(series, errors='coerce')

# ---------------------------------------------------------
# Feature Preparation (kept for compatibility in other calls)
# ---------------------------------------------------------
def prepare_ml_features(df: pd.DataFrame):
    """
    Returns dense numpy X, feature_names, and fitted encoder for categorical columns.
    Numeric columns are left unscaled here on purpose (some downstream steps scale).
    """
    if df is None or df.empty:
        return None, [], None

    num_cols, cat_cols = _split_columns(df, target='severity')
    all_feature_cols = num_cols + cat_cols
    if not all_feature_cols:
        return None, [], None

    feat = df[all_feature_cols].copy()
    encoder = None
    feature_names = []

    if cat_cols:
        encoder = _safe_ohe()
        # Robust to NaNs and mixed types
        cats = feat[cat_cols].astype(str).fillna("Missing")
        encoded = encoder.fit_transform(cats)
        encoded_names = encoder.get_feature_names_out(cat_cols)
        X = np.hstack([feat[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(feat), 0)),
                       encoded])
        feature_names = num_cols + list(encoded_names)
    else:
        X = feat[num_cols].to_numpy(dtype=float)
        feature_names = num_cols

    return X, feature_names, encoder

# ---------------------------------------------------------
# Model Comparison & ROC (multiclass One-vs-Rest)
# ---------------------------------------------------------
def compare_models(df: pd.DataFrame):
    X, feature_names, _ = prepare_ml_features(df)
    if X is None or 'severity' not in df.columns:
        empty = go.Figure()
        empty.add_annotation(text="No data available for ROC curves.", x=0.5, y=0.5, showarrow=False)
        return pd.DataFrame(), empty

    y = _norm_severity_series(df['severity'])
    mask = ~y.isna()
    X, y = X[mask], y[mask].astype(int)

    if len(X) < 10 or len(np.unique(y)) < 2:
        empty = go.Figure()
        empty.add_annotation(text="Not enough data/classes for ROC curves.", x=0.5, y=0.5, showarrow=False)
        return pd.DataFrame(), empty

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=120, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        # MLP benefits from scaling but we keep raw for speed here (Pipeline would be ideal)
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=42, early_stopping=True)
    }

    metrics = []
    roc_fig = go.Figure()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        metrics.append({'Model': name, 'Accuracy': acc})

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            # One-vs-rest curves
            for i in range(y_prob.shape[1]):
                # only plot if class i appears in y_test
                if np.any(y_test == i):
                    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                 name=f"{name} – class {i} (AUC={roc_auc:.2f})"))

    metrics_df = pd.DataFrame(metrics)
    if len(roc_fig.data) == 0:
        roc_fig.add_annotation(text="No ROC curves available.", x=0.5, y=0.5, showarrow=False)

    roc_fig.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"
    )
    return metrics_df, roc_fig

# ---------------------------------------------------------
# Severity Prediction (Pipeline with preprocessing)
# ---------------------------------------------------------
try:
    _cache_model = st.cache_resource  # best for sklearn objects
except AttributeError:
    _cache_model = st.cache_data

@_cache_model
def train_severity_prediction_model(df: pd.DataFrame):
    """
    Trains a RandomForest in a Pipeline that includes:
      - StandardScaler for numeric
      - OneHotEncoder for categoricals
    Returns (pipeline_model, accuracy, transformed_feature_names)
    """
    if df is None or df.empty or 'severity' not in df.columns or len(df) < 20:
        return None, None, None

    y = _norm_severity_series(df['severity'])
    mask = ~y.isna()
    df_ = df.loc[mask].copy()
    y = y.loc[mask].astype(int)

    if len(df_) < 20 or len(np.unique(y)) < 2:
        return None, None, None

    num_cols, cat_cols = _split_columns(df_, target='severity')

    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', _safe_ohe(), cat_cols)
        ],
        remainder='drop'
    )

    pipe = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df_[num_cols + cat_cols], y, test_size=0.30, random_state=42, stratify=y
    )

    # Fit preprocessor first to get feature names reliably
    pipe.named_steps['pre'].fit(X_train)
    feature_names = pipe.named_steps['pre'].get_feature_names_out()

    pipe.fit(X_train, y_train)
    acc = float(accuracy_score(y_test, pipe.predict(X_test)))
    return pipe, acc, list(feature_names)

# ---------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------
@st.cache_data
def perform_anomaly_detection(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 10:
        return None, None

    X, feature_names, _ = prepare_ml_features(df)
    if X is None:
        return None, None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.10, random_state=42)
    iso_labels = iso.fit_predict(Xs)

    svm = OneClassSVM(nu=0.10, kernel='rbf', gamma='scale')
    svm_labels = svm.fit_predict(Xs)

    out = df.copy()
    out['isolation_forest_anomaly'] = (iso_labels == -1)
    out['svm_anomaly'] = (svm_labels == -1)
    # Higher = more normal; lower = more anomalous. Flip sign for intuitive "risk".
    out['anomaly_score'] = -iso.decision_function(Xs)

    return out, feature_names

def plot_anomaly_scatter(anomaly_df, x_col, y_col, anomaly_column="isolation_forest_anomaly", axis_labels=None):
    if (
        anomaly_df is None
        or x_col not in anomaly_df.columns
        or y_col not in anomaly_df.columns
        or anomaly_column not in anomaly_df.columns
    ):
        raise ValueError("Required columns are missing in the DataFrame.")

    display_x = axis_labels.get(x_col, x_col) if axis_labels else x_col
    display_y = axis_labels.get(y_col, y_col) if axis_labels else y_col

    fig, ax = plt.subplots(figsize=(8, 5))
    normal = anomaly_df[~anomaly_df[anomaly_column]]
    anomaly = anomaly_df[anomaly_df[anomaly_column]]

    ax.scatter(normal[x_col], normal[y_col], label='Normal', alpha=0.5)
    ax.scatter(anomaly[x_col], anomaly[y_col], label='Anomaly', alpha=0.8, marker='x')

    ax.set_xlabel(display_x)
    ax.set_ylabel(display_y)
    ax.set_title("Anomaly Detection Scatter Plot")
    ax.legend()
    fig.tight_layout()
    return fig

# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------
@st.cache_data
def perform_clustering_analysis(df: pd.DataFrame, n_clusters=5, algorithm='kmeans'):
    if df is None or df.empty or len(df) < 10:
        return None, None, None, None

    X, feature_names, _ = prepare_ml_features(df)
    if X is None:
        return None, None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2D PCA for visualization
    pca2 = PCA(n_components=2, random_state=42)
    X_pca2 = pca2.fit_transform(X_scaled)

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
    if len(set(cluster_labels)) > 1 and len(np.unique(cluster_labels)) > 1:
        try:
            sil_score = float(silhouette_score(X_scaled, cluster_labels))
        except Exception:
            sil_score = None

    result_df = df.copy()
    result_df['cluster'] = cluster_labels
    result_df['pca_x'] = X_pca2[:, 0]
    result_df['pca_y'] = X_pca2[:, 1]

    return result_df, feature_names, sil_score, pca2

def plot_3d_clusters(clustered_df: pd.DataFrame):
    if clustered_df is None or not all(c in clustered_df.columns for c in ['pca_x', 'pca_y']):
        return go.Figure()

    # If we don’t have a 3rd axis, compute from numeric features
    if 'pca_z' not in clustered_df.columns:
        num_feats = clustered_df.select_dtypes(include=[np.number]).drop(columns=['cluster', 'pca_x', 'pca_y'], errors='ignore')
        if num_feats.shape[1] >= 3:
            pca3 = PCA(n_components=1, random_state=42)
            clustered_df = clustered_df.copy()
            clustered_df['pca_z'] = pca3.fit_transform(num_feats)[:, 0]
        else:
            clustered_df = clustered_df.copy()
            clustered_df['pca_z'] = 0.0

    hover_cols = [c for c in ["incident_date", "location", "incident_type", "severity"] if c in clustered_df.columns]

    fig = px.scatter_3d(
        clustered_df, x='pca_x', y='pca_y', z='pca_z',
        color=clustered_df['cluster'].astype(str),
        hover_data=hover_cols,
        title="Incident Clusters (3D PCA View)"
    ).update_layout(template="plotly_white")
    return fig

# ---------------------------------------------------------
# Feature Association Analysis
# ---------------------------------------------------------
def cramers_v(x, y):
    confusion = pd.crosstab(x, y, dropna=False)
    if confusion.size == 0:
        return np.nan
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan

def phi_coefficient(x, y):
    confusion = pd.crosstab(x, y, dropna=False)
    if confusion.shape != (2, 2):
        return np.nan
    a = confusion.iloc[0, 0]
    b = confusion.iloc[0, 1]
    c = confusion.iloc[1, 0]
    d = confusion.iloc[1, 1]
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denom if denom != 0 else np.nan

def eta_squared(groups, values):
    mask = ~pd.isna(groups) & ~pd.isna(values)
    groups = groups[mask]
    values = values[mask]
    if len(values) == 0:
        return np.nan
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return np.nan
    group_means = [values[groups == g].mean() for g in uniq]
    grand_mean = values.mean()
    ssm = sum([np.sum(groups == g) * (m - grand_mean) ** 2 for g, m in zip(uniq, group_means)])
    sst = np.sum((values - grand_mean) ** 2)
    return ssm / sst if sst != 0 else np.nan

def perform_correlation_analysis(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame(), None

    results = []
    cols = list(df.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            try:
                dt1, dt2 = df[c1].dtype, df[c2].dtype
                if str(dt1) in ["object", "category", "bool"] and str(dt2) in ["object", "category", "bool"]:
                    v = cramers_v(df[c1], df[c2])
                    if np.isnan(v):
                        continue
                    chi2, p, _, _ = chi2_contingency(pd.crosstab(df[c1], df[c2], dropna=False))
                    interp = "No association" if v < 0.10 else "Small association" if v < 0.30 else "Medium association" if v < 0.50 else "Large association"
                    results.append(dict(feature_1=c1, feature_2=c2, association_type="Categorical–Categorical",
                                        strength=float(v), significant=bool(p < 0.05), interpretation=interp))
                elif df[c1].nunique(dropna=True) == 2 and df[c2].nunique(dropna=True) == 2:
                    v = phi_coefficient(df[c1], df[c2])
                    if np.isnan(v):
                        continue
                    iv = abs(v)
                    interp = "No association" if iv < 0.10 else "Small association" if iv < 0.30 else "Medium association" if iv < 0.50 else "Strong association"
                    results.append(dict(feature_1=c1, feature_2=c2, association_type="Binary–Binary",
                                        strength=float(v), significant=bool(iv > 0.10), interpretation=interp))
                else:
                    # numerical–categorical (ANOVA + eta^2)
                    # decide which is numeric/categorical
                    if np.issubdtype(df[c1].dtype, np.number) and str(df[c2].dtype) in ['object', 'category', 'bool']:
                        num_col, cat_col = c1, c2
                    elif np.issubdtype(df[c2].dtype, np.number) and str(df[c1].dtype) in ['object', 'category', 'bool']:
                        num_col, cat_col = c2, c1
                    else:
                        continue

                    groups = df[cat_col]
                    values = pd.to_numeric(df[num_col], errors='coerce')
                    valid_groups = [g for g in pd.unique(groups) if np.sum(groups == g) > 1]
                    if len(valid_groups) < 2:
                        continue
                    vals_by_group = [values[groups == g].dropna() for g in valid_groups if np.sum(groups == g) > 1]
                    if len(vals_by_group) < 2 or any(len(vg) == 0 for vg in vals_by_group):
                        continue
                    anova = f_oneway(*vals_by_group)
                    eta2 = eta_squared(groups, values)
                    if np.isnan(eta2):
                        continue
                    interp = "Small effect" if eta2 < 0.06 else "Medium effect" if eta2 < 0.14 else "Large effect"
                    results.append(dict(feature_1=num_col, feature_2=cat_col, association_type="Numerical–Categorical",
                                        strength=float(eta2), significant=bool(anova.pvalue < 0.05), interpretation=interp))
            except Exception:
                continue

    associations_df = pd.DataFrame(results)
    if associations_df.empty:
        return associations_df, None

    try:
        heatmap_df = associations_df.pivot_table(index='feature_1', columns='feature_2', values='strength', aggfunc='mean')
        zmin = np.nanmin(heatmap_df.values)
        zmax = np.nanmax(heatmap_df.values)
        zmid = 0 if zmin < 0 else None
        fig = px.imshow(
            heatmap_df,
            color_continuous_scale='RdBu',
            aspect='auto',
            title='Feature Association Matrix',
            labels=dict(x="Feature 2", y="Feature 1", color="Strength"),
            zmin=zmin, zmax=zmax, zmid=zmid
        ).update_layout(template="plotly_white")
    except Exception:
        fig = None

    return associations_df, fig

# ---------------------------------------------------------
# Dashboard Helper Stubs / Profiles
# ---------------------------------------------------------
def forecast_incident_volume(*args, **kwargs):
    return pd.Series(dtype='float64'), pd.Series(dtype='float64')

def profile_location_risk(df: pd.DataFrame):
    if df is None or df.empty or 'location' not in df.columns or 'severity' not in df.columns:
        return pd.DataFrame(), None

    tmp = df.copy()
    tmp['sev_num'] = _norm_severity_series(tmp['severity'])
    location_counts = tmp['location'].value_counts(dropna=False).rename('incident_count')
    location_severity = tmp.groupby('location', dropna=False)['sev_num'].mean().rename('avg_severity')

    risk_df = pd.concat([location_counts, location_severity], axis=1).sort_values('incident_count', ascending=False)
    plot_df = risk_df.reset_index().rename(columns={'index': 'location'})

    fig = px.bar(
        plot_df, x='location', y='incident_count', color='avg_severity',
        title='Incident Risk by Location',
        labels={'incident_count': 'Incidents', 'avg_severity': 'Avg Severity'}
    ).update_layout(template="plotly_white")
    return risk_df, fig

def profile_incident_type_risk(df: pd.DataFrame):
    if df is None or df.empty or 'incident_type' not in df.columns or 'severity' not in df.columns:
        return pd.DataFrame(), None

    tmp = df.copy()
    tmp['sev_num'] = _norm_severity_series(tmp['severity'])
    type_counts = tmp['incident_type'].value_counts(dropna=False).rename('incident_count')
    type_severity = tmp.groupby('incident_type', dropna=False)['sev_num'].mean().rename('avg_severity')

    risk_df = pd.concat([type_counts, type_severity], axis=1).sort_values('incident_count', ascending=False)
    plot_df = risk_df.reset_index().rename(columns={'index': 'incident_type'})

    fig = px.bar(
        plot_df, x='incident_type', y='incident_count', color='avg_severity',
        title='Incident Risk by Type',
        labels={'incident_count': 'Incidents', 'avg_severity': 'Avg Severity'}
    ).update_layout(template="plotly_white")
    return risk_df, fig

def detect_seasonal_patterns(df: pd.DataFrame):
    if df is None or df.empty or 'incident_date' not in df.columns:
        return None
    ts = _ensure_datetime(df['incident_date'])
    monthly_counts = df.groupby(ts.dt.to_period('M')).size()
    if monthly_counts.empty:
        return None
    monthly_counts.index = monthly_counts.index.to_timestamp()
    fig = px.line(
        x=monthly_counts.index,
        y=monthly_counts.values,
        title='Monthly Incident Volume',
        labels={'x': 'Month', 'y': 'Incident Count'}
    ).update_layout(template="plotly_white")
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """Matplotlib-only correlation heatmap (no seaborn dependency)."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation heatmap",
                fontsize=12, ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        return fig

    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, interpolation='nearest', aspect='auto')
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Correlation', rotation=270, labelpad=15)

    # Annotate cells
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8)

    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig
