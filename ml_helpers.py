# ml_helpers.py
# Utilities and analytics helpers for the NDIS dashboard.
# - Re-exports feature builders from utils.ndis_enhanced_prep
# - Baseline visuals + models
# - Enhanced analytics (confusion matrix, carer network, participant journey, risk scorer, similarity, alerts)
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import Tuple, Dict, Any, Optional, List, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------
# Re-export feature preparation
# ---------------------------------------
try:
    from utils.ndis_enhanced_prep import (
        prepare_ndis_data as _prepare_ndis_data,
        create_comprehensive_features as _create_comprehensive_features,
    )
except Exception:
    _prepare_ndis_data = None
    _create_comprehensive_features = None


def prepare_ndis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper so legacy imports keep working."""
    if _prepare_ndis_data is None:
        raise ImportError("utils.ndis_enhanced_prep.prepare_ndis_data not found. Ensure utils/ exists with __init__.py.")
    return _prepare_ndis_data(df)


def create_comprehensive_features(df: pd.DataFrame):
    """Thin wrapper so legacy imports keep working."""
    if _create_comprehensive_features is None:
        raise ImportError("utils.ndis_enhanced_prep.create_comprehensive_features not found. Ensure utils/ exists with __init__.py.")
    return _create_comprehensive_features(df)


# ---------------------------------------
# Internal helpers
# ---------------------------------------
def _ensure_datetime(df: pd.DataFrame, date_col: str = "incident_datetime") -> pd.DataFrame:
    if date_col not in df.columns:
        base = "incident_date" if "incident_date" in df.columns else None
        if base is None:
            raise ValueError("No datetime column found. Expected 'incident_datetime' or 'incident_date'.")
        df = df.copy()
        df[base] = pd.to_datetime(df[base], errors="coerce")
        df["incident_datetime"] = df[base]
    else:
        df = df.copy()
        df["incident_datetime"] = pd.to_datetime(df["incident_datetime"], errors="coerce")
    return df


def _monthly_counts(df: pd.DataFrame, date_col: str = "incident_datetime") -> pd.Series:
    df = _ensure_datetime(df, date_col=date_col)
    s = (
        df.set_index("incident_datetime")
          .sort_index()
          .assign(count=1)
          .resample("MS")["count"]
          .sum()
          .asfreq("MS")
          .fillna(0)
    )
    s.index.name = "Month"
    return s


# ---------------------------------------
# 1) Incident volume forecasting (+ alias)
# ---------------------------------------
def incident_volume_forecasting(df, horizon=None, horizon_months=None, months=None, n_periods=None, ...):
    if horizon is None:
        horizon = horizon_months or months or n_periods or 6
) -> Tuple[go.Figure, pd.DataFrame]:
    """Forecast monthly incident volumes via SARIMAX (if available) with naive fallback."""
    y = _monthly_counts(df, date_col=date_col)

    def _naive_forecast(series: pd.Series, steps: int):
        last_mean = series[-12:].mean() if len(series) >= 12 else series.mean()
        pred = np.full(steps, float(last_mean))
        sd = series[-12:].std(ddof=1) if len(series) >= 12 else series.std(ddof=1)
        sd = 1.96 * (sd if np.isfinite(sd) and sd > 0 else max(1.0, np.sqrt(max(last_mean, 1))))
        return pred, pred - sd, pred + sd

    use_sarimax = True
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred_res = res.get_forecast(steps=periods)
        pred_mean = pred_res.predicted_mean.values
        conf = pred_res.conf_int(alpha=0.05).to_numpy()
        lower, upper = conf[:, 0], conf[:, 1]
    except Exception:
        use_sarimax = False
        pred_mean, lower, upper = _naive_forecast(y, periods)

    last_date = y.index[-1] if len(y) else pd.Timestamp.today().normalize()
    future_idx = pd.period_range(start=last_date.to_period("M") + 1, periods=periods, freq="M").to_timestamp()

    forecast_df = pd.DataFrame({
        "Month": future_idx.strftime("%Y-%m"),
        "Predicted Incidents": np.clip(np.round(pred_mean, 0).astype(int), 0, None),
        "Lower Bound": np.clip(np.round(lower, 0).astype(int), 0, None),
        "Upper Bound": np.clip(np.round(upper, 0).astype(int), 0, None),
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(x=y.index, y=y.values, name="Historical", opacity=0.6))
    fig.add_trace(go.Scatter(x=future_idx, y=pred_mean, mode="lines+markers",
                             name=("SARIMAX Forecast" if use_sarimax else "Naive Forecast")))
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_idx, future_idx[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", name="95% CI", showlegend=True, opacity=0.2, mode="lines"
    ))
    fig.update_layout(title="Incident Volume Forecast (Monthly)", xaxis_title="Month", yaxis_title="Incidents")
    return fig, forecast_df


def forecast_incident_volume(df: pd.DataFrame, periods: int = 6):
    """Backwards-compatible alias some of your code calls."""
    return incident_volume_forecasting(df, periods=periods)


# ---------------------------------------
# 2) Temporal patterns
# ---------------------------------------
def seasonal_temporal_patterns(df: pd.DataFrame, date_col: str = "incident_datetime") -> go.Figure:
    """Heatmap of incidents by day-of-week vs hour."""
    df = _ensure_datetime(df, date_col=date_col)
    work = df.copy()
    work["dow"] = work["incident_datetime"].dt.dayofweek  # 0=Mon
    work["hour"] = work["incident_datetime"].dt.hour
    pivot = work.pivot_table(index="dow", columns="hour", values="incident_id", aggfunc="count").fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        coloraxis="coloraxis"
    ))
    fig.update_layout(
        title="Temporal Pattern Heatmap (Day-of-Week × Hour)",
        xaxis_title="Hour",
        yaxis_title="Day of Week",
        coloraxis=dict(colorscale="Blues")
    )
    return fig


# ---------------------------------------
# 3) Time series with causes
# ---------------------------------------
def plot_time_with_causes(
    df: pd.DataFrame,
    by: str = "incident_type",
    date_col: str = "incident_datetime",
    top_k: int = 5
) -> go.Figure:
    """Stacked area by top-K categories over time (by 'incident_type' or 'contributing_factors')."""
    if by not in df.columns:
        by = "incident_type" if "incident_type" in df.columns else None
        if by is None:
            raise ValueError("Neither 'incident_type' nor the provided 'by' column is present.")

    _ = _monthly_counts(df, date_col=date_col)  # ensures datetime ok
    work = _ensure_datetime(df, date_col=date_col)

    top_vals = work[by].astype(str).value_counts().head(top_k).index.tolist()
    work["_cat"] = work[by].astype(str).where(work[by].astype(str).isin(top_vals), other="Other")

    ts = (
        work.set_index("incident_datetime")
            .assign(count=1)
            .groupby([pd.Grouper(freq="MS"), "_cat"])["count"]
            .sum()
            .reset_index()
    )

    fig = px.area(
        ts,
        x="incident_datetime",
        y="count",
        color="_cat",
        title=f"Incidents Over Time by {by.title()} (Top {top_k})",
        labels={"incident_datetime": "Month", "count": "Incidents", "_cat": by.title()}
    )
    return fig


# ---------------------------------------
# 4) Carer performance scatter
# ---------------------------------------
def plot_carer_performance_scatter(df: pd.DataFrame) -> go.Figure:
    """Each point = carer. X: total incidents, Y: % High/Critical, Size: medical attention rate."""
    work = df.copy()
    if "severity" not in work.columns:
        raise ValueError("Expected 'severity' column.")
    if "carer_id" not in work.columns:
        raise ValueError("Expected 'carer_id' column.")

    if "severity_numeric" not in work.columns:
        sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        work["severity_numeric"] = work["severity"].map(sev_map).fillna(2).astype(int)

    if "medical_attention_required_bin" not in work.columns:
        mar = work.get("medical_attention_required", pd.Series([0]*len(work)))
        work["medical_attention_required_bin"] = mar.astype(str).str.lower().isin(["yes","true","1"]).astype(int)

    g = work.groupby("carer_id").agg(
        incidents=("incident_id", "count"),
        high_crit=("severity_numeric", lambda s: (s >= 3).mean()),
        med_rate=("medical_attention_required_bin", "mean"),
    ).reset_index()

    fig = px.scatter(
        g, x="incidents", y="high_crit", size="med_rate", hover_data=["carer_id"],
        labels={"incidents":"# Incidents","high_crit":"% High/Critical","med_rate":"Medical Attention Rate"},
        title="Carer Risk/Performance Overview"
    )
    fig.add_hline(y=g["high_crit"].quantile(0.8), line_dash="dash", annotation_text="80th %ile High/Critical")
    fig.add_vline(x=g["incidents"].quantile(0.8), line_dash="dash", annotation_text="80th %ile Incidents")
    fig.update_yaxes(tickformat=".0%")
    return fig


# ---------------------------------------
# 5) Correlation heatmap
# ---------------------------------------
def correlation_analysis(df: pd.DataFrame, include: Optional[list] = None) -> go.Figure:
    """Heatmap of correlations for numeric columns (optionally restrict to a whitelist)."""
    num = df.select_dtypes(include=[np.number]).copy()
    if include:
        existing = [c for c in include if c in num.columns]
        if existing:
            num = num[existing]
    corr = num.corr(numeric_only=True).fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, coloraxis="coloraxis"
    ))
    fig.update_layout(title="Correlation Matrix", coloraxis=dict(colorscale="RdBu", cmin=-1, cmax=1))
    return fig


# ---------------------------------------
# 6) Clustering analysis (KMeans + PCA)
# ---------------------------------------
def clustering_analysis(features_df: pd.DataFrame, k: int = 4) -> Tuple[go.Figure, pd.Series]:
    """KMeans on engineered features. Returns a 2D projection scatter (PCA) and labels."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = features_df.to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)
    df_plot = pd.DataFrame({"pc1": XY[:,0], "pc2": XY[:,1], "cluster": labels})

    fig = px.scatter(df_plot, x="pc1", y="pc2", color="cluster", title=f"KMeans Clusters (k={k})")
    return fig, pd.Series(labels, index=features_df.index, name="cluster")


# ---------------------------------------
# 7) Predictive models comparison (baselines)
# ---------------------------------------
def predictive_models_comparison(
    df: pd.DataFrame,
    target: str = "reportable_bin",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train baseline models and return a dict suitable for the enhanced confusion matrix UI.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X, feature_names, features_df = create_comprehensive_features(df)
    y = df[target].copy() if target in df.columns else (df.get("severity_numeric", pd.Series([2]*len(df))) >= 3).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results: Dict[str, Dict[str, Any]] = {}

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test) if hasattr(rf, "predict_proba") else None
    results["RandomForest"] = {
        "model": rf,
        "accuracy": float((rf_pred == y_test).mean()),
        "y_test": y_test,
        "predictions": rf_pred,
        "probabilities": rf_proba,
    }

    try:
        logreg = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1)
    except TypeError:
        logreg = LogisticRegression(max_iter=2000, solver="saga")
    logreg.fit(X_train, y_train)
    lg_pred = logreg.predict(X_test)
    lg_proba = logreg.predict_proba(X_test) if hasattr(logreg, "predict_proba") else None
    results["LogisticRegression"] = {
        "model": logreg,
        "accuracy": float((lg_pred == y_test).mean()),
        "y_test": y_test,
        "predictions": lg_pred,
        "probabilities": lg_proba,
    }

    return results


# ---------------------------------------
# 8) Incident type risk profiling
# ---------------------------------------
def incident_type_risk_profiling(df: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """Summarise risk by incident_type: volume, % High/Critical, medical attention rate."""
    work = df.copy()
    if "incident_type" not in work.columns:
        work["incident_type"] = "Unknown"

    if "severity_numeric" not in work.columns:
        sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        work["severity_numeric"] = work["severity"].map(sev_map).fillna(2).astype(int)

    if "medical_attention_required_bin" not in work.columns:
        mar = work.get("medical_attention_required", pd.Series([0]*len(work)))
        work["medical_attention_required_bin"] = mar.astype(str).str.lower().isin(["yes","true","1"]).astype(int)

    g = work.groupby("incident_type").agg(
        incidents=("incident_id","count"),
        high_crit_rate=("severity_numeric", lambda s: (s >= 3).mean()),
        medical_rate=("medical_attention_required_bin","mean"),
    ).reset_index().sort_values("incidents", ascending=False)

    fig = px.bar(
        g, x="incident_type", y="incidents",
        hover_data={"high_crit_rate":":.1%", "medical_rate":":.1%"},
        title="Incident Type Risk Profile (volume + rates)"
    )
    fig.update_layout(xaxis_title="Incident Type", yaxis_title="Incidents")
    return fig, g


# Friendly alias to match your import style:
profile_incident_type_risk = incident_type_risk_profiling


# =====================================================================
# =============== ENHANCED ANALYTICS (your requested set) =============
# =====================================================================

# 9) Enhanced Confusion Matrix with ROC/PR + metrics table
def enhanced_confusion_matrix_analysis(y_test, y_pred, y_proba, target_names, model_name):
    """Enhanced confusion matrix with ROC/PR curves (binary) and per-class metrics table."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_curve, auc

    cm = confusion_matrix(y_test, y_pred)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'{model_name} - Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 'Performance Metrics'),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "table"}]]
    )

    # Heatmap
    fig.add_trace(
        go.Heatmap(z=cm, x=target_names, y=target_names, colorscale='Blues', showscale=False),
        row=1, col=1
    )

    # ROC / PR (binary only)
    if len(np.unique(y_test)) == 2 and y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'), row=1, col=2)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'), row=1, col=2)

        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        pr_auc = auc(recall, precision)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC = {pr_auc:.3f})'), row=2, col=1)

    # Per-class metrics table
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    metrics_data = []
    for i, class_name in enumerate(target_names):
        metrics_data.append([class_name, f"{precision[i]:.3f}", f"{recall[i]:.3f}", f"{f1[i]:.3f}", str(support[i])])

    fig.add_trace(
        go.Table(header=dict(values=['Class', 'Precision', 'Recall', 'F1-Score', 'Support']),
                 cells=dict(values=list(zip(*metrics_data)))),
        row=2, col=2
    )
    fig.update_layout(height=800, title=f"{model_name} - Comprehensive Performance Analysis")
    return fig


# 10) Carer-Participant Risk Network
def carer_risk_network_analysis(df: pd.DataFrame):
    """Analyze carer-participant risk network; returns (fig, risk_matrix)."""
    required = {"carer_id", "participant_id", "severity", "incident_id"}
    if not required.issubset(df.columns):
        return None, None

    work = df.copy()
    # severity numeric
    if "severity_numeric" not in work.columns:
        sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        work["severity_numeric"] = work["severity"].map(sev_map).fillna(2).astype(int)

    # medical attention binary
    if "medical_attention_required_bin" in work.columns:
        mar = work["medical_attention_required_bin"].astype(int)
    else:
        mar = work.get("medical_attention_required", pd.Series([0]*len(work)))
        mar = mar.astype(str).str.lower().isin(["yes","true","1"]).astype(int)
    work["medical_attention_required_bin"] = mar

    risk_matrix = work.groupby(['carer_id', 'participant_id']).agg(
        incident_count=('incident_id', 'count'),
        avg_severity=('severity_numeric', 'mean'),
        medical_rate=('medical_attention_required_bin', 'mean')
    ).round(3).reset_index()

    # composite risk
    denom = max(risk_matrix['incident_count'].max(), 1)
    risk_matrix['risk_score'] = (
        risk_matrix['avg_severity'] * 0.4 +
        risk_matrix['medical_rate'] * 0.3 +
        (risk_matrix['incident_count'] / denom) * 0.3
    )

    fig = px.scatter(
        risk_matrix,
        x='incident_count',
        y='avg_severity',
        size='medical_rate',
        color='risk_score',
        hover_data=['carer_id', 'participant_id'],
        title="Carer-Participant Risk Network",
        labels={'incident_count': 'Number of Incidents', 'avg_severity': 'Average Severity', 'risk_score': 'Risk Score'},
        color_continuous_scale='Reds'
    )
    fig.add_hline(y=risk_matrix['avg_severity'].quantile(0.8), line_dash="dash", line_color="red",
                  annotation_text="High Severity Threshold")
    fig.add_vline(x=risk_matrix['incident_count'].quantile(0.8), line_dash="dash", line_color="orange",
                  annotation_text="High Frequency Threshold")
    return fig, risk_matrix


# 11) Participant Journey Analysis
def participant_journey_analysis(df: pd.DataFrame, participant_id: Any):
    """Timeline markers by severity + numeric severity trend + location histogram for one participant."""
    if 'participant_id' not in df.columns or participant_id not in df['participant_id'].values:
        return None

    participant_data = df[df['participant_id'] == participant_id].copy()
    participant_data['incident_date'] = pd.to_datetime(participant_data.get('incident_date', participant_data.get('incident_datetime')), errors='coerce')
    participant_data = participant_data.sort_values('incident_date')

    colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Critical': 'red'}

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=('Incident Timeline', 'Severity Trend', 'Location Pattern'),
                        vertical_spacing=0.1, row_heights=[0.4, 0.3, 0.3])

    # Timeline
    for severity in participant_data['severity'].astype(str).unique():
        sd = participant_data[participant_data['severity'].astype(str) == severity]
        fig.add_trace(
            go.Scatter(
                x=sd['incident_date'], y=[severity] * len(sd),
                mode='markers',
                marker=dict(size=10, color=colors.get(severity, 'blue')),
                name=severity,
                text=sd.get('location', ''),
                hovertemplate='Date: %{x}<br>Severity: %{y}<br>Location: %{text}'
            ),
            row=1, col=1
        )

    # Severity trend
    sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    severity_numeric = participant_data['severity'].map(sev_map)
    fig.add_trace(
        go.Scatter(x=participant_data['incident_date'], y=severity_numeric,
                   mode='lines+markers', name='Severity Trend', line=dict(color='red', width=2)),
        row=2, col=1
    )

    # Location frequency
    location_counts = participant_data.get('location', pd.Series(dtype=str)).value_counts()
    fig.add_trace(
        go.Bar(x=location_counts.values, y=location_counts.index, orientation='h', name='Location Frequency'),
        row=3, col=1
    )

    fig.update_layout(height=900, title=f"Participant Journey Analysis - {participant_id}")
    return fig


# 12) Predictive Risk Scoring System
def create_predictive_risk_scoring(df: pd.DataFrame, trained_models: Dict[str, Dict[str, Any]], feature_names: List[str]):
    """Return a function calculate_risk_score(scenario_data) using the best model in trained_models."""
    if not trained_models:
        return None

    best_model_name = max(trained_models.keys(), key=lambda x: trained_models[x].get('accuracy', 0))
    best_model = trained_models[best_model_name]['model']

    def calculate_risk_score(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        vec = np.zeros(len(feature_names))
        mapping = {
            'hour': scenario_data.get('hour', 12),
            'is_weekend': 1 if scenario_data.get('day_type') == 'weekend' else 0,
            'is_kitchen': 1 if 'kitchen' in scenario_data.get('location', '').lower() else 0,
            'is_bathroom': 1 if any(k in scenario_data.get('location', '').lower() for k in ['bathroom','toilet','washroom','restroom']) else 0,
            'participant_incident_count': scenario_data.get('participant_history', 1),
            'carer_incident_count': scenario_data.get('carer_history', 1),
            'location_risk_score': scenario_data.get('location_risk', 2)
        }
        for i, fn in enumerate(feature_names):
            if fn in mapping:
                vec[i] = mapping[fn]

        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba([vec])[0]
            max_risk = float(np.max(proba))
        else:
            pred = best_model.predict([vec])[0]
            max_risk = float(pred) / 3.0  # crude normalization

        level = 'HIGH' if max_risk > 0.7 else 'MEDIUM' if max_risk > 0.4 else 'LOW'
        return {'risk_score': max_risk, 'risk_level': level, 'confidence': max_risk, 'model_used': best_model_name}

    return calculate_risk_score


# 13) Incident Similarity Analysis
def incident_similarity_analysis(df: pd.DataFrame, X: np.ndarray, feature_names: List[str]):
    """Return (find_similar_incidents, similarity_matrix) using cosine similarity on standardized features."""
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    similarity_matrix = cosine_similarity(X_scaled)

    def find_similar_incidents(incident_idx: int, top_k: int = 5):
        if incident_idx >= len(similarity_matrix):
            return None
        sims = similarity_matrix[incident_idx]
        indices = np.argsort(sims)[-top_k-1:-1][::-1]  # exclude self
        out = []
        for idx in indices:
            out.append({'index': int(idx), 'similarity_score': float(sims[idx]), 'incident_data': df.iloc[idx] if idx < len(df) else None})
        return out

    return find_similar_incidents, similarity_matrix


# 14) Real-Time Alert System (Simulation)
def generate_recommendation(scenario: Dict[str, Any], risk_assessment: Dict[str, Any]) -> List[str]:
    recs = {
        'Early Morning Kitchen Risk': [
            'Assign experienced carer for kitchen activities',
            'Ensure safety equipment is readily available',
            'Consider rescheduling to later hours if possible'
        ],
        'High-Risk Combination': [
            'Provide additional supervision',
            'Review recent incident patterns',
            'Consider alternative carer assignment'
        ],
        'Weekend Transport Risk': [
            'Ensure additional safety protocols',
            'Verify transport safety equipment',
            'Consider companion support'
        ]
    }
    return recs.get(scenario['name'], ['Review safety protocols', 'Increase supervision'])


def simulate_real_time_alerts(df: pd.DataFrame, risk_scoring_function: Callable[[Dict[str, Any]], Dict[str, Any]], alert_thresholds: Dict[str, float]):
    """Simulate real-time alerts for a few predefined scenarios."""
    alerts = []
    scenarios = [
        {'name': 'Early Morning Kitchen Risk', 'conditions': {'hour': [5,6,7,8], 'location_contains': 'kitchen', 'participant_history': '>= 3'}, 'severity': 'HIGH'},
        {'name': 'High-Risk Combination', 'conditions': {'participant_history': '>= 5', 'carer_history': '>= 10'}, 'severity': 'MEDIUM'},
        {'name': 'Weekend Transport Risk', 'conditions': {'day_type': 'weekend', 'location_contains': 'transport'}, 'severity': 'MEDIUM'}
    ]
    for sc in scenarios:
        # simple representative scenario input
        scenario_input = {'hour': 6, 'location': 'kitchen', 'day_type': 'weekday', 'participant_history': 5, 'carer_history': 8, 'location_risk': 3}
        risk = risk_scoring_function(scenario_input)
        if risk['risk_score'] > alert_thresholds.get(sc['severity'].lower(), 0.5):
            alerts.append({
                'scenario': sc['name'],
                'risk_score': risk['risk_score'],
                'risk_level': risk['risk_level'],
                'timestamp': pd.Timestamp.now(),
                'recommendation': generate_recommendation(sc, risk)
            })
    return alerts


# 15) Streamlit integration helpers (safe-imported)
def add_enhanced_features_to_dashboard(df: pd.DataFrame, X: np.ndarray, feature_names: List[str], trained_models: Dict[str, Dict[str, Any]]):
    """Render enhanced features inside a Streamlit app."""
    try:
        import streamlit as st
    except Exception:
        raise ImportError("Streamlit not available. This function must be called inside a Streamlit app.")

    st.markdown("### 🔬 Enhanced Analytics Features")

    # Enhanced model analysis
    if trained_models:
        st.markdown("#### 📊 Enhanced Model Performance Analysis")
        for model_name, model_data in trained_models.items():
            if 'predictions' in model_data and 'probabilities' in model_data:
                fig = enhanced_confusion_matrix_analysis(
                    model_data.get('y_test', []),
                    model_data['predictions'],
                    model_data['probabilities'],
                    ['Low', 'Medium', 'High'] if len(np.unique(model_data.get('y_test', []))) > 2 else ['No','Yes'],
                    model_name
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # Carer risk network
    st.markdown("#### 🕸️ Carer-Participant Risk Network")
    network_fig, risk_matrix = carer_risk_network_analysis(df)
    if network_fig is not None:
        st.plotly_chart(network_fig, use_container_width=True)
        high_risk_pairs = risk_matrix[risk_matrix['risk_score'] > risk_matrix['risk_score'].quantile(0.8)]
        if len(high_risk_pairs) > 0:
            st.markdown("##### 🚨 High-Risk Carer-Participant Pairs")
            st.dataframe(high_risk_pairs.sort_values('risk_score', ascending=False))

    # Participant journey
    if 'participant_id' in df.columns:
        st.markdown("#### 👤 Individual Participant Journey")
        participant_ids = df['participant_id'].dropna().unique()
        if len(participant_ids):
            selected_participant = st.selectbox("Select Participant", participant_ids)
            journey_fig = participant_journey_analysis(df, selected_participant)
            if journey_fig:
                st.plotly_chart(journey_fig, use_container_width=True)

    # Predictive risk scoring UI
    if trained_models:
        st.markdown("#### 🎯 Predictive Risk Scoring")
        risk_scorer = create_predictive_risk_scoring(df, trained_models, feature_names)
        if risk_scorer:
            col1, col2, col3 = st.columns(3)
            with col1:
                test_hour = st.slider("Hour", 0, 23, 8)
            with col2:
                test_location = st.selectbox("Location", ['kitchen', 'bathroom', 'living room', 'activity room'])
            with col3:
                test_history = st.slider("Participant History", 1, 20, 3)

            scenario = {
                'hour': test_hour,
                'location': test_location,
                'participant_history': test_history,
                'carer_history': 5,
                'location_risk': 3 if test_location in ['kitchen', 'bathroom'] else 1,
                'day_type': 'weekend' if st.checkbox("Weekend?", value=False) else 'weekday'
            }
            risk_result = risk_scorer(scenario)
            if risk_result['risk_level'] == 'HIGH':
                st.error(f"🚨 {risk_result['risk_level']} RISK - {risk_result['confidence']:.1%} confidence")
            elif risk_result['risk_level'] == 'MEDIUM':
                st.warning(f"⚠️ {risk_result['risk_level']} RISK - {risk_result['confidence']:.1%} confidence")
            else:
                st.success(f"✅ {risk_result['risk_level']} RISK - {risk_result['confidence']:.1%} confidence")

    # Real-time alerts simulation
    st.markdown("#### 🚨 Real-Time Alert System (Simulation)")
    alert_thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.3}
    if trained_models:
        risk_scorer = create_predictive_risk_scoring(df, trained_models, feature_names)
        if risk_scorer:
            alerts = simulate_real_time_alerts(df, risk_scorer, alert_thresholds)
            if alerts:
                st.warning(f"🚨 {len(alerts)} Active Alerts")
                for alert in alerts:
                    with st.expander(f"Alert: {alert['scenario']} - {alert['risk_level']}"):
                        st.write(f"**Risk Score:** {alert['risk_score']:.1%}")
                        st.write(f"**Timestamp:** {alert['timestamp']}")
                        st.write("**Recommendations:**")
                        for rec in alert['recommendation']:
                            st.write(f"• {rec}")
            else:
                st.success("✅ No active alerts")


def integrate_enhanced_features(existing_main_function):
    """Wrap your `main()` so it renders enhanced features after loading data into st.session_state.df."""
    def enhanced_main():
        existing_main_function()
        try:
            import streamlit as st
        except Exception:
            raise ImportError("Streamlit not available. integrate_enhanced_features must be used in a Streamlit app.")

        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            df = st.session_state.df
            X, feature_names, _ = create_comprehensive_features(df)
            trained_models = getattr(st.session_state, 'trained_models', {})
            add_enhanced_features_to_dashboard(df, X, feature_names, trained_models)

    return enhanced_main
