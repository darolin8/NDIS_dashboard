
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from typing import Tuple, Dict, Any, Optional

# ---------------------------------------
# Re-export feature preparation
# ---------------------------------------
try:
    from utils.ndis_enhanced_prep import (
        prepare_ndis_data as _prepare_ndis_data,
        create_comprehensive_features as _create_comprehensive_features,
    )
except Exception as e:
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
        # fallback to incident_date if present
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
# 1) Incident volume forecasting
# ---------------------------------------

def incident_volume_forecasting(
    df: pd.DataFrame,
    date_col: str = "incident_datetime",
    periods: int = 6,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Forecast monthly incident volumes with a light SARIMAX, falling back to a naive seasonal method.

    Returns:
        fig (plotly.graph_objects.Figure)
        forecast_df (pd.DataFrame): Month, Predicted Incidents, Lower Bound, Upper Bound
    """
    y = _monthly_counts(df, date_col=date_col)

    # Default: naive seasonal (last 12-month mean) forecast + simple CI
    def _naive_forecast(series: pd.Series, steps: int):
        last_mean = series[-12:].mean() if len(series) >= 12 else series.mean()
        pred = np.full(steps, float(last_mean))
        # crude CI based on recent std
        sd = series[-12:].std(ddof=1) if len(series) >= 12 else series.std(ddof=1)
        sd = 1.96 * (sd if np.isfinite(sd) and sd > 0 else max(1.0, np.sqrt(last_mean)))
        return pred, pred - sd, pred + sd

    use_sarimax = True
    try:
        # Try SARIMAX if available
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
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

    # Figure
    fig = go.Figure()
    fig.add_trace(go.Bar(x=y.index, y=y.values, name="Historical", opacity=0.6))
    fig.add_trace(go.Scatter(x=future_idx, y=pred_mean, mode="lines+markers",
                             name=("SARIMAX Forecast" if use_sarimax else "Naive Forecast")))
    fig.add_trace(go.Scatter(x=np.concatenate([future_idx, future_idx[::-1]]),
                             y=np.concatenate([upper, lower[::-1]]),
                             fill="toself", name="95% CI", showlegend=True, opacity=0.2, mode="lines"))
    fig.update_layout(title="Incident Volume Forecast (Monthly)", xaxis_title="Month", yaxis_title="Incidents")
    return fig, forecast_df


# Backwards-compatible alias used in some of your code
def forecast_incident_volume(df: pd.DataFrame, periods: int = 6):
    return incident_volume_forecasting(df, periods=periods)


# ---------------------------------------
# 2) Seasonal / Temporal patterns
# ---------------------------------------

def seasonal_temporal_patterns(df: pd.DataFrame, date_col: str = "incident_datetime") -> go.Figure:
    """
    Heatmap of incidents by day-of-week vs hour.
    """
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
# 3) Time series with causes/factors
# ---------------------------------------

def plot_time_with_causes(
    df: pd.DataFrame,
    by: str = "incident_type",
    date_col: str = "incident_datetime",
    top_k: int = 5
) -> go.Figure:
    """
    Stacked area by top-K categories over time.
    'by' can be 'incident_type' or 'contributing_factors'.
    """
    if by not in df.columns:
        by = "incident_type" if "incident_type" in df.columns else None
        if by is None:
            raise ValueError("Neither 'incident_type' nor the provided 'by' column is present.")

    s = _monthly_counts(df, date_col=date_col)
    work = _ensure_datetime(df, date_col=date_col)

    # pick top K categories
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
    """
    Each point = carer.
    X: total incidents, Y: % Critical/High, Size: medical attention rate
    """
    work = df.copy()
    if "severity" not in work.columns:
        raise ValueError("Expected 'severity' column.")
    if "carer_id" not in work.columns:
        raise ValueError("Expected 'carer_id' column.")

    # Make sure numeric helper exists
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
    # helpful reference lines
    fig.add_hline(y=g["high_crit"].quantile(0.8), line_dash="dash", annotation_text="80th %ile High/Critical")
    fig.add_vline(x=g["incidents"].quantile(0.8), line_dash="dash", annotation_text="80th %ile Incidents")
    fig.update_yaxes(tickformat=".0%")
    return fig


# ---------------------------------------
# 5) Correlation analysis (numeric)
# ---------------------------------------

def correlation_analysis(df: pd.DataFrame, include: Optional[list] = None) -> go.Figure:
    """
    Heatmap of correlations for numeric columns (optionally a whitelist).
    """
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
# 6) Clustering analysis (KMeans)
# ---------------------------------------

def clustering_analysis(features_df: pd.DataFrame, k: int = 4) -> Tuple[go.Figure, pd.Series]:
    """
    KMeans on engineered features. Returns a 2D projection scatter (PCA) and labels.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = features_df.to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    # quick 2D projection for display
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)
    df_plot = pd.DataFrame({"pc1": XY[:,0], "pc2": XY[:,1], "cluster": labels})

    fig = px.scatter(df_plot, x="pc1", y="pc2", color="cluster", title=f"KMeans Clusters (k={k})")
    return fig, pd.Series(labels, index=features_df.index, name="cluster")


# ---------------------------------------
# 7) Predictive models comparison
# ---------------------------------------

def predictive_models_comparison(
    df: pd.DataFrame,
    target: str = "reportable_bin",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train a couple of baseline models and return a dictionary shaped for your
    Enhanced Confusion Matrix UI:
        {
          "Model Name": {
              "model": fitted_model,
              "accuracy": float,
              "y_test": pd.Series,
              "predictions": np.ndarray,
              "probabilities": np.ndarray or None
          }, ...
        }
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Build features
    X, feature_names, features_df = create_comprehensive_features(df)

    # Choose/derive target
    y = df[target].copy() if target in df.columns else None
    if y is None:
        # Fallback: high/critical vs not
        y = (df.get("severity_numeric", pd.Series([2]*len(df))) >= 3).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results: Dict[str, Dict[str, Any]] = {}

    # Random Forest
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

    # Logistic Regression (saga for robustness)
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
    """
    Summarise risk by incident_type:
      - total incidents
      - % High/Critical
      - medical attention rate
    """
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
