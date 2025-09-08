# ml_helpers.py
# Utilities and analytics helpers for the NDIS dashboard.
# - Re-exports feature builders from utils.ndis_enhanced_prep (if present)
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
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class LeakageGuard(BaseEstimator, TransformerMixin):
    """
    Keeps only the columns chosen during fit (self.columns_out_).
    During inference it can accept either a named DataFrame or a raw ndarray.
    """
    def __init__(self, columns_out=None):
        self.columns_out_ = list(columns_out) if columns_out is not None else []

    def fit(self, X, y=None):
        # If X is a DataFrame, remember the columns; otherwise assume current list is correct
        if isinstance(X, pd.DataFrame):
            if not self.columns_out_:
                # default: keep numeric columns if nothing was provided
                self.columns_out_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X):
        # If a DataFrame with names is provided, use name-based selection
        if isinstance(X, pd.DataFrame):
            keep = [c for c in self.columns_out_ if c in X.columns]
            return X[keep].to_numpy(dtype=float)

        # Otherwise, try to align by position if the widths match
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == len(self.columns_out_):
            return arr.astype(float)

        # Fallback: nothing reliable to keep → empty (rare)
        return np.empty((arr.shape[0], 0), dtype=float)

# ---------------------------------------
# Optional enhanced prep from utils/ (no __init__.py required)
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
    """Thin wrapper; if utils version is missing, return df unchanged."""
    if _prepare_ndis_data is not None:
        return _prepare_ndis_data(df)
    return df.copy()


# ---------------------------------------
# Internal helpers (de-duplicated)
# ---------------------------------------
def ensure_incident_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a usable 'incident_datetime' from incident_date/incident_time if needed."""
    d = df.copy()
    if "incident_datetime" in d.columns:
        d["incident_datetime"] = pd.to_datetime(d["incident_datetime"], errors="coerce")
        return d

    if "incident_date" in d.columns:
        d["incident_datetime"] = pd.to_datetime(d["incident_date"], errors="coerce")
    else:
        d["incident_datetime"] = pd.NaT

    if "incident_time" in d.columns:
        t = pd.to_datetime(d["incident_time"], errors="coerce")
        mask = d["incident_datetime"].notna() & t.notna()
        d.loc[mask, "incident_datetime"] = pd.to_datetime(
            d.loc[mask, "incident_datetime"].dt.date.astype(str) + " " + t.dt.time.astype(str),
            errors="coerce"
        )
    return d


def _ensure_datetime(df: pd.DataFrame, date_col: str = "incident_datetime") -> pd.DataFrame:
    """Back-compat wrapper used by a few functions. Uses ensure_incident_datetime()."""
    if date_col == "incident_datetime":
        return ensure_incident_datetime(df)
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if "incident_datetime" not in out.columns:
        out["incident_datetime"] = out[date_col]
    return out


def _monthly_counts(df: pd.DataFrame, date_col: str = "incident_date", freq: str = "MS") -> pd.Series:
    """Aggregate 1-per-row incidents to monthly counts."""
    if date_col not in df.columns:
        if "incident_datetime" in df.columns:
            date_col = "incident_datetime"
        else:
            return pd.Series(dtype=float)
    dt = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dt.empty:
        return pd.Series(dtype=float)
    y = pd.Series(1, index=dt).resample(freq).sum().astype(int)
    return y.asfreq(freq, fill_value=0)


def _numeric_features(features_df: pd.DataFrame, sample: Optional[int] = None, random_state: int = 42):
    """Select, clean, and (optionally) downsample numeric features. Returns (X, index)."""
    X = (
        features_df.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if sample and len(X) > sample:
        X = X.sample(sample, random_state=random_state)
    return X, X.index


def _make_color_map(labels: np.ndarray) -> Dict[str, str]:
    """Stable discrete color map for cluster labels 0..k-1."""
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe
    labs = [str(x) for x in sorted(pd.Series(labels).unique(), key=lambda v: int(v))]
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labs)}


# ---------------------------------------
# Fallback feature builder (leak-safe)
# ---------------------------------------
def _fallback_features(df: pd.DataFrame):
    """
    Minimal, leak-safe features if utils.create_comprehensive_features is unavailable.
    Uses cumcount() within participant/carer ordered by incident_datetime (past-only).
    Returns (X, feature_names, features_df).
    """
    d = ensure_incident_datetime(df)

    # Work on a copy with stable ordering
    work = d.reset_index().rename(columns={"index": "__orig_idx"}).copy()
    # order each group by time for proper cumcount
    work["__order"] = pd.to_datetime(work["incident_datetime"], errors="coerce")
    work["__order"] = work["__order"].fillna(pd.Timestamp(0))

    # participant cumulative history (past incidents before this row)
    if "participant_id" in work.columns:
        work = work.sort_values(["participant_id", "__order", "__orig_idx"])
        work["participant_incident_count"] = work.groupby("participant_id").cumcount()
    else:
        work["participant_incident_count"] = 0

    # carer cumulative history
    if "carer_id" in work.columns:
        work = work.sort_values(["carer_id", "__order", "__orig_idx"])
        work["carer_incident_count"] = work.groupby("carer_id").cumcount()
    else:
        work["carer_incident_count"] = 0

    # restore original order
    work = work.sort_values("__orig_idx")

    base = pd.DataFrame(index=work.index)
    base["hour"] = pd.to_datetime(work["incident_datetime"], errors="coerce").dt.hour.fillna(0).astype(int)

    dow = pd.to_datetime(work["incident_datetime"], errors="coerce").dt.dayofweek
    base["is_weekend"] = (dow >= 5).astype(int)

    loc = work.get("location", pd.Series([""], index=work.index)).astype(str).str.lower()
    base["is_kitchen"] = loc.str.contains("kitchen", na=False).astype(int)
    base["is_bathroom"] = loc.str.contains("bath|toilet|washroom|restroom", regex=True, na=False).astype(int)

    base["participant_incident_count"] = work["participant_incident_count"].astype(int)
    base["carer_incident_count"] = work["carer_incident_count"].astype(int)

    # coarse location risk proxy
    base["location_risk_score"] = (
        3 * base["is_kitchen"] + 3 * base["is_bathroom"] + 1
    ).clip(lower=1)

    base = base.fillna(0).astype(float)
    return base.values, list(base.columns), base


def create_comprehensive_features(df: pd.DataFrame):
    """Use utils version if present, otherwise fall back."""
    if _create_comprehensive_features is not None:
        return _create_comprehensive_features(df)
    return _fallback_features(df)


# ---------------------------------------
# 1) Incident volume forecasting (+ alias)
# ---------------------------------------
def incident_volume_forecasting(
    df: pd.DataFrame,
    horizon: Optional[int] = None,
    horizon_months: Optional[int] = None,
    months: Optional[int] = None,
    n_periods: Optional[int] = None,
    date_col: str = "incident_date",
    season_length: int = 12,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Improved forecast with log transformation and stability fixes.
    Returns (plotly Figure, DataFrame with columns: actual, forecast, lower, upper).
    """
    # Resolve horizon robustly
    resolved = next((v for v in [horizon, horizon_months, months, n_periods] if v is not None), 6)
    try:
        H = int(resolved)
    except Exception:
        H = 6

    y = _monthly_counts(df, date_col=date_col, freq="MS")
    if y.empty:
        fig = go.Figure()
        fig.update_layout(title="Incident Forecast: no date data", xaxis_title="Month", yaxis_title="Incidents")
        return fig, pd.DataFrame(columns=["actual", "forecast", "lower", "upper"])

    # Ensure minimum data length
    if len(y) < 3:
        # Not enough data for SARIMAX, use simple mean forecast
        mean_val = y.mean()
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
        forecast = pd.Series([mean_val] * H, index=idx, name="forecast")
        std_val = max(y.std(), np.sqrt(mean_val))  # Ensure positive std
        conf_int = pd.DataFrame({
            "lower": np.maximum(0, forecast.values - 1.96 * std_val),
            "upper": forecast.values + 1.96 * std_val,
        }, index=idx)
        
        out = pd.DataFrame({"actual": y})
        fc_df = pd.concat([forecast, conf_int], axis=1)
        merged = out.join(fc_df, how="outer")
        merged.index.name = "date"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Forecast"))
        fig.add_trace(
            go.Scatter(
                x=list(conf_int.index) + list(conf_int.index[::-1]),
                y=list(conf_int["upper"].values) + list(conf_int["lower"].values[::-1]),
                fill="toself", opacity=0.2, line=dict(width=0), name="95% interval",
            )
        )
        fig.update_layout(title="Monthly Incident Forecast (Simple Mean)", xaxis_title="Month", yaxis_title="Incidents")
        return fig, merged

    use_sarimax = False

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        # Apply log transformation to prevent negative predictions
        y_positive = np.maximum(y, 0.1)  # Ensure positive values
        y_log = np.log1p(y_positive)  # log1p handles near-zero values better
        
        # More conservative seasonal parameters
        if len(y) >= 2 * season_length:
            # Full seasonal model only with sufficient data
            seasonal_order = (1, 1, 1, season_length)
            order = (1, 1, 1)
        elif len(y) >= season_length:
            # Simpler seasonal model
            seasonal_order = (0, 1, 1, season_length)
            order = (0, 1, 1)
        else:
            # No seasonality
            seasonal_order = (0, 0, 0, 0)
            order = (1, 1, 1)
        
        # Fit model with conservative settings
        model = SARIMAX(
            y_log, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
            concentrate_scale=True  # Better numerical stability
        )
        
        # Fit with better optimization settings
        res = model.fit(
            disp=False, 
            maxiter=200,
            method='lbfgs',  # More stable than default
            optim_score='harvey'  # Alternative scoring
        )
        
        # Get forecast in log space
        fc = res.get_forecast(steps=H)
        forecast_log = fc.predicted_mean
        conf_log = fc.conf_int(alpha=0.05)  # 95% interval
        
        # Transform back to original space
        forecast = np.expm1(forecast_log).rename("forecast")
        conf_int = pd.DataFrame({
            "lower": np.maximum(0, np.expm1(conf_log.iloc[:, 0]).values),  # Ensure non-negative
            "upper": np.expm1(conf_log.iloc[:, 1]).values
        }, index=forecast.index)
        
        # Additional sanity checks
        max_historical = y.max()
        forecast = np.minimum(forecast, max_historical * 3)  # Cap extreme forecasts
        conf_int["upper"] = np.minimum(conf_int["upper"], max_historical * 4)
        
        use_sarimax = True
        
    except Exception as e:
        print(f"SARIMAX failed: {e}")
        # Enhanced seasonal naive fallback
        try:
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
            
            if len(y) >= season_length:
                # Use seasonal pattern with trend adjustment
                last_season = y[-season_length:].values
                # Simple trend calculation
                if len(y) >= 2 * season_length:
                    recent_avg = y[-season_length:].mean()
                    older_avg = y[-2*season_length:-season_length].mean()
                    trend = max(-0.1, min(0.1, (recent_avg - older_avg) / older_avg))  # Cap trend
                else:
                    trend = 0
                
                # Apply trend to seasonal pattern
                base_vals = last_season * (1 + trend)
                vals = np.tile(base_vals, int(np.ceil(H / season_length)))[:H]
            else:
                # Simple trend from recent data
                if len(y) >= 3:
                    trend = (y.iloc[-1] - y.iloc[-3]) / 2
                    trend = max(-y.iloc[-1] * 0.1, min(y.iloc[-1] * 0.1, trend))  # Cap trend
                else:
                    trend = 0
                vals = [max(0.1, y.iloc[-1] + trend * i) for i in range(1, H + 1)]
            
            forecast = pd.Series(vals, index=idx, name="forecast")
            
            # Confidence intervals based on historical volatility
            historical_std = y.std()
            expanding_std = np.array([historical_std * np.sqrt(i) for i in range(1, H + 1)])
            
            conf_int = pd.DataFrame({
                "lower": np.maximum(0, forecast.values - 1.96 * expanding_std),
                "upper": forecast.values + 1.96 * expanding_std,
            }, index=idx)
            
        except Exception:
            # Ultimate fallback: flat forecast
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
            last_val = max(0.1, y.iloc[-1])
            forecast = pd.Series([last_val] * H, index=idx, name="forecast")
            std_val = max(y.std(), np.sqrt(last_val))
            conf_int = pd.DataFrame({
                "lower": np.maximum(0, forecast.values - 1.96 * std_val),
                "upper": forecast.values + 1.96 * std_val,
            }, index=idx)

    # Output frame
    out = pd.DataFrame({"actual": y})
    fc_df = pd.concat([forecast, conf_int], axis=1)
    merged = out.join(fc_df, how="outer")
    merged.index.name = "date"

    # Enhanced figure with better styling
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values, 
        mode="lines+markers", 
        name="Actual",
        line=dict(color='steelblue', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values, 
        mode="lines+markers", 
        name="Forecast",
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(conf_int.index) + list(conf_int.index[::-1]),
        y=list(conf_int["upper"].values) + list(conf_int["lower"].values[::-1]),
        fill="toself", 
        opacity=0.2, 
        line=dict(width=0), 
        name="95% interval",
        fillcolor='orange'
    ))
    
    model_type = 'SARIMAX (Log-transformed)' if use_sarimax else 'Enhanced Seasonal Naive'
    fig.update_layout(
        title=f"Monthly Incident Forecast ({model_type})",
        xaxis_title="Month", 
        yaxis_title="Incidents", 
        hovermode="x unified",
        showlegend=True,
        yaxis=dict(rangemode='tozero')  # Ensure y-axis starts at 0
    )
    
    return fig, merged


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
        xaxis_title="Hour", yaxis_title="Day of Week",
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

    df = _ensure_datetime(df, date_col=date_col)
    work = df.copy()

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
        ts, x="incident_datetime", y="count", color="_cat",
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
# 5) Correlation heatmap (now resizable)
# ---------------------------------------
def correlation_analysis(df: pd.DataFrame, include: Optional[list] = None, height: int = 900) -> go.Figure:
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
    fig.update_layout(
        title="Correlation Matrix",
        coloraxis=dict(colorscale="RdBu", cmin=-1, cmax=1),
        height=height
    )
    return fig


# ---------------------------------------
# 6) Clustering analysis (KMeans + PCA) — 2D
# ---------------------------------------
def clustering_analysis(df_or_features: pd.DataFrame, k: int = 4):
    """
    KMeans on engineered features. Accepts either the original df or a prebuilt feature frame.
    Returns (fig_2d_plotly, labels_series) and stores the palette in fig.layout.meta['cluster_color_map'].
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    use_df = df_or_features.copy()
    numeric_cols = use_df.select_dtypes(include=[np.number]).columns.tolist()
    looks_like_raw = ("incident_id" in use_df.columns) or ("incident_date" in use_df.columns) or (len(numeric_cols) < 2)

    if looks_like_raw:
        X, feature_names, features_df = create_comprehensive_features(use_df)
        use_df = pd.DataFrame(X, columns=feature_names, index=features_df.index)
    else:
        use_df = use_df.select_dtypes(include=[np.number]).copy()

    X = use_df.to_numpy(dtype=float)
    km = KMeans(n_clusters=int(k), n_init=10, random_state=42)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)
    df_plot = pd.DataFrame({"pc1": XY[:, 0], "pc2": XY[:, 1], "cluster": labels})
    df_plot["cluster_str"] = df_plot["cluster"].astype(str)

    base_palette = px.colors.qualitative.Safe + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    uniq = sorted(df_plot["cluster_str"].unique(), key=lambda s: int(s))
    discrete_map = {cid: base_palette[i % len(base_palette)] for i, cid in enumerate(uniq)}

    fig = px.scatter(
        df_plot, x="pc1", y="pc2",
        color="cluster_str",
        color_discrete_map=discrete_map,
        title=f"KMeans Clusters (k={k})"
    )
    fig.update_layout(meta={"cluster_color_map": discrete_map})

    return fig, pd.Series(labels, index=use_df.index, name="cluster")


# ---------------------------------------
# 6b) Clustering — 3D (PCA) with color matching to 2D
# ---------------------------------------
def plot_3d_clusters(
    features_df: pd.DataFrame,
    k: int = 4,
    sample: int = 2000,
    color_map: Optional[Dict[str, str]] = None
):
    """PCA->3D + KMeans clustering. Returns (fig, labels, df3d)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    X = (
        features_df
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if len(X) > sample:
        X = X.sample(sample, random_state=42)

    Xs = StandardScaler().fit_transform(X.values)
    Z = PCA(n_components=3, random_state=42).fit_transform(Xs)
    labels = KMeans(n_clusters=int(k), n_init=10, random_state=42).fit_predict(Xs)

    df3d = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "PC3": Z[:, 2], "cluster": labels})
    df3d["cluster_str"] = df3d["cluster"].astype(str)

    fig = px.scatter_3d(
        df3d, x="PC1", y="PC2", z="PC3",
        color="cluster_str", opacity=0.8,
        color_discrete_map=(color_map or {})
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Cluster")
    return fig, labels, df3d


# ---------------------------------------
# 7) Leak guard + predictive models comparison (baselines, leak-safe)
# ---------------------------------------

class EnhancedLeakageGuard(BaseEstimator, TransformerMixin):
    """
    Enhanced leakage detection with stricter rules and validation.
    """
    def __init__(self, 
                 drop_names=None, 
                 drop_patterns=None, 
                 corr_threshold=0.85,  # Lower threshold
                 variance_threshold=0.01,  # Remove near-constant features
                 max_unique_ratio=0.95):  # Remove features with too many unique values
        self.drop_names = set(drop_names or [])
        self.drop_patterns = [p.lower() for p in (drop_patterns or [])]
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.max_unique_ratio = max_unique_ratio
        
        self.columns_in_ = None
        self.columns_to_drop_ = []
        self.columns_out_ = []
        self.leakage_report_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        X.columns = [str(c) for c in X.columns]
        self.columns_in_ = list(X.columns)
        
        drops = set()
        self.leakage_report_ = {
            'explicit_drops': [],
            'pattern_drops': [],
            'correlation_drops': [],
            'variance_drops': [],
            'unique_ratio_drops': []
        }
        
        # 1. Explicit name drops
        for c in X.columns:
            if c in self.drop_names:
                drops.add(c)
                self.leakage_report_['explicit_drops'].append(c)
        
        # 2. Pattern-based drops (more aggressive)
        for c in X.columns:
            c_lower = c.lower()
            for pattern in self.drop_patterns:
                if pattern in c_lower:
                    drops.add(c)
                    self.leakage_report_['pattern_drops'].append((c, pattern))
                    break
        
        # 3. High correlation with target (if provided)
        if y is not None:
            y_series = pd.Series(y).astype(float)
            for c in X.columns:
                if c in drops:
                    continue
                try:
                    s = pd.to_numeric(X[c], errors='coerce')
                    if s.isna().all():
                        drops.add(c)
                        continue
                    
                    # Remove constant features
                    if s.std(ddof=0) <= self.variance_threshold:
                        drops.add(c)
                        self.leakage_report_['variance_drops'].append(c)
                        continue
                    
                    # Check correlation with target
                    corr = abs(s.corr(y_series))
                    if np.isfinite(corr) and corr >= self.corr_threshold:
                        drops.add(c)
                        self.leakage_report_['correlation_drops'].append((c, corr))
                        
                except Exception:
                    drops.add(c)
        
        # 4. Features with too many unique values (potential IDs)
        for c in X.columns:
            if c in drops:
                continue
            try:
                unique_ratio = X[c].nunique() / len(X)
                if unique_ratio >= self.max_unique_ratio:
                    drops.add(c)
                    self.leakage_report_['unique_ratio_drops'].append((c, unique_ratio))
            except Exception:
                pass
        
        self.columns_to_drop_ = sorted(drops)
        self.columns_out_ = [c for c in X.columns if c not in drops]
        
        # Warning if too many features dropped
        if len(self.columns_out_) < 3:
            warnings.warn(f"Only {len(self.columns_out_)} features remaining after leakage guard. "
                         "Model may be undertrained.")
        
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = [str(c) for c in X.columns]
        keep = [c for c in self.columns_out_ if c in X.columns]
        
        if len(keep) == 0:
            # Emergency fallback - create dummy features
            warnings.warn("No features passed leakage guard. Creating dummy features.")
            return np.ones((len(X), 1), dtype=float)
        
        return X[keep].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_out_, dtype=object)
    
    def print_leakage_report(self):
        """Print detailed report of what was dropped and why."""
        print("=== LEAKAGE GUARD REPORT ===")
        print(f"Input features: {len(self.columns_in_)}")
        print(f"Output features: {len(self.columns_out_)}")
        print(f"Dropped features: {len(self.columns_to_drop_)}")
        print()
        
        for category, items in self.leakage_report_.items():
            if items:
                print(f"{category.replace('_', ' ').title()}:")
                for item in items:
                    if isinstance(item, tuple):
                        print(f"  - {item[0]} (value: {item[1]:.3f})")
                    else:
                        print(f"  - {item}")
                print()


def predictive_models_comparison(
    df: pd.DataFrame,
    target: str = "reportable_bin",
    test_size: float = 0.25,
    random_state: int = 42,
    extra_leaky_features: Optional[List[str]] = None,
    split_strategy: str = "time",
    time_col: Optional[str] = "incident_datetime",
    group_col: Optional[str] = None,
    leak_corr_threshold: float = 0.80,  # Lowered threshold
    leak_name_patterns: Optional[List[str]] = None,
    validate_with_cv: bool = True
) -> Dict[str, Any]:
    """
    Enhanced model comparison with stricter leakage detection and cross-validation.
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
    
    # Import your feature creation function
    # Assuming this is available from your ml_helpers
    out = create_comprehensive_features(df)
    if isinstance(out, tuple) and len(out) == 3:
        _, _, features_df = out
    elif isinstance(out, pd.DataFrame):
        features_df = out
    else:
        features_df = pd.DataFrame(out)
    features_df = features_df.copy()

    # Target preparation
    if target in df.columns:
        y = df.loc[features_df.index, target].copy()
    else:
        y = (df.get("severity_numeric", pd.Series([2]*len(df))).loc[features_df.index] >= 3).astype(int)

    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Class balance: {y.mean():.1%} positive class")

    # Enhanced leaky feature detection
    comprehensive_leaky_features = set([
        target, "reportable", "reportable_bin",
        "severity", "severity_numeric", "severity_level",
        "medical_attention_required", "treatment_required",
        "notified_to_commission", "reporting_timeframe",
        "investigation_required", "incident_resolved",
        "actions_documented", "medical_outcome",
        "report_delay_hours", "within_24h",
        "outcome", "result", "resolution", "status",
        "follow_up", "action_taken", "investigation_outcome"
    ])
    
    if extra_leaky_features:
        comprehensive_leaky_features.update(extra_leaky_features)

    enhanced_leak_patterns = leak_name_patterns or [
        "report", "reportable", "notify", "notified",
        "investigat", "severity", "medical", "outcome",
        "resolution", "timeframe", "delay", "compliance",
        "follow", "action", "result", "status", "closed",
        "_id", "id_", "index", "row"  # Potential ID columns
    ]

    # Data splitting (same as before)
    df_dt = ensure_incident_datetime(df)
    if time_col not in df_dt.columns:
        time_col = "incident_date" if "incident_date" in df_dt.columns else None

    if split_strategy == "time" and time_col:
        dt = pd.to_datetime(df_dt.loc[features_df.index, time_col], errors="coerce")
        cutoff = dt.quantile(0.75)  # Use 75% for more training data
        mask_train = dt <= cutoff
        mask_test = dt > cutoff
        X_train_df, X_test_df = features_df[mask_train], features_df[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
    elif split_strategy == "group" and group_col and group_col in df.columns:
        groups = df.loc[features_df.index, group_col].astype(str).fillna("NA")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(features_df, y, groups))
        X_train_df, X_test_df = features_df.iloc[train_idx], features_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        strat = y if y.nunique() > 1 else None
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            features_df, y, test_size=test_size, random_state=random_state, stratify=strat
        )

    print(f"Training set size: {len(X_train_df)}")
    print(f"Test set size: {len(X_test_df)}")

    # Enhanced leakage guard
    guard = LeakageGuard(
        drop_names=comprehensive_leaky_features,
        drop_patterns=enhanced_leak_patterns,
        corr_threshold=leak_corr_threshold,
        variance_threshold=0.01,
        max_unique_ratio=0.90
    )

    # Model pipelines with more conservative settings
    rf_pipe = Pipeline([
        ("guard", guard),
        ("model", RandomForestClassifier(
            n_estimators=100,  # Reduced to prevent overfitting
            max_depth=5,       # Limit depth
            min_samples_split=10,  # Require more samples to split
            min_samples_leaf=5,    # Require more samples in leaf
            random_state=random_state,
            class_weight='balanced'  # Handle imbalanced data
        )),
    ])

    lr_pipe = Pipeline([
        ("guard", guard),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            solver="liblinear",  # Better for small datasets
            C=1.0,              # Regularization
            random_state=random_state,
            class_weight='balanced'
        )),
    ])

    results = {}

    # Fit models and get results
    for name, pipe in [("RandomForest", rf_pipe), ("LogisticRegression", lr_pipe)]:
        print(f"\n=== Training {name} ===")
        
        # Fit model
        pipe.fit(X_train_df, y_train)
        
        # Print leakage report
        pipe.named_steps["guard"].print_leakage_report()
        
        # Predictions
        pred = pipe.predict(X_test_df)
        proba = pipe.predict_proba(X_test_df) if hasattr(pipe, "predict_proba") else None
        
        # Test set performance
        test_accuracy = (pred == y_test).mean()
        
        # Cross-validation on training set
        cv_scores = None
        if validate_with_cv and len(X_train_df) > 20:
            try:
                cv = StratifiedKFold(n_splits=min(5, len(X_train_df)//10), shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(pipe, X_train_df, y_train, cv=cv, scoring='accuracy')
                print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            except Exception as e:
                print(f"CV failed: {e}")
        
        # Feature importance (for Random Forest)
        feature_importance = None
        if name == "RandomForest":
            try:
                feature_names = pipe.named_steps["guard"].get_feature_names_out()
                importances = pipe.named_steps["model"].feature_importances_
                feature_importance = dict(zip(feature_names, importances))
                
                # Show top features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                print("Top 5 Important Features:")
                for feat, imp in sorted_features[:5]:
                    print(f"  {feat}: {imp:.3f}")
            except Exception as e:
                print(f"Feature importance extraction failed: {e}")
        
        # Store results
        results[name] = {
            "model": pipe,
            "accuracy": float(test_accuracy),
            "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
            "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
            "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
            "y_test": y_test,
            "predictions": pred,
            "probabilities": proba,
            "feature_names": list(pipe.named_steps["guard"].get_feature_names_out()),
            "feature_importance": feature_importance,
            "leakage_report": pipe.named_steps["guard"].leakage_report_,
            "n_features_used": len(pipe.named_steps["guard"].columns_out_)
        }
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Features used: {len(pipe.named_steps['guard'].columns_out_)}")
        
        # Warning for perfect performance
        if test_accuracy >= 0.99:
            print("⚠️  WARNING: Perfect/near-perfect performance detected!")
            print("   This likely indicates data leakage or overfitting.")
            print("   Consider:")
            print("   - Reviewing feature engineering")
            print("   - Using time-based splits")
            print("   - Adding more regularization")

    return results


# Example usage with your data
def validate_models_thoroughly(df, target="reportable_bin"):
    """
    Run comprehensive model validation to detect leakage issues.
    """
    print("=== COMPREHENSIVE MODEL VALIDATION ===")
    
    results = improved_predictive_models_comparison(
        df, 
        target=target,
        leak_corr_threshold=0.75,  # Stricter threshold
        validate_with_cv=True,
        split_strategy="time"  # Use time-based split
    )
    
    print("\n=== VALIDATION SUMMARY ===")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {result['accuracy']:.3f}")
        if result['cv_mean'] is not None:
            print(f"  CV Accuracy: {result['cv_mean']:.3f} (+/- {result['cv_std']*2:.3f})")
        print(f"  Features Used: {result['n_features_used']}")
        
        # Check for leakage indicators
        if result['accuracy'] >= 0.99:
            print("  🚨 POTENTIAL LEAKAGE: Perfect performance")
        elif result['cv_mean'] and abs(result['accuracy'] - result['cv_mean']) > 0.1:
            print("  ⚠️  Large test/CV gap - possible overfitting")
        elif result['n_features_used'] < 5:
            print("  ⚠️  Very few features - model may be undertrained")
        else:
            print("  ✅ Performance looks reasonable")
    
    return results

# ---------------------------------------
# 8) Incident type risk profiling
# ---------------------------------------
def incident_type_risk_profiling(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
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
    return g, fig


# Friendly alias to match possible imports:
profile_incident_type_risk = incident_type_risk_profiling


# (Optional) Location risk profiling if your dashboard calls it
def profile_location_risk(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[go.Figure]]:
    """Summarise risk by location: volume, % High/Critical, medical attention rate."""
    work = df.copy()
    if "location" not in work.columns:
        work["location"] = "Unknown"

    if "severity_numeric" not in work.columns:
        sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        work["severity_numeric"] = work["severity"].map(sev_map).fillna(2).astype(int)

    if "medical_attention_required_bin" not in work.columns:
        mar = work.get("medical_attention_required", pd.Series([0]*len(work)))
        work["medical_attention_required_bin"] = mar.astype(str).str.lower().isin(["yes","true","1"]).astype(int)

    g = work.groupby("location").agg(
        incidents=("incident_id","count"),
        high_crit_rate=("severity_numeric", lambda s: (s >= 3).mean()),
        medical_rate=("medical_attention_required_bin","mean"),
    ).reset_index().sort_values("incidents", ascending=False)

    fig = px.bar(
        g.head(25), x="location", y="incidents",
        hover_data={"high_crit_rate":":.1%", "medical_rate":":.1%"},
        title="Location Risk Profile (Top 25)"
    )
    fig.update_layout(xaxis_title="Location", yaxis_title="Incidents")
    return g, fig


# =====================================================================
# =============== ENHANCED ANALYTICS (requested set) ==================
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
    participant_data['incident_date'] = pd.to_datetime(
        participant_data.get('incident_date', participant_data.get('incident_datetime')),
        errors='coerce'
    )
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
# 12) Predictive Risk Scoring System (robust to pipelines/name-based selectors)
def create_predictive_risk_scoring(
    df: pd.DataFrame,
    trained_models: Dict[str, Dict[str, Any]],
    feature_names: List[str],
):
    """
    Return calculate_risk_score(scenario) using the best model.
    Builds input in the schema the model expects (named DataFrame for pipelines,
    or width-matched ndarray otherwise) to avoid (1, 0) errors.
    """
    if not trained_models:
        return None

    # pick best by accuracy
    best_key = max(trained_models, key=lambda k: trained_models[k].get("accuracy", 0))
    best_blob = trained_models[best_key]
    model = best_blob["model"]

    # --- Figure out expected schema (names or just a count) ---
    expected_cols: List[str] = []
    expected_n = None

    # If it's a Pipeline, try to get names from a step
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            for name, step in model.steps:
                # custom selector / guard
                if hasattr(step, "columns_out_") and isinstance(step.columns_out_, (list, tuple)):
                    expected_cols = list(step.columns_out_)
                    break
                # many sklearn transformers expose feature names
                if hasattr(step, "get_feature_names_out"):
                    try:
                        expected_cols = list(step.get_feature_names_out())
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    # Fall back to names recorded during training, then to provided feature_names
    if not expected_cols:
        expected_cols = list(best_blob.get("feature_names", []) or feature_names)

    # Fall back to a feature count if no names
    try:
        expected_n = getattr(model, "n_features_in_", None)
    except Exception:
        expected_n = None
    if expected_n is None and hasattr(model, "steps"):
        # last step might have it
        try:
            expected_n = getattr(model.steps[-1][1], "n_features_in_", None)
        except Exception:
            expected_n = None

    # If we only know the count, synthesize placeholder names so we can build a DF
    if expected_n is not None and not expected_cols:
        expected_cols = [f"f_{i}" for i in range(int(expected_n))]

    # If we still have nothing, return a safe no-op scorer
    if not expected_cols and (expected_n is None or expected_n == 0):
        def _noop(_scenario):
            return {"risk_score": 0.0, "risk_level": "LOW", "confidence": 0.0, "model_used": best_key}
        return _noop

    # --- Build the callable scorer ---
    def calculate_risk_score(scenario: Dict[str, Any]) -> Dict[str, Any]:
        # Scenario → demo features
        mapping = {
            "hour": scenario.get("hour", 12),
            "is_weekend": 1 if scenario.get("day_type") == "weekend" else 0,
            "is_kitchen": 1 if "kitchen" in str(scenario.get("location", "")).lower() else 0,
            "is_bathroom": 1 if any(k in str(scenario.get("location", "")).lower()
                                     for k in ["bathroom", "toilet", "washroom", "restroom"]) else 0,
            "participant_incident_count": scenario.get("participant_history", 1),
            "carer_incident_count": scenario.get("carer_history", 1),
            "location_risk_score": scenario.get("location_risk", 2),
        }

        # Prefer a named row (DataFrame) so pipeline steps can select by column name
        if expected_cols:
            row = {col: float(mapping.get(col, 0.0)) for col in expected_cols}
            X_one = pd.DataFrame([row], columns=expected_cols)
        else:
            # Width-only fallback
            n = int(expected_n) if expected_n is not None else len(feature_names)
            vec = np.zeros(n, dtype=float)
            train_names = list(best_blob.get("feature_names", []) or feature_names)
            for i, fn in enumerate(train_names[:n]):
                vec[i] = float(mapping.get(fn, 0.0))
            X_one = vec.reshape(1, -1)

        # Predict (try DF first; if a step rejects DF, retry with ndarray)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_one)[0]
                score = float(np.max(proba))
            else:
                pred = model.predict(X_one)[0]
                score = float(pred) / 3.0
        except Exception:
            X_np = X_one.to_numpy(dtype=float) if isinstance(X_one, pd.DataFrame) else np.asarray(X_one, dtype=float)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_np)[0]
                score = float(np.max(proba))
            else:
                pred = model.predict(X_np)[0]
                score = float(pred) / 3.0

        level = "HIGH" if score > 0.7 else ("MEDIUM" if score > 0.4 else "LOW")
        return {"risk_score": score, "risk_level": level, "confidence": score, "model_used": best_key}

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


# 15) Streamlit integration helpers (optional)
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
