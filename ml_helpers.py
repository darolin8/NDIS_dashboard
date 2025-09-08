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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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


def _fallback_features(df: pd.DataFrame):
    """
    Minimal feature builder used if utils.create_comprehensive_features is unavailable.
    Matches the names your live risk-scorer expects so the UI keeps working.
    Returns (X, feature_names, features_df).
    """
    d = ensure_incident_datetime(df)

    base = pd.DataFrame(index=d.index)
    base["hour"] = d["incident_datetime"].dt.hour.fillna(0).astype(int)

    dow = d["incident_datetime"].dt.dayofweek
    base["is_weekend"] = (dow >= 5).astype(int)

    loc = d.get("location", pd.Series([""], index=d.index)).astype(str).str.lower()
    base["is_kitchen"] = loc.str.contains("kitchen", na=False).astype(int)
    base["is_bathroom"] = loc.str.contains("bath|toilet|washroom|restroom", regex=True, na=False).astype(int)

    # history counts (0 if ids missing)
    if "participant_id" in d.columns and "incident_id" in d.columns:
        base["participant_incident_count"] = d.groupby("participant_id")["incident_id"].transform("count")
    else:
        base["participant_incident_count"] = 0
    if "carer_id" in d.columns and "incident_id" in d.columns:
        base["carer_incident_count"] = d.groupby("carer_id")["incident_id"].transform("count")
    else:
        base["carer_incident_count"] = 0

    # coarse location risk proxy
    base["location_risk_score"] = (
        3*base["is_kitchen"] + 3*base["is_bathroom"] + 1
    ).clip(lower=1)

    base = base.fillna(0).astype(float)
    return base.values, list(base.columns), base


def create_comprehensive_features(df: pd.DataFrame):
    """
    Wrapper: use utils version if present, otherwise fall back to a minimal, safe feature set.
    """
    if _create_comprehensive_features is not None:
        return _create_comprehensive_features(df)
    return _fallback_features(df)


# ---------------------------------------
# Re-export feature preparation
# ---------------------------------------
#try:
    #from utils.ndis_enhanced_prep import (
        #prepare_ndis_data as _prepare_ndis_data,
        #create_comprehensive_features as _create_comprehensive_features,
    #)
#except Exception:
    #_prepare_ndis_data = None
    #_create_comprehensive_features = None


#def prepare_ndis_data(df: pd.DataFrame) -> pd.DataFrame:
    #"""Thin wrapper so legacy imports keep working."""
    #if _prepare_ndis_data is None:
        #raise ImportError("utils.ndis_enhanced_prep.prepare_ndis_data not found. Ensure utils/ exists with __init__.py.")
    #return _prepare_ndis_data(df)


#def create_comprehensive_features(df: pd.DataFrame):
    #"""Thin wrapper so legacy imports keep working."""
  #  if _create_comprehensive_features is None:
    #    raise ImportError("utils.ndis_enhanced_prep.create_comprehensive_features not found. Ensure utils/ exists with __init__.py.")
    #return _create_comprehensive_features(df)


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
    """
    Back-compat wrapper used by a few functions. Uses ensure_incident_datetime().
    """
    if date_col == "incident_datetime":
        return ensure_incident_datetime(df)
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if "incident_datetime" not in out.columns:
        out["incident_datetime"] = out[date_col]
    return out


def _monthly_counts(df: pd.DataFrame, date_col: str = "incident_date", freq: str = "MS") -> pd.Series:
    """
    Aggregate 1-per-row incidents to monthly counts.
    """
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
    """
    Select, clean, and (optionally) downsample numeric features. Returns (X, index).
    """
    X = (
        features_df.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if sample and len(X) > sample:
        X = X.sample(sample, random_state=random_state)
    return X, X.index


def _make_color_map(labels: np.ndarray) -> Dict[str, str]:
    """
    Stable discrete color map for cluster labels 0..k-1.
    Ensures consistent mapping across 2D/3D when labels are the same.
    """
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe
    labs = [str(x) for x in sorted(pd.Series(labels).unique(), key=lambda v: int(v))]
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labs)}


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
    Forecast monthly incident volumes. Tries SARIMAX; falls back to seasonal naive.
    Returns (plotly Figure, DataFrame with columns: actual, forecast, lower, upper).
    Accepts horizon via any of: horizon, horizon_months, months, n_periods.
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

    use_sarimax = False
    forecast = None
    conf_int = None

    # Try SARIMAX
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        seasonal_order = (0, 1, 1, season_length) if len(y) >= 2 * season_length else (0, 0, 0, 0)
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=H)
        forecast = fc.predicted_mean.rename("forecast")
        conf = fc.conf_int(alpha=0.2)  # 80% interval
        conf_int = pd.DataFrame({"lower": conf.iloc[:, 0].values, "upper": conf.iloc[:, 1].values}, index=forecast.index)
        use_sarimax = True
    except Exception:
        # Seasonal naive fallback
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
        if len(y) >= season_length:
            last_season = y[-season_length:].values
            vals = np.tile(last_season, int(np.ceil(H / season_length)))[:H]
        else:
            vals = np.repeat(y.iloc[-1], H)
        forecast = pd.Series(vals, index=idx, name="forecast")
        conf_int = pd.DataFrame(
            {
                "lower": np.maximum(0, forecast.values - np.sqrt(np.maximum(1, forecast.values))),
                "upper": forecast.values + np.sqrt(np.maximum(1, forecast.values)),
            },
            index=idx,
        )

    # Output frame
    out = pd.DataFrame({"actual": y})
    fc_df = pd.concat([forecast, conf_int], axis=1)
    merged = out.join(fc_df, how="outer")
    merged.index.name = "date"

    # Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Forecast"))
    fig.add_trace(
        go.Scatter(
            x=list(conf_int.index) + list(conf_int.index[::-1]),
            y=list(conf_int["upper"].values) + list(conf_int["lower"].values[::-1]),
            fill="toself",
            opacity=0.2,
            line=dict(width=0),
            name="80% interval",
        )
    )
    fig.update_layout(
        title=f"Monthly Incident Forecast ({'SARIMAX' if use_sarimax else 'Seasonal naive'})",
        xaxis_title="Month",
        yaxis_title="Incidents",
        hovermode="x unified",
    )
    return fig, merged


def forecast_incident_volume(df: pd.DataFrame, periods: int = 6):
    """Back-compat alias used elsewhere."""
    return incident_volume_forecasting(df, months=periods)


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

    # Ensure datetime for grouping
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

    # fixed palette so 3D can reuse
    base_palette = px.colors.qualitative.Safe + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    uniq = sorted(df_plot["cluster_str"].unique(), key=lambda s: int(s))
    discrete_map = {cid: base_palette[i % len(base_palette)] for i, cid in enumerate(uniq)}

    fig = px.scatter(
        df_plot, x="pc1", y="pc2",
        color="cluster_str",
        color_discrete_map=discrete_map,
        title=f"KMeans Clusters (k={k})"
    )
    # stash the palette for 3D
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
    """
    PCA->3D + KMeans clustering. Returns (fig, labels, df3d).
    If 'color_map' is provided, reuse the 2D palette (keys should be str cluster labels).
    """
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
    # fit in feature space (Xs), not on Z
    labels = KMeans(n_clusters=int(k), n_init=10, random_state=42).fit_predict(Xs)

    df3d = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "PC3": Z[:, 2], "cluster": labels})
    df3d["cluster_str"] = df3d["cluster"].astype(str)

    fig = px.scatter_3d(
        df3d,
        x="PC1", y="PC2", z="PC3",
        color="cluster_str",
        opacity=0.8,
        color_discrete_map=(color_map or {})
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Cluster")
    return fig, labels, df3d


# ---------------------------------------
# 7) Predictive models comparison (baselines)
# ---------------------------------------

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

def predictive_models_comparison(
    df: pd.DataFrame,
    target: str = "reportable_bin",
    test_size: float = 0.25,
    random_state: int = 42,
    extra_leaky_features: Optional[List[str]] = None,
    split_strategy: str = "random",       # "random" | "time" | "group"
    time_col: Optional[str] = "incident_date",
    group_col: Optional[str] = None,
    leak_corr_threshold: float = 0.98,
    leak_name_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train RF & LogReg; return dict with models, metrics, and training feature_names."""
    out = create_comprehensive_features(df)
    if isinstance(out, tuple) and len(out) == 3:
        _, _, features_df = out
    elif isinstance(out, pd.DataFrame):
        features_df = out
    else:
        features_df = pd.DataFrame(out)
    features_df = features_df.copy()

    # Target aligned to features_df.index
    if target in df.columns:
        y = df.loc[features_df.index, target].copy()
    else:
        y = (df.get("severity_numeric", pd.Series([2]*len(df))).loc[features_df.index] >= 3).astype(int)

    # Drop obvious post-outcome/leaky columns
    drop_cols = set([
        target, "reportable", "reportable_bin",
        "severity", "severity_numeric",
        "medical_attention_required",
        "notified_to_commission", "reporting_timeframe",
        "investigation_required", "incident_resolved",
    ])
    if extra_leaky_features:
        drop_cols.update(extra_leaky_features)

    leak_name_patterns = leak_name_patterns or [
        "report", "reportable", "notify", "notified",
        "investigat", "severity", "medical", "outcome",
        "resolution", "timeframe"
    ]
    by_name = [c for c in features_df.columns if any(p in c.lower() for p in leak_name_patterns)]
    drop_cols.update([c for c in by_name if c in features_df.columns])
    if drop_cols:
        features_df = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])

    # Auto-drop near-perfectly correlated columns
    to_drop_corr = []
    for c in features_df.columns:
        s = pd.to_numeric(features_df[c], errors="coerce")
        if s.std(ddof=0) == 0 or s.isna().all():
            to_drop_corr.append(c); continue
        try:
            corr = abs(s.corr(pd.to_numeric(y, errors="coerce")))
            if corr >= leak_corr_threshold:
                to_drop_corr.append(c)
        except Exception:
            pass
    if to_drop_corr:
        features_df = features_df.drop(columns=to_drop_corr)

    feature_names = features_df.columns.tolist()
    X = features_df.values

    # Split strategy
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    if split_strategy == "time" and time_col and time_col in df.columns:
        dt = pd.to_datetime(df.loc[features_df.index, time_col], errors="coerce")
        cutoff = dt.quantile(0.8)
        mask_train = dt <= cutoff
        mask_test  = dt > cutoff
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
    elif split_strategy == "group" and group_col and group_col in df.columns:
        groups = df.loc[features_df.index, group_col].astype(str).fillna("NA")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        strat = y if getattr(y, "nunique", lambda: 2)() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )

    # Models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    results: Dict[str, Any] = {}

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
        "feature_names": feature_names,
    }

    try:
        logreg = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1)
    except TypeError:
        logreg = LogisticRegression(max_iter=2000, solver="saga")
    lg_pred = logreg.fit(X_train, y_train).predict(X_test)
    lg_proba = logreg.predict_proba(X_test) if hasattr(logreg, "predict_proba") else None
    results["LogisticRegression"] = {
        "model": logreg,
        "accuracy": float((lg_pred == y_test).mean()),
        "y_test": y_test,
        "predictions": lg_pred,
        "probabilities": lg_proba,
        "feature_names": feature_names,
    }

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

def create_predictive_risk_scoring(
    df: pd.DataFrame,
    trained_models: Dict[str, Dict[str, Any]],
    feature_names: List[str],
):
    """
    Return calculate_risk_score(scenario) using the best model.
    Uses the model's TRAINING feature list to avoid n_features_in_ mismatches.
    """
    if not trained_models:
        return None

    best_model_name = max(trained_models, key=lambda k: trained_models[k].get('accuracy', 0))
    best_model = trained_models[best_model_name]['model']

    # Use training feature order saved with the model (fallback to arg)
    train_feats = list(trained_models[best_model_name].get('feature_names', feature_names))
    n_expected = getattr(best_model, "n_features_in_", len(train_feats))
    if len(train_feats) != n_expected:
        if len(train_feats) > n_expected:
            train_feats = train_feats[:n_expected]
        else:
            train_feats = train_feats + [f"__pad_{i}__" for i in range(n_expected - len(train_feats))]

    def calculate_risk_score(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            'hour': scenario_data.get('hour', 12),
            'is_weekend': 1 if scenario_data.get('day_type') == 'weekend' else 0,
            'is_kitchen': 1 if 'kitchen' in str(scenario_data.get('location', '')).lower() else 0,
            'is_bathroom': 1 if any(k in str(scenario_data.get('location', '')).lower()
                                    for k in ['bathroom','toilet','washroom','restroom']) else 0,
            'participant_incident_count': scenario_data.get('participant_history', 1),
            'carer_incident_count': scenario_data.get('carer_history', 1),
            'location_risk_score': scenario_data.get('location_risk', 2),
        }

        vec = np.zeros(len(train_feats), dtype=float)
        for i, fn in enumerate(train_feats):
            vec[i] = float(mapping.get(fn, 0.0))
        X_one = vec.reshape(1, -1)

        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(X_one)[0]
            score = float(np.max(proba))
        else:
            pred = best_model.predict(X_one)[0]
            score = float(pred) / 3.0

        level = 'HIGH' if score > 0.7 else ('MEDIUM' if score > 0.4 else 'LOW')
        return {'risk_score': score, 'risk_level': level, 'confidence': score, 'model_used': best_model_name}

    return calculate_risk_score


    def calculate_risk_score(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        # simple name→value mapping for your demo features
        mapping = {
            'hour': scenario_data.get('hour', 12),
            'is_weekend': 1 if scenario_data.get('day_type') == 'weekend' else 0,
            'is_kitchen': 1 if 'kitchen' in str(scenario_data.get('location', '')).lower() else 0,
            'is_bathroom': 1 if any(k in str(scenario_data.get('location', '')).lower()
                                    for k in ['bathroom','toilet','washroom','restroom']) else 0,
            'participant_incident_count': scenario_data.get('participant_history', 1),
            'carer_incident_count': scenario_data.get('carer_history', 1),
            'location_risk_score': scenario_data.get('location_risk', 2),
        }

        # build vector aligned to TRAINING feature order
        vec = np.zeros(len(train_feats), dtype=float)
        for i, fn in enumerate(train_feats):
            vec[i] = float(mapping.get(fn, 0.0))  # 0.0 for any features not in the mapping

        X_one = vec.reshape(1, -1)

        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(X_one)[0]
            # for binary classifiers this is usually prob of positive class; max() keeps it robust
            score = float(np.max(proba))
        else:
            pred = best_model.predict(X_one)[0]
            score = float(pred) / 3.0  # crude normalisation fallback

        level = 'HIGH' if score > 0.7 else ('MEDIUM' if score > 0.4 else 'LOW')
        return {'risk_score': score, 'risk_level': level, 'confidence': score, 'model_used': best_model_name}

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


# 15) Streamlit integration helpers (optional)
def add_enhanced_features_to_dashboard(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_names: List[str],
    trained_models: Dict[str, Dict[str, Any]],
):
    """Render enhanced features inside a Streamlit app."""
    try:
        import streamlit as st
    except Exception:
        raise ImportError("Streamlit not available. This function must be called inside a Streamlit app.")

    st.markdown("### 🔬 Enhanced Analytics Features")

    # 📊 Enhanced model analysis
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

    # 🕸️ Carer risk network
    st.markdown("#### 🕸️ Carer-Participant Risk Network")
    network_fig, risk_matrix = carer_risk_network_analysis(df)
    if network_fig is not None:
        st.plotly_chart(network_fig, use_container_width=True)
        high_risk_pairs = risk_matrix[risk_matrix['risk_score'] > risk_matrix['risk_score'].quantile(0.8)]
        if len(high_risk_pairs) > 0:
            st.markdown("##### 🚨 High-Risk Carer-Participant Pairs")
            st.dataframe(high_risk_pairs.sort_values('risk_score', ascending=False))

    # 👤 Participant journey
    if 'participant_id' in df.columns:
        st.markdown("#### 👤 Individual Participant Journey")
        participant_ids = df['participant_id'].dropna().unique()
        if len(participant_ids):
            selected_participant = st.selectbox("Select Participant", participant_ids)
            journey_fig = participant_journey_analysis(df, selected_participant)
            if journey_fig:
                st.plotly_chart(journey_fig, use_container_width=True)

    # 🎯 Predictive risk scoring UI
    if trained_models:
        st.markdown("#### 🎯 Predictive Risk Scoring")

        # ✅ Use training feature order saved with the (best) model
        best_key = max(trained_models, key=lambda k: trained_models[k].get('accuracy', 0))
        trained_feature_names = trained_models[best_key].get('feature_names', [])
        risk_scorer = create_predictive_risk_scoring(df, trained_models, trained_feature_names)

        if risk_scorer:
            # (optional) quick shape sanity caption
            try:
                n_expected = getattr(trained_models[best_key]['model'], 'n_features_in_', '?')
                st.caption(f"Model expects {n_expected} features; scorer using {len(trained_feature_names)}.")
            except Exception:
                pass

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
                'day_type': 'weekend' if st.checkbox("Weekend?", value=False, key="risk_ui_weekend") else 'weekday'
            }
            risk_result = risk_scorer(scenario)
            if risk_result['risk_level'] == 'HIGH':
                st.error(f"🚨 {risk_result['risk_level']} RISK - {risk_result['confidence']:.1%} confidence")
            elif risk_result['risk_level'] == 'MEDIUM':
                st.warning(f"⚠️ {risk_result['risk_level']} RISK - {risk_result['confidence']:.1%} confidence")
            else:
                st.success(f"✅ {risk_result['risk_level']} RISK - {risk_result['confidence']:.1%} confidence")

    # 🚨 Real-time alerts simulation
    st.markdown("#### 🚨 Real-Time Alert System (Simulation)")
    alert_thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.3}
    if trained_models:
        # ✅ Use training feature order saved with the (best) model
        best_key = max(trained_models, key=lambda k: trained_models[k].get('accuracy', 0))
        trained_feature_names = trained_models[best_key].get('feature_names', [])
        risk_scorer = create_predictive_risk_scoring(df, trained_models, trained_feature_names)

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
