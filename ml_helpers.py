"""
COMPREHENSIVE NDIS ANALYTICS SYSTEM
Participants & Carers Predictive Analytics with Advanced Visualizations

Sections:
- Overview (quick KPIs)
- Incident Volume Forecasting
- Location Risk Profiling (Table + Charts)
- Seasonal & Temporal Patterns
- Clustering Analysis (2D/3D PCA)
- Correlation Analysis (heatmap + strong pairs)
- Carer Performance Scatter (filters, 'All' options)
- (Optional) Feature Engineering + Model Comparison

Notes:
- No download/save UI per user request.
"""

import warnings
warnings.filterwarnings('ignore')

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats

# ML libs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, silhouette_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

# Time series
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------------------------------------------------------
# Page Config + Light CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NDIS Advanced Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:2.0rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:0.8rem; }
    .subtle { color:#666; }
    .metric-card { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:0.8rem;border-radius:12px;color:#fff;text-align:center; }
    .stAlert > div { padding:0.8rem;border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
SEV_MAP = {'low':0, 'minor':0, 'medium':1, 'moderate':1, 'high':2, 'major':2, 'critical':2}

def ensure_bool01(s):
    if s is None:
        return None
    if s.dtype == bool:
        return s.astype(int)
    return s.map(lambda v: 1 if str(v).strip().lower() in {"1","true","yes","y","t"} else 0).astype(int)

def norm_severity(series):
    if series is None:
        return pd.Series(dtype='float64')
    return series.astype(str).str.strip().str.lower().map(SEV_MAP)

def coerce_dates(df):
    if 'incident_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    if 'notification_date' in df.columns:
        df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
    if 'incident_time' in df.columns:
        # Accepts "HH:MM" or full datetime; keep only time if datetime given
        t = pd.to_datetime(df['incident_time'], errors='coerce')
        # If time-only strings, date defaults to today; OK—we only need hour/min later
        df['incident_time'] = t.dt.strftime('%H:%M')
        df.loc[t.isna(), 'incident_time'] = None
    return df

def plot_metric(label, value, suffix="", color_graph=None):
    # Simple, pretty metric card using CSS above
    st.markdown(
        f"<div class='metric-card'><div style='font-size:0.9rem;opacity:.9'>{label}</div>"
        f"<div style='font-size:1.8rem;font-weight:700'>{value}{suffix}</div></div>",
        unsafe_allow_html=True
    )

def load_data():
    st.sidebar.markdown("### 📥 Data")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        return df, "Uploaded CSV"

    # Fallback: local sample if present
    default_path = "/mnt/data/ndis_incident_1000.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        return df, "Sample: ndis_incident_1000.csv"

    st.info("Upload a CSV with columns like: incident_date, incident_time, location, severity, medical_attention_required, reportable, participant_id, carer_id, incident_type.")
    return pd.DataFrame(), None

# -----------------------------------------------------------------------------
# Feature Engineering (cached)
# -----------------------------------------------------------------------------
@st.cache_data
def create_comprehensive_features(df):
    if df.empty:
        return None, None, None

    features_df = df.copy()
    features_df = coerce_dates(features_df)

    # incident_datetime
    if 'incident_time' in features_df.columns and features_df['incident_time'].notna().any():
        # build datetime from date + time (HH:MM)
        time_obj = pd.to_datetime(features_df['incident_time'], format='%H:%M', errors='coerce')
        features_df['incident_datetime'] = pd.to_datetime(
            features_df['incident_date'].dt.strftime('%Y-%m-%d') + ' ' + time_obj.dt.strftime('%H:%M'),
            errors='coerce'
        )
    else:
        features_df['incident_datetime'] = features_df['incident_date']

    features_df = features_df.sort_values('incident_datetime').reset_index(drop=True)

    # ---- Temporal features
    if 'incident_datetime' in features_df.columns:
        dt = features_df['incident_datetime']
        features_df['hour'] = dt.dt.hour
        features_df['day_of_week'] = dt.dt.dayofweek
        features_df['day_of_year'] = dt.dt.dayofyear
        features_df['month'] = dt.dt.month
        features_df['quarter'] = dt.dt.quarter
        features_df['week_of_year'] = dt.dt.isocalendar().week.astype(int)

        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['is_early_morning'] = ((features_df['hour'] >= 5) & (features_df['hour'] <= 8)).astype(int)
        features_df['is_morning'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 11)).astype(int)
        features_df['is_afternoon'] = ((features_df['hour'] >= 12) & (features_df['hour'] <= 17)).astype(int)
        features_df['is_evening'] = ((features_df['hour'] >= 18) & (features_df['hour'] <= 22)).astype(int)
        features_df['is_night'] = ((features_df['hour'] >= 23) | (features_df['hour'] <= 4)).astype(int)
        features_df['is_business_hours'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 17)).astype(int)

        season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
        features_df['season'] = features_df['month'].map(season_map)
        features_df['is_summer'] = (features_df['season'] == 0).astype(int)
        features_df['is_autumn'] = (features_df['season'] == 1).astype(int)
        features_df['is_winter'] = (features_df['season'] == 2).astype(int)
        features_df['is_spring'] = (features_df['season'] == 3).astype(int)

    # ---- Location flags
    if 'location' in features_df.columns:
        f = features_df['location'].astype(str).str.lower()
        features_df['is_participant_home'] = f.str.contains(r'\b(participant home|own home|home)\b', regex=True).astype(int)
        features_df['is_kitchen'] = f.str.contains('kitchen').astype(int)
        features_df['is_bathroom'] = f.str.contains('bathroom|toilet', regex=True).astype(int)
        features_df['is_bedroom'] = f.str.contains('bedroom').astype(int)
        features_df['is_living_room'] = f.str.contains('living').astype(int)
        features_df['is_activity_room'] = f.str.contains('activity|day room', regex=True).astype(int)
        features_df['is_outdoor'] = f.str.contains('outdoor|garden|yard|park', regex=True).astype(int)
        features_df['is_transport'] = f.str.contains('transport|car|vehicle|road', regex=True).astype(int)
        features_df['is_community'] = f.str.contains('community|shop|store|mall', regex=True).astype(int)
        features_df['is_medical_facility'] = f.str.contains('hospital|clinic|medical', regex=True).astype(int)

        def _loc_risk(row):
            risk = 1
            if row['is_kitchen'] or row['is_bathroom']: risk += 3
            elif row['is_transport'] or row['is_medical_facility']: risk += 2
            elif row['is_activity_room'] or row['is_community']: risk += 1
            elif row['is_participant_home'] and not (row['is_kitchen'] or row['is_bathroom']):
                risk = max(1, risk - 1)
            return min(risk, 5)

        features_df['location_risk_score'] = features_df.apply(_loc_risk, axis=1)

    # ---- Participant features
    if 'participant_id' in features_df.columns:
        features_df = features_df.sort_values(['participant_id', 'incident_datetime']).reset_index(drop=True)
        features_df['participant_incident_count'] = features_df.groupby('participant_id').cumcount() + 1
        features_df['days_since_last_incident'] = (
            features_df.groupby('participant_id')['incident_datetime'].diff().dt.days.fillna(999)
        )

        sev_num = norm_severity(features_df['severity']) if 'severity' in features_df.columns else 0
        features_df['severity_numeric'] = sev_num if isinstance(sev_num, pd.Series) else 0

        features_df['participant_medical_rate'] = (
            features_df.groupby('participant_id')['medical_attention_required']
            .transform(lambda s: ensure_bool01(s).rolling(window=10, min_periods=1).mean() if s is not None else 0)
            if 'medical_attention_required' in features_df.columns else 0
        )
        features_df['participant_reportable_rate'] = (
            features_df.groupby('participant_id')['reportable']
            .transform(lambda s: ensure_bool01(s).rolling(window=10, min_periods=1).mean() if s is not None else 0)
            if 'reportable' in features_df.columns else 0
        )
        if 'severity_numeric' in features_df.columns:
            features_df['participant_high_severity_rate'] = (
                features_df.groupby('participant_id')['severity_numeric']
                .transform(lambda s: (s >= 2).rolling(window=10, min_periods=1).mean())
            )
        else:
            features_df['participant_high_severity_rate'] = 0

        features_df['participant_avg_days_between_incidents'] = (
            features_df.groupby('participant_id')['days_since_last_incident']
            .transform(lambda s: s.rolling(window=10, min_periods=1).mean())
        )
        if len(features_df) > 20:
            q80 = features_df['participant_incident_count'].quantile(0.8)
            features_df['is_high_risk_participant'] = (features_df['participant_incident_count'] >= q80).astype(int)
        else:
            features_df['is_high_risk_participant'] = 0

    # ---- Carer features
    if 'carer_id' in features_df.columns:
        features_df = features_df.sort_values(['carer_id', 'incident_datetime']).reset_index(drop=True)
        features_df['carer_incident_count'] = features_df.groupby('carer_id').cumcount() + 1
        features_df['carer_days_since_last_incident'] = (
            features_df.groupby('carer_id')['incident_datetime'].diff().dt.days.fillna(999)
        )
        if 'participant_id' in features_df.columns:
            features_df['carer_participant_incident_count'] = (
                features_df.groupby(['carer_id', 'participant_id']).cumcount() + 1
            )
        if 'incident_type' in features_df.columns:
            features_df['carer_same_incident_type_count'] = (
                features_df.groupby(['carer_id','incident_type']).cumcount() + 1
            )

        features_df['carer_medical_rate'] = (
            features_df.groupby('carer_id')['medical_attention_required']
            .transform(lambda s: ensure_bool01(s).rolling(window=10, min_periods=1).mean() if s is not None else 0)
            if 'medical_attention_required' in features_df.columns else 0
        )
        if 'severity_numeric' in features_df.columns:
            features_df['carer_high_severity_rate'] = (
                features_df.groupby('carer_id')['severity_numeric']
                .transform(lambda s: (s >= 2).rolling(window=10, min_periods=1).mean())
            )
        else:
            features_df['carer_high_severity_rate'] = 0
        features_df['carer_reportable_rate'] = (
            features_df.groupby('carer_id')['reportable']
            .transform(lambda s: ensure_bool01(s).rolling(window=10, min_periods=1).mean() if s is not None else 0)
            if 'reportable' in features_df.columns else 0
        )

        if len(features_df) > 20:
            q80c = features_df['carer_incident_count'].quantile(0.8)
            features_df['is_high_risk_carer'] = (features_df['carer_incident_count'] >= q80c).astype(int)
        else:
            features_df['is_high_risk_carer'] = 0

    # ---- Contextual
    features_df['high_risk_time_location'] = (
        (features_df.get('is_early_morning', 0) == 1) | (features_df.get('is_night', 0) == 1)
    ) & (
        (features_df.get('is_kitchen', 0) == 1) | (features_df.get('is_bathroom', 0) == 1)
    )
    features_df['high_risk_time_location'] = features_df['high_risk_time_location'].astype(int)

    if {'participant_id','carer_id'}.issubset(features_df.columns):
        features_df['high_risk_participant_carer'] = (
            (features_df['is_high_risk_participant'] == 1) &
            (features_df['is_high_risk_carer'] == 1)
        ).astype(int)

    # ---- Encode categoricals (optional)
    label_encoders = {}
    categorical_columns = []
    for col in ['location', 'incident_type', 'severity', 'carer_id', 'participant_id']:
        if col in features_df.columns:
            categorical_columns.append(col)

    for col in categorical_columns:
        le = LabelEncoder()
        features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].fillna('Unknown').astype(str))
        label_encoders[col] = le

    # ---- Feature set
    feature_columns = [
        'hour','day_of_week','month','quarter','season',
        'is_weekend','is_early_morning','is_morning','is_afternoon','is_evening','is_night',
        'is_business_hours','is_summer','is_autumn','is_winter','is_spring',
        'is_participant_home','is_kitchen','is_bathroom','is_bedroom','is_living_room',
        'is_activity_room','is_outdoor','is_transport','is_community','is_medical_facility',
        'location_risk_score',
        'high_risk_time_location'
    ]
    if 'participant_id' in features_df.columns:
        feature_columns += [
            'participant_incident_count','days_since_last_incident',
            'participant_medical_rate','participant_high_severity_rate','participant_reportable_rate',
            'participant_avg_days_between_incidents','is_high_risk_participant'
        ]
    if 'carer_id' in features_df.columns:
        feature_columns += [
            'carer_incident_count','carer_days_since_last_incident','carer_participant_incident_count',
            'carer_medical_rate','carer_high_severity_rate','carer_reportable_rate','is_high_risk_carer'
        ]
        if 'incident_type' in features_df.columns:
            feature_columns.append('carer_same_incident_type_count')
        if 'participant_id' in features_df.columns:
            feature_columns.append('high_risk_participant_carer')

    for col in categorical_columns:
        enc = f'{col}_encoded'
        if enc in features_df.columns:
            feature_columns.append(enc)

    feature_columns = [c for c in feature_columns if c in features_df.columns]
    X = features_df[feature_columns].fillna(0)

    return X, feature_columns, features_df

# -----------------------------------------------------------------------------
# Predictive Models (optional section)
# -----------------------------------------------------------------------------
def predictive_models_comparison(X, y, feature_names, target_name="severity"):
    st.markdown("### 🤖 Predictive Models Comparison")
    if X is None or len(X) < 20 or y is None or len(y) != len(X):
        st.warning("Insufficient data for model training.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    results = {}
    rows = []

    for name, model in models.items():
        try:
            if name == 'Logistic Regression':
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                y_proba = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            rows.append({'Model': name, 'Accuracy': acc, 'CV Mean': cv.mean(), 'CV Std': cv.std()})

            results[name] = {
                'model': model, 'predictions': y_pred, 'probabilities': y_proba,
                'accuracy': acc,
                'feature_importance': getattr(model, 'feature_importances_', None)
            }
        except Exception as e:
            st.warning(f"Model {name} failed: {e}")

    if rows:
        perf_df = pd.DataFrame(rows)
        st.dataframe(perf_df.style.format({'Accuracy':'{:.3f}','CV Mean':'{:.3f}','CV Std':'{:.3f}'}), use_container_width=True)

    # Feature importance (RF)
    rf_imp = results.get('Random Forest', {}).get('feature_importance')
    if rf_imp is not None:
        st.markdown("#### 🎯 Top Feature Importance (Random Forest)")
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_imp}).sort_values('Importance', ascending=False).head(15)
        fig = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='viridis', title="Top 15 Features")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    if results:
        st.markdown("#### 🎭 Confusion Matrices")
        cols = st.columns(2)
        labels = sorted(pd.Series(y).unique().tolist())
        for i, (name, res) in enumerate(results.items()):
            cm = confusion_matrix(y_test, res['predictions'], labels=labels)
            fig = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            color_continuous_scale='Blues', text_auto=True, title=f"{name}")
            fig.update_layout(height=380)
            cols[i % 2].plotly_chart(fig, use_container_width=True)

    return results, rows

# -----------------------------------------------------------------------------
# Incident Volume Forecasting (completed, no downloads)
# -----------------------------------------------------------------------------
def incident_volume_forecasting(df, periods=6):
    st.markdown("### 📈 Incident Volume Forecasting")
    if df.empty or 'incident_date' not in df.columns:
        st.warning("Date column required for forecasting.")
        return None

    try:
        df = df.copy()
        df = coerce_dates(df)
        series = df.groupby(df['incident_date'].dt.to_period('M')).size()
        series.index = series.index.to_timestamp()

        if len(series) < 6:
            st.warning("Need at least 6 months of data for reliable forecasting.")
            return None

        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        fitted = model.fit()
        forecast = fitted.forecast(periods)

        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=periods, freq='M')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines+markers',
                                 name='Historical', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast.values, mode='lines+markers',
                                 name='Forecast', line=dict(color='red', width=3, dash='dash')))

        forecast_upper = forecast * 1.1
        forecast_lower = forecast * 0.9

        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(forecast_upper) + list(forecast_lower[::-1]),
            fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'
        ))

        fig.update_layout(title=f"Incident Volume Forecast - Next {periods} Months",
                          xaxis_title="Date", yaxis_title="Number of Incidents",
                          hovermode='x unified', height=500)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            curr_avg = series.tail(6).mean()
            st.metric("Current 6-Month Avg", f"{curr_avg:.1f}")
        with c2:
            fc_avg = forecast.mean()
            st.metric("Forecast 6-Month Avg", f"{fc_avg:.1f}")
        with c3:
            trend = ((fc_avg - curr_avg) / curr_avg * 100) if curr_avg > 0 else 0
            st.metric("Trend", f"{trend:+.1f}%")

        # Forecast table (no download)
        forecast_df = pd.DataFrame({
            'Month': forecast_dates.strftime('%Y-%m'),
            'Predicted Incidents': forecast.round(0).astype(int),
            'Lower Bound': forecast_lower.round(0).astype(int),
            'Upper Bound': forecast_upper.round(0).astype(int)
        })
        st.markdown("#### 📋 Detailed Forecast")
        st.dataframe(forecast_df, use_container_width=True)
        return forecast_df

    except Exception as e:
        st.error(f"Forecasting failed: {e}")
        return None

# -----------------------------------------------------------------------------
# Location Risk Profiling
# -----------------------------------------------------------------------------
def location_risk_profiling(df):
    st.markdown("### 📍 Location Risk Profiling")
    if 'location' not in df.columns or 'severity' not in df.columns:
        st.warning("Location and severity columns required.")
        return None, None

    tmp = df.copy()
    tmp['severity_numeric'] = norm_severity(tmp['severity'])
    if 'medical_attention_required' in tmp.columns:
        tmp['medical_attention_required'] = ensure_bool01(tmp['medical_attention_required'])
    if 'reportable' in tmp.columns:
        tmp['reportable'] = ensure_bool01(tmp['reportable'])

    # Build metrics safely with only present columns
    agg_dict = {'severity_numeric': ['count','mean','std']}
    if 'medical_attention_required' in tmp.columns:
        agg_dict['medical_attention_required'] = 'mean'
    if 'reportable' in tmp.columns:
        agg_dict['reportable'] = 'mean'
    if 'participant_id' in tmp.columns:
        agg_dict['participant_id'] = pd.Series.nunique

    location_stats = tmp.groupby('location').agg(agg_dict)
    # Ensure all expected columns exist
    if ('medical_attention_required' not in location_stats.columns.get_level_values(0)):
        location_stats[('medical_attention_required', 'mean')] = 0.0
    if ('reportable' not in location_stats.columns.get_level_values(0)):
        location_stats[('reportable', 'mean')] = 0.0
    if ('participant_id' not in location_stats.columns.get_level_values(0)):
        location_stats[('participant_id', 'nunique')] = 1

    location_stats = location_stats.round(3)
    location_stats.columns = ['Incident_Count','Avg_Severity','Severity_Std','Medical_Rate','Reportable_Rate','Unique_Participants']
    location_stats['Incidents_Per_Participant'] = (
        location_stats['Incident_Count'] / location_stats['Unique_Participants'].replace(0, np.nan)
    ).fillna(0).round(2)
    location_stats['Risk_Score'] = (
        location_stats['Avg_Severity'] * 0.4 +
        location_stats['Medical_Rate'] * 0.3 +
        location_stats['Reportable_Rate'] * 0.3
    )

    location_stats = location_stats.sort_values('Risk_Score', ascending=False)

    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("#### 📊 Location Risk Analysis Table")
        display_df = location_stats.head(20).copy()
        styled = display_df.style.format({
            'Avg_Severity':'{:.2f}','Severity_Std':'{:.2f}',
            'Medical_Rate':'{:.1%}','Reportable_Rate':'{:.1%}',
            'Risk_Score':'{:.3f}','Incidents_Per_Participant':'{:.2f}'
        }).background_gradient(subset=['Risk_Score'], cmap='Reds')
        st.dataframe(styled, use_container_width=True)
    with c2:
        risk_score_fig = px.histogram(location_stats.reset_index(), x='Risk_Score', nbins=20,
                                     title="Risk Score Distribution",
                                     labels={'Risk_Score':'Risk Score','count':'Locations'})
        st.plotly_chart(risk_score_fig, use_container_width=True)

    st.markdown("#### 📊 Location Risk Visualizations")
    tab1, tab2, tab3 = st.tabs(["🎯 Top Risk Locations", "📈 Incident Volume", "🏥 Medical Attention Rate"])
    with tab1:
        top_risk = location_stats.head(15).reset_index()
        top_risk_fig = px.bar(top_risk, x='Risk_Score', y='location', orientation='h',
                              title="Top 15 Highest Risk Locations",
                              color='Risk_Score', color_continuous_scale='Reds',
                              labels={'Risk_Score':'Risk Score','location':'Location'})
        top_risk_fig.update_layout(height=600)
        st.plotly_chart(top_risk_fig, use_container_width=True)
    with tab2:
        top_vol = location_stats.sort_values('Incident_Count', ascending=False).head(15).reset_index()
        top_vol_fig = px.bar(top_vol, x='Incident_Count', y='location', orientation='h',
                             title="Top 15 Locations by Incident Volume",
                             color='Incident_Count', color_continuous_scale='Blues')
        top_vol_fig.update_layout(height=600)
        st.plotly_chart(top_vol_fig, use_container_width=True)
    with tab3:
        high_med = location_stats[location_stats['Medical_Rate'] > 0].sort_values('Medical_Rate', ascending=False).head(15).reset_index()
        if len(high_med) > 0:
            high_med_fig = px.bar(high_med, x='Medical_Rate', y='location', orientation='h',
                                  title="Top 15 by Medical Attention Rate",
                                  color='Medical_Rate', color_continuous_scale='Oranges')
            high_med_fig.update_layout(height=600)
            st.plotly_chart(high_med_fig, use_container_width=True)
        else:
            st.info("No locations with medical attention data available.")

    st.markdown("#### 💡 Location-Based Recommendations")
    recs = []
    high_risk = location_stats[location_stats['Risk_Score'] > location_stats['Risk_Score'].quantile(0.8)].head(5)
    if len(high_risk) > 0:
        locs = "', '".join(high_risk.index[:3])
        recs.append(f"🚨 **High Priority**: Locations '{locs}' require immediate safety protocol review.")
    high_vol = location_stats[location_stats['Incident_Count'] >= 10].head(3)
    if len(high_vol) > 0:
        recs.append(f"📊 **Volume Concern**: {len(high_vol)} locations have ≥10 incidents—review staffing levels.")
    high_med_locs = location_stats[location_stats['Medical_Rate'] > 0.5]
    if len(high_med_locs) > 0:
        recs.append(f"🏥 **Medical Resources**: {len(high_med_locs)} locations have >50% medical attention rate—ensure supplies.")

    recs.extend([
        "🛡️ **Safety Equipment**: Prioritise safety equipment installation in highest-risk locations.",
        "👥 **Staff Training**: Focus location-specific training on high-risk areas.",
        "📋 **Regular Audits**: Monthly audits for top 5 risk locations.",
        "📊 **Monitoring**: Real-time monitoring for locations with risk scores > 2.0."
    ])
    for r in recs:
        st.write(r)

    return location_stats, top_risk_fig

# -----------------------------------------------------------------------------
# Incident Type Risk Profiling
# -----------------------------------------------------------------------------
def incident_type_risk_profiling(df):
    st.markdown("### 🗂️ Incident Type Risk Profiling")
    if 'incident_type' not in df.columns or 'severity' not in df.columns:
        st.warning("Incident type and severity columns required.")
        return None, None

    tmp = df.copy()
    tmp['severity_numeric'] = norm_severity(tmp['severity'])
    if 'medical_attention_required' in tmp.columns:
        tmp['medical_attention_required'] = ensure_bool01(tmp['medical_attention_required'])
    if 'reportable' in tmp.columns:
        tmp['reportable'] = ensure_bool01(tmp['reportable'])

    agg_dict = {'severity_numeric': ['count', 'mean', 'std']}
    if 'medical_attention_required' in tmp.columns:
        agg_dict['medical_attention_required'] = 'mean'
    if 'reportable' in tmp.columns:
        agg_dict['reportable'] = 'mean'
    if 'participant_id' in tmp.columns:
        agg_dict['participant_id'] = pd.Series.nunique

    type_stats = tmp.groupby('incident_type').agg(agg_dict)
    if ('medical_attention_required' not in type_stats.columns.get_level_values(0)):
        type_stats[('medical_attention_required','mean')] = 0.0
    if ('reportable' not in type_stats.columns.get_level_values(0)):
        type_stats[('reportable','mean')] = 0.0
    if ('participant_id' not in type_stats.columns.get_level_values(0)):
        type_stats[('participant_id','nunique')] = 1

    type_stats = type_stats.round(3)
    type_stats.columns = ['Incident_Count', 'Avg_Severity', 'Severity_Std', 'Medical_Rate', 'Reportable_Rate', 'Unique_Participants']
    type_stats['Risk_Score'] = (
        type_stats['Avg_Severity'] * 0.4 +
        type_stats['Medical_Rate'] * 0.3 +
        type_stats['Reportable_Rate'] * 0.3
    )
    type_stats = type_stats.sort_values('Risk_Score', ascending=False)

    fig = px.bar(
        type_stats.reset_index(),
        x='Risk_Score', y='incident_type', orientation='h',
        title="Incident Type Risk Scores",
        color='Risk_Score', color_continuous_scale='Reds',
        labels={'Risk_Score': 'Risk Score', 'incident_type': 'Incident Type'}
    )
    fig.update_layout(height=600)
    return type_stats, fig

# -----------------------------------------------------------------------------
# Seasonal & Temporal Patterns
# -----------------------------------------------------------------------------
def seasonal_temporal_patterns(df):
    st.markdown("### 🌍 Seasonal & Temporal Pattern Detection")
    if 'incident_date' not in df.columns:
        st.warning("Date column required for temporal analysis.")
        return

    df = df.copy()
    df = coerce_dates(df)

    df['year_month'] = df['incident_date'].dt.to_period('M')
    df['month_name'] = df['incident_date'].dt.month_name()
    df['day_name'] = df['incident_date'].dt.day_name()
    df['hour'] = df['incident_date'].dt.hour if 'incident_time' not in df.columns else pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour

    st.markdown("#### 📅 Monthly Incident Trends")
    monthly = df.groupby('year_month').size()
    monthly.index = monthly.index.to_timestamp()

    xnum = np.arange(len(monthly))
    slope, intercept, r, p, se = stats.linregress(xnum, monthly.values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, mode='lines+markers',
                             name='Monthly Incidents', line=dict(color='#1f77b4', width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=monthly.index, y=(slope*xnum + intercept),
                             mode='lines', name=f'Trend (R²={r**2:.3f})',
                             line=dict(color='red', width=2, dash='dash')))
    if len(monthly) >= 6:
        ma = monthly.rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines',
                                 name='3-Month Moving Avg', line=dict(color='green', width=2)))
    fig.update_layout(title="Monthly Incident Volume with Trend",
                      xaxis_title="Date", yaxis_title="Incidents",
                      hovermode='x unified', height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🌤️ Seasonal Pattern Analysis")
    c1, c2 = st.columns(2)
    with c1:
        monthly_pattern = df.groupby(df['incident_date'].dt.month).size()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig = px.bar(x=month_names, y=monthly_pattern.reindex(range(1,13), fill_value=0).values,
                     title="Incidents by Month of Year", labels={'x':'Month','y':'Incidents'},
                     color=monthly_pattern.reindex(range(1,13), fill_value=0).values,
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        if len(monthly_pattern) > 0:
            pm = monthly_pattern.idxmax()
            lm = monthly_pattern.idxmin()
            st.info(f"Peak month: **{month_names[pm-1]}** ({monthly_pattern.max()} incidents)")
            st.info(f"Lowest month: **{month_names[lm-1]}** ({monthly_pattern.min()} incidents)")
    with c2:
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        day_pattern = df.groupby('day_name').size().reindex(day_order, fill_value=0)
        fig = px.bar(x=day_pattern.index, y=day_pattern.values, title="Incidents by Day of Week",
                     labels={'x':'Day of Week','y':'Incidents'},
                     color=day_pattern.values, color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
        weekend = day_pattern[['Saturday','Sunday']].sum()
        weekday = day_pattern[['Monday','Tuesday','Wednesday','Thursday','Friday']].sum()
        weekend_pct = 100 * weekend / (weekend + weekday) if (weekend + weekday) > 0 else 0
        st.info(f"Weekend incidents: **{weekend_pct:.1f}%** of total")

    # Hourly
    if df['hour'].notna().any():
        st.markdown("#### ⏰ Hourly Pattern Analysis")
        hourly = df.groupby('hour').size()
        hours = list(range(24))
        vals = [hourly.get(h, 0) for h in hours]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=vals, name='Hourly Incidents',
                             marker_color=vals, marker_colorscale='Reds'))
        fig.update_layout(title="Incidents by Hour of Day",
                          xaxis_title="Hour", yaxis_title="Incidents",
                          xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig, use_container_width=True)

        if len(hourly) > 0:
            peak_hour = hourly.idxmax()
            low_hour = hourly.idxmin()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Peak Hour", f"{int(peak_hour):02d}:00", f"{hourly.max()} incidents")
            with c2: st.metric("Lowest Hour", f"{int(low_hour):02d}:00", f"{hourly.min()} incidents")
            with c3:
                bh = hourly.loc[9:17].sum() if 9 in hourly.index and 17 in hourly.index else 0
                pct = 100 * bh / hourly.sum() if hourly.sum() > 0 else 0
                st.metric("Business Hours %", f"{pct:.1f}%")

    # Insights
    st.markdown("#### 💡 Temporal Insights & Recommendations")
    insights = []
    if len(monthly) > 6:
        recent_trend = monthly.tail(6).mean() - monthly.head(6).mean()
        if recent_trend > 0:
            insights.append(f"📈 **Increasing Trend**: Recent 6 months show {recent_trend:.1f} more incidents/month on average.")
        elif recent_trend < -1:
            insights.append(f"📉 **Decreasing Trend**: Recent 6 months show {abs(recent_trend):.1f} fewer incidents/month.")
        else:
            insights.append("📊 **Stable Trend**: Incident rates remain relatively stable.")

    if len(df) > 0:
        mp = df.groupby(df['incident_date'].dt.month).size()
        if len(mp) >= 12:
            summer = mp.reindex([12,1,2]).sum()
            winter = mp.reindex([6,7,8]).sum()
            if summer > winter * 1.2:
                insights.append("☀️ **Summer Pattern**: Significantly more incidents during summer months.")
            elif winter > summer * 1.2:
                insights.append("❄️ **Winter Pattern**: Significantly more incidents during winter months.")

    if 'hour' in df.columns and df['hour'].notna().any():
        hp = df.groupby('hour').size()
        needed = [23,0,1,2,3,4,5]
        if all(h in hp.index for h in needed):
            night_pct = 100 * hp.loc[needed].sum() / hp.sum()
            if night_pct > 25:
                insights.append("🌙 **Night Risk**: High nighttime incident rate—review night shift protocols.")

    recs = [
        "📊 **Monthly Monitoring**: Track monthly trends to plan seasonal staffing.",
        "📅 **Shift Planning**: Adjust staffing based on day-of-week patterns.",
        "⏰ **Time-Based Protocols**: Introduce measures for high-risk hours.",
        "🌤️ **Seasonal Prep**: Prepare resources for high-risk seasons.",
        "📈 **Trend Watch**: Monitor early changes in patterns."
    ]
    for line in insights + recs:
        st.write(line)

# -----------------------------------------------------------------------------
# Time vs Causes (stacked bars + total line)
# -----------------------------------------------------------------------------
def plot_time_with_causes(df, cause_col=None, top_n=5):
    if 'incident_date' not in df.columns:
        st.warning("Date column required for time + causes chart.")
        return
    if cause_col is None:
        if 'incident_type' in df.columns:
            cause_col = 'incident_type'
        elif 'contributing_factors' in df.columns:
            cause_col = 'contributing_factors'
        else:
            st.warning("No categorical cause column found (e.g., incident_type).")
            return

    df_ = df.copy()
    df_ = coerce_dates(df_)
    df_['period'] = df_['incident_date'].dt.to_period('M').dt.to_timestamp()

    # Top N causes overall
    top_causes = (
        df_[cause_col].fillna('Unknown').astype(str)
        .value_counts().head(top_n).index.tolist()
    )
    df_['cause_top'] = df_[cause_col].fillna('Unknown').astype(str)
    df_.loc[~df_['cause_top'].isin(top_causes), 'cause_top'] = 'Other'

    # Pivot for stacked bars
    pivot = (
        df_.groupby(['period','cause_top'])
           .size()
           .unstack(fill_value=0)
           .sort_index()
    )
    total = pivot.sum(axis=1)

    # Build combined figure with secondary y for total line
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in pivot.columns:
        fig.add_trace(
            go.Bar(x=pivot.index, y=pivot[col], name=col),
            secondary_y=False
        )
    fig.add_trace(
        go.Scatter(x=total.index, y=total.values, name='Total', mode='lines+markers', line=dict(width=3)),
        secondary_y=True
    )
    fig.update_layout(
        title=f"Incidents by Month and {cause_col.replace('_',' ').title()} (Top {top_n})",
        barmode='stack', hovermode='x unified', height=520
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Incidents (stacked)", secondary_y=False)
    fig.update_yaxes(title_text="Total", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Carer Performance Scatter (filters with 'All')
# -----------------------------------------------------------------------------
def plot_carer_performance_scatter(df):
    required = {'carer_id','incident_date','severity'}
    if not required.issubset(df.columns):
        st.info("Carer performance scatter needs columns: carer_id, incident_date, severity.")
        return

    df_ = df.copy()
    df_ = coerce_dates(df_)
    df_['severity_numeric'] = norm_severity(df_['severity'])
    if 'medical_attention_required' in df_.columns:
        df_['medical_attention_required'] = ensure_bool01(df_['medical_attention_required'])
    if 'reportable' in df_.columns:
        df_['reportable'] = ensure_bool01(df_['reportable'])

    # Filters
    carers = sorted(df_['carer_id'].dropna().astype(str).unique().tolist())
    parts = sorted(df_['participant_id'].dropna().astype(str).unique().tolist()) if 'participant_id' in df_.columns else []
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        carer_sel = st.selectbox("Carer", options=["All"] + carers, index=0)
    with col2:
        part_sel = st.selectbox("Participant", options=["All"] + parts if parts else ["All"], index=0)
    with col3:
        period_months = st.slider("Months", 1, 24, 6)

    cutoff = (df_['incident_date'].max() - pd.DateOffset(months=period_months)) if df_['incident_date'].notna().any() else None
    if cutoff is not None:
        df_ = df_[df_['incident_date'] >= cutoff]

    if carer_sel != "All":
        df_ = df_[df_['carer_id'].astype(str) == carer_sel]
    if part_sel != "All" and 'participant_id' in df_.columns:
        df_ = df_[df_['participant_id'].astype(str) == part_sel]

    if df_.empty:
        st.info("No data for selected filters.")
        return

    # Aggregate per carer
    grp = df_.groupby('carer_id')
    summary = pd.DataFrame({
        'incidents': grp.size(),
        'unique_participants': grp['participant_id'].nunique() if 'participant_id' in df_.columns else 1,
        'high_sev_rate': grp['severity_numeric'].apply(lambda s: (s >= 2).mean() if s.notna().any() else 0),
        'medical_rate': grp['medical_attention_required'].mean() if 'medical_attention_required' in df_.columns else 0,
        'reportable_rate': grp['reportable'].mean() if 'reportable' in df_.columns else 0
    }).reset_index()

    summary['incidents_per_participant'] = (
        summary['incidents'] / summary['unique_participants'].replace(0, np.nan)
    ).fillna(summary['incidents'])

    fig = px.scatter(
        summary,
        x='incidents_per_participant',
        y='high_sev_rate',
        size='medical_rate' if 'medical_attention_required' in df.columns else None,
        color='reportable_rate' if 'reportable' in df.columns else None,
        hover_data=['carer_id','incidents','unique_participants','medical_rate','reportable_rate'],
        labels={
            'incidents_per_participant': 'Incidents per Participant',
            'high_sev_rate': 'High Severity Rate'
        },
        title="Carer Performance: Incidents vs High Severity (size=Medical, color=Reportable)"
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Tip: Aim for lower high severity rate and moderate incidents-per-participant; investigate outliers.")

# -----------------------------------------------------------------------------
# Correlation Analysis (no downloads)
# -----------------------------------------------------------------------------
def correlation_analysis(X, feature_names, df):
    st.markdown("### 🔗 Correlation Analysis")
    if X is None:
        st.warning("No feature matrix provided.")
        return None

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
        if X_df.columns.isnull().any() and feature_names:
            X_df.columns = feature_names
    else:
        if not feature_names:
            st.warning("Feature names are required when X is not a DataFrame.")
            return None
        X_df = pd.DataFrame(X, columns=feature_names)

    X_num = X_df.select_dtypes(include=[np.number])
    if X_num.shape[1] < 3:
        st.warning("Insufficient numeric features for correlation analysis (need ≥ 3).")
        return None

    corr = X_num.corr().fillna(0)

    st.markdown("#### 🎯 Feature Correlation Heatmap")
    fig = px.imshow(corr, x=corr.columns, y=corr.index, color_continuous_scale="RdBu_r",
                    aspect="auto", title="Feature Correlation Matrix", zmin=-1, zmax=1)
    fig.update_layout(height=600, margin=dict(t=60, r=20, l=60, b=60),
                      coloraxis_colorbar=dict(title="ρ"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Strongest Correlated Feature Pairs")
    abs_thresh = st.slider("Absolute correlation threshold", 0.0, 1.0, 0.6, 0.05)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iat[i, j])
            if abs(r) >= abs_thresh:
                pairs.append({"Feature A": cols[i], "Feature B": cols[j], "Correlation (ρ)": r})

    if pairs:
        pairs_df = (
            pd.DataFrame(pairs)
              .sort_values(by="Correlation (ρ)", key=lambda s: s.abs(), ascending=False)
              .reset_index(drop=True)
        )
        st.dataframe(pairs_df.head(100), use_container_width=True)
    else:
        st.info("No pairs exceed the selected threshold.")

    avg_abs = corr.abs().replace(1.0, np.nan).mean().sort_values(ascending=False)
    likely_redundant = avg_abs[avg_abs >= 0.6]
    if not likely_redundant.empty:
        st.warning(
            f"Potential multicollinearity: {', '.join(likely_redundant.index[:8])} "
            f"(avg |ρ| ≥ 0.6 to other features). Consider feature selection/regularization."
        )

    return corr

# -----------------------------------------------------------------------------
# Clustering Analysis
# -----------------------------------------------------------------------------
def clustering_analysis(X, df, feature_names):
    st.markdown("### 📊 Clustering Analysis")
    if X is None or len(X) < 10:
        st.warning("Insufficient data for clustering analysis.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        algorithm = st.selectbox("🔍 Clustering Algorithm", ['K-Means','DBSCAN','Hierarchical'])
    with c2:
        if algorithm in ['K-Means','Hierarchical']:
            n_clusters = st.slider("📊 Number of Clusters", 2, 10, 5)
        else:
            eps = st.slider("🎯 DBSCAN Epsilon", 0.1, 2.0, 0.5, 0.1)
    with c3:
        show_3d = st.checkbox("📈 3D Visualization", value=True)

    if st.button("🔍 Perform Clustering Analysis", type="primary"):
        with st.spinner("Clustering..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca_2d = PCA(n_components=2)
            pca_3d = PCA(n_components=3)
            X2 = pca_2d.fit_transform(X_scaled)
            X3 = pca_3d.fit_transform(X_scaled)

            if algorithm == 'K-Means':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif algorithm == 'DBSCAN':
                model = DBSCAN(eps=eps, min_samples=5)
            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)

            labels = model.fit_predict(X_scaled)

            sil = None
            if len(set(labels)) > 1 and -1 not in labels:
                sil = silhouette_score(X_scaled, labels)

            dfc = df.copy()
            dfc['cluster'] = labels
            dfc['pca_1'] = X2[:,0]
            dfc['pca_2'] = X2[:,1]
            dfc['pca_3'] = X3[:,2]

            st.success("✅ Clustering complete!")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                k = len(set(labels)) - (1 if -1 in labels else 0)
                st.metric("Clusters", k)
            with c2:
                st.metric("Silhouette", f"{sil:.3f}" if sil is not None else "N/A")
            with c3:
                st.metric("Largest Cluster Size", pd.Series(labels).value_counts().iloc[0] if len(labels)>0 else 0)
            with c4:
                st.metric("Noise Points", int((labels == -1).sum()) if -1 in labels else 0)

            st.markdown("#### 🎯 2D PCA View")
            fig = px.scatter(x=X2[:,0], y=X2[:,1], color=[str(l) for l in labels],
                             title=f"2D PCA Clustering View - {algorithm}",
                             labels={'x':f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
                                     'y':f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})'},
                             hover_data={
                                 'Incident Date': df['incident_date'] if 'incident_date' in df.columns else None,
                                 'Location': df['location'] if 'location' in df.columns else None,
                                 'Severity': df['severity'] if 'severity' in df.columns else None
                             })
            fig.update_layout(height=580)
            st.plotly_chart(fig, use_container_width=True)

            if show_3d:
                st.markdown("#### 📊 3D PCA View")
                fig = px.scatter_3d(x=X3[:,0], y=X3[:,1], z=X3[:,2], color=[str(l) for l in labels],
                                    title=f"3D PCA Clustering View - {algorithm}",
                                    labels={'x':f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
                                            'y':f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
                                            'z':f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'})
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📋 Cluster Characteristics")
            rows = []
            for cl in sorted(set(labels)):
                if cl == -1:
                    continue
                sub = dfc[dfc['cluster'] == cl]
                if sub.empty: 
                    continue
                row = {'Cluster': cl, 'Size': len(sub), 'Size %': f"{100*len(sub)/len(dfc):.1f}%"}
                for col in ['location','incident_type','severity']:
                    if col in sub.columns and not sub[col].empty:
                        row[f'Most Common {col.title()}'] = sub[col].mode().iloc[0]
                for col in ['medical_attention_required','reportable']:
                    if col in sub.columns:
                        row[f'Avg {col.replace("_"," ").title()}'] = f"{ensure_bool01(sub[col]).mean():.1%}"
                if 'hour' in sub.columns:
                    row['Avg Hour'] = f"{sub['hour'].mean():.1f}"
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("#### 🎯 PCA Component Analysis")
            comp = pd.DataFrame(pca_3d.components_[:3].T, columns=['PC1','PC2','PC3'], index=feature_names)
            for pc in ['PC1','PC2','PC3']:
                st.write(f"**{pc} Top Contributors:**")
                top = comp[pc].abs().nlargest(5)
                for feat, contrib in top.items():
                    sign = "+" if comp.loc[feat, pc] > 0 else "-"
                    st.write(f"  {sign} {feat}: {abs(contrib):.3f}")

            st.markdown("#### 💡 Clustering Insights")
            sizes = pd.Series(labels).value_counts()
            if len(sizes) > 1:
                largest_pct = 100 * sizes.iloc[0] / len(labels)
                if largest_pct > 70:
                    st.write(f"📊 **Dominant Pattern**: One cluster contains {largest_pct:.1f}% of incidents.")
                elif largest_pct < 30 and len(sizes) >= 4:
                    st.write("📊 **Diverse Patterns**: Well-distributed clusters indicate diverse incident patterns.")
            if sil is not None:
                if sil > 0.7:
                    st.write(f"✅ **Excellent Clustering**: Silhouette {sil:.3f}.")
                elif sil > 0.5:
                    st.write(f"👍 **Good Clustering**: Silhouette {sil:.3f}.")
                elif sil > 0.25:
                    st.write(f"⚠️ **Moderate Clustering**: Silhouette {sil:.3f}.")
                else:
                    st.write(f"❌ **Poor Clustering**: Silhouette {sil:.3f}.")

# ----------------------------
# Reporter metrics (optional small KPIs using carer_id fields)
# ----------------------------
def plot_reporter_type_metrics(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'carer_id' in df.columns:
            value = df['carer_id'].nunique()
            plot_metric("Carers (unique)", value)
    with col2:
        if 'medical_attention_required' in df.columns:
            value = int(ensure_bool01(df['medical_attention_required']).sum())
            plot_metric("Medical Attention Required", value)
    with col3:
        if 'participant_age' in df.columns:
            avg_age = round(df['participant_age'].mean(), 1)
            plot_metric("Avg Participant Age", avg_age, suffix=" yrs")

# ----------------------------
# Main App
# ----------------------------
def main():
    st.markdown("<div class='main-header'>NDIS Advanced Analytics Dashboard</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        forecast_horizon = st.slider("Forecast months", 3, 12, 6, 1)
        top_n_causes = st.slider("Top N causes (time chart)", 3, 10, 5, 1)

    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV to begin. Expected columns like: incident_date, incident_time, location, severity, medical_attention_required, reportable, participant_id, carer_id, incident_type, notification_date.")
        return

    # Quick data preview
    with st.expander("👀 Data preview"):
        st.dataframe(df.head(30), use_container_width=True)

    # Optional KPI cards
    plot_reporter_type_metrics(df)

    # Forecasting
    incident_volume_forecasting(df, periods=forecast_horizon)

    # Location Risk
    location_risk_profiling(df)

    # Seasonal & temporal
    seasonal_temporal_patterns(df)

    # Time + causes combo
    st.markdown("### ⏰ Time vs Causes (Stacked Bars + Total Line)")
    plot_time_with_causes(df, cause_col=None, top_n=top_n_causes)

    # Carer performance scatter
    st.markdown("### 🧑‍⚕️ Carer Performance Scatter")
    plot_carer_performance_scatter(df)

    # Features for clustering / modeling / correlation
    X, feature_names, features_df = create_comprehensive_features(df)
    if X is not None and features_df is not None:
        # Correlation
        correlation_analysis(X, feature_names, features_df)

        # Clustering
        clustering_analysis(X, features_df, feature_names)

        # Predictive models (if severity available)
        if 'severity' in df.columns:
            y = norm_severity(df['severity']).fillna(0).astype(int)
            predictive_models_comparison(X, y, feature_names, target_name="severity")

if __name__ == "__main__":
    main()





# --- Export aliases for dashboard compatibility ---

# Hard aliases to differently-named helpers (these DO exist above)
prepare_ml_features        = create_comprehensive_features
compare_models             = predictive_models_comparison
forecast_incident_volume   = incident_volume_forecasting
profile_location_risk      = location_risk_profiling
detect_seasonal_patterns   = seasonal_temporal_patterns
perform_clustering_analysis= clustering_analysis

# If your dashboard expects plot_correlation_heatmap but you implemented correlation_analysis,
# alias it. (Your UI already guards for either a fig or a matrix.)
plot_correlation_heatmap   = correlation_analysis

# Optional functions — set to None if missing so imports don’t crash
try:
    analyze_cluster_characteristics
except NameError:
    analyze_cluster_characteristics = None

try:
    plot_3d_clusters
except NameError:
    plot_3d_clusters = None

try:
    train_severity_prediction_model
except NameError:
    train_severity_prediction_model = None

try:
    perform_anomaly_detection
except NameError:
    perform_anomaly_detection = None

try:
    plot_anomaly_scatter
except NameError:
    plot_anomaly_scatter = None

# Dashboard imports this name; ensure it exists. If your real function is named
# `incident_type_risk_profiling`, alias it; if you don’t have it, provide a harmless stub.
try:
    incident_type_risk_profiling
except NameError:
    def incident_type_risk_profiling(df):
        return (pd.DataFrame(), None)

profile_incident_type_risk = incident_type_risk_profiling
                    # or None




