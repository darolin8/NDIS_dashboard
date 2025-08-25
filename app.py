import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st

from dashboard_pages import PAGE_TO_RENDERER

st.set_page_config(page_title="NDIS Executive Dashboard", page_icon="📊", layout="wide")

page = st.sidebar.radio("Select a page:", list(PAGE_TO_RENDERER.keys()))

renderer = PAGE_TO_RENDERER[page]
renderer()


# Optional minimal CSS tune-up (colors align with your NDIS palette)
st.markdown("""
<style>
:root {
  --primary: #003F5C;
  --secondary: #2F9E7D;
  --accent: #F59C2F;
  --critical: #DC2626;
  --text: #1B1B1B;
  --bg: #F7F9FA;
}
.main > div { background-color: var(--bg); padding-top: 0.8rem; }
h1, h2, h3 { color: var(--primary) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# ML Helper Functions (kept from your earlier code, just compact)
# =========================
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

# Association rules
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

# Forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

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
    # time-based
    if 'incident_date' in features_df.columns:
        features_df['day_of_week'] = features_df['incident_date'].dt.dayofweek
        features_df['month'] = features_df['incident_date'].dt.month
        features_df['hour'] = pd.to_datetime(features_df['incident_time'], format='%H:%M', errors='coerce').dt.hour
    num_cols = [c for c in [
        'day_of_week','month','hour',
        'location_encoded','incident_type_encoded','contributing_factors_encoded','reported_by_encoded',
        'reporting_delay_hours','age_at_incident'
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
    sev_map = {'Low':0,'Medium':1,'High':2,'Critical':3}
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
def find_association_rules(df: pd.DataFrame):
    if not MLXTEND_AVAILABLE or df.empty or len(df) < 20:
        return None, None
    transactions = []
    for _, row in df.iterrows():
        t = []
        if pd.notna(row.get('location')): t.append(f"location_{row['location']}")
        if pd.notna(row.get('incident_type')): t.append(f"type_{row['incident_type']}")
        if pd.notna(row.get('severity')): t.append(f"severity_{row['severity']}")
        if pd.notna(row.get('contributing_factors')): t.append(f"factor_{row['contributing_factors']}")
        if row.get('medical_attention_required') == 'Yes': t.append('medical_required')
        if row.get('reportable') == 'Yes': t.append('reportable')
        if bool(row.get('same_day_reporting', False)): t.append('same_day_reported')
        if t: transactions.append(t)
    if not transactions:
        return None, None
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_enc = pd.DataFrame(te_ary, columns=te.columns_)
    fi = apriori(df_enc, min_support=0.1, use_colnames=True)
    if fi.empty:
        return None, None
    rules = association_rules(fi, metric="confidence", min_threshold=0.5)
    return fi, rules

@st.cache_data
def time_series_forecast(df: pd.DataFrame, periods=30):
    if not STATSMODELS_AVAILABLE or df.empty:
        return None, None
    daily = df.groupby(df['incident_date'].dt.date).size().reset_index()
    daily.columns = ['date','incident_count']
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.set_index('date').sort_index()
    if len(daily) < 30:
        return None, None
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(date_range, fill_value=0)
    model = ExponentialSmoothing(daily['incident_count'], trend='add', seasonal=None)
    fitted = model.fit()
    fc = fitted.forecast(periods)
    fc_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=periods, freq='D')
    fc_df = pd.DataFrame({'date': fc_dates, 'forecast': fc})
    return daily.to_frame(), fc_df

# =========================
# Data loading + prep
# =========================
@st.cache_data
def load_incident_data():
    # Try mounted path first (per your upload)
    paths = [
        '/mnt/data/ndis_incidents_synthetic.csv',
        'ndis_incidents_synthetic.csv',
        './ndis_incidents_synthetic.csv',
        'data/ndis_incidents_synthetic.csv'
    ]
    df = None
    for p in paths:
        try:
            df = pd.read_csv(p)
            st.sidebar.success(f"✅ Data loaded from: {p}")
            break
        except Exception:
            continue
    if df is None:
        st.sidebar.error("CSV not found. Using generated sample data.")
        df = create_sample_data()

    # parse & enrich
    df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
    df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
    df['reporting_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
    df['same_day_reporting'] = df['reporting_delay_hours'] <= 24
    df['age_at_incident'] = (df['incident_date'] - df['dob']).dt.days / 365.25
    df['incident_month'] = df['incident_date'].dt.month_name()
    df['incident_year'] = df['incident_date'].dt.year
    return df

def create_sample_data(n=500):
    rng = np.random.default_rng(42)
    sample = {
        'incident_id': [f'INC-2024-{i:04d}' for i in range(1, n+1)],
        'participant_name': [f'Participant {i}' for i in range(1, n+1)],
        'ndis_number': rng.integers(400000000, 500000000, n),
        'dob': pd.date_range('1950-01-01', '2010-12-31', periods=n).strftime('%d/%m/%Y'),
        'incident_date': pd.date_range('2023-01-01', '2024-12-31', periods=n).strftime('%d/%m/%Y'),
        'incident_time': [f'{rng.integers(0,24):02d}:{rng.integers(0,60):02d}' for _ in range(n)],
        'notification_date': pd.date_range('2023-01-02', '2025-01-31', periods=n).strftime('%d/%m/%Y'),
        'location': rng.choice(['Group Home','Transport Vehicle','Day Program','Community Access','Therapy Clinic'], n),
        'incident_type': rng.choice(['Injury','Missing Person','Death','Restrictive Practices','Transport Incident','Medication Error'], n),
        'subcategory': rng.choice(['Fall','Unexplained absence','Natural causes','Unauthorised','Vehicle crash','Wrong dose'], n),
        'severity': rng.choice(['Critical','High','Medium','Low'], n, p=[0.1,0.2,0.4,0.3]),
        'reportable': rng.choice(['Yes','No'], n, p=[0.7,0.3]),
        'description': ['Sample incident description' for _ in range(n)],
        'immediate_action': ['Immediate action taken' for _ in range(n)],
        'actions_taken': ['Follow-up actions completed' for _ in range(n)],
        'contributing_factors': rng.choice(['Staff error','Equipment failure','Environmental factors','Participant behavior','System failure'], n),
        'reported_by': [f'Staff Member {i} (Support Worker)' for i in range(1, n+1)],
        'injury_type': rng.choice(['No physical injury','Minor injury','Major injury'], n, p=[0.6,0.3,0.1]),
        'injury_severity': rng.choice(['None','Mild','Moderate','Severe'], n, p=[0.5,0.3,0.15,0.05]),
        'treatment_required': rng.choice(['Yes','No'], n, p=[0.3,0.7]),
        'medical_attention_required': rng.choice(['Yes','No'], n, p=[0.25,0.75]),
        'medical_treatment_type': rng.choice(['None','First aid','GP visit','Hospital'], n, p=[0.6,0.25,0.1,0.05]),
        'medical_outcome': rng.choice(['No treatment required','Treated and released','Ongoing monitoring'], n, p=[0.7,0.25,0.05])
    }
    return pd.DataFrame(sample)

# =========================
# Load + Filter
# =========================
df = load_incident_data()

st.sidebar.title("🏥 NDIS Dashboard")
st.sidebar.markdown("---")

# Filters
min_date = df['incident_date'].min()
max_date = df['incident_date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

locations = ['All'] + sorted(df['location'].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

severities = st.sidebar.multiselect(
    "Severity",
    df['severity'].dropna().unique().tolist(),
    default=df['severity'].dropna().unique().tolist()
)

incident_types = st.sidebar.multiselect(
    "Incident Type",
    df['incident_type'].dropna().unique().tolist(),
    default=df['incident_type'].dropna().unique().tolist()
)

filtered_df = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['incident_date'] >= pd.Timestamp(date_range[0])) &
        (filtered_df['incident_date'] <= pd.Timestamp(date_range[1]))
    ]
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
if severities:
    filtered_df = filtered_df[filtered_df['severity'].isin(severities)]
if incident_types:
    filtered_df = filtered_df[filtered_df['incident_type'].isin(incident_types)]

# =========================
# Router
# =========================
page = st.sidebar.selectbox(
    "Dashboard Pages",
    ["Executive Summary", "Operational Performance", "Compliance & Investigation", "🤖 Machine Learning Analytics", "Risk Analysis"],
    index=0
)

renderer = PAGE_TO_RENDERER[page]
if page == "🤖 Machine Learning Analytics":
    renderer(
        filtered_df,
        train_severity_prediction_model=train_severity_prediction_model,
        prepare_ml_features=prepare_ml_features,
        perform_anomaly_detection=perform_anomaly_detection,
        find_association_rules=find_association_rules,
        time_series_forecast=time_series_forecast,
        MLXTEND_AVAILABLE=MLXTEND_AVAILABLE,
        STATSMODELS_AVAILABLE=STATSMODELS_AVAILABLE
    )
elif page == "Executive Summary":
    renderer(df, filtered_df)
else:
    renderer(filtered_df)

# =========================
# Footer
# =========================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Data Summary:** {len(df)} total incidents")
with col2:
    if not df.empty:
        date_range_str = f"{df['incident_date'].min().strftime('%d/%m/%Y')} - {df['incident_date'].max().strftime('%d/%m/%Y')}"
        st.markdown(f"**Date Range:** {date_range_str}")
with col3:
    from datetime import datetime
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Sidebar quick actions
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
if st.sidebar.button("📊 Export Current View"):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="ndis_incidents_filtered.csv",
        mime="text/csv"
    )
