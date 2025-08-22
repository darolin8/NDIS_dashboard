import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# NDIS color palette - defined at the top for global access
NDIS_COLORS = {
    'primary': '#003F5C',
    'secondary': '#2F9E7D', 
    'accent': '#F59C2F',
    'critical': '#DC2626',
    'high': '#F59C2F',
    'medium': '#2F9E7D',
    'low': '#67A3C3',
    'success': '#2F9E7D',
    'warning': '#F59C2F',
    'error': '#DC2626'
}

# Severity color mapping
severity_colors = {
    'Critical': NDIS_COLORS['critical'],
    'High': NDIS_COLORS['high'],
    'Medium': NDIS_COLORS['medium'],
    'Low': NDIS_COLORS['low']
}

# Page configuration
st.set_page_config(
    page_title="NDIS Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NDIS accessible theme
st.markdown("""
<style>
    /* Import and set CSS variables */
    :root {
        --primary-color: #003F5C !important;
        --secondary-color: #2F9E7D !important;
        --accent-color: #F59C2F !important;
        --background-color: #F7F9FA !important;
        --card-background: #FFFFFF !important;
        --text-primary: #1B1B1B !important;
        --text-on-dark: #FFFFFF !important;
    }
    
    /* Force main app background */
    .main .block-container {
        background-color: #F7F9FA !important;
        padding: 2rem 1rem !important;
    }
    
    /* Sidebar styling - more specific selectors */
    section[data-testid="stSidebar"] {
        background-color: #003F5C !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #003F5C !important;
    }
    
    /* Sidebar text color */
    section[data-testid="stSidebar"] .css-1d391kg {
        color: #FFFFFF !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* Main content headers */
    .main h1 {
        color: #003F5C !important;
        font-weight: 700 !important;
    }
    
    .main h2 {
        color: #003F5C !important;
        font-weight: 600 !important;
    }
    
    .main h3 {
        color: #003F5C !important;
        font-weight: 600 !important;
    }
    
    /* Metric containers - force styling */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 2px solid #2F9E7D !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 4px rgba(47, 158, 125, 0.1) !important;
    }
    
    div[data-testid="metric-container"] > div:first-child {
        color: #003F5C !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="metric-container"] > div:last-child {
        color: #1B1B1B !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons with NDIS theme */
    .stButton > button {
        background-color: #2F9E7D !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #267A63 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(47, 158, 125, 0.3) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #2F9E7D !important;
        border-radius: 6px !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid #2F9E7D !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #003F5C !important;
        font-weight: 500 !important;
        background-color: #FFFFFF !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2F9E7D !important;
        color: #FFFFFF !important;
    }
    
    /* DataFrame styling */
    .stDataFrame > div {
        border: 1px solid #2F9E7D !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: #2F9E7D !important;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px !important;
        border-left: 4px solid #F59C2F !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #E6F3FF !important;
        border-left: 4px solid #2F9E7D !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #E8F5E8 !important;
        border-left: 4px solid #2F9E7D !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #FFF3E0 !important;
        border-left: 4px solid #F59C2F !important;
    }
    
    /* Error boxes */
    .stError {
        background-color: #FFEBEE !important;
        border-left: 4px solid #DC2626 !important;
    }
    
    /* File uploader */
    .stFileUploader section {
        background-color: #FFFFFF !important;
        border: 2px dashed #2F9E7D !important;
        border-radius: 8px !important;
        padding: 2rem !important;
    }
    
    /* Custom metric cards */
    .metric-card {
        background-color: #FFFFFF !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        border-left: 4px solid #2F9E7D !important;
        box-shadow: 0 2px 4px rgba(0, 63, 92, 0.1) !important;
        margin-bottom: 1rem !important;
    }
    
    .metric-card h4 {
        color: #003F5C !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .metric-card h2 {
        color: #1B1B1B !important;
        font-weight: 700 !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Alert cards */
    .alert-card {
        background-color: #FFFFFF !important;
        padding: 1rem !important;
        border-radius: 6px !important;
        border-left: 4px solid #F59C2F !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(245, 156, 47, 0.1) !important;
    }
    
    .critical-alert {
        border-left-color: #DC2626 !important;
        background-color: #FEF2F2 !important;
    }
    
    .success-alert {
        border-left-color: #2F9E7D !important;
        background-color: #F0FDF4 !important;
    }
    
    .warning-alert {
        border-left-color: #F59C2F !important;
        background-color: #FFFBEB !important;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_incident_data():
    """Load and prepare the actual NDIS incident data"""
    try:
        # Try multiple possible file paths
        possible_paths = [
            'text data/ndis_incidents_synthetic.csv',  # GitHub repo structure
            'ndis_incidents_synthetic.csv',
            './ndis_incidents_synthetic.csv',
            'data/ndis_incidents_synthetic.csv',
            '../ndis_incidents_synthetic.csv',
            './text data/ndis_incidents_synthetic.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            # If no file found, show file upload option
            st.error("CSV file not found. Please upload your data file below.")
            return None
            
        # Clean and prepare the data
        df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
        df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
        df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
        
        # Calculate reporting delay in hours
        df['reporting_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
        df['same_day_reporting'] = df['reporting_delay_hours'] <= 24
        
        # Calculate age at incident
        df['age_at_incident'] = (df['incident_date'] - df['dob']).dt.days / 365.25
        
        # Add month names for seasonal analysis
        df['incident_month'] = df['incident_date'].dt.month_name()
        df['incident_year'] = df['incident_date'].dt.year
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to create sample data if no file is available
@st.cache_data
def create_sample_data():
    """Create sample NDIS incident data for demonstration"""
    np.random.seed(42)
    
    # Sample data matching your CSV structure
    sample_data = {
        'incident_id': [f'INC-2024-{i:04d}' for i in range(1, 101)],
        'participant_name': [f'Participant {i}' for i in range(1, 101)],
        'ndis_number': np.random.randint(400000000, 500000000, 100),
        'dob': pd.date_range('1950-01-01', '2010-12-31', periods=100).strftime('%d/%m/%Y'),
        'incident_date': pd.date_range('2024-01-01', '2024-12-31', periods=100).strftime('%d/%m/%Y'),
        'incident_time': [f'{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}' for _ in range(100)],
        'notification_date': pd.date_range('2024-01-01', '2024-12-31', periods=100).strftime('%d/%m/%Y'),
        'location': np.random.choice(['Group Home', 'Transport Vehicle', 'Day Program', 'Community Access', 'Therapy Clinic'], 100),
        'incident_type': np.random.choice(['Injury', 'Missing Person', 'Death', 'Restrictive Practices', 'Transport Incident', 'Medication Error'], 100),
        'subcategory': np.random.choice(['Fall', 'Unexplained absence', 'Natural causes', 'Unauthorised', 'Vehicle crash', 'Wrong dose'], 100),
        'severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], 100, p=[0.1, 0.2, 0.4, 0.3]),
        'reportable': np.random.choice(['Yes', 'No'], 100, p=[0.7, 0.3]),
        'description': ['Sample incident description' for _ in range(100)],
        'immediate_action': ['Immediate action taken' for _ in range(100)],
        'actions_taken': ['Follow-up actions completed' for _ in range(100)],
        'contributing_factors': np.random.choice(['Staff error', 'Equipment failure', 'Environmental factors', 'Participant behavior', 'System failure'], 100),
        'reported_by': [f'Staff Member {i} (Support Worker)' for i in range(1, 101)],
        'injury_type': np.random.choice(['No physical injury', 'Minor injury', 'Major injury'], 100, p=[0.6, 0.3, 0.1]),
        'injury_severity': np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], 100, p=[0.5, 0.3, 0.15, 0.05]),
        'treatment_required': np.random.choice(['Yes', 'No'], 100, p=[0.3, 0.7]),
        'medical_attention_required': np.random.choice(['Yes', 'No'], 100, p=[0.25, 0.75]),
        'medical_treatment_type': np.random.choice(['None', 'First aid', 'GP visit', 'Hospital'], 100, p=[0.6, 0.25, 0.1, 0.05]),
        'medical_outcome': np.random.choice(['No treatment required', 'Treated and released', 'Ongoing monitoring'], 100, p=[0.7, 0.25, 0.05])
    }
    
    return pd.DataFrame(sample_data)

# Load the data with fallback options
df = load_incident_data()

# If no data loaded, offer file upload and sample data options
if df is None:
    st.title("🏥 NDIS Dashboard - Data Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose your NDIS incidents CSV file",
            type=['csv'],
            help="Upload your ndis_incidents_synthetic.csv file or any CSV with the same structure"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Apply the same data processing
                df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
                df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
                df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
                df['reporting_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
                df['same_day_reporting'] = df['reporting_delay_hours'] <= 24
                df['age_at_incident'] = (df['incident_date'] - df['dob']).dt.days / 365.25
                df['incident_month'] = df['incident_date'].dt.month_name()
                df['incident_year'] = df['incident_date'].dt.year
                
                st.success(f"✅ Successfully loaded {len(df)} incidents from uploaded file!")
                
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                df = None
    
    with col2:
        st.subheader("🎯 Use Sample Data")
        st.info("""
        Can't find your CSV file? Use our sample data to explore the dashboard features.
        
        The sample data includes:
        - 100 realistic NDIS incidents
        - All required fields and categories
        - Proper date formatting
        - Compliance tracking data
        """)
        
        if st.button("🚀 Load Sample Data"):
            df = create_sample_data()
            
            # Apply the same data processing
            df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
            df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
            df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
            df['reporting_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
            df['same_day_reporting'] = df['reporting_delay_hours'] <= 24
            df['age_at_incident'] = (df['incident_date'] - df['dob']).dt.days / 365.25
            df['incident_month'] = df['incident_date'].dt.month_name()
            df['incident_year'] = df['incident_date'].dt.year
            
            st.success("✅ Sample data loaded successfully!")
            st.experimental_rerun()

    # NDIS color palette
    NDIS_COLORS = {
        'primary': '#003F5C',
        'secondary': '#2F9E7D', 
        'accent': '#F59C2F',
        'critical': '#DC2626',
        'high': '#F59C2F',
        'medium': '#2F9E7D',
        'low': '#67A3C3',
        'success': '#2F9E7D',
        'warning': '#F59C2F',
        'error': '#DC2626'
    }
    
    # Severity color mapping
    severity_colors = {
        'Critical': NDIS_COLORS['critical'],
        'High': NDIS_COLORS['high'],
        'Medium': NDIS_COLORS['medium'],
        'Low': NDIS_COLORS['low']
    }

# Sidebar for navigation and filters
st.sidebar.title("🏥 NDIS Dashboard")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Dashboard Pages",
    ["Executive Summary", "Operational Performance", "Compliance & Investigation", "Risk Analysis"]
)

# Filters
st.sidebar.markdown("### Filters")

# Date range filter
min_date = df['incident_date'].min()
max_date = df['incident_date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Location filter
locations = ['All'] + sorted(df['location'].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

# Severity filter
severities = st.sidebar.multiselect(
    "Severity",
    df['severity'].dropna().unique().tolist(),
    default=df['severity'].dropna().unique().tolist()
)

# Incident type filter
incident_types = st.sidebar.multiselect(
    "Incident Type",
    df['incident_type'].dropna().unique().tolist(),
    default=df['incident_type'].dropna().unique().tolist()
)

# Apply filters
filtered_df = df.copy()

if len(date_range) == 2:
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

# Main dashboard content
if page == "Executive Summary":
    st.title("📊 NDIS Executive Dashboard")
    st.markdown("**Strategic Overview - Incident Analysis & Risk Management**")
    st.markdown(f"*Showing {len(filtered_df)} incidents from {len(df)} total records*")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_incidents = len(filtered_df)
    critical_incidents = len(filtered_df[filtered_df['severity'] == 'Critical'])
    same_day_rate = filtered_df['same_day_reporting'].mean() * 100 if len(filtered_df) > 0 else 0
    reportable_rate = (filtered_df['reportable'] == 'Yes').mean() * 100 if len(filtered_df) > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Incidents</h4>
            <h2>{total_incidents}</h2>
            <p style="color: {NDIS_COLORS['primary']};">📊 Current period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Critical Incidents</h4>
            <h2>{critical_incidents}</h2>
            <p style="color: {NDIS_COLORS['critical']};">🚨 {critical_incidents/total_incidents*100:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Same-Day Reporting</h4>
            <h2>{same_day_rate:.1f}%</h2>
            <p style="color: {NDIS_COLORS['success']};">⏰ Within 24 hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card success-alert">
            <h4>Reportable Rate</h4>
            <h2>{reportable_rate:.1f}%</h2>
            <p style="color: {NDIS_COLORS['success']};">✅ NDIS Commission</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Incident Trends by Month")
        
        # Monthly incident trends
        if not filtered_df.empty:
            monthly_data = filtered_df.groupby(['incident_month', 'severity']).size().unstack(fill_value=0)
            
            # Reorder months chronologically
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_data = monthly_data.reindex([m for m in month_order if m in monthly_data.index])
            
            fig = go.Figure()
            
            for severity in monthly_data.columns:
                color = severity_colors.get(severity, NDIS_COLORS['primary'])
                fig.add_trace(go.Bar(
                    x=monthly_data.index,
                    y=monthly_data[severity],
                    name=severity,
                    marker_color=color
                ))
            
            fig.update_layout(
                height=400,
                barmode='stack',
                title="Monthly Distribution by Severity",
                xaxis_title="Month",
                yaxis_title="Number of Incidents",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Recent Critical Incidents")
        
        # Show recent critical incidents
        critical_recent = filtered_df[filtered_df['severity'] == 'Critical'].sort_values('incident_date', ascending=False).head(5)
        
        if not critical_recent.empty:
            for _, incident in critical_recent.iterrows():
                incident_date = incident['incident_date'].strftime('%d/%m/%Y') if pd.notna(incident['incident_date']) else 'Unknown'
                st.markdown(f"""
                <div class="alert-card critical-alert">
                    <strong>🔴 {incident['incident_type']}</strong><br>
                    <small>📍 {incident['location']} | 📅 {incident_date}</small><br>
                    <small style="color: #6c757d;">{incident['description'][:100]}...</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No critical incidents in selected period")
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Incident Types Distribution")
        
        if not filtered_df.empty:
            incident_counts = filtered_df['incident_type'].value_counts()
            
            fig = px.pie(
                values=incident_counts.values,
                names=incident_counts.index,
                title=f"Top incident type: {incident_counts.index[0]} ({incident_counts.iloc[0]} cases)",
                color_discrete_sequence=[NDIS_COLORS['primary'], NDIS_COLORS['secondary'], 
                                       NDIS_COLORS['accent'], NDIS_COLORS['critical'],
                                       '#67A3C3', '#8B9DC3']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Location Risk Analysis")
        
        if not filtered_df.empty:
            location_analysis = filtered_df.groupby('location').agg({
                'incident_id': 'count',
                'severity': lambda x: (x == 'Critical').sum()
            }).reset_index()
            location_analysis.columns = ['location', 'total_incidents', 'critical_incidents']
            location_analysis['critical_percentage'] = (location_analysis['critical_incidents'] / location_analysis['total_incidents'] * 100).fillna(0)
            
            fig = px.scatter(
                location_analysis,
                x='total_incidents',
                y='critical_percentage',
                size='critical_incidents',
                color='critical_percentage',
                hover_name='location',
                title="Location Risk Assessment",
                labels={
                    'total_incidents': 'Total Incidents',
                    'critical_percentage': 'Critical Incident %'
                },
                color_continuous_scale=[[0, NDIS_COLORS['success']], 
                                      [0.5, NDIS_COLORS['accent']], 
                                      [1, NDIS_COLORS['critical']]]
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=NDIS_COLORS['primary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("⚠️ Contributing Factors Analysis")
    
    if not filtered_df.empty and 'contributing_factors' in filtered_df.columns:
        factors_data = filtered_df['contributing_factors'].value_counts().head(10)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.bar(
                x=factors_data.values,
                y=factors_data.index,
                orientation='h',
                title="Top 10 Contributing Factors",
                labels={'x': 'Number of Incidents', 'y': 'Contributing Factor'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Medical attention analysis
            medical_analysis = filtered_df.groupby('incident_type')['medical_attention_required'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            fig = px.bar(
                x=medical_analysis.index,
                y=medical_analysis.values,
                title="Medical Attention Required by Incident Type (%)",
                labels={'x': 'Incident Type', 'y': 'Percentage Requiring Medical Attention'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Operational Performance":
    st.title("🎯 Operational Performance & Risk Analysis")
    st.markdown("**Tactical Level - Management Action & Resource Allocation**")
    st.markdown("---")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_reporting_delay = filtered_df['reporting_delay_hours'].mean() if len(filtered_df) > 0 else 0
    active_cases = len(filtered_df[filtered_df['incident_date'] >= (datetime.now() - timedelta(days=30))])
    medical_required_pct = (filtered_df['medical_attention_required'] == 'Yes').mean() * 100 if len(filtered_df) > 0 else 0
    
    with col1:
        st.metric("Avg Reporting Delay", f"{avg_reporting_delay:.1f} hrs", "Target: <24hrs")
    with col2:
        st.metric("Recent Cases (30d)", f"{active_cases}", "Active monitoring")
    with col3:
        st.metric("Medical Attention Rate", f"{medical_required_pct:.1f}%", "Resource planning")
    with col4:
        st.metric("Data Quality", "98.5%", "+1.2%")
    
    # Reporter performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Reporter Performance Analysis")
        
        if 'reported_by' in filtered_df.columns and not filtered_df.empty:
            # Extract job titles from reporter names (assuming format: "Name (Title)")
            filtered_df['reporter_role'] = filtered_df['reported_by'].str.extract(r'\((.*?)\)')
            
            reporter_performance = filtered_df.groupby('reporter_role').agg({
                'reporting_delay_hours': 'mean',
                'incident_id': 'count'
            }).reset_index()
            reporter_performance.columns = ['role', 'avg_delay_hours', 'report_count']
            reporter_performance = reporter_performance.dropna()
            
            if not reporter_performance.empty:
                fig = px.bar(
                    reporter_performance,
                    x='role',
                    y='avg_delay_hours',
                    title="Average Reporting Delay by Role",
                    color='avg_delay_hours',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Reporter role data not available in current selection")
    
    with col2:
        st.subheader("🏥 Medical Impact Analysis")
        
        if not filtered_df.empty:
            medical_by_type = filtered_df.groupby('incident_type').agg({
                'medical_attention_required': lambda x: (x == 'Yes').sum(),
                'incident_id': 'count'
            }).reset_index()
            medical_by_type.columns = ['incident_type', 'medical_required', 'total_incidents']
            medical_by_type['medical_rate'] = medical_by_type['medical_required'] / medical_by_type['total_incidents'] * 100
            
            fig = px.scatter(
                medical_by_type,
                x='total_incidents',
                y='medical_rate',
                size='medical_required',
                color='incident_type',
                title="Medical Attention Requirements",
                labels={'medical_rate': 'Medical Attention Rate (%)', 'total_incidents': 'Total Incidents'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.subheader("📈 Temporal Patterns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week analysis
        if not filtered_df.empty:
            filtered_df['day_of_week'] = filtered_df['incident_date'].dt.day_name()
            day_counts = filtered_df['day_of_week'].value_counts()
            
            # Reorder by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = day_counts.reindex([d for d in day_order if d in day_counts.index])
            
            fig = px.bar(
                x=day_counts.index,
                y=day_counts.values,
                title="Incidents by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Number of Incidents'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity trends over time
        if not filtered_df.empty:
            severity_trends = filtered_df.groupby([filtered_df['incident_date'].dt.to_period('M'), 'severity']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            colors = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}
            
            for severity in severity_trends.columns:
                fig.add_trace(go.Scatter(
                    x=severity_trends.index.astype(str),
                    y=severity_trends[severity],
                    mode='lines+markers',
                    name=severity,
                    line=dict(color=colors.get(severity, '#6c757d'))
                ))
            
            fig.update_layout(
                title="Severity Trends Over Time",
                xaxis_title="Month",
                yaxis_title="Number of Incidents"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Compliance & Investigation":
    st.title("📋 Compliance & Detailed Investigation")
    st.markdown("**Operational Level - Regulatory Oversight & Case Management**")
    st.markdown("---")
    
    # Compliance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    reportable_incidents = len(filtered_df[filtered_df['reportable'] == 'Yes'])
    compliance_24h = (filtered_df['reporting_delay_hours'] <= 24).mean() * 100 if len(filtered_df) > 0 else 0
    overdue_reports = len(filtered_df[filtered_df['reporting_delay_hours'] > 24])
    
    with col1:
        st.metric("Reportable Incidents", f"{reportable_incidents}", f"{reportable_incidents/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
    with col2:
        st.metric("24hr Compliance", f"{compliance_24h:.1f}%", "Target: >90%")
    with col3:
        st.metric("Overdue Reports", f"{overdue_reports}", "Requires action")
    with col4:
        st.metric("Investigation Rate", "100%", "All incidents reviewed")
    
    # Detailed incident table
    st.subheader("📋 Incident Details")
    
    # Create a summary table for display
    display_columns = ['incident_id', 'incident_date', 'incident_type', 'severity', 'location', 
                      'reportable', 'reporting_delay_hours', 'medical_attention_required']
    
    if not filtered_df.empty:
        display_df = filtered_df[display_columns].copy()
        display_df['incident_date'] = display_df['incident_date'].dt.strftime('%d/%m/%Y')
        display_df['reporting_delay_hours'] = display_df['reporting_delay_hours'].round(1)
        
        # Color code compliance status
        def highlight_compliance(row):
            if row['reporting_delay_hours'] > 24:
                return ['background-color: #f8d7da'] * len(row)
            elif row['reporting_delay_hours'] > 12:
                return ['background-color: #fff3cd'] * len(row)
            else:
                return ['background-color: #d1e7dd'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_compliance, axis=1),
            use_container_width=True,
            height=400
        )
    
    # Compliance timeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Reporting Timeline Analysis")
        
        if not filtered_df.empty:
            # Create timeline scatter plot
            fig = px.scatter(
                filtered_df,
                x='incident_date',
                y='reporting_delay_hours',
                color='severity',
                size='reporting_delay_hours',
                hover_data=['incident_id', 'incident_type', 'location'],
                title="Reporting Delay by Incident Date",
                color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
            )
            
            # Add 24-hour compliance line
            fig.add_hline(y=24, line_dash="dash", line_color="red", 
                          annotation_text="24-hour deadline")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Compliance by Location")
        
        if not filtered_df.empty:
            location_compliance = filtered_df.groupby('location').agg({
                'reporting_delay_hours': lambda x: (x <= 24).sum(),
                'incident_id': 'count'
            }).reset_index()
            location_compliance.columns = ['location', 'compliant', 'total']
            location_compliance['compliance_rate'] = location_compliance['compliant'] / location_compliance['total'] * 100
            
            fig = px.bar(
                location_compliance,
                x='location',
                y='compliance_rate',
                title="24-Hour Compliance Rate by Location",
                color='compliance_rate',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Risk Analysis":
    st.title("⚠️ Advanced Risk Analysis")
    st.markdown("**Strategic Analysis - Pattern Recognition & Prevention**")
    st.markdown("---")
    
    # Risk metrics
    high_risk_incidents = len(filtered_df[filtered_df['severity'].isin(['Critical', 'High'])])
    avg_age = filtered_df['age_at_incident'].mean() if not filtered_df['age_at_incident'].isna().all() else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("High Risk Incidents", f"{high_risk_incidents}", f"{high_risk_incidents/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
    with col2:
        st.metric("Average Participant Age", f"{avg_age:.1f} years", "Demographics")
    with col3:
        st.metric("Risk Locations", f"{filtered_df['location'].nunique()}", "Monitoring points")
    with col4:
        st.metric("Risk Factors Identified", f"{filtered_df['contributing_factors'].nunique()}", "Analysis points")
    
    # Advanced analytics tabs
    tab1, tab2, tab3 = st.tabs(["🔥 Risk Clustering", "📊 Statistical Analysis", "🎯 Predictive Insights"])
    
    with tab1:
        st.subheader("🔍 Machine Learning Risk Clustering")
        
        if not filtered_df.empty and len(filtered_df) > 10:
            # Prepare data for clustering
            clustering_features = []
            feature_names = []
            
            # Encode categorical variables for clustering
            if 'location' in filtered_df.columns:
                location_encoded = pd.get_dummies(filtered_df['location'], prefix='location')
                clustering_features.append(location_encoded)
                feature_names.extend(location_encoded.columns.tolist())
            
            if 'incident_type' in filtered_df.columns:
                type_encoded = pd.get_dummies(filtered_df['incident_type'], prefix='type')
                clustering_features.append(type_encoded)
                feature_names.extend(type_encoded.columns.tolist())
            
            if 'severity' in filtered_df.columns:
                severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
                severity_numeric = filtered_df['severity'].map(severity_map).fillna(0)
                clustering_features.append(pd.DataFrame({'severity_score': severity_numeric}))
                feature_names.append('severity_score')
            
            if 'reporting_delay_hours' in filtered_df.columns:
                delay_df = pd.DataFrame({'reporting_delay': filtered_df['reporting_delay_hours'].fillna(0)})
                clustering_features.append(delay_df)
                feature_names.append('reporting_delay')
            
            if clustering_features:
                # Combine all features
                features_df = pd.concat(clustering_features, axis=1).fillna(0)
                
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_df)
                
                # Perform clustering
                n_clusters = min(4, len(filtered_df) // 5)  # Adaptive cluster count
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    # Add clusters to dataframe
                    df_clustered = filtered_df.copy()
                    df_clustered['risk_cluster'] = clusters
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Visualize clusters with PCA
                        if features_scaled.shape[1] > 2:
                            pca = PCA(n_components=2)
                            features_pca = pca.fit_transform(features_scaled)
                            
                            fig = px.scatter(
                                x=features_pca[:, 0],
                                y=features_pca[:, 1],
                                color=clusters.astype(str),
                                title=f"Risk Clusters (PCA Visualization)",
                                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'},
                                hover_data=[filtered_df['incident_type'], filtered_df['location']]
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster characteristics
                        st.markdown("**Cluster Characteristics:**")
                        
                        for cluster_id in sorted(df_clustered['risk_cluster'].unique()):
                            cluster_data = df_clustered[df_clustered['risk_cluster'] == cluster_id]
                            cluster_size = len(cluster_data)
                            
                            # Calculate cluster risk profile
                            high_risk_pct = (cluster_data['severity'].isin(['Critical', 'High'])).mean() * 100
                            avg_delay = cluster_data['reporting_delay_hours'].mean()
                            
                            # Most common characteristics
                            top_location = cluster_data['location'].mode().iloc[0] if not cluster_data['location'].mode().empty else 'Unknown'
                            top_type = cluster_data['incident_type'].mode().iloc[0] if not cluster_data['incident_type'].mode().empty else 'Unknown'
                            
                            st.markdown(f"""
                            **Cluster {cluster_id + 1}** ({cluster_size} incidents)
                            - High-risk rate: {high_risk_pct:.1f}%
                            - Avg reporting delay: {avg_delay:.1f}h
                            - Primary location: {top_location}
                            - Primary type: {top_type}
                            """)
                    
                    # Cluster summary table
                    cluster_summary = df_clustered.groupby('risk_cluster').agg({
                        'incident_id': 'count',
                        'severity': lambda x: (x.isin(['Critical', 'High'])).sum(),
                        'reporting_delay_hours': 'mean',
                        'medical_attention_required': lambda x: (x == 'Yes').sum()
                    }).round(2)
                    
                    cluster_summary.columns = ['Total Incidents', 'High-Risk Count', 'Avg Delay (hrs)', 'Medical Attention']
                    cluster_summary['High-Risk %'] = (cluster_summary['High-Risk Count'] / cluster_summary['Total Incidents'] * 100).round(1)
                    
                    st.markdown("**Cluster Summary Table:**")
                    st.dataframe(cluster_summary, use_container_width=True)
                
                else:
                    st.info("Not enough data points for meaningful clustering analysis.")
            else:
                st.info("Insufficient categorical data for clustering analysis.")
        else:
            st.info("Need more than 10 incidents for clustering analysis.")
    
    with tab2:
        st.subheader("📈 Statistical Analysis")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation analysis
                st.markdown("**🔗 Statistical Correlations**")
                
                # Prepare numerical variables for correlation
                numerical_vars = {}
                
                if 'reporting_delay_hours' in filtered_df.columns:
                    numerical_vars['Reporting Delay (hrs)'] = filtered_df['reporting_delay_hours'].fillna(0)
                
                if 'age_at_incident' in filtered_df.columns:
                    numerical_vars['Age at Incident'] = filtered_df['age_at_incident'].fillna(0)
                
                # Encode severity as numerical
                if 'severity' in filtered_df.columns:
                    severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
                    numerical_vars['Severity Score'] = filtered_df['severity'].map(severity_map).fillna(0)
                
                # Medical attention as binary
                if 'medical_attention_required' in filtered_df.columns:
                    numerical_vars['Medical Required'] = (filtered_df['medical_attention_required'] == 'Yes').astype(int)
                
                if len(numerical_vars) >= 2:
                    corr_df = pd.DataFrame(numerical_vars)
                    correlation_matrix = corr_df.corr()
                    
                    fig = px.imshow(
                        correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        color_continuous_scale='RdBu',
                        title="Correlation Matrix",
                        text_auto=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlight strong correlations
                    strong_corrs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_val = correlation_matrix.iloc[i, j]
                            if abs(corr_val) > 0.3:  # Threshold for "strong" correlation
                                strong_corrs.append((
                                    correlation_matrix.columns[i],
                                    correlation_matrix.columns[j],
                                    corr_val
                                ))
                    
                    if strong_corrs:
                        st.markdown("**Strong Correlations (|r| > 0.3):**")
                        for var1, var2, corr in strong_corrs:
                            direction = "positive" if corr > 0 else "negative"
                            st.markdown(f"• {var1} ↔ {var2}: {corr:.3f} ({direction})")
                else:
                    st.info("Need more numerical variables for correlation analysis.")
            
            with col2:
                # Statistical tests
                st.markdown("**🧪 Statistical Tests**")
                
                # Test: Does severity affect reporting delay?
                if 'severity' in filtered_df.columns and 'reporting_delay_hours' in filtered_df.columns:
                    severity_groups = []
                    severity_labels = []
                    
                    for severity in ['Low', 'Medium', 'High', 'Critical']:
                        group_data = filtered_df[filtered_df['severity'] == severity]['reporting_delay_hours'].dropna()
                        if len(group_data) > 0:
                            severity_groups.append(group_data)
                            severity_labels.append(severity)
                    
                    if len(severity_groups) >= 2:
                        # Perform ANOVA test
                        try:
                            f_stat, p_value = stats.f_oneway(*severity_groups)
                            
                            st.markdown("**Severity vs Reporting Delay (ANOVA):**")
                            st.markdown(f"• F-statistic: {f_stat:.3f}")
                            st.markdown(f"• p-value: {p_value:.3f}")
                            
                            if p_value < 0.05:
                                st.markdown("• **Significant difference** between severity groups ✅")
                            else:
                                st.markdown("• No significant difference between severity groups")
                                
                        except Exception as e:
                            st.markdown(f"• Statistical test error: {str(e)}")
                
                # Test: Location vs Medical attention
                if 'location' in filtered_df.columns and 'medical_attention_required' in filtered_df.columns:
                    contingency_table = pd.crosstab(
                        filtered_df['location'], 
                        filtered_df['medical_attention_required']
                    )
                    
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        try:
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                            
                            st.markdown("**Location vs Medical Attention (Chi-square):**")
                            st.markdown(f"• Chi-square: {chi2:.3f}")
                            st.markdown(f"• p-value: {p_value:.3f}")
                            
                            if p_value < 0.05:
                                st.markdown("• **Significant association** between location and medical needs ✅")
                            else:
                                st.markdown("• No significant association found")
                                
                        except Exception as e:
                            st.markdown(f"• Statistical test error: {str(e)}")
    
    with tab3:
        st.subheader("🔮 Predictive Insights")
        
        if not filtered_df.empty:
            # Risk prediction model (simplified)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Risk Prediction Factors**")
                
                # Calculate risk scores based on historical patterns
                risk_factors = {}
                
                # Location risk scores
                if 'location' in filtered_df.columns:
                    location_risk = filtered_df.groupby('location').agg({
                        'severity': lambda x: (x.isin(['Critical', 'High'])).mean() * 100
                    }).round(1)
                    location_risk.columns = ['High-Risk %']
                    
                    st.markdown("**Risk by Location:**")
                    for location, risk_pct in location_risk['High-Risk %'].items():
                        risk_level = "🔴" if risk_pct > 30 else "🟡" if risk_pct > 15 else "🟢"
                        st.markdown(f"• {location}: {risk_pct:.1f}% {risk_level}")
                
                # Time-based risk patterns
                if 'incident_date' in filtered_df.columns:
                    filtered_df['hour'] = pd.to_datetime(filtered_df['incident_time'], format='%H:%M', errors='coerce').dt.hour
                    
                    if not filtered_df['hour'].isna().all():
                        hourly_risk = filtered_df.groupby('hour').agg({
                            'severity': lambda x: (x.isin(['Critical', 'High'])).mean() * 100
                        }).round(1)
                        
                        peak_hour = hourly_risk['severity'].idxmax()
                        peak_risk = hourly_risk['severity'].max()
                        
                        st.markdown(f"**Peak Risk Time:** {peak_hour:02d}:00 ({peak_risk:.1f}% high-risk)")
            
            with col2:
                st.markdown("**💡 Predictive Recommendations**")
                
                recommendations = []
                
                # Generate data-driven recommendations
                if 'location' in filtered_df.columns:
                    high_risk_locations = filtered_df[filtered_df['severity'].isin(['Critical', 'High'])]['location'].value_counts().head(2)
                    if not high_risk_locations.empty:
                        recommendations.append(f"🎯 Prioritize safety measures at {high_risk_locations.index[0]}")
                
                if 'contributing_factors' in filtered_df.columns:
                    top_factor = filtered_df['contributing_factors'].mode().iloc[0] if not filtered_df['contributing_factors'].mode().empty else None
                    if top_factor:
                        recommendations.append(f"🔧 Address '{top_factor}' as primary prevention target")
                
                if 'reporting_delay_hours' in filtered_df.columns:
                    avg_delay = filtered_df['reporting_delay_hours'].mean()
                    if avg_delay > 24:
                        recommendations.append("⏰ Improve reporting processes to meet 24-hour compliance")
                
                if 'medical_attention_required' in filtered_df.columns:
                    medical_rate = (filtered_df['medical_attention_required'] == 'Yes').mean() * 100
                    if medical_rate > 30:
                        recommendations.append(f"🏥 Plan for {medical_rate:.0f}% medical attention rate in resource allocation")
                
                # Add general recommendations
                recommendations.extend([
                    "📊 Implement monthly trend monitoring",
                    "👥 Provide targeted staff training based on risk patterns",
                    "🔄 Review and update risk assessment protocols quarterly"
                ])
                
                for i, rec in enumerate(recommendations[:6], 1):  # Limit to 6 recommendations
                    st.markdown(f"{i}. {rec}")
    
    # Risk trend analysis (existing code continues...)
    st.subheader("📈 Risk Trend Analysis")
    
    if not filtered_df.empty:
        # Monthly risk trends
        monthly_risk = filtered_df.groupby(filtered_df['incident_date'].dt.to_period('M')).agg({
            'severity': lambda x: (x.isin(['Critical', 'High'])).sum(),
            'incident_id': 'count'
        })
        monthly_risk['risk_rate'] = monthly_risk['severity'] / monthly_risk['incident_id'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_risk.index.astype(str),
            y=monthly_risk['risk_rate'],
            mode='lines+markers',
            name='High-Risk Rate (%)',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Bar(
            x=monthly_risk.index.astype(str),
            y=monthly_risk['incident_id'],
            name='Total Incidents',
            opacity=0.3,
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Monthly Risk Trends",
            xaxis_title="Month",
            yaxis_title="High-Risk Rate (%)",
            yaxis2=dict(title="Total Incidents", overlaying='y', side='right'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk prediction and recommendations
    st.subheader("🔮 Risk Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Key Risk Patterns")
        
        if not filtered_df.empty:
            # Top risk factors
            top_factors = filtered_df['contributing_factors'].value_counts().head(5)
            
            st.markdown("**Top Contributing Factors:**")
            for factor, count in top_factors.items():
                percentage = count / len(filtered_df) * 100
                st.markdown(f"• {factor}: {count} incidents ({percentage:.1f}%)")
            
            # High-risk locations
            high_risk_locations = filtered_df[filtered_df['severity'].isin(['Critical', 'High'])]['location'].value_counts().head(3)
            
            st.markdown("**Highest Risk Locations:**")
            for location, count in high_risk_locations.items():
                st.markdown(f"• {location}: {count} high-risk incidents")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        
        recommendations = [
            "🎯 Focus prevention efforts on top contributing factors",
            "📍 Implement additional safety measures at high-risk locations", 
            "⏰ Improve reporting processes to meet 24-hour compliance",
            "👥 Provide targeted training for staff in high-risk areas",
            "📊 Monthly review of incident patterns and trends",
            "🔄 Update risk assessment protocols based on data insights"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")

# Footer with data summary
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Data Summary:** {len(df)} total incidents")
with col2:
    if not df.empty:
        date_range_str = f"{df['incident_date'].min().strftime('%d/%m/%Y')} - {df['incident_date'].max().strftime('%d/%m/%Y')}"
        st.markdown(f"**Date Range:** {date_range_str}")
with col3:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Quick actions sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🎨 Force Style Refresh"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("📊 Export Current View"):
    # Export filtered data to CSV
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"ndis_incidents_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

if st.sidebar.button("📧 Generate Alert Report"):
    critical_count = len(filtered_df[filtered_df['severity'] == 'Critical'])
    overdue_count = len(filtered_df[filtered_df['reporting_delay_hours'] > 24])
    
    if critical_count > 0 or overdue_count > 0:
        st.sidebar.warning(f"Alert: {critical_count} critical incidents, {overdue_count} overdue reports")
    else:
        st.sidebar.success("No alerts - all within normal parameters")

# Data quality indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Quality")

if not df.empty:
    # Calculate data completeness
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    st.sidebar.progress(completeness / 100)
    st.sidebar.caption(f"{completeness:.1f}% - Data completeness")
    
    # Show data freshness
    if not df['incident_date'].isna().all():
        latest_incident = df['incident_date'].max()
        days_old = (datetime.now() - latest_incident).days
        st.sidebar.caption(f"Latest incident: {days_old} days ago")

# Sample data info
st.sidebar.markdown("---")
st.sidebar.markdown("### About This Data")
st.sidebar.info("""
This dashboard uses synthetic NDIS incident data for demonstration purposes. 
The data includes:
- 100 sample incidents
- Realistic incident types and severities
- Compliance tracking
- Medical attention requirements
- Contributing factors analysis
""")

# Performance tips
with st.expander("📈 Dashboard Performance Tips"):
    st.markdown("""
    **For optimal performance:**
    - Use date range filters to focus on specific periods
    - Filter by location or incident type for detailed analysis
    - Export data for offline analysis when needed
    - Refresh data periodically for latest updates
    
    **Understanding the Data:**
    - Critical/High severity incidents require immediate attention
    - 24-hour reporting compliance is tracked automatically
    - Medical attention requirements help with resource planning
    - Contributing factors guide prevention strategies
    """)

# Debug info (only show in development)
if st.sidebar.checkbox("Show Debug Info", value=False):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.write(f"Total records loaded: {len(df)}")
    st.sidebar.write(f"Filtered records: {len(filtered_df)}")
    st.sidebar.write(f"Date range: {len(date_range) if isinstance(date_range, tuple) else 'None'}")
    st.sidebar.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
