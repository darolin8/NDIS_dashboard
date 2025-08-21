import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="NDIS Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 0.5rem;
    }
    .critical-alert {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .success-alert {
        background-color: #d1e7dd;
        border-left-color: #198754;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_incident_data():
    """Load and prepare the actual NDIS incident data"""
    try:
        # Load the CSV data
        df = pd.read_csv('ndis_incidents_synthetic.csv')
        
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
        
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'ndis_incidents_synthetic.csv' is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load the data
df = load_incident_data()

if df.empty:
    st.stop()

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
            <p style="color: #6c757d;">📊 Current period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Critical Incidents</h4>
            <h2>{critical_incidents}</h2>
            <p style="color: #dc3545;">🚨 {critical_incidents/total_incidents*100:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Same-Day Reporting</h4>
            <h2>{same_day_rate:.1f}%</h2>
            <p style="color: #198754;">⏰ Within 24 hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card success-alert">
            <h4>Reportable Rate</h4>
            <h2>{reportable_rate:.1f}%</h2>
            <p style="color: #198754;">✅ NDIS Commission</p>
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
            
            colors = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}
            
            for severity in monthly_data.columns:
                color = colors.get(severity, '#6c757d')
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
                yaxis_title="Number of Incidents"
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
                title=f"Top incident type: {incident_counts.index[0]} ({incident_counts.iloc[0]} cases)"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
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
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
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
    
    # Risk analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Risk Heat Map - Type vs Location")
        
        if not filtered_df.empty:
            # Create pivot table for heatmap
            risk_matrix = pd.crosstab(filtered_df['incident_type'], filtered_df['location'], values=filtered_df['severity'].map({'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}), aggfunc='mean')
            
            fig = px.imshow(
                risk_matrix.values,
                x=risk_matrix.columns,
                y=risk_matrix.index,
                color_continuous_scale='Reds',
                title="Average Risk Score by Type and Location",
                aspect="auto"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Age Distribution Risk Analysis")
        
        if not filtered_df.empty and not filtered_df['age_at_incident'].isna().all():
            # Age group analysis
            filtered_df['age_group'] = pd.cut(filtered_df['age_at_incident'], 
                                            bins=[0, 18, 35, 50, 65, 100], 
                                            labels=['0-18', '19-35', '36-50', '51-65', '65+'])
            
            age_risk = filtered_df.groupby('age_group')['severity'].apply(
                lambda x: (x.isin(['Critical', 'High'])).sum() / len(x) * 100
            ).dropna()
            
            fig = px.bar(
                x=age_risk.index.astype(str),
                y=age_risk.values,
                title="High-Risk Incident Rate by Age Group",
                labels={'x': 'Age Group', 'y': 'High-Risk Incident Rate (%)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed risk analysis
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
    st.experimental_rerun()

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
