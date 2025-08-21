import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import altair as alt

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

# Sample data generation (replace with your actual data source)
@st.cache_data
def load_incident_data():
    """Load and prepare incident data"""
    np.random.seed(42)
    
    # Monthly incident data with seasonal patterns
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    incident_data = pd.DataFrame({
        'month': months,
        'critical': [3, 2, 4, 1, 2, 3, 5, 4, 8, 9, 6, 4],
        'major': [12, 15, 18, 14, 16, 13, 20, 22, 28, 31, 24, 18],
        'minor': [28, 31, 34, 29, 32, 27, 38, 41, 52, 55, 45, 35],
        'same_day_reporting': [89, 91, 88, 93, 87, 94, 85, 86, 82, 81, 89, 92]
    })
    incident_data['total'] = incident_data['critical'] + incident_data['major'] + incident_data['minor']
    
    # Incident types data
    incident_types = pd.DataFrame({
        'incident_type': ['Missing Person/Unexplained Absence', 'Injury', 'Neglect', 
                         'Restrictive Practice', 'Abuse', 'Other'],
        'count': [89, 67, 34, 23, 12, 18],
        'percentage': [36.5, 27.4, 13.9, 9.4, 4.9, 7.4]
    })
    
    # Provider compliance data
    compliance_data = pd.DataFrame({
        'provider': ['Large Org A', 'Small Org B', 'Medium Org C', 'Sole Trader D', 'Large Org E'],
        'on_time': [94, 87, 91, 96, 89],
        'late': [6, 13, 9, 4, 11],
        'total_reports': [156, 73, 122, 24, 198]
    })
    
    # Risk matrix data
    risk_data = pd.DataFrame({
        'category': ['Physical Safety', 'Medication', 'Behavioral', 'Environmental', 'Procedural'],
        'likelihood': [3, 2, 4, 2, 3],
        'impact': [4, 3, 2, 2, 3],
        'incidents': [89, 23, 67, 45, 34]
    })
    risk_data['risk_score'] = risk_data['likelihood'] * risk_data['impact']
    
    # Location performance data
    location_data = pd.DataFrame({
        'location': ['Day Program Centers', 'Transport Vehicles', 'Community Access', 
                    'Therapy Clinics', 'Residential Care', 'Family Homes'],
        'total_incidents': [120, 89, 67, 45, 78, 56],
        'critical_incidents': [12, 53, 8, 15, 9, 4],
        'critical_percentage': [10, 60, 12, 33, 12, 7]
    })
    
    return incident_data, incident_types, compliance_data, risk_data, location_data

# Load data
incident_data, incident_types, compliance_data, risk_data, location_data = load_incident_data()

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
time_period = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"],
    index=3
)

provider_filter = st.sidebar.selectbox(
    "Provider Type",
    ["All Providers", "Large Organizations", "Small Organizations", "Sole Traders"]
)

severity_filter = st.sidebar.multiselect(
    "Incident Severity",
    ["Critical", "Major", "Minor"],
    default=["Critical", "Major", "Minor"]
)

# Main dashboard content
if page == "Executive Summary":
    st.title("📊 NDIS Executive Dashboard")
    st.markdown("**Strategic Overview - Incident Analysis & Risk Management**")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Total Incidents</h4>
            <h2>687</h2>
            <p style="color: #dc3545;">📈 +22.4% vs previous period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Missing Person Events</h4>
            <h2>89</h2>
            <p style="color: #dc3545;">📈 +15.8% - Primary concern</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Same-Day Reporting</h4>
            <h2>87.2%</h2>
            <p style="color: #ffc107;">📉 -2.8% needs attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card success-alert">
            <h4>Reportable Rate</h4>
            <h2>100%</h2>
            <p style="color: #198754;">✅ Compliance maintained</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Seasonal Incident Patterns")
        st.caption("Notable Sep-Oct peak detected - 40% above baseline")
        
        # Create stacked area chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=incident_data['month'], y=incident_data['critical'],
            mode='lines', stackgroup='one', name='Critical',
            line=dict(color='#dc3545'), fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=incident_data['month'], y=incident_data['major'],
            mode='lines', stackgroup='one', name='Major',
            line=dict(color='#fd7e14'), fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=incident_data['month'], y=incident_data['minor'],
            mode='lines', stackgroup='one', name='Minor',
            line=dict(color='#ffc107'), fill='tonexty'
        ))
        
        fig.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Month",
            yaxis_title="Number of Incidents"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Priority Risk Alerts")
        
        alerts = [
            ("🔴", "Missing person incident at Transport Vehicle - immediate response activated", "1 hour ago"),
            ("📊", "September-October incident spike detected - 40% above baseline", "2 hours ago"),
            ("⏰", "3 reports approaching 24-hour deadline", "4 hours ago"),
            ("🚗", "Transport Vehicles showing 60% critical incident concentration", "6 hours ago"),
            ("📋", "8 follow-up tasks overdue across Day Program Centers", "1 day ago")
        ]
        
        for icon, message, time in alerts:
            st.markdown(f"""
            <div class="alert-card">
                <strong>{icon} {message}</strong><br>
                <small style="color: #6c757d;">{time}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Incident Types Distribution")
        
        fig = px.pie(
            incident_types, 
            values='count', 
            names='incident_type',
            title="Primary concern: Missing Person/Unexplained Absence (36.5%)"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Location Risk Analysis")
        
        # Create scatter plot for location risk
        fig = px.scatter(
            location_data,
            x='total_incidents',
            y='critical_percentage',
            size='critical_incidents',
            color='critical_percentage',
            hover_name='location',
            title="Transport Vehicles: High-risk concentration",
            labels={
                'total_incidents': 'Total Incidents',
                'critical_percentage': 'Critical Incident %'
            },
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk matrix
    st.subheader("⚠️ Risk Assessment Matrix")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for i, (col, risk) in enumerate(zip([col1, col2, col3, col4, col5], risk_data.itertuples())):
        risk_level = "🔴 High" if risk.risk_score >= 9 else "🟡 Medium" if risk.risk_score >= 6 else "🟢 Low"
        
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h5>{risk.category}</h5>
                <p><strong>Risk Score: {risk.risk_score}</strong></p>
                <p>Likelihood: {risk.likelihood}/5</p>
                <p>Impact: {risk.impact}/5</p>
                <p>{risk.incidents} incidents</p>
                <p>{risk_level}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "Operational Performance":
    st.title("🎯 Operational Performance & Risk Analysis")
    st.markdown("**Tactical Level - Management Action & Resource Allocation**")
    st.markdown("---")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Resolution Time", "3.2 days", "-0.5 days")
    with col2:
        st.metric("Active Cases", "47", "+5")
    with col3:
        st.metric("Staff Utilization", "78%", "+3%")
    with col4:
        st.metric("Resource Efficiency", "92%", "+1.2%")
    
    # Reporter performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Reporter Performance Analysis")
        
        reporter_data = pd.DataFrame({
            'role': ['Support Worker', 'Team Leader', 'Manager', 'External Reporter'],
            'avg_delay_hours': [2.1, 1.3, 0.8, 4.2],
            'reports_count': [156, 89, 45, 23]
        })
        
        fig = px.bar(
            reporter_data,
            x='role',
            y='avg_delay_hours',
            title="Average Notification Delay by Role",
            color='avg_delay_hours',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Medical Impact Correlation")
        
        medical_data = pd.DataFrame({
            'incident_type': ['Injury', 'Missing Person', 'Neglect', 'Abuse', 'Other'],
            'medical_required': [85, 45, 60, 75, 30],
            'hospital_visits': [45, 12, 25, 35, 8]
        })
        
        fig = px.scatter(
            medical_data,
            x='medical_required',
            y='hospital_visits',
            size='medical_required',
            color='incident_type',
            title="Healthcare Resource Requirements"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Workload trends
    st.subheader("📈 Workload & Resolution Trends")
    
    workload_data = pd.DataFrame({
        'week': [f'Week {i}' for i in range(1, 13)],
        'new_cases': [23, 31, 28, 35, 42, 38, 29, 33, 45, 52, 38, 31],
        'resolved_cases': [21, 29, 30, 33, 40, 36, 31, 35, 43, 48, 40, 33],
        'active_cases': [23, 25, 23, 25, 27, 29, 27, 25, 27, 31, 29, 27]
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=workload_data['week'], y=workload_data['new_cases'], name='New Cases'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(x=workload_data['week'], y=workload_data['resolved_cases'], name='Resolved Cases'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=workload_data['week'], y=workload_data['active_cases'], 
                  mode='lines+markers', name='Active Cases', line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_layout(title="Case Management Workload Analysis")
    fig.update_yaxes(title_text="Cases (New/Resolved)", secondary_y=False)
    fig.update_yaxes(title_text="Active Cases", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Compliance & Investigation":
    st.title("📋 Compliance & Detailed Investigation")
    st.markdown("**Operational Level - Regulatory Oversight & Case Management**")
    st.markdown("---")
    
    # Compliance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Reportable Rate", "100%", "Maintained")
    with col2:
        st.metric("24hr Compliance", "94.2%", "+1.8%")
    with col3:
        st.metric("Investigation Complete", "100%", "On target")
    with col4:
        st.metric("Overdue Tasks", "3", "-2")
    
    # Compliance timeline
    st.subheader("⏰ Compliance Timeline Analysis")
    
    timeline_data = pd.DataFrame({
        'report_id': [f'INC-{1000+i}' for i in range(20)],
        'incident_date': pd.date_range('2024-01-01', periods=20, freq='3D'),
        'report_date': pd.date_range('2024-01-01', periods=20, freq='3D') + pd.Timedelta(hours=np.random.randint(1, 48, 20)),
        'severity': np.random.choice(['Critical', 'Major', 'Minor'], 20),
        'compliance_status': np.random.choice(['Compliant', 'At Risk', 'Overdue'], 20, p=[0.8, 0.15, 0.05])
    })
    
    timeline_data['hours_to_report'] = (timeline_data['report_date'] - timeline_data['incident_date']).dt.total_seconds() / 3600
    
    fig = px.scatter(
        timeline_data,
        x='incident_date',
        y='hours_to_report',
        color='compliance_status',
        size='hours_to_report',
        hover_data=['report_id', 'severity'],
        title="Reporting Timeline Compliance",
        color_discrete_map={
            'Compliant': 'green',
            'At Risk': 'orange', 
            'Overdue': 'red'
        }
    )
    
    # Add 24-hour compliance line
    fig.add_hline(y=24, line_dash="dash", line_color="red", 
                  annotation_text="24-hour deadline")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Investigation pipeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Investigation Pipeline")
        
        pipeline_data = pd.DataFrame({
            'stage': ['Initial Review', 'Investigation', 'Analysis', 'Report', 'Closure'],
            'completed': [100, 100, 98, 95, 92],
            'in_progress': [0, 0, 2, 5, 8],
            'pending': [0, 0, 0, 0, 0]
        })
        
        fig = px.bar(
            pipeline_data,
            x='stage',
            y=['completed', 'in_progress', 'pending'],
            title="Investigation Stage Completion Rates",
            color_discrete_map={
                'completed': 'green',
                'in_progress': 'orange',
                'pending': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Compliance by Provider")
        
        fig = px.bar(
            compliance_data,
            x='provider',
            y=['on_time', 'late'],
            title="Reporting Timeliness by Provider",
            color_discrete_map={'on_time': 'green', 'late': 'red'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Risk Analysis":
    st.title("⚠️ Advanced Risk Analysis")
    st.markdown("**Strategic Analysis - Pattern Recognition & Prevention**")
    st.markdown("---")
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("High Risk Incidents", "23", "+3")
    with col2:
        st.metric("Risk Score (Avg)", "6.2", "-0.3")
    with col3:
        st.metric("Prevention Actions", "18", "+5")
    with col4:
        st.metric("Risk Reduction", "12%", "+4%")
    
    # Risk heat map
    st.subheader("🔥 Risk Heat Map Analysis")
    
    # Create risk matrix visualization
    risk_matrix = np.random.rand(5, 5) * 100
    locations = ['Day Centers', 'Transport', 'Community', 'Therapy', 'Residential']
    incident_types = ['Injury', 'Missing', 'Neglect', 'Abuse', 'Other']
    
    fig = px.imshow(
        risk_matrix,
        x=locations,
        y=incident_types,
        color_continuous_scale='Reds',
        title="Risk Concentration by Location and Incident Type"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictive trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Predictive Risk Trends")
        
        # Generate predictive data
        future_dates = pd.date_range('2024-01-01', periods=52, freq='W')
        predicted_risk = 50 + 20 * np.sin(np.arange(52) * 2 * np.pi / 52) + np.random.normal(0, 5, 52)
        
        trend_data = pd.DataFrame({
            'week': future_dates,
            'predicted_risk': predicted_risk,
            'confidence_upper': predicted_risk + 10,
            'confidence_lower': predicted_risk - 10
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['week'], y=trend_data['predicted_risk'],
            mode='lines', name='Predicted Risk Score',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['week'], y=trend_data['confidence_upper'],
            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['week'], y=trend_data['confidence_lower'],
            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
            name='Confidence Interval', fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig.update_layout(title="52-Week Risk Prediction")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Factors Analysis")
        
        factor_data = pd.DataFrame({
            'factor': ['Staff Ratio', 'Time of Day', 'Day of Week', 'Weather', 'Activity Type'],
            'correlation': [0.72, 0.68, 0.45, 0.23, 0.81],
            'significance': ['High', 'High', 'Medium', 'Low', 'High']
        })
        
        fig = px.bar(
            factor_data,
            x='factor',
            y='correlation',
            color='significance',
            title="Risk Factor Correlation Analysis",
            color_discrete_map={
                'High': 'red',
                'Medium': 'orange',
                'Low': 'green'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d;">
    <small>NDIS Incident Management System | Last updated: {}</small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

# Quick actions sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

if st.sidebar.button("📊 Export Report"):
    st.sidebar.success("Report exported successfully!")

if st.sidebar.button("🚨 Generate Alert"):
    st.sidebar.warning("Alert system activated!")

# Data quality indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Quality")
st.sidebar.progress(0.95)
st.sidebar.caption("95% - Data freshness: 2 minutes ago")
