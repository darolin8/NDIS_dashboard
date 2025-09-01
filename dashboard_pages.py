import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta

# ================= UTILITY FUNCTIONS =================

def calculate_trend(current_value, previous_value):
    if previous_value == 0:
        return 0, "→"
    change_pct = ((current_value - previous_value) / previous_value) * 100
    if change_pct > 0:
        return change_pct, "↗️"
    elif change_pct < 0:
        return abs(change_pct), "↘️"
    else:
        return 0, "→"

# ================= EXECUTIVE SUMMARY PLOTS =================

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph="rgba(0,104,201,0.2)"):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        value=value,
        gauge={"axis": {"visible": False}},
        number={"prefix": prefix, "suffix": suffix, "font.size": 28},
        title={"text": label, "font": {"size": 24}},
    ))
    if show_graph:
        if "High Severity" in label:
            trend_data = [random.randint(5, 25) for _ in range(30)]
        elif "Total" in label:
            trend_data = [random.randint(20, 80) for _ in range(30)]
        else:
            trend_data = [random.randint(0, 50) for _ in range(30)]
        fig.add_trace(go.Scatter(
            y=trend_data,
            hoverinfo="skip",
            fill="tozeroy",
            fillcolor=color_graph,
            line={"color": color_graph},
        ))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )
    st.plotly_chart(fig, use_container_width=True, key=label.replace(" ", "_")+"_metric")

def plot_gauge(indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound):
    fig = go.Figure(go.Indicator(
        value=indicator_number,
        mode="gauge+number",
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": indicator_suffix, "font.size": 26},
        gauge={
            "axis": {"range": [0, max_bound], "tickwidth": 1},
            "bar": {"color": indicator_color},
            "steps": [
                {"range": [0, max_bound*0.5], "color": "lightgray"},
                {"range": [max_bound*0.5, max_bound*0.8], "color": "gray"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": max_bound*0.9
            }
        },
        title={"text": indicator_title, "font": {"size": 20}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True, key=indicator_title.replace(" ", "_")+"_gauge")

def plot_severity_distribution(df):
    if df.empty or 'severity' not in df.columns:
        st.warning("No data available for severity distribution")
        return
    severity_counts = df['severity'].value_counts()
    colors = {'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'}
    fig = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Incident Severity Distribution",
        color=severity_counts.index,
        color_discrete_map=colors,
        height=400
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=12
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    st.plotly_chart(fig, use_container_width=True, key="severity_dist")

def plot_top_incidents_by_volume_severity(df):
    if df.empty or 'incident_type' not in df.columns or 'severity' not in df.columns:
        st.warning("No data available for top incidents analysis")
        return
    top_incidents = df['incident_type'].value_counts().head(5).index
    filtered_df = df[df['incident_type'].isin(top_incidents)]
    severity_counts = filtered_df.groupby(['incident_type', 'severity']).size().reset_index(name='count')
    fig = px.bar(
        severity_counts,
        x='incident_type',
        y='count',
        color='severity',
        title="Top 5 Incident Types by Volume & Severity",
        labels={'incident_type': 'Incident Type', 'count': 'Number of Incidents'},
        color_discrete_map={'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'},
        height=400
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title="Severity",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key="top_incidents_volume_severity")

def plot_monthly_incidents_by_severity(df):
    if df.empty or 'incident_date' not in df.columns or 'severity' not in df.columns:
        st.warning("No data available for monthly trends")
        return
    df = df.copy()
    df['year_month'] = df['incident_date'].dt.to_period('M').astype(str)
    monthly_severity = df.groupby(['year_month', 'severity']).size().reset_index(name='count')
    fig = px.bar(
        monthly_severity,
        x='year_month',
        y='count',
        color='severity',
        title="Monthly Incidents by Severity",
        labels={'year_month': 'Month', 'count': 'Number of Incidents'},
        color_discrete_map={'High': '#FF2B2B', 'Moderate': '#FF8700', 'Low': '#29B09D'},
        height=400
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title="Severity",
        showlegend=True,
        xaxis_title="Month",
        yaxis_title="Number of Incidents"
    )
    st.plotly_chart(fig, use_container_width=True, key="monthly_incidents_severity")

def plot_incident_types_bar(df):
    if df.empty or 'incident_type' not in df.columns:
        st.warning("No data available for incident types")
        return
    incident_counts = df['incident_type'].value_counts().head(10)
    fig = px.bar(
        x=incident_counts.values,
        y=incident_counts.index,
        orientation='h',
        title="Top 10 Incident Types",
        labels={'x': 'Number of Incidents', 'y': 'Incident Type'},
        color=incident_counts.values,
        color_continuous_scale='Viridis',
        height=400
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, key="incident_types_bar")

def plot_location_analysis(df):
    if df.empty or 'location' not in df.columns:
        st.warning("No data available for location analysis")
        return
    location_counts = df['location'].value_counts().head(8)
    fig = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title="Incidents by Location",
        labels={'x': 'Location', 'y': 'Number of Incidents'},
        color=location_counts.values,
        color_continuous_scale='Blues',
        height=400
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="location_analysis")

def plot_incident_trends(df):
    if df.empty or 'incident_date' not in df.columns:
        st.warning("No data available for incident trends")
        return
    daily_counts = df.groupby(df['incident_date'].dt.date).size().reset_index(name='count')
    daily_counts.columns = ['date', 'incidents']
    fig = px.line(
        daily_counts,
        x='date',
        y='incidents',
        title="Daily Incident Trends",
        markers=True
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Incidents",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True, key="incident_trends")

def plot_weekday_analysis(df):
    if df.empty or 'incident_weekday' not in df.columns:
        st.warning("No data available for weekday analysis")
        return
    weekday_counts = df['incident_weekday'].value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = weekday_counts.reindex(day_order, fill_value=0)
    fig = px.bar(
        x=weekday_counts.index,
        y=weekday_counts.values,
        title="Incidents by Day of Week",
        labels={'x': 'Day of Week', 'y': 'Number of Incidents'},
        color=weekday_counts.values,
        color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig, use_container_width=True, key="weekday_analysis")

def plot_time_analysis(df):
    if df.empty or 'incident_time' not in df.columns:
        st.warning("No time data available for analysis")
        return
    df = df.copy()
    df['incident_hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour
    hourly_counts = df['incident_hour'].value_counts().sort_index()
    fig = px.line(
        x=hourly_counts.index,
        y=hourly_counts.values,
        title="Incidents by Hour of Day",
        markers=True
    )
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Number of Incidents",
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    st.plotly_chart(fig, use_container_width=True, key="time_analysis")

def plot_reportable_analysis(df):
    if df.empty or 'reportable' not in df.columns:
        st.warning("No data available for reportable analysis")
        return
    reportable_counts = df['reportable'].value_counts()
    reportable_labels = ['Not Reportable', 'Reportable']
    if len(reportable_counts) == 2 and 0 in reportable_counts.index and 1 in reportable_counts.index:
        reportable_counts.index = reportable_labels
    elif True in reportable_counts.index or False in reportable_counts.index:
        reportable_counts.index = ['Reportable' if x else 'Not Reportable' for x in reportable_counts.index]
    fig = px.pie(
        values=reportable_counts.values,
        names=reportable_counts.index,
        title="Reportable Incidents Distribution",
        color_discrete_sequence=['#90EE90', '#FFB6C1']
    )
    st.plotly_chart(fig, use_container_width=True, key="reportable_analysis")

def plot_medical_outcomes(df):
    if df.empty or 'treatment_required' not in df.columns or 'medical_attention_required' not in df.columns:
        st.warning("No data available for medical outcomes")
        return
    medical_summary = {
        'Treatment Required': df['treatment_required'].sum(),
        'Medical Attention Required': df['medical_attention_required'].sum(),
        'No Medical Intervention': len(df) - df[['treatment_required', 'medical_attention_required']].any(axis=1).sum()
    }
    fig = px.bar(
        x=list(medical_summary.keys()),
        y=list(medical_summary.values()),
        title="Medical Intervention Requirements",
        labels={'x': 'Medical Outcome', 'y': 'Number of Cases'},
        color=list(medical_summary.values()),
        color_continuous_scale='RdYlBu_r',
        height=400
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True, key="medical_outcomes")

def plot_reporter_type_metrics(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'reported_by' in df.columns:
            value = df['reported_by'].nunique()
            plot_metric("Reporter Types", value, color_graph="#5B8FF9")
    with col2:
        if 'medical_attention_required' in df.columns:
            value = int(df['medical_attention_required'].sum())
            plot_metric("Medical Attention Required", value, color_graph="#F6BD16")
    with col3:
        if 'participant_age' in df.columns:
            avg_age = df['participant_age'].mean()
            plot_metric("Avg Participant Age", avg_age, suffix=" yrs", color_graph="#5AD8A6")

def plot_reporter_performance_scatter(df):
    if df.empty or not {'reported_by','notification_date','incident_date'}.issubset(df.columns):
        st.warning("No data available for reporter performance analysis")
        return
    df = df.copy()
    perf = (
        df.groupby('reported_by')
        .agg(
            avg_delay=('notification_date', lambda x: (x - df.loc[x.index, 'incident_date']).dt.days.mean()),
            total_incidents=('incident_date', 'count')
        ).reset_index()
    )
    fig = px.scatter(
        perf,
        x='avg_delay',
        y='total_incidents',
        color='reported_by',
        size='total_incidents',
        size_max=60,
        labels={
            'avg_delay': 'Average Notification Delay (Days)',
            'total_incidents': 'Total Incidents',
            'reported_by': 'Reporter Type'
        },
        title='Reporter Performance Analysis',
        opacity=0.7
    )
    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='lightblue', griddash='dash')
    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='lightblue', griddash='dash')
    fig.update_traces(marker=dict(line=dict(width=2, color='rgba(0,0,0,0.3)')))
    fig.update_layout(
        legend_title_text='Reporter Type',
        xaxis=dict(zeroline=True, zerolinecolor='lightblue', zerolinewidth=2),
        yaxis=dict(zeroline=True, zerolinecolor='lightblue', zerolinewidth=2),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True, key="reporter_performance_scatter")

def plot_serious_injury_age_severity(df):
    if df.empty or 'severity' not in df.columns or 'participant_age' not in df.columns:
        st.info("No high severity incidents found for age analysis")
        return
    serious_df = df[df['severity'].str.lower() == 'high']
    if not serious_df.empty:
        fig = px.histogram(
            serious_df,
            x='participant_age',
            color='severity',
            nbins=20,
            title="High Severity Incidents: Age Distribution"
        )
        st.plotly_chart(fig, use_container_width=True, key="serious_injury_age_severity")
    else:
        st.info("No high severity incidents found for age analysis")

def display_executive_kpi_cards(df):
    if df.empty or 'incident_type' not in df.columns or 'severity' not in df.columns or 'incident_date' not in df.columns:
        st.warning("No data available for KPI cards")
        return
    top_incident_type = df['incident_type'].value_counts().index[0] if len(df) > 0 else "N/A"
    most_common_severity = df['severity'].value_counts().index[0] if len(df) > 0 else "N/A"
    current_date = df['incident_date'].max()
    current_month = current_date.to_period('M')
    previous_month = current_month - 1
    latest_month_incidents = len(df[df['incident_date'].dt.to_period('M') == current_month])
    previous_month_incidents = len(df[df['incident_date'].dt.to_period('M') == previous_month])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Top Incident Type", top_incident_type, help="Most frequently occurring incident type")
    with col2:
        change = latest_month_incidents - previous_month_incidents
        delta_color = "inverse" if change > 0 else "normal"
        st.metric(f"Latest Month ({current_month})", latest_month_incidents, delta=change, delta_color=delta_color, help="Number of incidents in the current month")
    with col3:
        st.metric(f"Previous Month ({previous_month})", previous_month_incidents, help="Number of incidents in the previous month")
    with col4:
        st.metric("Most Common Severity", most_common_severity, help="Most frequently occurring severity level")

def display_recent_critical_incidents_card(df):
    if df.empty or 'severity' not in df.columns or 'incident_date' not in df.columns:
        st.warning("No data available for recent critical incidents")
        return
    thirty_days_ago = datetime.now() - timedelta(days=30)
    critical_incidents = df[
        (df['severity'].str.lower() == 'high') &
        (df['incident_date'] >= thirty_days_ago)
    ].sort_values('incident_date', ascending=False).head(5)
    st.markdown("### 🚨 Recent Critical Incidents")
    if critical_incidents.empty:
        st.info("No critical incidents in the last 30 days")
        return
    for idx, incident in critical_incidents.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; height: 60px;">
                        <div style="width: 20px; height: 20px; background-color: #FF2B2B;
                                    border-radius: 50%; margin-right: 10px;"></div>
                        <div style="width: 2px; height: 40px; background-color: #FF2B2B;
                                    margin-left: 9px;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                incident_date = incident['incident_date'].strftime('%b %d')
                incident_type = incident.get('incident_type', 'Unknown')
                location = incident.get('location', 'Unknown')
                description = incident.get('description', 'Sample incident description...')
                st.markdown(f"**{incident_type}** - {location}")
                st.markdown(f"<small>{description[:80]}...</small>", unsafe_allow_html=True)
            with col3:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 10px;">
                        <div style="font-size: 12px; color: #666;">📅</div>
                        <div style="font-weight: bold; font-size: 14px;">{incident_date}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("---")

# ========== OPERATIONAL PERFORMANCE FUNCTIONS ==========

def display_operational_performance_cards(df):
    """Display operational performance cards with trend indicators"""
    if df.empty or 'incident_date' not in df.columns:
        st.warning("No data available for operational performance cards")
        return
    
    # Calculate current month and previous month data
    current_date = df['incident_date'].max()
    current_month = current_date.to_period('M')
    previous_month = current_month - 1
    
    current_df = df[df['incident_date'].dt.to_period('M') == current_month]
    previous_df = df[df['incident_date'].dt.to_period('M') == previous_month]
    
    st.markdown("### 📈 Operational Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Location Reportable Rate
        if 'location' in df.columns and 'reportable' in df.columns:
            current_reportable_rate = (current_df['reportable'].sum() / len(current_df) * 100) if len(current_df) > 0 else 0
            previous_reportable_rate = (previous_df['reportable'].sum() / len(previous_df) * 100) if len(previous_df) > 0 else 0
            
            trend_pct, trend_arrow = calculate_trend(current_reportable_rate, previous_reportable_rate)
            
            st.metric(
                label="🏢 Location Reportable Rate",
                value=f"{current_reportable_rate:.1f}%",
                delta=f"{trend_arrow} {trend_pct:.1f}%",
                delta_color="inverse",
                help="Percentage of incidents that are reportable by location"
            )
    
    with col2:
        # Average Participant Age
        if 'participant_age' in df.columns:
            current_avg_age = current_df['participant_age'].mean() if len(current_df) > 0 else 0
            previous_avg_age = previous_df['participant_age'].mean() if len(previous_df) > 0 else 0
            
            trend_pct, trend_arrow = calculate_trend(current_avg_age, previous_avg_age)
            
            st.metric(
                label="👥 Average Participant Age",
                value=f"{current_avg_age:.1f} yrs",
                delta=f"{trend_arrow} {trend_pct:.1f}%",
                delta_color="normal",
                help="Average age of participants involved in incidents"
            )
    
    with col3:
        # Medical Attention Rate
        if 'medical_attention_required' in df.columns:
            current_medical_rate = (current_df['medical_attention_required'].sum() / len(current_df) * 100) if len(current_df) > 0 else 0
            previous_medical_rate = (previous_df['medical_attention_required'].sum() / len(previous_df) * 100) if len(previous_df) > 0 else 0
            
            trend_pct, trend_arrow = calculate_trend(current_medical_rate, previous_medical_rate)
            
            st.metric(
                label="🏥 Medical Attention Rate",
                value=f"{current_medical_rate:.1f}%",
                delta=f"{trend_arrow} {trend_pct:.1f}%",
                delta_color="inverse",
                help="Percentage of incidents requiring medical attention"
            )

def plot_reporter_type_metrics(df):
    """Display reporter type related metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'reported_by' in df.columns:
            value = df['reported_by'].nunique()
            plot_metric("Reporter Types", value, color_graph="#5B8FF9")
    
    with col2:
        if 'medical_attention_required' in df.columns:
            value = int(df['medical_attention_required'].sum())
            plot_metric("Medical Attention Required", value, color_graph="#F6BD16")
    
    with col3:
        if 'participant_age' in df.columns:
            avg_age = df['participant_age'].mean()
            plot_metric("Avg Participant Age", avg_age, suffix=" yrs", color_graph="#5AD8A6")

def plot_reporter_performance_scatter(df):
    """Create reporter performance scatter plot"""
    if df.empty or not {'reported_by','notification_date','incident_date'}.issubset(df.columns):
        st.warning("No data available for reporter performance analysis")
        return
    df = df.copy()
    perf = (
        df.groupby('reported_by')
        .agg(
            avg_delay=('notification_date', lambda x: (x - df.loc[x.index, 'incident_date']).dt.days.mean()),
            total_incidents=('incident_date', 'count')
        ).reset_index()
    )
    fig = px.scatter(
        perf,
        x='avg_delay',
        y='total_incidents',
        color='reported_by',
        size='total_incidents',
        size_max=60,
        labels={
            'avg_delay': 'Average Notification Delay (Days)',
            'total_incidents': 'Total Incidents',
            'reported_by': 'Reporter Type'
        },
        title='Reporter Performance Analysis',
        opacity=0.7
    )
    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='lightblue', griddash='dash')
    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='lightblue', griddash='dash')
    fig.update_traces(marker=dict(line=dict(width=2, color='rgba(0,0,0,0.3)')))
    fig.update_layout(
        legend_title_text='Reporter Type',
        xaxis=dict(zeroline=True, zerolinecolor='lightblue', zerolinewidth=2),
        yaxis=dict(zeroline=True, zerolinecolor='lightblue', zerolinewidth=2),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True, key="reporter_performance_scatter")

def plot_serious_injury_age_severity(df):
    """Create serious injury age and severity analysis"""
    if df.empty or 'severity' not in df.columns or 'participant_age' not in df.columns:
        st.info("No high severity incidents found for age analysis")
        return
    serious_df = df[df['severity'].str.lower() == 'high']
    if not serious_df.empty:
        fig = px.histogram(
            serious_df, 
            x='participant_age', 
            color='severity', 
            nbins=20,
            title="High Severity Incidents: Age Distribution"
        )
        st.plotly_chart(fig, use_container_width=True, key="serious_injury_age_severity")
    else:
        st.info("No high severity incidents found for age analysis")

# ================= INVESTIGATION/COMPLIANCE FUNCTIONS =================

def apply_investigation_rules(df):
    def requires_investigation(row):
        if str(row.get('severity', '')).lower() == 'high':
            return True
        if row.get('reportable', False):
            return True
        serious_types = ['unethical behavior', 'assault', 'unauthorized restraints']
        if str(row.get('incident_type', '')).strip().lower() in serious_types:
            return True
        if row.get('medical_attention_required', False) or row.get('treatment_required', False):
            return True
        return False
    def action_completed(row):
        if str(row.get('medical_outcome', '')).strip().lower() == 'recovered':
            return True
        if str(row.get('severity', '').lower()) == 'low' and not (row.get('medical_attention_required', False) or row.get('treatment_required', False)):
            return True
        if row.get('actions_documented', False):
            return True
        return False
    df = df.copy()
    df['investigation_required'] = df.apply(requires_investigation, axis=1)
    df['action_complete'] = df.apply(action_completed, axis=1)
    return df

def display_compliance_investigation_cards(df):
    if df.empty or 'incident_date' not in df.columns or 'reportable' not in df.columns:
        st.warning("No data available for compliance cards")
        return
    if 'investigation_required' not in df.columns:
        df = apply_investigation_rules(df)
    current_date = df['incident_date'].max()
    current_month = current_date.to_period('M')
    previous_month = current_month - 1
    current_df = df[df['incident_date'].dt.to_period('M') == current_month]
    previous_df = df[df['incident_date'].dt.to_period('M') == previous_month]
    current_df = current_df.copy()
    previous_df = previous_df.copy()
    current_df['report_delay_hours'] = (current_df['notification_date'] - current_df['incident_date']).dt.total_seconds() / 3600
    previous_df['report_delay_hours'] = (previous_df['notification_date'] - previous_df['incident_date']).dt.total_seconds() / 3600
    st.markdown("### 📋 Compliance & Investigation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_reportable = int(current_df['reportable'].sum()) if len(current_df) > 0 else 0
        previous_reportable = int(previous_df['reportable'].sum()) if len(previous_df) > 0 else 0
        change = current_reportable - previous_reportable
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric(
            label="📊 Reportable Incidents",
            value=current_reportable,
            delta=f"{trend_arrow} {abs(change)}",
            delta_color="inverse" if change > 0 else "normal",
            help="Number of incidents that require regulatory reporting"
        )
    with col2:
        current_compliance = int((current_df['report_delay_hours'] <= 24).sum()) if len(current_df) > 0 else 0
        previous_compliance = int((previous_df['report_delay_hours'] <= 24).sum()) if len(previous_df) > 0 else 0
        change = current_compliance - previous_compliance
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric(
            label="⏱️ 24hr Compliance",
            value=current_compliance,
            delta=f"{trend_arrow} {abs(change)}",
            delta_color="normal" if change > 0 else "inverse",
            help="Number of incidents reported within 24 hours"
        )
    with col3:
        current_overdue = int((current_df['report_delay_hours'] > 24).sum()) if len(current_df) > 0 else 0
        previous_overdue = int((previous_df['report_delay_hours'] > 24).sum()) if len(previous_df) > 0 else 0
        change = current_overdue - previous_overdue
        trend_arrow = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        st.metric(
            label="⚠️ Overdue Reports",
            value=current_overdue,
            delta=f"{trend_arrow} {abs(change)}",
            delta_color="inverse" if change > 0 else "normal",
            help="Number of incidents with reporting delays > 24 hours"
        )
    with col4:
        current_total = len(current_df)
        previous_total = len(previous_df)
        current_investigation_rate = (current_df['investigation_required'].sum() / current_total * 100) if current_total > 0 else 0
        previous_investigation_rate = (previous_df['investigation_required'].sum() / previous_total * 100) if previous_total > 0 else 0
        trend_pct, trend_arrow = calculate_trend(current_investigation_rate, previous_investigation_rate)
        st.metric(
            label="🔍 Investigation Rate",
            value=f"{current_investigation_rate:.1f}%",
            delta=f"{trend_arrow} {trend_pct:.1f}%",
            delta_color="inverse",
            help="Percentage of incidents requiring formal investigation"
        )

def plot_compliance_metrics_poly(df):
    if df.empty or 'reportable' not in df.columns or 'incident_date' not in df.columns:
        st.warning("No data available for compliance metrics")
        return
    total = len(df)
    if total == 0:
        st.warning("No data available for compliance metrics")
        return
    if 'investigation_required' not in df.columns:
        df = apply_investigation_rules(df)
    df = df.copy()
    reportable_count = int(df['reportable'].sum())
    df['report_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
    compliance_24h_count = int((df['report_delay_hours'] <= 24).sum())
    overdue_count = int((df['report_delay_hours'] > 24).sum())
    inv_required = int(df['investigation_required'].sum())
    inv_rate = inv_required / total * 100 if total > 0 else 0
    action_complete = int(df['action_complete'].sum())
    breach_count = overdue_count
    inv_status_pct = (action_complete / inv_required * 100) if inv_required > 0 else 0
    inv_status_suffix = f" ({action_complete}/{inv_required})"
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        plot_metric("Reportable Incidents", reportable_count, color_graph="#5B8FF9")
    with col2:
        plot_metric("24hr Compliance", compliance_24h_count, suffix=f" ({compliance_24h_count/total*100:.1f}%)" if total else "", color_graph="#5AD8A6")
    with col3:
        plot_metric("Overdue Reports", overdue_count, color_graph="#F6BD16")
    with col4:
        plot_metric("Investigation Rate", inv_rate, suffix="%", color_graph="#E86452")
    with col5:
        plot_metric("Investigation Status", inv_status_pct, suffix=inv_status_suffix, color_graph="#6DC8EC")
    with col6:
        plot_metric("Compliance Breach", breach_count, color_graph="#FF2B2B")

def plot_reporting_delay_by_date(df):
    if df.empty or not {'incident_date','notification_date'}.issubset(df.columns):
        st.warning("No data available for reporting delay analysis")
        return
    df = df.copy()
    df['report_delay'] = (df['notification_date'] - df['incident_date']).dt.days
    agg = df.groupby('incident_date').agg(avg_delay=('report_delay', 'mean')).reset_index()
    fig = px.line(
        agg,
        x='incident_date',
        y='avg_delay',
        title="Average Reporting Delay by Incident Date",
        labels={'incident_date': 'Incident Date', 'avg_delay': 'Average Delay (Days)'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="reporting_delay_by_date")

def plot_24h_compliance_rate_by_location(df):
    if df.empty or not {'location','notification_date','incident_date'}.issubset(df.columns):
        st.warning("No data available for compliance rate by location")
        return
    df = df.copy()
    df['within_24h'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() <= 24*3600
    compliance = df.groupby('location')['within_24h'].mean().reset_index()
    compliance['within_24h'] = compliance['within_24h'] * 100
    fig = px.bar(
        compliance,
        x='location',
        y='within_24h',
        labels={'within_24h': '% Within 24hr', 'location': 'Location'},
        title="24 Hour Compliance Rate by Location",
        color='within_24h',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True, key="compliance_location")

def plot_investigation_pipeline(df):
    if df.empty or 'investigation_required' not in df.columns or 'action_complete' not in df.columns:
        st.warning("No data available for investigation pipeline")
        return
    all_incidents = len(df)
    required = df['investigation_required'].sum()
    complete = df['action_complete'].sum()
    values = [all_incidents, required, complete]
    names = ['All Incidents', 'Required Investigation', 'Action Complete']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig = px.bar(
        x=names,
        y=values,
        title="Investigation Pipeline",
        labels={'x': 'Stage', 'y': 'Count'},
        color=names,
        color_discrete_sequence=colors
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True, key="investigation_pipeline")

def plot_contributing_factors_by_month(df):
    if df.empty or 'contributing_factors' not in df.columns or 'incident_date' not in df.columns:
        st.warning("No data available for contributing factors analysis")
        return
    df = df.copy()
    df['month_year'] = df['incident_date'].dt.to_period('M').astype(str)
    try:
        fac_df = df.groupby(['month_year', 'contributing_factors']).size().unstack(fill_value=0)
        if not fac_df.empty:
            fig = px.imshow(
                fac_df.T,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Month-Year", y="Contributing Factor", color="Count"),
                title="Contributing Factors by Month-Year",
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="contributing_factors_month")
        else:
            st.info("No contributing factors data available for heatmap")
    except Exception as e:
        st.warning(f"Unable to create contributing factors heatmap: {str(e)}")

# ================= PAGE SECTIONS =================

def display_executive_summary_section(df):
    st.header("📊 Executive Summary")
    display_executive_kpi_cards(df)
    st.markdown("---")
    display_recent_critical_incidents_card(df)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_incidents = len(df)
        plot_metric("Total Incidents", total_incidents, show_graph=True, color_graph="rgba(0,104,201,0.2)")
    with col2:
        high_severity = len(df[df['severity'].str.lower() == 'high']) if 'severity' in df.columns else 0
        plot_metric("High Severity Incidents", high_severity, show_graph=True, color_graph="rgba(255,43,43,0.2)")
    with col3:
        reportable = int(df['reportable'].sum()) if 'reportable' in df.columns else 0
        plot_metric("Reportable Incidents", reportable, show_graph=True, color_graph="rgba(255,135,0,0.2)")
    with col4:
        avg_age = df['participant_age'].mean() if 'participant_age' in df.columns else 0
        plot_metric("Avg Participant Age", avg_age, suffix=" yrs", show_graph=True, color_graph="rgba(90,216,166,0.2)")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        plot_severity_distribution(df)
    with col2:
        plot_top_incidents_by_volume_severity(df)
    plot_monthly_incidents_by_severity(df)
    plot_location_analysis(df)
    plot_incident_trends(df)
    col1, col2 = st.columns(2)
    with col1:
        plot_weekday_analysis(df)
    with col2:
