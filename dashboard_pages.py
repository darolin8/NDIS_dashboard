import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- EXECUTIVE SUMMARY PAGE VISUALS ---------

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
    if df.empty:
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

def plot_incident_types_bar(df):
    if df.empty:
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
    if df.empty:
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

def plot_monthly_trends(df):
    if df.empty:
        st.warning("No data available for monthly trends")
        return
    df['year_month'] = df['incident_date'].dt.to_period('M')
    monthly_counts = df.groupby(['year_month', 'severity']).size().reset_index(name='count')
    monthly_counts['year_month'] = monthly_counts['year_month'].astype(str)
    fig = px.line(
        monthly_counts,
        x='year_month',
        y='count',
        color='severity',
        title="Monthly Incident Trends by Severity",
        markers=True,
        height=400
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Incidents",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True, key="monthly_trends")

def plot_medical_outcomes(df):
    if df.empty:
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
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-15
    )
    st.plotly_chart(fig, use_container_width=True, key="medical_outcomes")

def plot_incident_trends(df):
    if df.empty:
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
    if df.empty:
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
    if df.empty:
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

# ---------- OPERATIONAL PERFORMANCE & RISK ANALYSIS PAGE ---------
def plot_reporter_type_metrics(df):
    if 'reported_by' in df.columns:
        value = df['reported_by'].nunique()
        plot_metric("Reporter Types", value, color_graph="#5B8FF9")
    if 'medical_attention_required' in df.columns:
        value = int(df['medical_attention_required'].sum())
        plot_metric("Medical Attention Required", value, color_graph="#F6BD16")
    if 'participant_age' in df.columns:
        avg_age = df['participant_age'].mean()
        plot_metric("Avg Participant Age", avg_age, suffix=" yrs", color_graph="#5AD8A6")

def plot_reporter_performance_scatter(df):
    if {'reported_by','notification_date','incident_date'}.issubset(df.columns):
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
                'avg_delay': 'Average Notification Delay',
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

    df['investigation_required'] = df.apply(requires_investigation, axis=1)
    df['action_complete'] = df.apply(action_completed, axis=1)
    return df

# ---------- COMPLIANCE & INVESTIGATION PAGE ---------
def plot_compliance_metrics_poly(df):
    total = len(df)
    reportable_count = int(df['reportable'].sum())
    df = df.copy()
    df['report_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
    compliance_24h_count = int((df['report_delay_hours'] <= 24).sum())
    overdue_count = int((df['report_delay_hours'] > 24).sum())
    inv_required = int(df['investigation_required'].sum()) if 'investigation_required' in df.columns else 0
    inv_rate = inv_required / total * 100 if total > 0 else 0
    action_complete = int(df['action_complete'].sum()) if 'action_complete' in df.columns else 0
    action_progress = f"{action_complete}/{inv_required}" if inv_required > 0 else "0/0"
    breach_count = overdue_count

    plot_metric("Reportable Incidents", reportable_count, color_graph="#5B8FF9")
    plot_metric("24hr Compliance", compliance_24h_count, suffix=f" ({compliance_24h_count/total*100:.1f}%)" if total else "", color_graph="#5AD8A6")
    plot_metric("Overdue Reports", overdue_count, color_graph="#F6BD16")
    plot_metric("Investigation Rate", inv_rate, suffix="%", color_graph="#E86452")
    plot_metric("Investigation Status", action_progress, color_graph="#6DC8EC")
    plot_metric("Compliance Breach", breach_count, color_graph="#FF2B2B")

def plot_reporting_delay_by_date(df):
    if {'incident_date','notification_date'}.issubset(df.columns):
        df = df.copy()
        df['report_delay'] = (df['notification_date'] - df['incident_date']).dt.days
        agg = df.groupby('incident_date').agg(avg_delay=('report_delay', 'mean')).reset_index()
        fig = px.line(agg, x='incident_date', y='avg_delay', title="Reporting Delay by Incident Date")
        st.plotly_chart(fig, use_container_width=True, key="reporting_delay_by_date")

def plot_24h_compliance_rate_by_location(df):
    if {'location','notification_date','incident_date'}.issubset(df.columns):
        df = df.copy()
        df['within_24h'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() <= 24*3600
        compliance = df.groupby('location')['within_24h'].mean().reset_index()
        fig = px.bar(compliance, x='location', y='within_24h',
                     labels={'within_24h':'% Within 24hr'}, title="24 Hours Compliance Rate by Location")
        st.plotly_chart(fig, use_container_width=True, key="compliance_location")

def plot_investigation_pipeline(df):
    all_incidents = len(df)
    required = df['investigation_required'].sum() if 'investigation_required' in df.columns else 0
    complete = df['action_complete'].sum() if 'action_complete' in df.columns else 0
    values = [all_incidents, required, complete]
    names = ['All Incidents', 'Required Investigation', 'Action Complete']
    fig = px.bar(x=names, y=values, title="Investigation Pipeline", labels={'x':'Stage', 'y':'Count'})
    st.plotly_chart(fig, use_container_width=True, key="investigation_pipeline")

def plot_serious_injury_age_severity(df):
    if 'severity' in df.columns and 'participant_age' in df.columns:
        serious_df = df[df['severity'].str.lower() == 'high']
        fig = px.histogram(serious_df, x='participant_age', color='severity', nbins=20,
                           title="Serious Injury: Age and Severity Pattern")
        st.plotly_chart(fig, use_container_width=True, key="serious_injury_age_severity")

def plot_contributing_factors_by_month(df):
    if 'contributing_factors' in df.columns and 'incident_date' in df.columns:
        df = df.copy()
        df['month_year'] = df['incident_date'].dt.to_period('M').astype(str)
        fac_df = df.groupby(['month_year', 'contributing_factors']).size().unstack(fill_value=0)
        fig = px.imshow(fac_df.T, text_auto=True, aspect="auto",
                        labels=dict(x="Month-Year", y="Contributing Factor", color="Count"),
                        title="Contributing Factors by Month-Year")
        st.plotly_chart(fig, use_container_width=True, key="contributing_factors_month")
