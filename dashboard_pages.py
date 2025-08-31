import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------- DASHBOARD PLOTS -------------------

def plot_metric(label, value, show_graph=False, color_graph="rgba(0, 104, 201, 0.2)"):
    st.metric(label, value)

def plot_gauge(value, color, suffix, label, max_value):
    st.progress(min(int(value), max_value), text=f"{label}: {value:.1f}{suffix}")

def plot_severity_distribution(df):
    if 'severity' in df.columns:
        fig = px.pie(df, names='severity', title="Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)

def plot_incident_types_bar(df):
    if 'incident_type' in df.columns:
        ct = df['incident_type'].value_counts().reset_index()
        ct.columns = ['incident_type', 'count']
        fig = px.bar(ct, x='incident_type', y='count', title="Incident Types")
        st.plotly_chart(fig, use_container_width=True)

def plot_location_analysis(df):
    if 'location' in df.columns:
        ct = df['location'].value_counts().reset_index()
        ct.columns = ['location', 'count']
        fig = px.bar(ct, x='location', y='count', title="Location Analysis")
        st.plotly_chart(fig, use_container_width=True)

def plot_monthly_trends(df):
    if 'incident_month_name' in df.columns and 'incident_year' in df.columns:
        ct = df.groupby(['incident_year', 'incident_month_name']).size().reset_index(name='count')
        ct = ct.sort_values(['incident_year', 'incident_month_name'])
        fig = px.bar(ct, x='incident_month_name', y='count', color='incident_year', barmode='group',
                     title="Monthly Incident Trends")
        st.plotly_chart(fig, use_container_width=True)

def plot_medical_outcomes(df):
    if 'medical_outcome' in df.columns:
        ct = df['medical_outcome'].value_counts().reset_index()
        ct.columns = ['medical_outcome', 'count']
        fig = px.bar(ct, x='medical_outcome', y='count', title="Medical Outcomes")
        st.plotly_chart(fig, use_container_width=True)

# --------------- OPERATIONAL PERFORMANCE & RISK ANALYSIS ----------------

def apply_investigation_rules(df):
    """Adds investigation_required and action_complete columns to df as per business rules."""
    def requires_investigation(row):
        # High Severity
        if str(row.get('severity', '')).lower() == 'high':
            return True
        # Reportable = True
        if row.get('reportable', False):
            return True
        # Serious incident types
        serious_types = ['unethical behavior', 'assault', 'unauthorized restraints']
        if str(row.get('incident_type', '')).strip().lower() in serious_types:
            return True
        # Injury with treatment/medical attention
        if row.get('medical_attention_required', False) or row.get('treatment_required', False):
            return True
        return False

    def action_completed(row):
        # Medical outcome = recovered
        if str(row.get('medical_outcome', '')).strip().lower() == 'recovered':
            return True
        # Low severity + no treatment
        if str(row.get('severity', '').lower()) == 'low' and not (row.get('medical_attention_required', False) or row.get('treatment_required', False)):
            return True
        # Documented actions (if you have such a boolean column)
        if row.get('actions_documented', False):
            return True
        return False

    df['investigation_required'] = df.apply(requires_investigation, axis=1)
    df['action_complete'] = df.apply(action_completed, axis=1)
    return df

def plot_reporter_type_metric(df):
    if 'reported_by' in df.columns:
        st.metric("Reporter Types", df['reported_by'].nunique())
    if 'medical_attention_required' in df.columns:
        st.metric("Medical Attention Required", int(df['medical_attention_required'].sum()))
    if 'participant_age' in df.columns:
        st.metric("Avg Participant Age", f"{df['participant_age'].mean():.1f} yrs")

def plot_reporter_performance_scatter(df):
    if {'reported_by','notification_date','incident_date'}.issubset(df.columns):
        perf = (
            df.groupby('reported_by')
            .agg(
                avg_delay=('notification_date', lambda x: (x - df.loc[x.index, 'incident_date']).dt.days.mean()),
                total_incidents=('incident_date', 'count')
            ).reset_index()
        )
        fig = px.scatter(perf, x='avg_delay', y='total_incidents', color='reported_by',
                         labels={'avg_delay': 'Avg Notification Delay (days)', 'total_incidents': 'Total Incidents'},
                         title='Reporter Performance Analysis')
        st.plotly_chart(fig, use_container_width=True)

def plot_incident_heatmap(df):
    if {'incident_type','severity'}.issubset(df.columns):
        cross_tab = pd.crosstab(df['incident_type'], df['severity'])
        fig = px.imshow(cross_tab, text_auto=True, aspect="auto",
                        labels=dict(x="Severity", y="Incident Type", color="Count"),
                        title="Incident Type x Severity Heatmap")
        st.plotly_chart(fig, use_container_width=True)

def plot_avg_reporting_day_by_role(df):
    if {'reported_by','notification_date','incident_date'}.issubset(df.columns):
        role_delays = (
            df.assign(report_delay=(df['notification_date'] - df['incident_date']).dt.days)
            .groupby('reported_by')['report_delay'].mean()
            .reset_index()
        )
        fig = px.bar(role_delays, x='reported_by', y='report_delay',
                     labels={'report_delay': 'Avg Reporting Delay (days)', 'reported_by': 'Reporter Type'},
                     title='Average Reporting Day by Role')
        st.plotly_chart(fig, use_container_width=True)

def plot_medical_attention_vs_total(df):
    if {'medical_attention_required','incident_type'}.issubset(df.columns):
        sum_df = df.groupby('incident_type').agg(
            total_incidents=('medical_attention_required', 'count'),
            medical_attention=('medical_attention_required', 'sum')
        ).reset_index()
        fig = px.bar(sum_df, x='incident_type', y=['total_incidents', 'medical_attention'],
                     barmode='group', title="Medical Attention Required vs Total Incidents")
        st.plotly_chart(fig, use_container_width=True)

def plot_temporal_patterns(df):
    if {'incident_weekday','severity'}.issubset(df.columns):
        fig = px.histogram(df, x='incident_weekday', color='severity', barmode='group',
                           category_orders={'incident_weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
                           title="Incidents by Day of Week & Severity")
        st.plotly_chart(fig, use_container_width=True)

def plot_reporting_delay_by_date(df):
    if {'incident_date','notification_date'}.issubset(df.columns):
        df = df.copy()
        df['report_delay'] = (df['notification_date'] - df['incident_date']).dt.days
        agg = df.groupby('incident_date').agg(avg_delay=('report_delay', 'mean')).reset_index()
        fig = px.line(agg, x='incident_date', y='avg_delay', title="Reporting Delay by Incident Date")
        st.plotly_chart(fig, use_container_width=True)

def plot_24h_compliance_rate_by_location(df):
    if {'location','notification_date','incident_date'}.issubset(df.columns):
        df = df.copy()
        df['within_24h'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() <= 24*3600
        compliance = df.groupby('location')['within_24h'].mean().reset_index()
        fig = px.bar(compliance, x='location', y='within_24h',
                     labels={'within_24h':'% Within 24hr'}, title="24 Hours Compliance Rate by Location")
        st.plotly_chart(fig, use_container_width=True)

def plot_investigation_pipeline(df):
    # Columns: 'investigation_required', 'action_complete'
    if 'investigation_required' in df.columns:
        all_incidents = len(df)
        required = df['investigation_required'].sum()
        complete = df['action_complete'].sum() if 'action_complete' in df.columns else 0
        values = [all_incidents, required, complete]
        names = ['All Incidents', 'Required Investigation', 'Action Complete']
        fig = px.bar(x=names, y=values, title="Investigation Pipeline")
        st.plotly_chart(fig, use_container_width=True)

def plot_serious_injury_age_severity(df):
    if 'severity' in df.columns and 'participant_age' in df.columns:
        serious_df = df[df['severity'].str.lower() == 'high']
        fig = px.histogram(serious_df, x='participant_age', color='severity', nbins=20,
                           title="Serious Injury: Age and Severity Pattern")
        st.plotly_chart(fig, use_container_width=True)

def plot_contributing_factors_by_month(df):
    if 'contributing_factors' in df.columns and 'incident_date' in df.columns:
        df = df.copy()
        df['month_year'] = df['incident_date'].dt.to_period('M')
        fac_df = df.groupby(['month_year', 'contributing_factors']).size().unstack(fill_value=0)
        fig = px.imshow(fac_df.T, text_auto=True, aspect="auto",
                        labels=dict(x="Month-Year", y="Contributing Factor", color="Count"),
                        title="Contributing Factors by Month-Year")
        st.plotly_chart(fig, use_container_width=True)

# --- COMPLIANCE & INVESTIGATION METRICS ---

def plot_compliance_metrics(df):
    # Reportable incidents
    reportable_count = int(df['reportable'].sum())
    total = len(df)
    # 24 hr compliance: notification_date within 24 hours of incident_date
    df = df.copy()
    df['report_delay_hours'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / 3600
    compliance_24h_count = int((df['report_delay_hours'] <= 24).sum())
    overdue_count = int((df['report_delay_hours'] > 24).sum())
    # Investigation rate
    inv_required = int(df['investigation_required'].sum()) if 'investigation_required' in df.columns else 0
    inv_rate = inv_required / total * 100 if total > 0 else 0
    # Investigation status
    action_complete = int(df['action_complete'].sum()) if 'action_complete' in df.columns else 0
    action_progress = f"{action_complete}/{inv_required}" if inv_required > 0 else "0/0"
    # Compliance breach count
    breach_count = overdue_count
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Reportable Incidents", reportable_count)
    col2.metric("24hr Compliance", f"{compliance_24h_count} ({compliance_24h_count/total*100:.1f}%)" if total else "0")
    col3.metric("Overdue Reports", overdue_count)
    col4.metric("Investigation Rate", f"{inv_rate:.1f}%" )
    col5.metric("Investigation Status", action_progress)
    col6.metric("Compliance Breach", breach_count)

def plot_reporting_delay_by_incident_date(df):
    # Average report delay by date
    df = df.copy()
    df['report_delay'] = (df['notification_date'] - df['incident_date']).dt.total_seconds()/3600
    by_date = df.groupby('incident_date').agg(avg_delay=('report_delay', 'mean')).reset_index()
    fig = px.line(by_date, x='incident_date', y='avg_delay', title="Reporting Delay by Incident Date", labels={'avg_delay':'Avg Delay (hours)'})
    st.plotly_chart(fig, use_container_width=True)

def plot_24hr_compliance_by_location(df):
    df = df.copy()
    df['within_24h'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() <= 24*3600
    compliance = df.groupby('location')['within_24h'].mean().reset_index()
    compliance['percentage'] = compliance['within_24h']*100
    fig = px.bar(compliance, x='location', y='percentage', labels={'percentage':'% within 24hr'}, title="24hr Compliance Rate by Location")
    st.plotly_chart(fig, use_container_width=True)

def plot_investigation_pipeline(df):
    all_incidents = len(df)
    required = df['investigation_required'].sum() if 'investigation_required' in df.columns else 0
    complete = df['action_complete'].sum() if 'action_complete' in df.columns else 0
    values = [all_incidents, required, complete]
    names = ['All Incidents', 'Required Investigation', 'Action Complete']
    fig = px.bar(x=names, y=values, title="Investigation Pipeline", labels={'x':'Stage', 'y':'Count'})
    st.plotly_chart(fig, use_container_width=True)

def plot_serious_injury_age_severity(df):
    if 'severity' in df.columns and 'participant_age' in df.columns:
        serious_df = df[df['severity'].str.lower() == 'high']
        fig = px.histogram(serious_df, x='participant_age', color='severity', nbins=20,
                           title="Serious Injury: Age and Severity Pattern")
        st.plotly_chart(fig, use_container_width=True)

def plot_contributing_factors_by_month(df):
    if 'contributing_factors' in df.columns and 'incident_date' in df.columns:
        df = df.copy()
        df['month_year'] = df['incident_date'].dt.to_period('M').astype(str)
        fac_df = df.groupby(['month_year', 'contributing_factors']).size().unstack(fill_value=0)
        fig = px.imshow(fac_df.T, text_auto=True, aspect="auto",
                        labels=dict(x="Month-Year", y="Contributing Factor", color="Count"),
                        title="Contributing Factors by Month-Year")
        st.plotly_chart(fig, use_container_width=True)
