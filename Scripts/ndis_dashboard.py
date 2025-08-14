import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="NDIS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 NDIS Incident Management Dashboard")


@st.cache_data
def load_data():
    """Load and preprocess NDIS incidents data"""
    df = pd.read_csv("/Users/darolinvinisha/PycharmProjects/MD651/Using Ollama/ndis_incidents_synthetic.csv")

    # Convert date columns
    df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
    df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')

    # Calculate notification delay
    df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.days

    # Add time-based columns
    df['month'] = df['incident_date'].dt.month_name()
    df['day_of_week'] = df['incident_date'].dt.day_name()
    df['hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour

    return df


# Load data
try:
    df = load_data()
    st.success(
        f"✅ Successfully loaded {len(df)} incidents from {df['incident_date'].min().strftime('%B %Y')} to {df['incident_date'].max().strftime('%B %Y')}")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters & Controls")

# Date range filter
if not df['incident_date'].isna().all():
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(df['incident_date'].min().date(), df['incident_date'].max().date()),
        min_value=df['incident_date'].min().date(),
        max_value=df['incident_date'].max().date()
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['incident_date'].dt.date >= start_date) & (df['incident_date'].dt.date <= end_date)]

# Multi-select filters
severity_filter = st.sidebar.multiselect(
    "⚠️ Severity Level",
    options=df['severity'].unique(),
    default=df['severity'].unique()
)

incident_type_filter = st.sidebar.multiselect(
    "📋 Incident Type",
    options=sorted(df['incident_type'].unique()),
    default=df['incident_type'].unique()
)

location_filter = st.sidebar.multiselect(
    "📍 Location",
    options=sorted(df['location'].unique()),
    default=df['location'].unique()
)

reportable_filter = st.sidebar.radio(
    "📊 Reportable Status",
    options=['All', 'Reportable Only', 'Non-Reportable Only'],
    index=0
)

# Apply filters
filtered_df = df[
    (df['severity'].isin(severity_filter)) &
    (df['incident_type'].isin(incident_type_filter)) &
    (df['location'].isin(location_filter))
    ]

if reportable_filter == 'Reportable Only':
    filtered_df = filtered_df[filtered_df['reportable'] == 'Yes']
elif reportable_filter == 'Non-Reportable Only':
    filtered_df = filtered_df[filtered_df['reportable'] == 'No']

# Alert for critical incidents
critical_count = len(filtered_df[filtered_df['severity'] == 'Critical'])
if critical_count > 0:
    st.markdown(f"""
    <div class="alert-box alert-critical">
        <strong>⚠️ ALERT:</strong> {critical_count} critical incidents found in current filter selection. 
        Immediate review recommended.
    </div>
    """, unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 Total Incidents", len(filtered_df),
              delta=f"{len(filtered_df) - len(df) if len(filtered_df) != len(df) else ''}")

with col2:
    critical_pct = (critical_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("🚨 Critical", critical_count, delta=f"{critical_pct:.1f}%")

with col3:
    reportable_count = len(filtered_df[filtered_df['reportable'] == 'Yes'])
    reportable_pct = (reportable_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("📋 Reportable", reportable_count, delta=f"{reportable_pct:.1f}%")

with col4:
    unique_participants = filtered_df['ndis_number'].nunique()
    st.metric("👥 Participants", unique_participants)

with col5:
    avg_delay = filtered_df['notification_delay'].mean()
    st.metric("⏱️ Avg Delay (days)", f"{avg_delay:.1f}" if pd.notna(avg_delay) else "N/A")

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Analysis", "📋 Data"])

with tab1:
    # Main visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Top incident types
        incident_counts = filtered_df['incident_type'].value_counts().head(8)
        fig1 = px.bar(
            x=incident_counts.values,
            y=incident_counts.index,
            orientation='h',
            title="🔝 Top Incident Types",
            color=incident_counts.values,
            color_continuous_scale='Blues',
            text=incident_counts.values
        )
        fig1.update_traces(textposition='inside')
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Severity distribution with custom colors
        severity_counts = filtered_df['severity'].value_counts()
        colors = {'Critical': '#ff4444', 'High': '#ff8800', 'Medium': '#ffcc00', 'Low': '#44ff44'}
        fig2 = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="⚠️ Severity Distribution",
            color=severity_counts.index,
            color_discrete_map=colors
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

    # Location analysis
    st.subheader("📍 Incidents by Location")
    location_counts = filtered_df['location'].value_counts().head(10)
    fig3 = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title="Top 10 Locations for Incidents",
        color=location_counts.values,
        color_continuous_scale='Reds'
    )
    fig3.update_xaxes(tickangle=45)
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("📈 Incident Trends")

    if not filtered_df['incident_date'].isna().all():
        # Monthly trends
        monthly_trends = filtered_df.groupby(filtered_df['incident_date'].dt.to_period('M')).size()
        monthly_trends.index = monthly_trends.index.astype(str)

        fig_trend = px.line(
            x=monthly_trends.index,
            y=monthly_trends.values,
            title="📊 Monthly Incident Trends",
            markers=True
        )
        fig_trend.update_xaxes(tickangle=45)
        st.plotly_chart(fig_trend, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Day of week analysis
            dow_counts = filtered_df['day_of_week'].value_counts()
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = dow_counts.reindex(dow_order, fill_value=0)

            fig_dow = px.bar(
                x=dow_counts.index,
                y=dow_counts.values,
                title="📅 Incidents by Day of Week",
                color=dow_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        with col2:
            # Hour analysis (if time data available)
            if not filtered_df['hour'].isna().all():
                hour_counts = filtered_df['hour'].value_counts().sort_index()
                fig_hour = px.bar(
                    x=hour_counts.index,
                    y=hour_counts.values,
                    title="🕐 Incidents by Hour of Day",
                    color=hour_counts.values,
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig_hour, use_container_width=True)

with tab3:
    st.subheader("🔍 Advanced Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Notification delay analysis
        if not filtered_df['notification_delay'].isna().all():
            st.write("⏱️ **Notification Delay Analysis**")
            delay_stats = filtered_df['notification_delay'].describe()
            st.write(f"- Average delay: {delay_stats['mean']:.1f} days")
            st.write(f"- Median delay: {delay_stats['50%']:.1f} days")
            st.write(f"- Max delay: {delay_stats['max']:.0f} days")

            # Delays over 1 day
            late_notifications = filtered_df[filtered_df['notification_delay'] > 1]
            if len(late_notifications) > 0:
                st.warning(f"⚠️ {len(late_notifications)} incidents had notification delays > 1 day")

    with col2:
        # High-risk analysis
        st.write("🚨 **Risk Assessment**")

        # Participants with multiple incidents
        participant_counts = filtered_df['participant_name'].value_counts()
        repeat_participants = participant_counts[participant_counts > 1]

        if len(repeat_participants) > 0:
            st.write(f"👥 {len(repeat_participants)} participants with multiple incidents")
            st.write("Top repeat participants:")
            for name, count in repeat_participants.head(5).items():
                st.write(f"- {name}: {count} incidents")
        else:
            st.success("✅ No repeat incidents for any participant")

    # Contributing factors analysis
    st.subheader("🎯 Contributing Factors")
    if 'contributing_factors' in filtered_df.columns:
        factors = filtered_df['contributing_factors'].value_counts().head(10)
        fig_factors = px.bar(
            x=factors.values,
            y=factors.index,
            orientation='h',
            title="Top Contributing Factors",
            color=factors.values,
            color_continuous_scale='Oranges'
        )
        fig_factors.update_layout(height=400)
        st.plotly_chart(fig_factors, use_container_width=True)

with tab4:
    st.subheader("📋 Incident Data")

    # Search functionality
    search_term = st.text_input("🔍 Search incidents (description, actions, etc.)")

    # Column selection
    available_columns = ['incident_id', 'participant_name', 'incident_date', 'incident_type',
                         'severity', 'location', 'reportable', 'description', 'immediate_action']
    display_columns = st.multiselect(
        "Select columns to display",
        options=[col for col in available_columns if col in filtered_df.columns],
        default=[col for col in available_columns[:7] if col in filtered_df.columns]
    )

    # Apply search filter
    display_df = filtered_df.copy()
    if search_term:
        search_cols = ['description', 'immediate_action', 'actions_taken', 'contributing_factors']
        search_mask = pd.Series(False, index=display_df.index)
        for col in search_cols:
            if col in display_df.columns:
                search_mask |= display_df[col].str.contains(search_term, case=False, na=False)
        display_df = display_df[search_mask]

    # Display data
    if display_columns:
        st.dataframe(
            display_df[display_columns].sort_values('incident_date', ascending=False)
            if 'incident_date' in display_columns else display_df[display_columns],
            use_container_width=True,
            height=400
        )

        # Download options
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered CSV",
                csv_data,
                f"ndis_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

        with col2:
            # Summary statistics
            if st.button("📊 Generate Summary Report"):
                st.subheader("📈 Summary Statistics")
                st.write(f"**Total incidents:** {len(display_df)}")
                st.write(f"**Critical incidents:** {len(display_df[display_df['severity'] == 'Critical'])}")
                st.write(f"**Reportable incidents:** {len(display_df[display_df['reportable'] == 'Yes'])}")
                st.write(f"**Unique participants:** {display_df['ndis_number'].nunique()}")
                st.write(
                    f"**Most common incident type:** {display_df['incident_type'].mode().iloc[0] if len(display_df) > 0 else 'N/A'}")
                st.write(
                    f"**Most common location:** {display_df['location'].mode().iloc[0] if len(display_df) > 0 else 'N/A'}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown(f"**Records displayed:** {len(filtered_df)} of {len(df)}")
with col3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()