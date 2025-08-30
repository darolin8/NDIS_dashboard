"""
Dashboard visualization functions for incident management
Includes traditional analytics and ML-enhanced visualizations
"""

import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    """Display a metric with optional mini graph"""
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        # Generate realistic trend data based on the context
        if "High Severity" in label:
            trend_data = [random.randint(5, 25) for _ in range(30)]
        elif "Total" in label:
            trend_data = [random.randint(20, 80) for _ in range(30)]
        else:
            trend_data = [random.randint(0, 50) for _ in range(30)]
            
        fig.add_trace(
            go.Scatter(
                y=trend_data,
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={"color": color_graph},
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound):
    """Create a gauge chart for KPI visualization"""
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
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
            title={
                "text": indicator_title,
                "font": {"size": 20},
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_severity_distribution(df):
    """Create a pie chart for incident severity distribution"""
    if df.empty:
        st.warning("No data available for severity distribution")
        return
        
    severity_counts = df['severity'].value_counts()
    
    colors = {
        'High': '#FF2B2B',
        'Moderate': '#FF8700', 
        'Low': '#29B09D'
    }
    
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_incident_types_bar(df):
    """Create a horizontal bar chart for incident types"""
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_location_analysis(df):
    """Create a bar chart for incident locations"""
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
    
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_trends(df):
    """Create a line chart showing monthly incident trends"""
    if df.empty:
        st.warning("No data available for monthly trends")
        return
        
    # Group by year-month for better trending
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_medical_outcomes(df):
    """Create charts showing medical treatment outcomes"""
    if df.empty:
        st.warning("No data available for medical outcomes")
        return
        
    # Create summary of medical outcomes
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_incident_trends(df):
    """Create a comprehensive incident trend analysis"""
    if df.empty:
        st.warning("No data available for incident trends")
        return
        
    # Daily incident counts
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_weekday_analysis(df):
    """Analyze incidents by day of week"""
    if df.empty:
        st.warning("No data available for weekday analysis")
        return
        
    weekday_counts = df['incident_weekday'].value_counts()
    
    # Reorder days of week
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_time_analysis(df):
    """Analyze incidents by time of day"""
    if df.empty or 'incident_time' not in df.columns:
        st.warning("No time data available for analysis")
        return
        
    # Convert time to hour
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_reportable_analysis(df):
    """Analyze reportable vs non-reportable incidents"""
    if df.empty:
        st.warning("No data available for reportable analysis")
        return
        
    reportable_counts = df['reportable'].value_counts()
    reportable_counts.index = ['Not Reportable', 'Reportable']
    
    fig = px.pie(
        values=reportable_counts.values,
        names=reportable_counts.index,
        title="Reportable Incidents Distribution",
        color_discrete_sequence=['#90EE90', '#FFB6C1']
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_scatter(clustered_df, x_col='pca_x', y_col='pca_y', color_col='cluster', 
                        title="Incident Clusters", hover_cols=None):
    """Create an enhanced scatter plot for cluster visualization"""
    if clustered_df is None or clustered_df.empty:
        st.warning("No clustered data available")
        return
    if hover_cols is None:
        hover_cols = ['incident_type', 'location', 'severity']
    clustered_df_copy = clustered_df.copy()
    clustered_df_copy[color_col] = clustered_df_copy[color_col].astype(str)
    fig = px.scatter(
        clustered_df_copy,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        hover_data=hover_cols,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    # Add cluster centroids if PCA data
    if 'pca_x' in clustered_df_copy.columns and 'pca_y' in clustered_df_copy.columns:
        centroids = clustered_df_copy.groupby(color_col).agg({
            'pca_x': 'mean',
            'pca_y': 'mean'
        }).reset_index()
        fig.add_scatter(
            x=centroids['pca_x'],
            y=centroids['pca_y'],
            mode='markers',
            marker=dict(
                size=15,
                symbol='x',
                color='black',
                line=dict(width=2, color='white')
            ),
            name='Centroids',
            showlegend=True
        )
    fig.update_layout(
        xaxis_title='First Principal Component' if x_col == 'pca_x' else x_col,
        yaxis_title='Second Principal Component' if y_col == 'pca_y' else y_col,
        legend_title="Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_characteristics_heatmap(cluster_analysis):
    """Create a heatmap showing cluster characteristics"""
    if cluster_analysis is None:
        st.warning("No cluster analysis data available")
        return
    clusters = list(cluster_analysis.keys())
    metrics = ['size', 'avg_medical_attention', 'avg_reportable']
    heatmap_data = []
    for cluster_id in clusters:
        row = []
        for metric in metrics:
            if metric in cluster_analysis[cluster_id]:
                value = cluster_analysis[cluster_id][metric]
                # Normalize size metric
                if metric == 'size':
                    max_size = max([cluster_analysis[c]['size'] for c in clusters])
                    value = value / max_size if max_size > 0 else 0
                row.append(value)
            else:
                row.append(0)
        heatmap_data.append(row)
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['Relative Size', 'Medical Attention %', 'Reportable %'],
        y=[f'Cluster {c}' for c in clusters],
        colorscale='RdYlBu_r',
        text=[[f'{val:.2f}' for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Normalized Value")
    ))
    fig.update_layout(
        title="Cluster Characteristics Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Clusters",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_distribution_sunburst(clustered_df):
    """Create a sunburst chart showing cluster distribution by incident type and severity"""
    if clustered_df is None or clustered_df.empty:
        st.warning("No clustered data available")
        return
    sunburst_data = clustered_df.groupby(['cluster', 'incident_type', 'severity']).size().reset_index(name='count')
    sunburst_data['cluster'] = 'Cluster ' + sunburst_data['cluster'].astype(str)
    fig = px.sunburst(
        sunburst_data,
        path=['cluster', 'incident_type', 'severity'],
        values='count',
        title="Cluster Distribution: Type → Severity",
        color='count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_anomaly_timeline(anomaly_df):
    """Create a timeline showing anomalies over time"""
    if anomaly_df is None or anomaly_df.empty:
        st.warning("No anomaly data available")
        return
    timeline_data = anomaly_df.copy()
    timeline_data['is_anomaly'] = (
        timeline_data['isolation_forest_anomaly'] | 
        timeline_data['svm_anomaly']
    )
    fig = px.scatter(
        timeline_data,
        x='incident_date',
        y='anomaly_score',
        color='is_anomaly',
        title="Anomaly Detection Timeline",
        hover_data=['incident_type', 'location', 'severity'],
        color_discrete_map={True: '#FF2B2B', False: '#29B09D'},
        labels={'is_anomaly': 'Anomaly Detected'}
    )
    threshold = timeline_data['anomaly_score'].quantile(0.1)
    fig.add_hline(
        y=threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Anomaly Threshold"
    )
    fig.update_layout(
        xaxis_title="Incident Date",
        yaxis_title="Anomaly Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance from trained model"""
    if model is None or feature_names is None:
        st.warning("No model or feature names available")
        return
    try:
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        fig = px.bar(
            feature_importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f"Top {top_n} Feature Importances",
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot feature importance: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plot confusion matrix for classification results"""
    from sklearn.metrics import confusion_matrix
    if labels is None:
        labels = ['Low', 'Moderate', 'High']
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast_with_confidence(historical_data, forecast_data, confidence_intervals=None):
    """Enhanced forecast visualization with confidence intervals"""
    if historical_data is None or forecast_data is None:
        st.warning("No forecast data available")
        return
    fig = go.Figure()
    hist_df = historical_data.reset_index() if hasattr(historical_data, 'index') else historical_data
    fig.add_trace(go.Scatter(
        x=hist_df.iloc[:, 0],
        y=hist_df.iloc[:, 1],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#0068C9')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_data.iloc[:, 0],
        y=forecast_data.iloc[:, 1],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#FF8700', dash='dash')
    ))
    if confidence_intervals is not None:
        fig.add_trace(go.Scatter(
            x=forecast_data.iloc[:, 0],
            y=confidence_intervals['upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_data.iloc[:, 0],
            y=confidence_intervals['lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(255, 135, 0, 0.2)'
        ))
    fig.update_layout(
        title="Incident Forecast with Historical Trend",
        xaxis_title="Date",
        yaxis_title="Number of Incidents",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_severity_prediction_probabilities(probabilities, class_names):
    """Plot prediction probabilities for severity classification"""
    if probabilities is None:
        st.warning("No probability data available")
        return
    prob_df = pd.DataFrame({
        'Severity': class_names,
        'Probability': probabilities
    })
    fig = px.bar(
        prob_df,
        x='Severity',
        y='Probability',
        title="Severity Prediction Probabilities",
        color='Probability',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_layout(
        yaxis_title="Probability",
        showlegend=False,
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    
def plot_correlation_matrix(df, features=None):
    """Plot correlation matrix for numerical features"""
    if df.empty:
        st.warning("No data available for correlation analysis")
        return
    if features is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = numerical_cols[:10]  # Limit to first 10 for readability
    if not features:
        st.warning("No numerical features found for correlation analysis")
        return
    corr_matrix = df[features].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        xaxis={'side': 'bottom'}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_incident_severity_trends(df):
    """Enhanced severity trends with statistical insights"""
    if df.empty:
        st.warning("No data available for severity trends")
        return
    df['year_month'] = df['incident_date'].dt.to_period('M')
    monthly_severity = df.groupby(['year_month', 'severity']).size().reset_index(name='count')
    monthly_severity['year_month'] = monthly_severity['year_month'].astype(str)
    fig = px.area(
        monthly_severity,
        x='year_month',
        y='count',
        color='severity',
        title="Incident Severity Trends Over Time",
        color_discrete_map={'Low': '#29B09D', 'Moderate': '#FF8700', 'High': '#FF2B2B'}
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Incidents",
        xaxis_tickangle=-45,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_location_risk_assessment(df):
    """Advanced location-based risk assessment"""
    if df.empty:
        st.warning("No data available for location risk assessment")
        return
    location_metrics = df.groupby('location').agg({
        'severity': lambda x: (x == 'High').mean(),  # High severity rate
        'medical_attention_required': lambda x: x.sum() if x.dtype == bool else (x == True).sum(),
        'incident_id': 'count'  # Total incidents
    }).reset_index()
    location_metrics.columns = ['location', 'high_severity_rate', 'medical_cases', 'total_incidents']
    location_metrics = location_metrics[location_metrics['total_incidents'] >= 5]
    if location_metrics.empty:
        st.warning("Insufficient data for location risk assessment")
        return
    fig = px.scatter(
        location_metrics,
        x='total_incidents',
        y='high_severity_rate',
        size='medical_cases',
        hover_data=['location'],
        title="Location Risk Assessment Matrix",
        labels={
            'total_incidents': 'Total Incidents',
            'high_severity_rate': 'High Severity Rate',
            'medical_cases': 'Medical Cases'
        }
    )
    median_incidents = location_metrics['total_incidents'].median()
    median_severity_rate = location_metrics['high_severity_rate'].median()
    fig.add_vline(x=median_incidents, line_dash="dash", line_color="gray", annotation_text="Median Incidents")
    fig.add_hline(y=median_severity_rate, line_dash="dash", line_color="gray", annotation_text="Median Severity Rate")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

