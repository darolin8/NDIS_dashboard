import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Chart functions (your improved code) ---

def plot_feature_importance(feature_importances, top_n=10):
    fi = feature_importances.sort_values(ascending=False)[:top_n]
    fig = px.bar(
        fi[::-1],
        orientation="h",
        labels={"value": "Importance", "index": "Feature"},
        title="Top Feature Importances"
    )
    fig.update_layout(height=360, plot_bgcolor='white')
    return fig

def plot_predicted_severity_distribution(preds):
    df = pd.DataFrame({'Severity': preds})
    fig = px.histogram(df, x='Severity', color='Severity', title="Predicted Severity Distribution")
    fig.update_layout(height=320, plot_bgcolor='white')
    return fig

def plot_anomaly_score_distribution(anomaly_scores):
    fig = px.histogram(pd.DataFrame({'score': anomaly_scores}), x='score', nbins=25, title="Anomaly Score Distribution")
    fig.update_layout(height=320, plot_bgcolor='white')
    return fig

def plot_anomaly_rate_by_severity(df):
    if not all(col in df.columns for col in ['is_anomaly', 'severity']):
        return go.Figure()
    summary = df.groupby('severity')['is_anomaly'].mean().reset_index()
    fig = px.bar(summary, x='severity', y='is_anomaly', labels={'is_anomaly':'Anomaly Rate'}, title="Anomaly Rate by Severity")
    fig.update_layout(height=320, plot_bgcolor='white')
    return fig

def plot_support_confidence(support, confidence, labels):
    df = pd.DataFrame({'support': support, 'confidence': confidence, 'rule': labels})
    fig = px.scatter(df, x='support', y='confidence', text='rule', title="Support vs Confidence of Rules")
    fig.update_traces(textposition='top center')
    fig.update_layout(height=400, plot_bgcolor='white')
    return fig

def plot_incident_forecast(dates, actual, predicted):
    df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Actual': actual, 'Predicted': predicted})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], mode='lines+markers', name='Predicted'))
    fig.update_layout(title="Incident Forecast", xaxis_title="Date", yaxis_title="Incidents", height=350, plot_bgcolor='white')
    return fig

def plot_location_risk(df):
    if not all(col in df.columns for col in ['location', 'incident_id', 'severity']):
        return go.Figure()
    risk = df.groupby('location').agg(
        total=('incident_id','count'),
        critical=('severity', lambda x: (x=='Critical').sum())
    ).reset_index()
    risk['critical_pct'] = np.where(risk['total']>0, risk['critical']/risk['total']*100, 0)
    fig = px.scatter(
        risk,
        x='total', y='critical_pct', size='critical', color='critical_pct',
        hover_name='location',
        color_continuous_scale=[[0, '#2F9E7D'], [0.5, '#F59C2F'], [1, '#DC2626']],
        labels={'total':'Total Incidents', 'critical_pct':'% Critical'},
        title="Location Risk Assessment (Critical % vs Total)"
    )
    fig.update_layout(height=420, plot_bgcolor='white')
    return fig

def plot_advanced_clustering(df, feature_cols, n_clusters=4):
    if not all(col in df.columns for col in feature_cols):
        return go.Figure()
    X = df[feature_cols].dropna().values
    if X.shape[0] == 0:
        return go.Figure()
    pca = PCA(n_components=2).fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    plot_df = pd.DataFrame({
        'PC1': pca[:,0], 'PC2': pca[:,1], 'Cluster': cluster_labels,
        'Severity': df.loc[df[feature_cols].dropna().index, 'severity'].values if 'severity' in df else None
    })
    fig = px.scatter(
        plot_df, x='PC1', y='PC2', color='Cluster', symbol='Severity' if 'Severity' in plot_df else None,
        title="Incident Clusters (PCA projection)",
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
    )
    fig.update_layout(height=420, plot_bgcolor='white')
    return fig

# --- Main ML Analytics Page with chart calls ---

def render_ml_analytics(
    filtered_df: pd.DataFrame, 
    feature_importances=None, 
    preds=None, 
    anomaly_scores=None, 
    association_rules=None, 
    forecast=None,
    clustering_cols=None
):
    st.title("🤖 Machine Learning Analytics")
    st.write("Advanced analytics and ML insights for NDIS incident data")

    # 1. Feature Importance
    if feature_importances is not None and len(feature_importances) > 0:
        st.subheader("Top Features for Severity Prediction")
        st.plotly_chart(plot_feature_importance(feature_importances), use_container_width=True)

    # 2. Predicted Severity Distribution
    if preds is not None and len(preds) > 0:
        st.subheader("Predicted Severity Distribution")
        st.plotly_chart(plot_predicted_severity_distribution(preds), use_container_width=True)

    # 3. Anomaly Score Distribution
    if anomaly_scores is not None and len(anomaly_scores) > 0:
        st.subheader("Anomaly Score Distribution")
        st.plotly_chart(plot_anomaly_score_distribution(anomaly_scores), use_container_width=True)
        # 4. Anomaly Rate by Severity
        if "is_anomaly" in filtered_df.columns and "severity" in filtered_df.columns:
            st.subheader("Anomaly Rate by Severity")
            st.plotly_chart(plot_anomaly_rate_by_severity(filtered_df), use_container_width=True)

    # 5. Association Rules: Support & Confidence
    if association_rules is not None and all(k in association_rules for k in ['support','confidence','labels']):
        st.subheader("Association Rule Support & Confidence")
        st.plotly_chart(
            plot_support_confidence(
                association_rules['support'], 
                association_rules['confidence'], 
                association_rules['labels']
            ), 
            use_container_width=True
        )

    # 6. Incident Forecast (time series)
    if forecast is not None and all(k in forecast for k in ['dates','actual','predicted']):
        st.subheader("Incident Forecast (Next 30 Days)")
        st.plotly_chart(
            plot_incident_forecast(
                forecast['dates'], 
                forecast['actual'], 
                forecast['predicted']
            ), 
            use_container_width=True
        )

    # 7. Location Risk Assessment (Critical % vs Total)
    if all(c in filtered_df.columns for c in ['incident_id','severity','location']):
        st.subheader("Location Risk Assessment (Critical % vs Total)")
        st.plotly_chart(plot_location_risk(filtered_df), use_container_width=True)

    # 8. Advanced Clustering (only if you specify clustering columns)
    if clustering_cols and all(c in filtered_df.columns for c in clustering_cols):
        st.subheader("Advanced Clustering of Incidents")
        st.plotly_chart(
            plot_advanced_clustering(filtered_df, clustering_cols, n_clusters=4), 
            use_container_width=True
        )
    st.write(filtered_df.head())

# --- Other pages ---

def render_executive_summary(filtered_df):
    st.header("Executive Summary")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # 1. Number of incidents by month and severity (custom colors)
    st.subheader("Incidents by Month and Severity")
    if "incident_month" not in filtered_df.columns:
        filtered_df['incident_month'] = filtered_df['incident_date'].dt.strftime('%b')
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    severity_colors = {
        "Critical": "#DC2626",  # red
        "High": "#F59C2F",      # orange
        "Medium": "#2F9E7D",    # teal
        "Low": "#37d67a"        # green
    }
    count_month_sev = (
        filtered_df.groupby(['incident_month','severity'])
        .size().reset_index(name='count')
    )
    count_month_sev['incident_month'] = pd.Categorical(count_month_sev['incident_month'], categories=month_order, ordered=True)
    fig1 = px.bar(
        count_month_sev.sort_values('incident_month'),
        x="incident_month",
        y="count",
        color="severity",
        color_discrete_map=severity_colors,
        barmode="stack",
        category_orders={"incident_month": month_order},
        labels={"incident_month": "Month", "count": "Incidents", "severity": "Severity"},
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Location based incident analysis
    st.subheader("Incidents by Location")
    location_counts = filtered_df["location"].value_counts().reset_index()
    location_counts.columns = ["Location", "Count"]
    fig2 = px.bar(location_counts, x="Location", y="Count", color="Count",
                  color_continuous_scale="Blues", title="Incidents by Location")
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Total incident by severity (with custom color order)
    st.subheader("Total Incidents by Severity")
    sev_counts = filtered_df["severity"].value_counts().reindex(["Critical", "High", "Medium", "Low"]).fillna(0).reset_index()
    sev_counts.columns = ["Severity", "Count"]
    fig3 = px.bar(
        sev_counts, x="Severity", y="Count", color="Severity", 
        color_discrete_map=severity_colors,
        title="Total Incidents by Severity"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Top 5 incident type by volume
    st.subheader("Top 5 Incident Types by Volume")
    top_types = filtered_df["incident_type"].value_counts().head(5).reset_index()
    top_types.columns = ["Incident Type", "Count"]
    fig4 = px.bar(top_types, x="Incident Type", y="Count", color="Count",
                  color_continuous_scale="Viridis", title="Top 5 Incident Types by Volume")
    st.plotly_chart(fig4, use_container_width=True)

    # 5. Medical attention rate (gauge chart)
    st.subheader("Medical Attention Rate")
    if "medical_attention_required" in filtered_df.columns:
        total = len(filtered_df)
        med = (filtered_df["medical_attention_required"].str.lower() == "yes").sum()
        med_rate = med / total * 100 if total else 0
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=med_rate,
            number={"suffix": "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Medical Attention Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2F9E7D"},
                'steps': [
                    {'range': [0, 50], 'color': "#F7F9FA"},
                    {'range': [50, 80], 'color': "#F59C2F"},
                    {'range': [80, 100], 'color': "#DC2626"}
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig5, use_container_width=True)

    # 6. Critical incident rate (gauge)
    st.subheader("Critical Incident Rate")
    if "severity" in filtered_df.columns:
        crit = (filtered_df["severity"] == "Critical").sum()
        crit_rate = crit / len(filtered_df) * 100 if len(filtered_df) else 0
        fig6 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=crit_rate,
            number={"suffix": "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Critical Incident Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#DC2626"},
                'steps': [
                    {'range': [0, 10], 'color': "#F7F9FA"},
                    {'range': [10, 20], 'color': "#F59C2F"},
                    {'range': [20, 100], 'color': "#DC2626"}
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            }
        ))
        st.plotly_chart(fig6, use_container_width=True)

    # 7. Same day reporting rate (gauge)
    st.subheader("Same Day Reporting Rate")
    if "same_day_reporting" in filtered_df.columns:
        same_day = filtered_df["same_day_reporting"].sum()
        same_day_rate = same_day / len(filtered_df) * 100 if len(filtered_df) else 0
        fig7 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=same_day_rate,
            number={"suffix": "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Same Day Reporting Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2F9E7D"},
                'steps': [
                    {'range': [0, 60], 'color': "#DC2626"},
                    {'range': [60, 80], 'color': "#F59C2F"},
                    {'range': [80, 100], 'color': "#2F9E7D"}
                ],
                'threshold': {
                    'line': {'color': "#2F9E7D", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        st.plotly_chart(fig7, use_container_width=True)

    st.write("Preview of filtered data:", filtered_df.head())


def render_operational_performance(filtered_df):
    st.header("Operational Performance")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    if "location" in filtered_df.columns:
        location_counts = filtered_df["location"].value_counts().reset_index()
        location_counts.columns = ["Location", "Count"]
        fig = px.bar(location_counts, x="Location", y="Count", title="Incidents by Location")
        st.plotly_chart(fig, use_container_width=True)
    st.write(filtered_df.head())

def render_compliance_investigation(filtered_df):
    st.header("Compliance & Investigation")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    if "reportable" in filtered_df.columns:
        fig = px.pie(filtered_df, names="reportable", title="Reportable Incidents")
        st.plotly_chart(fig, use_container_width=True)
    st.write(filtered_df.head())


def render_risk_analysis(filtered_df):
    st.header("Risk Analysis")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    if "severity" in filtered_df.columns and "location" in filtered_df.columns:
        crits = filtered_df[filtered_df["severity"] == "Critical"]
        location_counts = crits["location"].value_counts().reset_index()
        location_counts.columns = ["Location", "Critical Count"]
        fig = px.bar(location_counts, x="Location", y="Critical Count", title="Critical Incidents by Location")
        st.plotly_chart(fig, use_container_width=True)
    st.write(filtered_df.head())


PAGE_TO_RENDERER = {
    "Executive Summary": render_executive_summary,
    "Operational Performance": render_operational_performance,
    "Compliance & Investigation": render_compliance_investigation,
    "🤖 Machine Learning Analytics": render_ml_analytics,
    "Risk Analysis": render_risk_analysis,
}
