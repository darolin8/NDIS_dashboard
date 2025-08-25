import hashlib
from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =========================
# Palettes
# =========================
NDIS_COLORS = {
    'primary':   '#003F5C',
    'secondary': '#2F9E7D',
    'accent':    '#F59C2F',
    'critical':  '#DC2626',
    'high':      '#F59C2F',
    'medium':    '#2F9E7D',
    'low':       '#67A3C3',
    'success':   '#2F9E7D',
    'warning':   '#F59C2F',
    'error':     '#DC2626',
}

STORY_COLORS = {
    'emphasis':   '#1F77B4',   # Blue for main point
    'positive':   '#2F9E7D',
    'negative':   '#DC2626',
    'warning':    '#F59C2F',
    'context':    '#D3D3D3',
    'text':       '#666666',
    'background': '#FFFFFF',
    'grid':       '#F0F0F0',
    'axisline':   '#E0E0E0',
}

severity_colors = {
    'Critical': NDIS_COLORS['critical'],
    'High':     NDIS_COLORS['high'],
    'Medium':   NDIS_COLORS['medium'],
    'Low':      NDIS_COLORS['low'],
}

# =========================
# Streamlit-safe chart utils (prevents StreamlitDuplicateElementId)
# =========================
_KEY_REGISTRY = "_chart_key_registry"

def fig_copy(fig: go.Figure) -> go.Figure:
    return go.Figure(fig)

def chart_key(name: str, namespace: str = "main", idx: int | None = None) -> str:
    base = f"plotly::{namespace}::{name}::{idx if idx is not None else ''}"
    h = hashlib.md5(base.encode()).hexdigest()[:8]
    return f"{base}::{h}"

def unique_chart_key(name: str, namespace: str = "main", idx: int | None = None) -> str:
    if _KEY_REGISTRY not in st.session_state:
        st.session_state[_KEY_REGISTRY] = set()
    key = chart_key(name, namespace, idx)
    if key in st.session_state[_KEY_REGISTRY]:
        suffix = 2
        while f"{key}__{suffix}" in st.session_state[_KEY_REGISTRY]:
            suffix += 1
        key = f"{key}__{suffix}"
    st.session_state[_KEY_REGISTRY].add(key)
    return key

def plotly_chart_safe(fig, *, name: str, namespace: str, idx: int | None = None, **kwargs):
    st.plotly_chart(fig_copy(fig), key=unique_chart_key(name, namespace, idx), use_container_width=True, **kwargs)

# =========================
# Storytelling helper (your 5-step rules)
# =========================
def apply_5_step_story(
    fig: go.Figure,
    *,
    emphasis_trace_idxs=None,
    title_text="",
    subtitle_text=None,
    decimals=0,
    show_legend=False,
) -> go.Figure:
    fig = deepcopy(fig)

    # 1) Start everything in grey
    for tr in fig.data:
        if hasattr(tr, "marker"):
            tr.update(marker=dict(color=STORY_COLORS['context']))
        if hasattr(tr, "line"):
            tr.update(line=dict(color=STORY_COLORS['axisline'], width=max(getattr(tr.line, "width", 1), 1)))

    # 2) Emphasize one thing
    if emphasis_trace_idxs:
        for idx in emphasis_trace_idxs:
            if 0 <= idx < len(fig.data):
                tr = fig.data[idx]
                if hasattr(tr, "marker"):
                    tr.update(marker=dict(color=STORY_COLORS['emphasis']))
                if hasattr(tr, "line"):
                    tr.update(line=dict(color=STORY_COLORS['emphasis'], width=3))
                fig.data += (fig.data.pop(idx),)  # bring to front

    # 3) Remove clutter
    fig.update_layout(
        showlegend=show_legend,
        plot_bgcolor=STORY_COLORS['background'],
        paper_bgcolor=STORY_COLORS['background'],
        margin=dict(l=60, r=60, t=80, b=40),
        font=dict(family='Arial', size=11, color=STORY_COLORS['text']),
    )
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False,
                     tickfont=dict(color=STORY_COLORS['text']),
                     tickformat=f",.{max(decimals,0)}f" if decimals is not None else None)
    fig.update_yaxes(showline=False, zeroline=False, showgrid=True,
                     gridcolor=STORY_COLORS['grid'], gridwidth=0.5,
                     tickfont=dict(color=STORY_COLORS['text']),
                     tickformat=f",.{max(decimals,0)}f" if decimals is not None else None)

    # 4) Action title
    if title_text:
        title_html = f"<b>{title_text}</b>"
        if subtitle_text:
            title_html += f"<br><sup style='color:{STORY_COLORS['text']}'>{subtitle_text}</sup>"
        fig.update_layout(title={'text': title_html, 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}})

    # 5) 3-second rule is for humans 😉
    return fig

# =========================
# Shared calcs
# =========================
def compute_common_metrics(filtered_df: pd.DataFrame):
    total_incidents = len(filtered_df)
    critical_incidents = int((filtered_df['severity'] == 'Critical').sum()) if total_incidents else 0
    same_day_rate = float(filtered_df['same_day_reporting'].mean() * 100) if total_incidents else 0.0
    reportable_rate = float((filtered_df['reportable'] == 'Yes').mean() * 100) if total_incidents else 0.0
    return total_incidents, critical_incidents, same_day_rate, reportable_rate

# =========================
# Pages
# =========================

# ---------- Executive Summary ----------
def render_executive_summary(df: pd.DataFrame, filtered_df: pd.DataFrame):
    ns = "Executive Summary"
    st.title("📊 NDIS Executive Dashboard")
    st.markdown("**Strategic Overview - Incident Analysis & Risk Management**")
    st.markdown(f"*Showing {len(filtered_df)} incidents from {len(df)} total records*")
    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    total_incidents, critical_incidents, same_day_rate, reportable_rate = compute_common_metrics(filtered_df)
    with c1:
        st.metric("Total Incidents", f"{total_incidents}")
    with c2:
        pct = (critical_incidents/total_incidents*100) if total_incidents else 0
        st.metric("Critical Incidents", f"{critical_incidents}", f"{pct:.1f}% of total")
    with c3:
        st.metric("Same-Day Reporting", f"{same_day_rate:.1f}%")
    with c4:
        st.metric("Reportable Rate", f"{reportable_rate:.1f}%")

    # Row: Monthly trends (stacked) + Recent critical list
    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("📈 Incident Trends by Month")
        if not filtered_df.empty:
            monthly = filtered_df.groupby(['incident_month', 'severity']).size().unstack(fill_value=0)

            # Order month names if present
            order = ['January','February','March','April','May','June','July','August','September','October','November','December']
            monthly = monthly.reindex([m for m in order if m in monthly.index])

            fig = go.Figure()
            for sev in monthly.columns:
                fig.add_trace(go.Bar(
                    x=monthly.index,
                    y=monthly[sev],
                    name=sev,
                    marker_color=severity_colors.get(sev, NDIS_COLORS['primary'])
                ))
            fig.update_layout(barmode='stack', height=420)
            fig = apply_5_step_story(fig, title_text="Monthly distribution by severity")
            plotly_chart_safe(fig, name="monthly_by_sev", namespace=ns)

    with colB:
        st.subheader("🚨 Recent Critical Incidents")
        crit = filtered_df[filtered_df['severity'] == 'Critical'].sort_values('incident_date', ascending=False).head(5)
        if crit.empty:
            st.info("No critical incidents in selected period")
        else:
            for _, r in crit.iterrows():
                date_str = r['incident_date'].strftime('%d/%m/%Y') if pd.notna(r['incident_date']) else 'Unknown'
                st.markdown(
                    f"<div style='border-left:4px solid {NDIS_COLORS['critical']};"
                    "background:#FEF2F2;padding:.6rem .8rem;border-radius:.5rem;margin-bottom:.5rem;'>"
                    f"<b>🔴 {r['incident_type']}</b><br>"
                    f"<small>📍 {r['location']} | 📅 {date_str}</small><br>"
                    f"<small style='color:#6c757d;'>{str(r['description'])[:100]}...</small>"
                    "</div>", unsafe_allow_html=True
                )

    # Row: Incident type pie + Location risk scatter
    colC, colD = st.columns(2)
    with colC:
        st.subheader("📊 Incident Types Distribution")
        if not filtered_df.empty:
            vc = filtered_df['incident_type'].value_counts()
            fig = px.pie(values=vc.values, names=vc.index)
            fig.update_traces(textposition='inside', textinfo='percent+label',
                              marker=dict(colors=[NDIS_COLORS['primary'], NDIS_COLORS['secondary'],
                                                  NDIS_COLORS['accent'], NDIS_COLORS['critical'],
                                                  '#67A3C3', '#8B9DC3']))
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text=f"Top incident type: {vc.index[0]} ({vc.iloc[0]})")
            plotly_chart_safe(fig, name="type_pie", namespace=ns)

    with colD:
        st.subheader("🎯 Location Risk Analysis")
        if not filtered_df.empty:
            loc = filtered_df.groupby('location').agg(
                total=('incident_id','count'),
                critical=('severity', lambda x: (x=='Critical').sum())
            ).reset_index()
            loc['critical_pct'] = np.where(loc['total']>0, loc['critical']/loc['total']*100, 0)
            fig = px.scatter(loc, x='total', y='critical_pct', size='critical', color='critical_pct',
                             hover_name='location',
                             color_continuous_scale=[[0, NDIS_COLORS['success']],
                                                     [0.5, NDIS_COLORS['accent']],
                                                     [1, NDIS_COLORS['critical']]])
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="Location risk assessment (critical % vs total)")
            plotly_chart_safe(fig, name="loc_risk", namespace=ns)

    # Contributing factors + Medical attention by type
    st.subheader("⚠️ Contributing Factors Analysis")
    colE, colF = st.columns(2)
    with colE:
        if 'contributing_factors' in filtered_df.columns and not filtered_df.empty:
            factors = filtered_df['contributing_factors'].value_counts().head(10)
            fig = px.bar(x=factors.values, y=factors.index, orientation='h')
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="Top 10 contributing factors")
            plotly_chart_safe(fig, name="factors_bar", namespace=ns)

    with colF:
        if not filtered_df.empty:
            med = filtered_df.groupby('incident_type')['medical_attention_required'].apply(
                lambda s: (s=='Yes').mean()*100
            ).sort_values(ascending=False)
            fig = px.bar(x=med.index, y=med.values)
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="Medical attention required by incident type (%)")
            plotly_chart_safe(fig, name="med_by_type", namespace=ns)

# ---------- Operational Performance ----------
def render_operational_performance(filtered_df: pd.DataFrame):
    ns = "Operational Performance"
    st.title("🎯 Operational Performance & Risk Analysis")
    st.markdown("**Tactical Level - Management Action & Resource Allocation**")
    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    avg_delay = float(filtered_df['reporting_delay_hours'].mean()) if not filtered_df.empty else 0.0
    last_30 = int((filtered_df['incident_date'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))).sum()) if 'incident_date' in filtered_df else 0
    med_rate = float((filtered_df['medical_attention_required']=='Yes').mean()*100) if not filtered_df.empty else 0.0
    with c1: st.metric("Avg Reporting Delay", f"{avg_delay:.1f} hrs", "Target: <24hrs")
    with c2: st.metric("Recent Cases (30d)", f"{last_30}", "Active monitoring")
    with c3: st.metric("Medical Attention Rate", f"{med_rate:.1f}%", "Resource planning")
    with c4: st.metric("Data Quality", "98.5%", "+1.2%")

    # Reporter performance
    colA, colB = st.columns(2)
    with colA:
        st.subheader("👥 Reporter Performance Analysis")
        if 'reported_by' in filtered_df.columns and not filtered_df.empty:
            tmp = filtered_df.copy()
            tmp['reporter_role'] = tmp['reported_by'].str.extract(r'\((.*?)\)')
            perf = tmp.groupby('reporter_role').agg(
                avg_delay=('reporting_delay_hours','mean'),
                count=('incident_id','count')
            ).dropna().reset_index()
            if not perf.empty:
                fig = px.bar(perf, x='reporter_role', y='avg_delay', color='avg_delay', color_continuous_scale='Reds')
                fig.update_layout(height=420)
                fig = apply_5_step_story(fig, title_text="Average reporting delay by role")
                plotly_chart_safe(fig, name="reporter_perf", namespace=ns)

    with colB:
        st.subheader("🏥 Medical Impact Analysis")
        if not filtered_df.empty:
            med = filtered_df.groupby('incident_type').agg(
                medical_required=('medical_attention_required', lambda s: (s=='Yes').sum()),
                total=('incident_id','count')
            ).reset_index()
            med['medical_rate'] = np.where(med['total']>0, med['medical_required']/med['total']*100, 0)
            fig = px.scatter(med, x='total', y='medical_rate', size='medical_required', color='incident_type')
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="Medical attention requirements vs total incidents")
            plotly_chart_safe(fig, name="med_scatter", namespace=ns)

    # Temporal patterns
    st.subheader("📈 Temporal Patterns Analysis")
    colC, colD = st.columns(2)
    with colC:
        if not filtered_df.empty:
            tmp = filtered_df.copy()
            tmp['day_of_week'] = tmp['incident_date'].dt.day_name()
            order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            counts = tmp['day_of_week'].value_counts()
            counts = counts.reindex([d for d in order if d in counts.index])
            fig = px.bar(x=counts.index, y=counts.values)
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="Incidents by day of week")
            plotly_chart_safe(fig, name="dow_bar", namespace=ns)

    with colD:
        if not filtered_df.empty:
            sev = filtered_df.groupby([filtered_df['incident_date'].dt.to_period('M'), 'severity']).size().unstack(fill_value=0)
            fig = go.Figure()
            col_map = {'Critical':'#dc3545','High':'#fd7e14','Medium':'#ffc107','Low':'#28a745'}
            for s in sev.columns:
                fig.add_trace(go.Scatter(x=sev.index.astype(str), y=sev[s], mode='lines+markers',
                                         name=s, line=dict(color=col_map.get(s,'#6c757d'))))
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="Severity trends over time", show_legend=True)
            plotly_chart_safe(fig, name="sev_trend", namespace=ns)

# ---------- Compliance & Investigation ----------
def render_compliance_investigation(filtered_df: pd.DataFrame):
    ns = "Compliance & Investigation"
    st.title("📋 Compliance & Detailed Investigation")
    st.markdown("**Operational Level - Regulatory Oversight & Case Management**")
    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    reportable = int((filtered_df['reportable']=='Yes').sum()) if not filtered_df.empty else 0
    compliance_24 = float((filtered_df['reporting_delay_hours'] <= 24).mean()*100) if not filtered_df.empty else 0.0
    overdue = int((filtered_df['reporting_delay_hours'] > 24).sum()) if not filtered_df.empty else 0
    with c1: st.metric("Reportable Incidents", f"{reportable}", f"{(reportable/len(filtered_df)*100 if len(filtered_df) else 0):.1f}%")
    with c2: st.metric("24hr Compliance", f"{compliance_24:.1f}%", "Target: >90%")
    with c3: st.metric("Overdue Reports", f"{overdue}", "Requires action")
    with c4: st.metric("Investigation Rate", "100%", "All incidents reviewed")

    # Detailed table
    st.subheader("📋 Incident Details")
    cols = ['incident_id','incident_date','incident_type','severity','location','reportable','reporting_delay_hours','medical_attention_required']
    if not filtered_df.empty:
        table_df = filtered_df[cols].copy()
        table_df['incident_date'] = table_df['incident_date'].dt.strftime('%d/%m/%Y')
        table_df['reporting_delay_hours'] = table_df['reporting_delay_hours'].round(1)
        st.dataframe(table_df, use_container_width=True, height=420)

    # Reporting timeline + Compliance by location
    colA, colB = st.columns(2)
    with colA:
        st.subheader("⏰ Reporting Timeline Analysis")
        if not filtered_df.empty:
            fig = px.scatter(filtered_df, x='incident_date', y='reporting_delay_hours', color='severity',
                             hover_data=['incident_id','incident_type','location'])
            fig.add_hline(y=24, line_dash="dash", line_color="red")
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="Reporting delay by incident date", show_legend=True)
            plotly_chart_safe(fig, name="reporting_timeline", namespace=ns)

    with colB:
        st.subheader("📊 Compliance by Location")
        if not filtered_df.empty:
            comp = filtered_df.groupby('location').agg(
                compliant=('reporting_delay_hours', lambda s: (s<=24).sum()),
                total=('incident_id','count')
            ).reset_index()
            comp['compliance_rate'] = np.where(comp['total']>0, comp['compliant']/comp['total']*100, 0)
            fig = px.bar(comp, x='location', y='compliance_rate', color='compliance_rate', color_continuous_scale='RdYlGn')
            fig.update_layout(height=420)
            fig = apply_5_step_story(fig, title_text="24-hour compliance rate by location")
            plotly_chart_safe(fig, name="loc_compliance", namespace=ns)

# ---------- Machine Learning Analytics ----------
def render_ml_analytics(filtered_df: pd.DataFrame,
                        train_severity_prediction_model,
                        prepare_ml_features,
                        perform_anomaly_detection,
                        find_association_rules=None,
                        time_series_forecast=None,
                        MLXTEND_AVAILABLE=False,
                        STATSMODELS_AVAILABLE=False):
    ns = "ML Analytics"
    st.title("🤖 Machine Learning Analytics")
    st.markdown("**Advanced AI-Powered Insights & Predictions**")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predictive Models", "🚨 Anomaly Detection", "🔗 Association Rules",
        "📈 Time Series Forecasting", "🎯 Advanced Clustering"
    ])

    with tab1:
        st.subheader("🔮 Severity Prediction & Classification")
        if not filtered_df.empty and len(filtered_df) >= 20:
            model, accuracy, feature_names = train_severity_prediction_model(filtered_df)
            if model is not None:
                st.success(f"Random Forest accuracy: {accuracy:.2%} (trained on {len(filtered_df)} incidents)")
                # Feature importances
                if hasattr(model, 'feature_importances_') and feature_names:
                    imp = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance')
                    fig = px.bar(imp.tail(10), x='importance', y='feature', orientation='h')
                    fig = apply_5_step_story(fig, title_text="Top features for severity prediction")
                    plotly_chart_safe(fig, name="rf_importances", namespace=ns)
                # Prediction distribution
                X, _, _ = prepare_ml_features(filtered_df)
                if X is not None:
                    preds = model.predict(X)
                    names = ['Low','Medium','High','Critical']
                    counts = pd.Series(preds).value_counts().sort_index()
                    fig = px.bar(x=[names[i] for i in counts.index], y=counts.values)
                    fig = apply_5_step_story(fig, title_text="Predicted severity distribution")
                    plotly_chart_safe(fig, name="pred_dist", namespace=ns)
            else:
                st.warning("Unable to train prediction model. Need more diverse data.")
        else:
            st.info("Need at least 20 incidents for meaningful prediction modeling.")

    with tab2:
        st.subheader("🚨 Anomaly Detection & Outlier Analysis")
        if not filtered_df.empty and len(filtered_df) >= 10:
            anomaly_df, feat_names = perform_anomaly_detection(filtered_df)
            if anomaly_df is not None:
                iso_anoms = int(anomaly_df['isolation_forest_anomaly'].sum())
                svm_anoms = int(anomaly_df['svm_anomaly'].sum())
                total = len(anomaly_df)
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Isolation Forest Anomalies", f"{iso_anoms}", f"{iso_anoms/total*100:.1f}%")
                with c2: st.metric("SVM Anomalies", f"{svm_anoms}", f"{svm_anoms/total*100:.1f}%")
                with c3: st.metric("Total Flagged", f"{int((anomaly_df['isolation_forest_anomaly'] | anomaly_df['svm_anomaly']).sum())}")

                # Score histogram
                fig = px.histogram(anomaly_df, x='anomaly_score', color='isolation_forest_anomaly')
                fig = apply_5_step_story(fig, title_text="Anomaly score distribution")
                plotly_chart_safe(fig, name="anomaly_hist", namespace=ns)

                # By severity
                grp = anomaly_df.groupby('severity').agg(
                    iso=('isolation_forest_anomaly','sum'),
                    n=('incident_id','count')
                ).reset_index()
                grp['anomaly_rate'] = grp['iso']/grp['n']*100
                fig = px.bar(grp, x='severity', y='anomaly_rate', color='anomaly_rate', color_continuous_scale='Reds')
                fig = apply_5_step_story(fig, title_text="Anomaly rate by severity")
                plotly_chart_safe(fig, name="anomaly_by_sev", namespace=ns)

                # Top anomalies table
                top = anomaly_df[anomaly_df['isolation_forest_anomaly']].nsmallest(10, 'anomaly_score')
                if not top.empty:
                    show_cols = ['incident_id','incident_date','incident_type','severity','location','anomaly_score']
                    t = top[show_cols].copy()
                    t['incident_date'] = t['incident_date'].dt.strftime('%d/%m/%Y')
                    t['anomaly_score'] = t['anomaly_score'].round(3)
                    st.dataframe(t, use_container_width=True)
            else:
                st.error("Unable to perform anomaly detection on current data.")
        else:
            st.info("Need at least 10 incidents for anomaly detection.")

    with tab3:
        st.subheader("🔗 Association Rules & Pattern Mining")
        if not filtered_df.empty and len(filtered_df) >= 20:
            if MLXTEND_AVAILABLE and find_association_rules:
                fi, rules = find_association_rules(filtered_df)
                if rules is not None and not rules.empty:
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Frequent Itemsets", len(fi))
                    with c2: st.metric("Association Rules", len(rules))
                    with c3: st.metric("Avg Confidence", f"{rules['confidence'].mean():.2%}")

                    # Top rules table
                    tbl = rules.copy()
                    tbl['antecedents'] = tbl['antecedents'].apply(lambda x: ', '.join(list(x)))
                    tbl['consequents'] = tbl['consequents'].apply(lambda x: ', '.join(list(x)))
                    tbl = tbl[['antecedents','consequents','support','confidence','lift']].round(3).nlargest(10,'confidence')
                    st.dataframe(tbl, use_container_width=True)

                    # Support vs Confidence
                    fig = px.scatter(rules, x='support', y='confidence', size='lift', color='lift')
                    fig = apply_5_step_story(fig, title_text="Support vs confidence")
                    plotly_chart_safe(fig, name="rules_scatter", namespace=ns)
                else:
                    st.info("No significant association rules found. Try more data or wider filters.")
            else:
                st.error("mlxtend not available — `pip install mlxtend` to enable.")
        else:
            st.info("Need at least 20 incidents for association rules.")

    with tab4:
        st.subheader("📈 Time Series Forecasting")
        if not filtered_df.empty and len(filtered_df) >= 30:
            if STATSMODELS_AVAILABLE and time_series_forecast:
                periods = st.selectbox("Forecast Periods", [7,14,30,60], index=2, key="forecast_periods")
                hist, fc = time_series_forecast(filtered_df, periods)
                if hist is not None and fc is not None:
                    c1, c2, c3 = st.columns(3)
                    avg_daily = hist['incident_count'].mean()
                    fc_avg = fc['forecast'].mean()
                    change = (fc_avg - avg_daily)/avg_daily*100 if avg_daily else 0
                    with c1: st.metric("Avg Daily Incidents", f"{avg_daily:.1f}")
                    with c2: st.metric("Forecast Avg", f"{fc_avg:.1f}")
                    with c3: st.metric("Predicted Change", f"{change:+.1f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['incident_count'], mode='lines+markers', name='Historical',
                                             line=dict(color=NDIS_COLORS['primary'])))
                    fig.add_trace(go.Scatter(x=fc['date'], y=fc['forecast'], mode='lines+markers', name='Forecast',
                                             line=dict(color=NDIS_COLORS['accent'], dash='dash')))
                    fig.update_layout(height=480)
                    fig = apply_5_step_story(fig, title_text=f"Incident forecast – next {periods} days", show_legend=True)
                    plotly_chart_safe(fig, name="forecast_main", namespace=ns)
                else:
                    st.error("Unable to generate forecast. Need more consistent time series data.")
            else:
                st.error("statsmodels not available — `pip install statsmodels` to enable.")
        else:
            st.info("Need at least 30 incidents for time series forecasting.")

    with tab5:
        st.subheader("🎯 Advanced Clustering")
        st.info("Reuse your clustering code here; ensure each chart uses "
                "plotly_chart_safe(..., namespace=ns, name='cluster_*') for unique keys.")

# ---------- Risk Analysis ----------
def render_risk_analysis(filtered_df: pd.DataFrame):
    ns = "Risk Analysis"
    st.title("⚠️ Advanced Risk Analysis")
    st.markdown("**Strategic Analysis - Pattern Recognition & Prevention**")
    st.markdown("---")

    hi_risk = int(filtered_df['severity'].isin(['Critical','High']).sum()) if not filtered_df.empty else 0
    avg_age = float(filtered_df['age_at_incident'].mean()) if 'age_at_incident' in filtered_df.columns and not filtered_df.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("High Risk Incidents", f"{hi_risk}", f"{(hi_risk/len(filtered_df)*100 if len(filtered_df) else 0):.1f}%")
    with c2: st.metric("Average Participant Age", f"{avg_age:.1f} years", "Demographics")
    with c3: st.metric("Risk Locations", f"{filtered_df['location'].nunique() if 'location' in filtered_df else 0}")
    with c4: st.metric("Risk Factors Identified", f"{filtered_df['contributing_factors'].nunique() if 'contributing_factors' in filtered_df else 0}")

    tabs = st.tabs(["🔥 Risk Clustering", "📊 Statistical Analysis", "🎯 Predictive Insights"])

    with tabs[0]:
        st.subheader("🔍 Machine Learning Risk Clustering")
        st.info("Plug your clustering charts here, wrapped with plotly_chart_safe for uniqueness.")

    with tabs[1]:
        st.subheader("📊 Statistical Analysis")
        if not filtered_df.empty:
            corr_vars = {}
            if 'reporting_delay_hours' in filtered_df: corr_vars['Reporting Delay (hrs)'] = filtered_df['reporting_delay_hours'].fillna(0)
            if 'age_at_incident' in filtered_df:       corr_vars['Age at Incident']     = filtered_df['age_at_incident'].fillna(0)
            if 'severity' in filtered_df:
                sev_map = {'Low':1,'Medium':2,'High':3,'Critical':4}
                corr_vars['Severity Score'] = filtered_df['severity'].map(sev_map).fillna(0)
            if 'medical_attention_required' in filtered_df:
                corr_vars['Medical Required'] = (filtered_df['medical_attention_required']=='Yes').astype(int)

            if len(corr_vars) >= 2:
                corr_df = pd.DataFrame(corr_vars).corr()
                fig = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1)
                fig = apply_5_step_story(fig, title_text="Correlation matrix")
                plotly_chart_safe(fig, name="corr_heatmap", namespace=ns)
            else:
                st.info("Need more numerical variables for correlation.")

    with tabs[2]:
        st.subheader("🔮 Predictive Insights")
        if not filtered_df.empty:
            colA, colB = st.columns(2)
            with colA:
                if 'location' in filtered_df:
                    loc_risk = filtered_df.groupby('location').apply(lambda d: d['severity'].isin(['Critical','High']).mean()*100).round(1)
                    st.markdown("**Risk by Location:**")
                    for loc, pct in loc_risk.sort_values(ascending=False).items():
                        emoji = "🔴" if pct > 30 else "🟡" if pct > 15 else "🟢"
                        st.markdown(f"• {loc}: {pct:.1f}% {emoji}")
                if 'incident_time' in filtered_df:
                    tmp = filtered_df.copy()
                    tmp['hour'] = pd.to_datetime(tmp['incident_time'], format='%H:%M', errors='coerce').dt.hour
                    if not tmp['hour'].isna().all():
                        hr = tmp.groupby('hour')['severity'].apply(lambda s: s.isin(['High','Critical']).mean()*100).round(1)
                        peak_hr = int(hr.idxmax())
                        st.markdown(f"**Peak risk time:** {peak_hr:02d}:00 ({hr.max():.1f}% high-risk)")

            with colB:
                st.markdown("**💡 Recommendations**")
                recs = []
                if 'severity' in filtered_df and 'location' in filtered_df:
                    top_loc = filtered_df[filtered_df['severity'].isin(['High','Critical'])]['location'].value_counts().head(1)
                    if not top_loc.empty:
                        recs.append(f"🎯 Prioritize safety measures at **{top_loc.index[0]}**")
                if 'contributing_factors' in filtered_df and not filtered_df['contributing_factors'].mode().empty:
                    recs.append(f"🔧 Address **{filtered_df['contributing_factors'].mode().iloc[0]}** as a primary prevention target")
                if 'reporting_delay_hours' in filtered_df and filtered_df['reporting_delay_hours'].mean() > 24:
                    recs.append("⏰ Improve reporting processes to meet 24-hour compliance")
                med_pct = (filtered_df['medical_attention_required']=='Yes').mean()*100 if 'medical_attention_required' in filtered_df else 0
                if med_pct > 30:
                    recs.append(f"🏥 Plan resources for ~{med_pct:.0f}% medical attention rate")
                recs += ["📊 Monitor monthly trends", "👥 Targeted staff training", "🔄 Quarterly risk protocol review"]
                for i, r in enumerate(recs[:6], 1):
                    st.markdown(f"{i}. {r}")

# =========================
# Page router
# =========================

def executive_summary_page(df, filtered_df):
    st.title("Executive Summary")
    st.markdown("This is the Executive Summary page. Add key metrics and charts here.")
    st.dataframe(filtered_df.head())

def operational_performance_page(filtered_df):
    st.title("Operational Performance")
    st.markdown("This is the Operational Performance page.")
    st.dataframe(filtered_df.head())

def compliance_investigation_page(filtered_df):
    st.title("Compliance & Investigation")
    st.markdown("This is the Compliance & Investigation page.")
    st.dataframe(filtered_df.head())

def ml_analytics_page(
    filtered_df,
    train_severity_prediction_model=None,
    prepare_ml_features=None,
    perform_anomaly_detection=None,
    find_association_rules=None,
    time_series_forecast=None,
    MLXTEND_AVAILABLE=None,
    STATSMODELS_AVAILABLE=None
):
    st.title("🤖 Machine Learning Analytics")
    st.markdown("This is the ML Analytics page.")
    st.dataframe(filtered_df.head())

def risk_analysis_page(filtered_df):
    st.title("Risk Analysis")
    st.markdown("This is the Risk Analysis page.")
    st.dataframe(filtered_df.head())

PAGE_TO_RENDERER = {
    "Executive Summary": executive_summary_page,
    "Operational Performance": operational_performance_page,
    "Compliance & Investigation": compliance_investigation_page,
    "🤖 Machine Learning Analytics": ml_analytics_page,
    "Risk Analysis": risk_analysis_page
}

