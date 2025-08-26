import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import hashlib
from copy import deepcopy

# =========================
# Storytelling Colors & Chart Function
# =========================
STORY_COLORS = {
    'context': '#D3D3D3',
    'emphasis': '#1F77B4',
    'negative': '#DC2626',
    'grid': '#F3F3F3',
    'axisline': '#CCCCCC',
    'background': '#FFFFFF',
    'text': '#333333'
}

NDIS_COLORS = {
    'critical': '#DC2626',
    'high': '#fd7e14',
    'medium': '#ffc107',
    'low': '#28a745',
    'primary': '#1F77B4',
    'secondary': '#636EFA',
    'accent': '#00B4D8',
    'success': '#43AA8B'
}
severity_colors = {
    'Critical': NDIS_COLORS['critical'],
    'High': NDIS_COLORS['high'],
    'Medium': NDIS_COLORS['medium'],
    'Low': NDIS_COLORS['low']
}

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

def _numeric_width(w):
    import numpy as _np
    if isinstance(w, (list, tuple, _np.ndarray)):
        nums = [v for v in w if isinstance(v, (int, float))]
        return max(nums) if nums else 1
    if isinstance(w, (int, float)):
        try:
            if _np.isnan(w):
                return 1
        except Exception:
            pass
        return max(w, 1)
    return 1

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
    axis_color = STORY_COLORS.get('axisline', '#E0E0E0')
    for tr in fig.data:
        ttype = getattr(tr, 'type', None)
        marker_line_w = 1
        try:
            mlw = getattr(getattr(tr, "marker", None), "line", None)
            marker_line_w = _numeric_width(getattr(mlw, "width", None))
        except Exception:
            pass
        if hasattr(tr, "marker"):
            if ttype == "pie":
                n = 0
                try:
                    n = len(getattr(tr, "labels", []) or [])
                except Exception:
                    pass
                if not n:
                    try:
                        n = len(getattr(tr, "values", []) or [])
                    except Exception:
                        n = 1
                tr.update(marker=dict(
                    colors=[STORY_COLORS['context']] * n,
                    line=dict(color=axis_color, width=marker_line_w),
                ))
            elif ttype not in ["sunburst", "treemap", "funnelarea"]:
                tr.update(marker=dict(
                    color=STORY_COLORS['context'],
                    line=dict(color=axis_color, width=marker_line_w),
                ))
        if hasattr(tr, "line") and getattr(tr, "line", None) is not None:
            w = _numeric_width(getattr(tr.line, "width", None))
            tr.update(line=dict(color=axis_color, width=w))
    if emphasis_trace_idxs:
        for idx in list(emphasis_trace_idxs):
            if 0 <= idx < len(fig.data):
                tr = fig.data[idx]
                ttype = getattr(tr, 'type', None)
                if hasattr(tr, "marker"):
                    if ttype == "pie":
                        n = 0
                        try:
                            n = len(getattr(tr, "labels", []) or [])
                        except Exception:
                            pass
                        if not n:
                            try:
                                n = len(getattr(tr, "values", []) or [])
                            except Exception:
                                n = 1
                        tr.update(marker=dict(
                            colors=[STORY_COLORS['emphasis']] * n,
                            line=dict(color=STORY_COLORS['emphasis'], width=3),
                        ))
                    else:
                        tr.update(marker=dict(
                            color=STORY_COLORS['emphasis'],
                            line=dict(color=STORY_COLORS['emphasis'], width=3),
                        ))
                if hasattr(tr, "line") and getattr(tr, "line", None) is not None:
                    tr.update(line=dict(color=STORY_COLORS['emphasis'], width=3))
                nd = list(fig.data)
                nd.append(nd.pop(idx))
                fig.data = tuple(nd)
    fig.update_layout(
        showlegend=show_legend,
        plot_bgcolor=STORY_COLORS['background'],
        paper_bgcolor=STORY_COLORS['background'],
        margin=dict(l=60, r=60, t=80, b=40),
        font=dict(family='Arial', size=11, color=STORY_COLORS['text']),
    )
    fig.update_xaxes(
        showline=False, zeroline=False, showgrid=False,
        tickfont=dict(color=STORY_COLORS['text']),
        tickformat=f",.{max(decimals,0)}f" if decimals is not None else None,
    )
    fig.update_yaxes(
        showline=False, zeroline=False, showgrid=True,
        gridcolor=STORY_COLORS['grid'], gridwidth=0.5,
        tickfont=dict(color=STORY_COLORS['text']),
        tickformat=f",.{max(decimals,0)}f" if decimals is not None else None,
    )
    if title_text:
        title_html = f"<b>{title_text}</b>"
        if subtitle_text:
            title_html += f"<br><sup style='color:{STORY_COLORS['text']}'>{subtitle_text}</sup>"
        fig.update_layout(title={'text': title_html, 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}})
    return fig

def create_storytelling_chart(data, chart_type="bar", emphasis_category=None, title_text="", subtitle_text=None):
    if chart_type == "pie":
        return create_storytelling_chart(data, chart_type="bar", emphasis_category=emphasis_category, title_text=title_text, subtitle_text=subtitle_text)
    if chart_type == "bar":
        colors = [STORY_COLORS['context']] * len(data)
        emphasis_idx = None
        if emphasis_category and emphasis_category in data.index:
            emphasis_idx = data.index.get_loc(emphasis_category)
            colors[emphasis_idx] = STORY_COLORS['emphasis']
        fig = go.Figure(go.Bar(
            x=data.index,
            y=data.values,
            marker_color=colors,
            text=data.values,
            textposition='auto'
        ))
        if emphasis_idx is not None:
            fig = apply_5_step_story(fig, emphasis_trace_idxs=[0], title_text=title_text, subtitle_text=subtitle_text)
        else:
            fig = apply_5_step_story(fig, title_text=title_text, subtitle_text=subtitle_text)
    elif chart_type == "line":
        fig = go.Figure()
        emphasis_idx = None
        for i, col in enumerate(data.columns):
            color = STORY_COLORS['emphasis'] if col == emphasis_category else STORY_COLORS['context']
            width = 3 if col == emphasis_category else 1
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines+markers',
                line=dict(color=color, width=width),
                name=col,
                text=data[col] if col == emphasis_category else None,
                textposition='top center'
            ))
            if col == emphasis_category:
                emphasis_idx = i
        if emphasis_idx is not None:
            fig = apply_5_step_story(fig, emphasis_trace_idxs=[emphasis_idx], title_text=title_text, subtitle_text=subtitle_text)
        else:
            fig = apply_5_step_story(fig, title_text=title_text, subtitle_text=subtitle_text)
    return fig

def compute_common_metrics(filtered_df: pd.DataFrame):
    total_incidents = len(filtered_df)
    critical_incidents = int((filtered_df['severity'] == 'Critical').sum()) if total_incidents else 0
    same_day_rate = float(filtered_df['same_day_reporting'].mean() * 100) if total_incidents else 0.0
    reportable_rate = float((filtered_df['reportable'] == 'Yes').mean() * 100) if total_incidents else 0.0
    return total_incidents, critical_incidents, same_day_rate, reportable_rate

def render_executive_summary(df: pd.DataFrame, filtered_df: pd.DataFrame):
    ns = "Executive Summary"
    st.title("📊 NDIS Executive Dashboard")
    st.markdown("**Strategic Overview - Incident Analysis & Risk Management**")
    st.markdown(f"*Showing {len(filtered_df)} incidents from {len(df)} total records*")
    st.markdown("---")
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
    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader("📈 Incident Trends by Month")
        if not filtered_df.empty:
            monthly = filtered_df.groupby(['incident_month', 'severity']).size().unstack(fill_value=0)
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
    colC, colD = st.columns(2)
    with colC:
        st.subheader("📊 Incident Types Distribution")
        if not filtered_df.empty:
            vc = filtered_df['incident_type'].value_counts()
            top_type = vc.index[0] if not vc.empty else None
            fig = create_storytelling_chart(
                vc,
                chart_type="bar",
                emphasis_category=top_type,
                title_text=f"Top incident type: {top_type} ({vc.iloc[0]})" if top_type else "Incident type distribution"
            )
            plotly_chart_safe(fig, name="type_story_bar", namespace=ns)
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

def render_operational_performance(df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.title("Operational Performance")
    st.markdown("This is the Operational Performance page.")
    st.dataframe(filtered_df.head())

def render_compliance_investigation(df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.title("Compliance & Investigation")
    st.markdown("This is the Compliance & Investigation page.")
    st.dataframe(filtered_df.head())

def render_ml_analytics(df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.title("🤖 Machine Learning Analytics")
    st.markdown("This is the ML Analytics page.")
    st.dataframe(filtered_df.head())

def render_risk_analysis(df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.title("Risk Analysis")
    st.markdown("This is the Risk Analysis page.")
    st.dataframe(filtered_df.head())

PAGE_TO_RENDERER = {
    "Executive Summary": render_executive_summary,
    "Operational Performance": render_operational_performance,
    "Compliance & Investigation": render_compliance_investigation,
    "🤖 Machine Learning Analytics": render_ml_analytics,
    "Risk Analysis": render_risk_analysis
}
