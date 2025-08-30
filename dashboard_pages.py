import hashlib
from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# NEW: Import mapping for geospatial visualization
import pydeck as pdk

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

# 
def _numeric_width(w):
    """Return a safe, positive numeric width from None / scalar / list-like."""
    import numpy as _np
    if isinstance(w, (list, tuple, _np.ndarray)):
        nums = [v for v in w if isinstance(v, (int, float))]
        return max(nums) if nums else 1
    if isinstance(w, (int, float)):
        try:
            if _np.isnan(w):  # handles float('nan')
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

    # 1) Start grey, respecting trace schemas
    for tr in fig.data:
        ttype = getattr(tr, 'type', None)

        # Work out a safe marker line width if marker.line exists
        marker_line_w = 1
        try:
            mlw = getattr(getattr(tr, "marker", None), "line", None)
            marker_line_w = _numeric_width(getattr(mlw, "width", None))
        except Exception:
            pass

        if hasattr(tr, "marker"):
            if ttype == "pie":
                # pie requires marker.colors (plural)
                # match the number of slices
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
                # bar/scatter/box/violin/etc. accept marker.color
                tr.update(marker=dict(
                    color=STORY_COLORS['context'],
                    line=dict(color=axis_color, width=marker_line_w),
                ))

        # Line-based styling (scatter, line, polar, etc.)
        if hasattr(tr, "line") and getattr(tr, "line", None) is not None:
            w = _numeric_width(getattr(tr.line, "width", None))
            tr.update(line=dict(color=axis_color, width=w))

    # 2) Emphasize selected traces + bring to front
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
                nd.append(nd.pop(idx))  # bring to front
                fig.data = tuple(nd)

    # 3) De-clutter
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

    # 4) Action title
    if title_text:
        title_html = f"<b>{title_text}</b>"
        if subtitle_text:
            title_html += f"<br><sup style='color:{STORY_COLORS['text']}'>{subtitle_text}</sup>"
        fig.update_layout(title={'text': title_html, 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}})

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

    # NEW: Map visualization for incident locations (if lat/lon present)
    if 'lat' in filtered_df.columns and 'lon' in filtered_df.columns:
        st.subheader("🗺️ Incident Location Map")
        map_df = filtered_df[['lat','lon','location','incident_type','severity']].dropna(subset=['lat','lon'])
        if not map_df.empty:
            st.map(map_df, latitude='lat', longitude='lon', size=None, color=None)
            # Optionally, for more advanced mapping:
            # st.pydeck_chart(
            #     pdk.Deck(
            #         initial_view_state=pdk.ViewState(
            #             latitude=map_df['lat'].mean(),
            #             longitude=map_df['lon'].mean(),
            #             zoom=10
            #         ),
            #         layers=[
            #             pdk.Layer(
            #                 "ScatterplotLayer",
            #                 data=map_df,
            #                 get_position='[lon, lat]',
            #                 get_color='[200, 30, 0, 160]',
            #                 get_radius=100,
            #             ),
            #         ],
            #     )
            # )

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

# ... rest of the code remains unchanged (Operational Performance, Compliance, ML, Risk Analysis, router, etc.) ...

PAGE_TO_RENDERER = {
    "Executive Summary": render_executive_summary,
    "Operational Performance": render_operational_performance,
    "Compliance & Investigation": render_compliance_investigation,
    "🤖 Machine Learning Analytics": render_ml_analytics,
    "Risk Analysis": render_risk_analysis
}
