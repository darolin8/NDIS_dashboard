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
# Pages (STUBS)
# =========================

def render_executive_summary(df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.title("Executive Summary")
    st.markdown("This is the Executive Summary page. Add key metrics and charts here.")
    st.dataframe(filtered_df.head())

def render_operational_performance(filtered_df: pd.DataFrame):
    st.title("Operational Performance")
    st.markdown("This is the Operational Performance page.")
    st.dataframe(filtered_df.head())

def render_compliance_investigation(filtered_df: pd.DataFrame):
    st.title("Compliance & Investigation")
    st.markdown("This is the Compliance & Investigation page.")
    st.dataframe(filtered_df.head())

def render_ml_analytics(
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

def render_risk_analysis(filtered_df):
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
