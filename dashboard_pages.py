import hashlib
from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# Palettes (colors)
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
    'emphasis':   '#1F77B4',
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
# Streamlit-safe chart utils
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
# Storytelling helper
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
    for tr in fig.data:
        ttype = getattr(tr, 'type', None)
        if ttype == "pie":
            n = len(getattr(tr, "labels", [])) or len(getattr(tr, "values", [])) or 1
            try:
                tr.update(marker=dict(colors=[STORY_COLORS['context']] * n))
            except Exception:
                pass
        elif hasattr(tr, "marker"):
            try:
                tr.update(marker=dict(color=STORY_COLORS['context']))
            except Exception:
                pass
        if hasattr(tr, "line") and getattr(tr, "line", None) is not None:
            line_props = {}
            if hasattr(tr.line, "color"):
                line_props["color"] = STORY_COLORS['axisline']
            if hasattr(tr.line, "width"):
                width = getattr(tr.line, "width", 1)
                line_props["width"] = max(width if width else 1, 1)
            if line_props:
                try:
                    tr.update(line=line_props)
                except Exception:
                    pass

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

# =========================
# Shared calcs
# =========================
def compute_common_metrics(filtered_df: pd.DataFrame):
    total_incidents = len(filtered_df)
    critical_incidents = int((filtered_df['severity'] == 'Critical').sum()) if total_incidents else 0
    same_day_rate = float(filtered_df['notification_date'].eq(filtered_df['incident_date']).mean() * 100) if total_incidents else 0.0
    reportable_rate = float((filtered_df['reportable'] == 'Yes').mean() * 100) if total_incidents else 0.0
    return total_incidents, critical_incidents, same_day_rate, reportable_rate

# =========================
# Chart functions (only valid keys in colorbar!)
# =========================

def create_heatmap(df, title=""):
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale=[
            [0, '#FFFFFF'],
            [0.5, '#67A3C3'],
            [1, '#003F5C']
        ],
        colorbar=dict(
            title="Value",
            thickness=15,
            tickmode="auto"
        ),
        text=df.values.round(2).astype(str),
        hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title={'text': f"<b>{title}</b>", 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}},
        xaxis=dict(tickfont=dict(size=10, color='#666666'), side='bottom'),
        yaxis=dict(tickfont=dict(size=10, color='#666666'), autorange='reversed'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=100, t=60, b=60),
        height=400
    )
    return fig

def create_progress_chart(value, target, title="", subtitle=""):
    percentage = (value / target) * 100 if target else 0
    remaining = 100 - percentage
    if percentage >= 90:
        color = '#2F9E7D'
    elif percentage >= 70:
        color = '#F59C2F'
    else:
        color = '#DC2626'
    fig = go.Figure(go.Pie(
        values=[percentage, remaining],
        hole=0.7,
        marker=dict(colors=[color, '#F0F0F0']),
        textinfo='none',
        hoverinfo='skip'
    ))
    fig.add_annotation(
        text=f'<b style="font-size:36px;color:{color}">{percentage:.0f}%</b><br>' +
             f'<span style="font-size:14px;color:#666666">{value:,.0f} / {target:,.0f}</span>',
        x=0.5, y=0.5,
        font=dict(size=20),
        showarrow=False
    )
    fig.update_layout(
        title={
            'text': f"<b>{title}</b><br><sup style='color:#666666'>{subtitle}</sup>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': '#333333'}
        },
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),
        height=250,
        paper_bgcolor='white'
    )
    return fig

def create_bullet_chart(actual, target, ranges, title="", subtitle=""):
    fig = go.Figure()
    colors = ['#F0F0F0', '#D3D3D3', '#B0B0B0']
    widths = [ranges[0], ranges[1]-ranges[0], ranges[2]-ranges[1]]
    for i, (width, color) in enumerate(zip(widths, colors)):
        fig.add_trace(go.Bar(
            x=[width],
            y=[0],
            orientation='h',
            marker_color=color,
            width=0.4,
            base=ranges[i-1] if i > 0 else 0,
            showlegend=False,
            hoverinfo='skip'
        ))
    fig.add_trace(go.Bar(
        x=[actual],
        y=[0],
        orientation='h',
        marker_color='#003F5C',
        width=0.2,
        name='Actual',
        text=f'{actual:.0f}',
        textposition='outside',
        textfont=dict(size=12, color='#003F5C', weight='bold'),
        hovertemplate=f'Actual: {actual}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[target],
        y=[0],
        mode='markers',
        marker=dict(
            symbol='line-ns',
            size=20,
            line_width=3,
            color='#DC2626'
        ),
        name='Target',
        hovertemplate=f'Target: {target}<extra></extra>'
    ))
    fig.update_layout(
        title={'text': f"<b>{title}</b><br><sup style='color:#666666'>{subtitle}</sup>", 'x': 0, 'xanchor': 'left', 'font': {'size': 14, 'color': '#333333'}},
        xaxis=dict(showgrid=False, showline=True, linecolor='#D3D3D3', tickfont=dict(size=10, color='#666666'), range=[0, ranges[2] * 1.1]),
        yaxis=dict(showgrid=False, showline=False, showticklabels=False, range=[-0.5, 0.5]),
        plot_bgcolor='white', paper_bgcolor='white', height=150, margin=dict(l=20, r=60, t=60, b=40), showlegend=False, barmode='overlay'
    )
    return fig

def create_dot_plot(df, category_col, value_cols, title=""):
    fig = go.Figure()
    colors = ['#003F5C', '#F59C2F', '#2F9E7D', '#DC2626', '#67A3C3']
    for i, col in enumerate(value_cols):
        fig.add_trace(go.Scatter(
            x=df[col],
            y=df[category_col],
            mode='markers+text',
            marker=dict(size=10, color=colors[i % len(colors)]),
            text=df[col].round(0).astype(int),
            textposition='middle right',
            textfont=dict(size=10, color=colors[i % len(colors)]),
            name=col,
            hovertemplate='%{text}<extra></extra>'
        ))
    fig.update_layout(
        title={'text': f"<b>{title}</b>", 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}},
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#F0F0F0', showline=True, linecolor='#D3D3D3', tickfont=dict(size=10, color='#666666')),
        yaxis=dict(tickfont=dict(size=11, color='#333333')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=100, r=60, t=60, b=40),
        showlegend=True,
        legend=dict(orientation='h', y=1.1, x=0, font=dict(size=10, color='#666666'))
    )
    return fig

def create_diverging_bar_chart(df, category_col, value_col, title="", center_value=0):
    df_sorted = df.copy()
    df_sorted['abs_val'] = df_sorted[value_col].abs()
    df_sorted = df_sorted.sort_values('abs_val', ascending=True)
    colors = ['#2F9E7D' if x > center_value else '#DC2626' for x in df_sorted[value_col]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted[value_col],
        y=df_sorted[category_col],
        orientation='h',
        marker_color=colors,
        text=df_sorted[value_col].apply(lambda x: f'{x:+.1f}%'),
        textposition='outside',
        textfont=dict(size=10, color='#333333'),
        hovertemplate='%{y}: %{x:+.1f}%<extra></extra>'
    ))
    fig.update_layout(
        title={'text': f"<b>{title}</b>", 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}},
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#F0F0F0', showline=False, zeroline=False, ticksuffix='%', tickfont=dict(size=10, color='#666666')),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color='#333333')),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=120, r=60, t=60, b=40), height=400
    )
    return fig

def create_waterfall_chart(categories, values, title="", subtitle=""):
    cumulative = [0]
    for val in values[:-1]:
        cumulative.append(cumulative[-1] + val)
    colors = []
    for i, val in enumerate(values):
        if i == 0 or i == len(values) - 1:
            colors.append('#003F5C')
        elif val > 0:
            colors.append('#2F9E7D')
        else:
            colors.append('#DC2626')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:+.0f}' if i not in [0, len(values)-1] else f'{v:.0f}' for i, v in enumerate(values)],
        textposition='outside',
        textfont=dict(size=11, color='#333333'),
        showlegend=False,
        hovertemplate='%{x}: %{y:+.0f}<extra></extra>'
    ))
    fig.update_layout(
        title={'text': f"<b>{title}</b><br><sup style='color:#666666'>{subtitle}</sup>", 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}},
        barmode='stack',
        xaxis=dict(showgrid=False, showline=True, linecolor='#D3D3D3', tickfont=dict(size=10, color='#666666')),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#F0F0F0', showline=False, tickfont=dict(size=10, color='#666666'), title=""),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=60, r=60, t=80, b=40), height=400
    )
    return fig

def create_horizontal_bar_chart(df, category_col, value_col, highlight_category=None, title="", subtitle=""):
    df_sorted = df.sort_values(value_col, ascending=True)
    colors = ['#D3D3D3'] * len(df_sorted)
    if highlight_category and highlight_category in df_sorted[category_col].tolist():
        idx = df_sorted[category_col].tolist().index(highlight_category)
        colors[idx] = '#1F77B4'
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted[value_col],
        y=df_sorted[category_col],
        orientation='h',
        marker_color=colors,
        text=df_sorted[value_col].round(0).astype(int),
        textposition='outside',
        textfont=dict(size=11, color='#666666'),
        hovertemplate='%{y}: %{x}<extra></extra>'
    ))
    fig.update_layout(
        title={'text': f"<b>{title}</b><br><sup style='color: #666666'>{subtitle}</sup>", 'x': 0, 'xanchor': 'left', 'font': {'size': 16, 'color': '#333333'}},
        xaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color='#666666')),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=150, r=50, t=80, b=40), height=400, showlegend=False
    )
    return fig

# =========================
# Pages
# =========================

def render_executive_summary(filtered_df: pd.DataFrame):
    ns = "Executive Summary"
    st.title("📊 NDIS Executive Dashboard")
    st.markdown("**Strategic Overview - Incident Analysis & Risk Management**")
    st.markdown(f"*Showing {len(filtered_df)} incidents*")
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
            filtered_df['incident_date'] = pd.to_datetime(filtered_df['incident_date'], errors='coerce', dayfirst=True)
            filtered_df['incident_month'] = filtered_df['incident_date'].dt.strftime('%B')
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
            fig = go.Figure(go.Bar(
                x=vc.index,
                y=vc.values,
                marker_color=[
                    NDIS_COLORS['primary'], NDIS_COLORS['secondary'], NDIS_COLORS['accent'],
                    NDIS_COLORS['critical'], '#67A3C3', '#8B9DC3'
                ][:len(vc)],
                text=vc.values,
                textposition='auto'
            ))
            fig = apply_5_step_story(fig, title_text=f"Top incident type: {vc.index[0]} ({vc.iloc[0]})")
            plotly_chart_safe(fig, name="type_bar", namespace=ns)
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

def render_operational_performance(filtered_df: pd.DataFrame):
    st.title("📊 Operational Performance")

    if filtered_df.empty:
        st.info("No data for the selected period.")
        return

    total = len(filtered_df)
    critical = (filtered_df['severity'] == "Critical").sum()
    st.subheader("Critical Incident Progress")
    st.plotly_chart(
        create_progress_chart(
            value=critical,
            target=total,
            title="Critical Incidents",
            subtitle=f"{critical} of {total} incidents"
        ),
        use_container_width=True
    )

    reportable_rate = (filtered_df['reportable'] == "Yes").mean() * 100
    st.subheader("Reportable Rate vs Target")
    st.plotly_chart(
        create_bullet_chart(
            actual=reportable_rate,
            target=90,
            ranges=[60, 80, 100],
            title="Reportable Incidents (%)",
            subtitle=f"Current: {reportable_rate:.1f}% | Target: 90%"
        ),
        use_container_width=True
    )

    st.subheader("Location vs Severity Dot Plot")
    by_loc_sev = (
        filtered_df.groupby(['location', 'severity'])
        .size().unstack(fill_value=0)
        .reset_index()
    )
    sev_cols = [col for col in by_loc_sev.columns if col != 'location']
    if len(sev_cols) > 0:
        st.plotly_chart(
            create_dot_plot(
                by_loc_sev,
                category_col='location',
                value_cols=sev_cols,
                title="Incidents per Location by Severity"
            ),
            use_container_width=True
        )

    st.subheader("Incident Type Deviation")
    type_counts = filtered_df['incident_type'].value_counts().head(10)
    type_dev = type_counts - type_counts.mean()
    df_type_dev = pd.DataFrame({
        'Type': type_counts.index,
        'Deviation': type_dev.values
    })
    st.plotly_chart(
        create_diverging_bar_chart(
            df_type_dev,
            category_col='Type',
            value_col='Deviation',
            title="Deviation from Avg. by Type"
        ),
        use_container_width=True
    )

    st.subheader("Monthly Incident Change Waterfall")
    filtered_df['month'] = pd.to_datetime(filtered_df['incident_date'], dayfirst=True).dt.strftime('%b %Y')
    monthly_counts = filtered_df.groupby('month').size().sort_index()
    months = monthly_counts.index.tolist()
    if len(monthly_counts) > 1:
        values = [monthly_counts.iloc[0]] + [monthly_counts.iloc[i] - monthly_counts.iloc[i-1] for i in range(1, len(monthly_counts))]
        st.plotly_chart(
            create_waterfall_chart(
                categories=months,
                values=values,
                title="Incident Count Changes by Month"
            ),
            use_container_width=True
        )

    st.subheader("Incident Type x Severity Heatmap")
    heatmap_df = (
        filtered_df.groupby(['incident_type', 'severity'])
        .size().unstack(fill_value=0)
    )
    st.plotly_chart(
        create_heatmap(
            heatmap_df,
            title="Incident Types by Severity"
        ),
        use_container_width=True
    )

    st.markdown("### Raw Data Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

def render_compliance_investigation(filtered_df: pd.DataFrame):
    st.title("🕵️ Compliance & Investigation")
    st.markdown("This dashboard will show compliance and investigation metrics.")
    if filtered_df.empty:
        st.info("No data for the selected period.")
    else:
        # Show a horizontal bar: incidents by subcategory
        st.subheader("Incidents by Subcategory")
        subcat_counts = filtered_df['subcategory'].value_counts().head(10)
        df_subcat = pd.DataFrame({'Subcategory': subcat_counts.index, 'Count': subcat_counts.values})
        st.plotly_chart(
            create_horizontal_bar_chart(
                df_subcat,
                category_col='Subcategory',
                value_col='Count',
                highlight_category=df_subcat.iloc[0]['Subcategory'],
                title="Most Frequent Subcategories",
                subtitle="Top 10 subcategories"
            ),
            use_container_width=True
        )

        # Show compliance rate (fake calculation for demo)
        if 'compliant' in filtered_df.columns:
            compliance_rate = (filtered_df['compliant'] == True).mean() * 100
            st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        st.dataframe(filtered_df.head(15), use_container_width=True)

def render_ml_analytics(filtered_df: pd.DataFrame, **kwargs):
    st.title("🤖 Machine Learning Analytics")
    st.markdown("This dashboard will show ML-based analytics and predictions.")
    if filtered_df.empty:
        st.info("No data for the selected period.")
    else:
        # Example: Heatmap by severity/location
        st.subheader("Severity by Location Heatmap")
        heatmap_df = (
            filtered_df.groupby(['location', 'severity'])
            .size().unstack(fill_value=0)
        )
        st.plotly_chart(
            create_heatmap(heatmap_df, title="Severity by Location"),
            use_container_width=True
        )
        st.dataframe(filtered_df.head(15), use_container_width=True)

def render_risk_analysis(filtered_df: pd.DataFrame):
    st.title("⚠️ Risk Analysis")
    st.markdown("This dashboard is for risk analysis. More analytics will be added here.")
    if filtered_df.empty:
        st.info("No data for the selected period.")
    else:
        # Use diverging bar: e.g. location with highest deviation from mean incident count
        st.subheader("Incident Count Deviation by Location")
        loc_counts = filtered_df['location'].value_counts().head(10)
        loc_dev = loc_counts - loc_counts.mean()
        df_loc_dev = pd.DataFrame({'Location': loc_counts.index, 'Deviation': loc_dev.values})
        st.plotly_chart(
            create_diverging_bar_chart(df_loc_dev, category_col='Location', value_col='Deviation', title="Deviation from Avg. by Location"),
            use_container_width=True
        )
        st.dataframe(filtered_df.head(15), use_container_width=True)

PAGE_TO_RENDERER = {
    "Executive Summary": render_executive_summary,
    "Operational Performance": render_operational_performance,
    "Compliance & Investigation": render_compliance_investigation,
    "ML Analytics": render_ml_analytics,
    "🤖 Machine Learning Analytics": render_ml_analytics,
    "Risk Analysis": render_risk_analysis,
}
