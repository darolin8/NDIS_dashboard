# theme.py — full UI kit for the NDIS Incident Dashboard
# -----------------------------------------------
# Provides: css(), set_plotly_theme(), kpi_card(), section_title(),
# severity_color_map(), compliance_band_color_map(), divider(), and extras.

from __future__ import annotations
import json
import plotly.io as pio
import streamlit as st
from typing import Dict, Any, Optional

# ---- Brand & palette -------------------------------------------------------
BRAND = {
    "primary": "#2563eb",   # blue-600
    "primaryDark": "#1d4ed8", # blue-700
    "accent": "#22c55e",    # green-500
    "warning": "#f59e0b",   # amber-500
    "danger":  "#ef4444",   # red-500
    "muted":   "#64748b",   # slate-500
    "ink":     "#0b1220",   # near-black background (alt)
}

SEVERITY_COLORS = {
    "Low": BRAND["accent"],
    "Moderate": BRAND["warning"],
    "High": BRAND["danger"],
    "Critical": "#991b1b",
}

COMPLIANCE_BANDS = {
    "Within 24h": BRAND["accent"],
    "24–48h": BRAND["warning"],
    ">48h": BRAND["danger"],
}

# ---- Public helpers --------------------------------------------------------

def severity_color_map() -> Dict[str, str]:
    return SEVERITY_COLORS.copy()

def compliance_band_color_map() -> Dict[str, str]:
    return COMPLIANCE_BANDS.copy()

# ---- CSS & page theming ----------------------------------------------------

_GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap"
)


def css():
    """Inject global CSS (dark/light aware) and small utility classes."""
    st.markdown(
        f"""
        <link href="{_GOOGLE_FONTS}" rel="stylesheet">
        <style>
            :root {{
                --bg: #ffffff; --panel: #f8fafc; --text: #0f172a; --muted: {BRAND['muted']};
                --primary: {BRAND['primary']}; --primary-dark: {BRAND['primaryDark']};
                --accent: {BRAND['accent']}; --warning: {BRAND['warning']}; --danger: {BRAND['danger']};
                --ring: rgba(37, 99, 235, .35);
            }}
            @media (prefers-color-scheme: dark) {{
                :root {{
                    --bg: #0b1220; --panel: #0f172a; --text: #e2e8f0; --muted: #94a3b8;
                }}
            }}
            html, body, [data-testid="stAppViewContainer"] {{
                background: var(--bg) !important;
                color: var(--text);
                font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            }}
            .small-muted {{ color: var(--muted); font-size: 12px; }}
            .kpi {{
                padding: 14px 16px; border-radius: 14px; background: var(--panel);
                border: 1px solid rgba(148,163,184,.25);
            }}
            .kpi .value {{ font-size: 30px; font-weight: 700; }}
            .kpi .delta-up {{ color: var(--accent); }}
            .kpi .delta-down {{ color: var(--danger); }}

            .title-wrap h2 {{ margin-bottom: 0; }}
            .title-sub {{ color: var(--muted); margin-top: 4px; }}

            .pill {{
                display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius: 9999px;
                background: rgba(148,163,184,.15); color: var(--text); font-size: 12px; border:1px solid rgba(148,163,184,.25);
            }}
            .pill .dot {{ width:8px; height:8px; border-radius: 9999px; display:inline-block; }}

            .card {{ background: var(--panel); border-radius: 16px; padding: 16px; border:1px solid rgba(148,163,184,.25) }}
            .notice.info {{ border-left: 4px solid var(--primary); }}
            .notice.warn {{ border-left: 4px solid var(--warning); }}
            .notice.ok {{ border-left: 4px solid var(--accent); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- Plotly template -------------------------------------------------------

def _plotly_template() -> Dict[str, Any]:
    base = {
        "layout": {
            "font": {"family": "Inter, system-ui, sans-serif", "size": 13},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(255,255,255,1)",
            "colorway": [
                BRAND["primary"], "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#14b8a6", "#a3e635"
            ],
            "xaxis": {"gridcolor": "rgba(148,163,184,.25)", "zerolinecolor": "rgba(148,163,184,.25)"},
            "yaxis": {"gridcolor": "rgba(148,163,184,.25)", "zerolinecolor": "rgba(148,163,184,.25)"},
            "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "xanchor": "left"},
            "margin": {"t": 50, "r": 20, "b": 50, "l": 60},
        }
    }
    return base


def set_plotly_theme(name: str = "ndis_light"):
    """Register and set a clean Plotly template (light). Honors dark mode via your CSS."""
    template = _plotly_template()
    pio.templates[name] = template
    pio.templates.default = name


# ---- UI atoms --------------------------------------------------------------

def _fmt_number(value, prefix: str = "", suffix: str = "") -> str:
    try:
        if isinstance(value, float):
            s = f"{value:,.2f}"
        else:
            s = f"{int(value):,}"
    except Exception:
        s = str(value)
    return f"{prefix}{s}{suffix}"


def kpi_card(label: str, value, prefix: str = "", suffix: str = "", *, delta: Optional[float] = None, help_text: Optional[str] = None):
    """Simple KPI block with optional delta percentage."""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        cls = "delta-up" if delta >= 0 else "delta-down"
        delta_html = f"<div class='small-muted {cls}'>{arrow} {abs(delta):.1f}%</div>"

    st.markdown(
        f"""
        <div class='kpi'>
            <div class='small-muted'>{label}</div>
            <div class='value'>{_fmt_number(value, prefix, suffix)}</div>
            {delta_html}
            {f"<div class='small-muted'>{help_text}</div>" if help_text else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: Optional[str] = None, *, badge: Optional[str] = None, badge_color: Optional[str] = None):
    """Big section header with optional pill badge."""
    pill_html = ""
    if badge:
        dot = f"<span class='dot' style='background:{badge_color or BRAND['primary']}'></span>" if badge_color else ""
        pill_html = f"<span class='pill'>{dot}{badge}</span>"

    st.markdown(
        f"""
        <div class='title-wrap'>
            <h2>{title} {pill_html}</h2>
            {f"<div class='title-sub'>{subtitle}</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown("---")


# ---- Extras ----------------------------------------------------------------

def badge(text: str, color: str = BRAND["primary"]):
    st.markdown(f"<span class='pill'><span class='dot' style='background:{color}'></span>{text}</span>", unsafe_allow_html=True)


def info_box(text: str, kind: str = "info"):
    kind = kind.lower()
    cls = {"info": "info", "warn": "warn", "ok": "ok"}.get(kind, "info")
    st.markdown(f"<div class='card notice {cls}'>{text}</div>", unsafe_allow_html=True)


def theme_state() -> Dict[str, Any]:
    """Expose palette (useful for custom charts or debugging)."""
    return {
        "brand": BRAND,
        "severity": SEVERITY_COLORS,
        "compliance": COMPLIANCE_BANDS,
        "plotly_default": pio.templates.default,
    }


# ---- Quick self-test when run directly ------------------------------------
if __name__ == "__main__":
    set_plotly_theme()
    css()
    st.title("Theme Kit Preview")
    section_title("Preview Section", "Subtitle example", badge="LIVE", badge_color=BRAND["accent"]) 
    kpi_card("Total Incidents", 1000)
    info_box("This is an informational callout. Use it for tips or context.")
    info_box("Heads up — compliance trending above threshold.", kind="warn")
    info_box("All systems nominal.", kind="ok")
    st.write("State:")
    st.json(theme_state())
