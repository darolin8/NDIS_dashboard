# incident_mapping.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
import re
import math
import hashlib

# ---------------------------------------------
# Config / constants
# ---------------------------------------------
AUS_CENTER = {"lat": -33.865, "lon": 151.209}  # around Sydney
AUS_ZOOM = 8

SEVERITY_ORDER = ["Low", "Medium", "Moderate", "High", "Critical"]
SEVERITY_RANK = {"Low": 0, "Medium": 1, "Moderate": 1, "High": 2, "Critical": 3}
SEVERITY_COLORS = {
    "Low": "#2F9E7D",
    "Medium": "#F59C2F",
    "Moderate": "#F59C2F",
    "High": "#DC2626",
    "Critical": "#7F1D1D",
}

# Suburb centroids you need for your addresses
SUBURB_CENTROIDS = {
    "SYDNEY": (-33.8688, 151.2093),
    "PARRAMATTA": (-33.8136, 151.0017),
    "LIVERPOOL": (-33.9200, 150.9230),
    "CAMPBELLTOWN": (-34.0660, 150.8140),
    "BONDI": (-33.8915, 151.2767),
    "BLACKTOWN": (-33.7680, 150.9080),
    "PENRITH": (-33.7510, 150.6920),
    "WOLLONGONG": (-34.4278, 150.8931),
    "NEWCASTLE": (-32.9283, 151.7817),
    "CHATSWOOD": (-33.7960, 151.1830),
}

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "incident_date" in df.columns and not np.issubdtype(df["incident_date"].dtype, np.datetime64):
        with pd.option_context("mode.chained_assignment", None):
            df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    return df

def _severity_category(s):
    if pd.isna(s):
        return "Low"
    s = str(s)
    if s not in SEVERITY_RANK:
        if s.lower().startswith("crit"):
            return "Critical"
        if s.lower().startswith("high"):
            return "High"
        if s.lower().startswith("mod"):
            return "Moderate"
        if s.lower().startswith("med"):
            return "Medium"
        return "Low"
    return s

def _extract_suburb_postcode(addr: str):
    """
    Parse '..., <Suburb>, NSW <postcode>' from the end of the address string.
    Returns (SUBURB_UPPER, postcode_str) or (None, None) if not found.
    """
    if not isinstance(addr, str):
        return None, None
    m = re.search(r",\s*([A-Za-z ]+),\s*NSW\s*(\d{4})\b", addr, flags=re.I)
    if not m:
        return None, None
    suburb = m.group(1).strip().upper()
    pcode = m.group(2).strip()
    return suburb, pcode

def _stable_jitter(address: str, lat: float, jitter_m: float = 250.0):
    """
    Deterministic tiny offset per address to avoid overplot.
    ~250 m default. Uses MD5(address) as seed.
    """
    if not isinstance(address, str) or pd.isna(lat):
        return 0.0, 0.0
    seed_int = int(hashlib.md5(address.encode("utf-8")).hexdigest(), 16) % (10**8)
    rng = np.random.default_rng(seed_int)
    # Convert metres to degrees
    lat_deg = jitter_m / 111_000.0
    lon_deg = jitter_m / (111_000.0 * max(0.2, math.cos(math.radians(lat))))  # protect near poles
    return rng.uniform(-lat_deg, lat_deg), rng.uniform(-lon_deg, lon_deg)

def _add_latlon_from_address(df: pd.DataFrame, address_col: str = "address", jitter_m: float = 250.0):
    """
    Fill/derive latitude & longitude from a free-text 'address' column by
    mapping suburb centroids, then applying small stable jitter per address.
    """
    if address_col not in df.columns:
        return df

    out = df.copy()
    if "latitude" not in out.columns:
        out["latitude"] = np.nan
    if "longitude" not in out.columns:
        out["longitude"] = np.nan

    # Only fill missing coords
    mask = out["latitude"].isna() | out["longitude"].isna()
    if not mask.any():
        return out

    sub_pcode = out.loc[mask, address_col].apply(_extract_suburb_postcode)
    subs = sub_pcode.apply(lambda t: t[0])
    # Map centroids for found suburbs
    latlon = subs.apply(lambda s: SUBURB_CENTROIDS.get(s) if isinstance(s, str) else None)

    # Assign centroids
    out.loc[mask, "latitude"] = [
        (latlon.iloc[i][0] if latlon.iloc[i] else np.nan) for i in range(latlon.shape[0])
    ]
    out.loc[mask, "longitude"] = [
        (latlon.iloc[i][1] if latlon.iloc[i] else np.nan) for i in range(latlon.shape[0])
    ]

    # Apply stable jitter so repeated addresses separate a bit
    need_jitter = out[address_col].notna() & out["latitude"].notna() & out["longitude"].notna()
    for idx in out.index[need_jitter]:
        lat = out.at[idx, "latitude"]
        lon = out.at[idx, "longitude"]
        jlat, jlon = _stable_jitter(str(out.at[idx, address_col]), lat, jitter_m=jitter_m)
        out.at[idx, "latitude"] = lat + jlat
        out.at[idx, "longitude"] = lon + jlon

    return out

def _fallback_geocode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Priority:
    1) Use existing latitude/longitude if present.
    2) Derive from 'address' column (your case).
    3) Derive from 'location' column if present (city keyword match).
    """
    if {"latitude", "longitude"}.issubset(df.columns) and df["latitude"].notna().any():
        return df
    # Address-based fill
    tmp = _add_latlon_from_address(df, address_col="address", jitter_m=250.0)
    if tmp["latitude"].notna().any() and tmp["longitude"].notna().any():
        return tmp

    # Fallback: try 'location' keywords (very light)
    if "location" in df.columns:
        out = df.copy()
        if "latitude" not in out.columns:
            out["latitude"] = np.nan
        if "longitude" not in out.columns:
            out["longitude"] = np.nan

        def key_lookup(loc):
            if not isinstance(loc, str):
                return np.nan, np.nan
            for k, (lat, lon) in SUBURB_CENTROIDS.items():
                if k.lower() in loc.lower():
                    return lat, lon
            return np.nan, np.nan

        ll = out["location"].apply(key_lookup)
        out["latitude"] = out["latitude"].fillna(ll.apply(lambda x: x[0]))
        out["longitude"] = out["longitude"].fillna(ll.apply(lambda x: x[1]))
        return out

    return df

def _center_from_data(df: pd.DataFrame):
    if {"latitude", "longitude"}.issubset(df.columns):
        lat_ok = df["latitude"].dropna()
        lon_ok = df["longitude"].dropna()
        if not lat_ok.empty and not lon_ok.empty:
            return {"lat": float(lat_ok.median()), "lon": float(lon_ok.median())}
    return AUS_CENTER

def _filter_df(df: pd.DataFrame, date_range, severities_selected, types_selected):
    out = df.copy()
    if "incident_date" in out.columns and (date_range is not None):
        start, end = date_range
        if isinstance(start, (pd.Timestamp, datetime)) and isinstance(end, (pd.Timestamp, datetime)):
            out = out[(out["incident_date"] >= pd.to_datetime(start)) & (out["incident_date"] <= pd.to_datetime(end))]
    if ("severity" in out.columns) and severities_selected:
        out = out[out["severity"].isin(severities_selected)]
    if ("incident_type" in out.columns) and types_selected:
        out = out[out["incident_type"].isin(types_selected)]
    return out

# ---------------------------------------------
# Public renderer for your PAGE_TO_RENDERER
# ---------------------------------------------
def render_incident_mapping(df: pd.DataFrame, filtered_df: pd.DataFrame | None = None):
    """
    Streamlit page: Incident Map (address-aware)
    - Reads df['address'] like '..., Bondi, NSW 2026'
    - Derives lat/lon from suburb centroids with stable jitter per address
    - Tabs: Points + Density (severity-weighted)
    """
    st.markdown("## 🗺️ Incident Map")

    if df is None or df.empty:
        st.warning("No data available.")
        return

    base = filtered_df if (filtered_df is not None and not filtered_df.empty) else df
    base = _ensure_datetime(base).copy()

    # Normalise severity naming if present
    if "severity" in base.columns:
        base["severity"] = base["severity"].apply(_severity_category)
        base["severity"] = pd.Categorical(base["severity"], categories=SEVERITY_ORDER, ordered=True)

    # Derive coordinates (address -> suburb centroid + jitter)
    geo = _fallback_geocode(base)
    geo = geo.dropna(subset=["latitude", "longitude"]).copy()
    if geo.empty:
        st.error(
            "Could not derive coordinates. Ensure your data has an 'address' column like '..., Suburb, NSW 2000', "
            "or provide latitude/longitude columns."
        )
        return

    with st.expander("Filters", expanded=True):
        # Date
        date_range = None
        if "incident_date" in geo.columns:
            min_d = pd.to_datetime(geo["incident_date"].min())
            max_d = pd.to_datetime(geo["incident_date"].max())
            date_range = st.date_input(
                "Date range",
                value=(min_d.date(), max_d.date()) if pd.notna(min_d) and pd.notna(max_d) else None,
                min_value=min_d.date() if pd.notna(min_d) else None,
                max_value=max_d.date() if pd.notna(max_d) else None,
            )

        # Severity
        sevs = sorted([s for s in geo.get("severity", pd.Series(dtype=object)).dropna().unique()],
                      key=lambda x: SEVERITY_ORDER.index(x) if x in SEVERITY_ORDER else 0)
        selected_sev = st.multiselect("Severity", options=sevs, default=sevs)

        # Type
        types = sorted(geo.get("incident_type", pd.Series(dtype=object)).dropna().unique())
        selected_types = st.multiselect("Incident type", options=types, default=types)

        style = st.selectbox(
            "Base map style",
            options=["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"],
            index=0,
        )
        point_size = st.slider("Point size", 4, 18, 9)
        heat_radius = st.slider("Heatmap radius (px)", 8, 60, 28)

    geo_f = _filter_df(geo, date_range, selected_sev, selected_types)
    if geo_f.empty:
        st.info("No incidents match the current filters.")
        return

    center = _center_from_data(geo_f)
    tabs = st.tabs(["• Points", "• Density heatmap"])

    # Points
    with tabs[0]:
        hover_cols = [c for c in ["incident_date", "incident_type", "severity", "address", "report_id"] if c in geo_f.columns]
        fig_points = px.scatter_mapbox(
            geo_f,
            lat="latitude",
            lon="longitude",
            color="severity" if "severity" in geo_f.columns else None,
            color_discrete_map=SEVERITY_COLORS if "severity" in geo_f.columns else None,
            hover_data=hover_cols,
            zoom=10,
        )
        fig_points.update_traces(marker=dict(size=point_size))
        fig_points.update_layout(
            mapbox_style=style,
            mapbox_center=center,
            mapbox_zoom=AUS_ZOOM,
            margin=dict(l=0, r=0, t=0, b=0),
            legend_title_text="Severity" if "severity" in geo_f.columns else None,
        )
        st.plotly_chart(fig_points, use_container_width=True)

    # Density
    with tabs[1]:
        z_values = geo_f["severity"].map(SEVERITY_RANK) if "severity" in geo_f.columns else pd.Series(1, index=geo_f.index)
        fig_heat = px.density_mapbox(
            geo_f.assign(sev_weight=z_values),
            lat="latitude",
            lon="longitude",
            z="sev_weight",
            radius=heat_radius,
            center=center,
            zoom=AUS_ZOOM,
        )
        fig_heat.update_layout(mapbox_style=style, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Incidents shown", len(geo_f))
    with c2:
        if "severity" in geo_f.columns:
            crit = int((geo_f["severity"] == "Critical").sum())
            st.metric("Critical incidents", crit)
        else:
            st.metric("Critical incidents", "—")
    with c3:
        if "address" in geo_f.columns:
            top_area = (
                geo_f["address"]
                .apply(lambda s: _extract_suburb_postcode(s)[0] if isinstance(s, str) else None)
                .dropna()
                .value_counts()
            )
            st.metric("Top suburb", top_area.index[0].title() if len(top_area) else "—")
        else:
            st.metric("Top suburb", "—")
