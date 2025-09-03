import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import pickle

st.set_page_config(page_title="NDIS Incidents Map", layout="wide")

st.title("NDIS Incident Mapping Dashboard")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def geocode_addresses(addresses):
    geolocator = Nominatim(user_agent="ndis_incidents_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    address_to_coords = {}
    for address in addresses:
        location = geocode(address)
        if location:
            address_to_coords[address] = (location.latitude, location.longitude)
        else:
            address_to_coords[address] = (None, None)
    return address_to_coords

# Load data
df = load_data("ndis_incident_1000.csv")

# Geocode and cache unique addresses
address_to_coords = geocode_addresses(df["address"].unique())
df["latitude"] = df["address"].map(lambda x: address_to_coords[x][0])
df["longitude"] = df["address"].map(lambda x: address_to_coords[x][1])
df = df.dropna(subset=["latitude", "longitude"])

# Sidebar filters
incident_types = st.sidebar.multiselect(
    "Incident Type", options=sorted(df["incident_type"].unique()), default=list(df["incident_type"].unique())
)
severities = st.sidebar.multiselect(
    "Severity", options=sorted(df["severity"].unique()), default=list(df["severity"].unique())
)

filtered_df = df[(df["incident_type"].isin(incident_types)) & (df["severity"].isin(severities))]

# Folium map
if not filtered_df.empty:
    center_lat = filtered_df["latitude"].mean()
    center_lon = filtered_df["longitude"].mean()
else:
    center_lat, center_lon = -33.8688, 151.2093  # Fallback (Sydney)

incident_map = folium.Map(location=[center_lat, center_lon], zoom_start=8)

for _, row in filtered_df.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"ID: {row['incident_id']}<br>Type: {row['incident_type']}<br>Severity: {row['severity']}<br>Address: {row['address']}",
        icon=folium.Icon(color="red" if row['severity'].lower() == "critical" else "blue")
    ).add_to(incident_map)

# Streamlit display
st.subheader("Incident Map (interactive)")
st_data = st_folium(incident_map, width=1000, height=600)

st.subheader("Filtered Data")
st.dataframe(filtered_df)