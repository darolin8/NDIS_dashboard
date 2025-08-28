import pandas as pd
from geopy.geocoders import Nominatim
import time

# Set this to the path of your CSV file
CSV_PATH = "text data/incidents_1000.csv"
OUTPUT_CSV = "text data/incidents_1000_geocoded.csv"

# Load your data
df = pd.read_csv(CSV_PATH)

# Function to determine if a string is an address (not just a CareHome/Clinic/etc)
def is_address(loc):
    if pd.isnull(loc):
        return False
    s = str(loc).strip()
    # Looks like an address if it has a number, a comma, and isn't a known type
    return any(char.isdigit() for char in s) and ',' in s

address_mask = df['location'].apply(is_address)
address_df = df[address_mask].copy()

# Setup geopy Nominatim geocoder
geolocator = Nominatim(user_agent="ndis_dashboard_app")

def geocode_address(address):
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return pd.Series([location.latitude, location.longitude])
    except Exception as e:
        print(f"Error geocoding {address}: {str(e)}")
    return pd.Series([None, None])

# Geocode with delay to avoid rate limits
latitudes = []
longitudes = []

for addr in address_df['location']:
    lat, lon = geocode_address(addr)
    latitudes.append(lat)
    longitudes.append(lon)
    print(f"Geocoded: {addr} => {lat}, {lon}")
    time.sleep(1)  # Nominatim recommends at least 1 second between requests

address_df['lat'] = latitudes
address_df['lon'] = longitudes

# Merge geocoded data back into main DataFrame
df = df.merge(address_df[['location', 'lat', 'lon']], on='location', how='left')

# Save the output
df.to_csv(OUTPUT_CSV, index=False)
print(f"Geocoded data saved to {OUTPUT_CSV}")

# Preview the result
print(df[['location', 'lat', 'lon']].head(20))