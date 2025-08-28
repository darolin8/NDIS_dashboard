import numpy as np
import pandas as pd
import re

def add_lat_lon(df):
    # Define a simple lookup for repeated place names
    location_coords = {
        "Group Home": (-33.8707, 151.2069),
        "Transport Vehicle": (-33.8688, 151.2093),
        "Day Program": (-33.8710, 151.2000),
        "Community Access": (-33.8730, 151.2050),
        "Therapy Clinic": (-33.8720, 151.2070),
        "Care Home 1": (-33.8670, 151.2100),
        "Care Home 2": (-33.8655, 151.2130),
        "Care Home 3": (-33.8620, 151.2170),
        "Care Home 4": (-33.8610, 151.2150),
        "Care Home 5": (-33.8640, 151.2190),
    }
    def get_coords(loc):
        if loc in location_coords:
            return location_coords[loc]
        # For addresses like "123 George St, Newtown NSW 2042"
        if re.search(r'\d+ [\w ]+, [\w ]+ NSW \d+', str(loc)):
            # Sydney bounding box
            lat = np.random.uniform(-33.95, -33.84)
            lon = np.random.uniform(151.10, 151.26)
            return (lat, lon)
        # Default fallback
        return (np.nan, np.nan)
    coords = df['location'].apply(get_coords)
    df['lat'] = coords.apply(lambda x: x[0])
    df['lon'] = coords.apply(lambda x: x[1])
    return df