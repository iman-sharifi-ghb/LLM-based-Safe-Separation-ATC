from math import floor
from typing import Optional
import numpy as np

SECONDS_IN_A_DAY = 86400
EARTH_RADIUS_NM = 3440 # Earth's radius in nautical miles

def haversine_distance(lat1, lon1, lat2, lon2, radius=6371e3):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = radius * c
    return distance/1852 # Convert to nautical miles

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def degto180(deg):
    """Converts degree to -180 to 180 range."""
    return (deg + 180) % 360 - 180

def time_fraction_to_seconds(time_fraction):
    # Convert to total seconds
    return np.round(time_fraction * SECONDS_IN_A_DAY)

def seconds_to_hms(seconds):
    hour = seconds // 3600  # Get whole hours
    minute = (seconds % 3600) // 60  # Get whole minutes
    second = seconds % 60
    return int(hour), int(minute), int(second)


def metric_spd_to_knots_rounded(speed):
    return np.round(speed * 1.94384449)  # Convert m/s to knots

def meters_to_feet_rounded(meters): 
    return np.round(meters / 0.3048)


def calculate_heading(coordinate1, coordinate2):
    # Convert latitudes and longitudes from degrees to radians

    lat1 = np.radians(coordinate1.latitude)
    lon1 = np.radians(coordinate1.longitude)
    lat2 = np.radians(coordinate2.latitude)
    lon2 = np.radians(coordinate2.longitude)
    
    # Calculate the differences in coordinates
    dlon = lon2 - lon1
    
    # Apply the formula to calculate the initial bearing
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.atan2(x, y)
    
    # Convert the bearing from radians to degrees
    initial_bearing = np.degrees(initial_bearing)
    
    # Normalize the bearing to a 0-360 range
    compass_bearing = (initial_bearing + 360) % 360
    
    return compass_bearing