import os
import math, random, json

# ========================
# PARAMETERS
# ========================
CENTER_LAT, CENTER_LON = 33.15516, -96.869125
RADIUS_NM = 3.99
RADIUS_M = RADIUS_NM * 1852.0

BASE_WAYPOINTS = {
    "WP1": (33.20470, -96.91000),
    "WP2": (33.17895, -96.91973),
    "WP3": (33.18389, -96.88552),  # merge point (fixed)
    "WP4": (33.15826, -96.81852),
    "WP5": (33.1211,  -96.91112),
    "WP6": (33.10562, -96.88237),
    "WP7": (33.13934, -96.87712),  # merge point (fixed)
    "WP8": (33.18607, -96.83017),
    "WP9": (33.16731, -96.84644)   # intersection (fixed)
}

# Utility: geographic offset
def destination_point(lat, lon, bearing_deg, distance_m):
    R = 6371000.0
    brg = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(distance_m/R) +
                     math.cos(lat1)*math.sin(distance_m/R)*math.cos(brg))
    lon2 = lon1 + math.atan2(math.sin(brg)*math.sin(distance_m/R)*math.cos(lat1),
                             math.cos(distance_m/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def initial_bearing(lat1, lon1, lat2, lon2):
    # Calculate initial bearing from point 1 to point 2
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    diff_lon_rad = math.radians(lon2 - lon1)
    x = math.sin(diff_lon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(diff_lon_rad)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def distance_m(lat1, lon1, lat2, lon2):
    # Haversine distance in meters
    R = 6371000.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ========================
# HELPER FUNCTION TO ROTATE POINT AROUND CENTER
# ========================
def rotate_point_around_center(lat, lon, center_lat, center_lon, angle_deg):
    # Compute distance and bearing from center to point
    dist = distance_m(center_lat, center_lon, lat, lon)
    brg = initial_bearing(center_lat, center_lon, lat, lon)
    # Add rotation angle
    new_brg = (brg + angle_deg) % 360
    # Compute new point from center with rotated bearing
    return destination_point(center_lat, center_lon, new_brg, dist)

# ========================
# GENERATE NEW WAYPOINT POSITIONS
# ========================
def randomize_waypoints(
    base_wps,
    fixed_points=("WP3", "WP7", "WP9"),
    radius_m=RADIUS_M,
    apply_global_rotation=True,
    apply_global_scaling=True,
    scale_range=(0.9, 1.1)
):
    """
    Randomize waypoints by moving or rotating non-fixed waypoints relative to their reference points WP3, WP7, WP9.
    Fixed points remain unchanged.
    """
    if apply_global_rotation:
        rotation_angle = random.uniform(-120, 120)
        rotated_wps = {}
        for name, (lat, lon) in base_wps.items():
            rlat, rlon = rotate_point_around_center(lat, lon, CENTER_LAT, CENTER_LON, rotation_angle)
            rotated_wps[name] = (rlat, rlon)
        base_wps = rotated_wps
        print(f"Applied global rotation of {rotation_angle:.2f} degrees to base waypoints.")

    # Optional: apply global scaling (uniform radial expansion/contraction from center)
    if apply_global_scaling:
        scale_factor = random.uniform(*scale_range)
        scaled_wps = {}
        for name, (lat, lon) in base_wps.items():
            dist = distance_m(CENTER_LAT, CENTER_LON, lat, lon)
            brg = initial_bearing(CENTER_LAT, CENTER_LON, lat, lon)
            new_dist = dist * scale_factor
            # Keep inside bounding circle
            if new_dist > radius_m:
                new_dist = radius_m * 0.98
            slat, slon = destination_point(CENTER_LAT, CENTER_LON, brg, new_dist)
            scaled_wps[name] = (slat, slon)
        base_wps = scaled_wps
        print(f"Applied global scaling factor of {scale_factor:.3f} to base waypoints.")

    new_wps = {}
    # Reference points
    ref_points = {
        "WP1": "WP3",
        "WP2": "WP3",
        "WP5": "WP7",
        "WP6": "WP7",
        "WP4": "WP9",
        "WP8": "WP9"
    }
    for name, (lat, lon) in base_wps.items():
        if name in fixed_points:
            new_wps[name] = {"latitude": lat, "longitude": lon}
        else:
            ref_name = ref_points.get(name)
            ref_lat, ref_lon = base_wps[ref_name]
            # Compute initial bearing and distance from reference point to original waypoint
            brg = initial_bearing(ref_lat, ref_lon, lat, lon)
            dist = distance_m(ref_lat, ref_lon, lat, lon)
            # Perturb bearing by ±20 degrees
            brg_new = (brg + random.uniform(-20, 20)) % 360
            # Vary distance by ±10%
            dist_new = dist * random.uniform(0.9, 1.1)
            # Compute new position from reference point
            nlat, nlon = destination_point(ref_lat, ref_lon, brg_new, dist_new)
            # Check if new position is outside the global bounding circle
            dist_from_center = distance_m(CENTER_LAT, CENTER_LON, nlat, nlon)
            if dist_from_center > radius_m:
                # Compute bearing from center to new point
                center_to_new_brg = initial_bearing(CENTER_LAT, CENTER_LON, nlat, nlon)
                # Place at min(original_distance_from_center * 1.1, RADIUS_M * 0.98) along that bearing from center
                original_dist_center = distance_m(CENTER_LAT, CENTER_LON, lat, lon)
                proj_dist = min(original_dist_center * 1.1, radius_m * 0.98)
                nlat, nlon = destination_point(CENTER_LAT, CENTER_LON, center_to_new_brg, proj_dist)
            new_wps[name] = {"latitude": round(nlat, 5), "longitude": round(nlon, 5)}
    return new_wps

# ========================
# ROUTE GENERATOR
# ========================
def make_path(names, alt="FL150", spd=25):
    return [{"name": n, "speed": spd, "altitude": alt, "end": False} for n in names]

def make_distribution(spawn=5):
    return [{
        "name": "Monday",
        "spawn": {"type": "rv_discrete", "parameters": {"values": [[spawn], [1]]}},
        "spacing": {
            "type": "rv_discrete",
            "parameters": {"values": [[40,45,50,55,60,65,70,75,80,85],[0.1]*10]}
        },
        "direction_forward_probability": 1,
        "flight_prefix_probabilities": [1],
        "aircraft_probabilities": [1],
        "path_probabilities": ["1"]
    }]

def build_routes(wps_dict, add_new=False):
    routes = []

    # existing sequences
    base_sequences = [
        ("AA", "Amzn", ["WP1","WP3","WP9","WP4"]),
        ("AB", "Xwing", ["WP2","WP3","WP9","WP4"]),
        ("AC", "Amzn", ["WP5","WP7","WP9","WP8"]),
        ("AD", "Xwing", ["WP6","WP7","WP9","WP8"])
    ]

    for prefix, aircraft, seq in base_sequences:
        route = {
            "flight_prefix": [prefix],
            "aircraft": [aircraft],
            "distribution": make_distribution(),
            "end_waypoint_1": {"name": seq[-1], "speed": 25, "end": False},
            "end_waypoint_2": {"name": seq[0], "speed": 25, "end": False},
            "possible_paths": [make_path(seq)]
        }
        routes.append(route)

    # Optionally add a new route WP10 → WP3 → WP9 → WP4
    if add_new:
        # Use WP3 as reference for bearings and distances of WP1 and WP2
        wp1 = wps_dict["WP1"]
        wp2 = wps_dict["WP2"]
        wp3 = wps_dict["WP3"]
        brg1 = initial_bearing(wp3["latitude"], wp3["longitude"], wp1["latitude"], wp1["longitude"])
        brg2 = initial_bearing(wp3["latitude"], wp3["longitude"], wp2["latitude"], wp2["longitude"])
        # Ensure the sector is in the correct direction
        # Find the smallest positive angular sector between brg1 and brg2
        def sector_bounds(b1, b2):
            # Returns (min_brg, max_brg) such that going from min_brg to max_brg is the smaller angle
            diff = (b2 - b1) % 360
            if diff <= 180:
                return b1, b2
            else:
                return b2, b1
        min_brg, max_brg = sector_bounds(brg1, brg2)
        # Pick random bearing within the sector
        if min_brg <= max_brg:
            rand_brg = random.uniform(min_brg, max_brg)
        else:
            # Sector wraps around 360
            span = (max_brg + 360) - min_brg
            rand_brg = (min_brg + random.uniform(0, span)) % 360
        # Distance: average of WP1 and WP2 from WP3, with small random variation (±5%)
        dist1 = distance_m(wp3["latitude"], wp3["longitude"], wp1["latitude"], wp1["longitude"])
        dist2 = distance_m(wp3["latitude"], wp3["longitude"], wp2["latitude"], wp2["longitude"])
        avg_dist = (dist1 + dist2) / 2.0
        rand_dist = avg_dist * random.uniform(0.95, 1.05)
        lat, lon = destination_point(wp3["latitude"], wp3["longitude"], rand_brg, rand_dist)
        wps_dict["WP10"] = {"latitude": round(lat,5), "longitude": round(lon,5)}
        new_route = {
            "flight_prefix": ["AE"],
            "aircraft": ["Amzn"],
            "distribution": make_distribution(),
            "end_waypoint_1": {"name": "WP4", "speed": 25, "end": False},
            "end_waypoint_2": {"name": "WP10", "speed": 25, "end": False},
            "possible_paths": [make_path(["WP10","WP3","WP9","WP4"])]
        }
        routes.append(new_route)

    return routes, wps_dict

# ========================
# MAIN DRIVER
# ========================

def generate_airspace_config(out_dir="routes/matc", add_new=True, scale_range=(0.75, 1.1)):
    """
    Generate a randomized airspace configuration and write it to a JSON file.
    Returns the path of the generated JSON file.
    """
    waypoints = randomize_waypoints(
        BASE_WAYPOINTS,
        apply_global_rotation=True,
        apply_global_scaling=True,
        scale_range=scale_range
    )
    routes, wps_final = build_routes(waypoints, add_new=add_new)

    config = {
        "meta": {
            "description": "Generated airspace configuration with fixed merge/intersection points.",
            "center": [CENTER_LAT, CENTER_LON],
            "radius_nm": RADIUS_NM
        },
        "waypoints": [{"name": k, **v} for k,v in wps_final.items()],
        "routes": routes
    }

    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    outpath = os.path.join(out_dir, "auto_airspace_config.json")
    with open(outpath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {outpath}")
    return outpath


if __name__ == "__main__":
    generate_airspace_config()