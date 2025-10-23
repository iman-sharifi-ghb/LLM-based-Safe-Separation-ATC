
import functools
from typing import List

from .structure import BoundingCircle, Waypoint, ATCRoute
from bluesky import stack

def calculate_bounding_circle(waypoints: list[Waypoint]) -> BoundingCircle:
    min_lat = float('inf')
    min_lon = float('inf')
    max_lat = float('-inf')
    max_lon = float('-inf')
    

    for waypoint in waypoints:
        latitude = waypoint.coordinate.latitude
        longitude = waypoint.coordinate.longitude

        min_lat = min(min_lat, latitude)
        min_lon = min(min_lon, longitude)
        max_lat = max(max_lat, latitude)
        max_lon = max(max_lon, longitude)

    BB_PADDING = 0.001
    bounding_circle = BoundingCircle(
        (min_lat - BB_PADDING, min_lon - BB_PADDING),
        (max_lat + BB_PADDING, max_lon + BB_PADDING)
    )
    return bounding_circle


def generate_route_line_commands(flight_routes: List[ATCRoute]):
    commands = []
    for route in flight_routes:
        # Build the full route (endpoints and intermediary waypoints)
        for path in route.possible_paths:
            # Prepend the starting endpoint and append the ending endpoint
            # full_path = [route.end_waypoint_1] + [rw for rw in path] + [route.end_waypoint_2]
            full_path = [rw for rw in path]
            # Draw lines between consecutive waypoints in the full path
            for i in range(len(full_path) - 1):
                wp1 = full_path[i].waypoint if hasattr(full_path[i], "waypoint") else full_path[i]
                wp2 = full_path[i + 1].waypoint if hasattr(full_path[i + 1], "waypoint") else full_path[i + 1]
                commands.append(f"LINE {wp1.name}-{wp2.name},{wp1.coordinate.latitude},{wp1.coordinate.longitude},{wp2.coordinate.latitude},{wp2.coordinate.longitude}")
    return commands