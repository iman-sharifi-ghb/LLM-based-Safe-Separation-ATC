from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from scipy.stats import rv_continuous, rv_discrete
import random
import copy
from geopy.distance import distance


# region: Basic

@dataclass
class Coordinate:
    latitude: float
    longitude: float

    def __eq__(self, other):
        if isinstance(other, Coordinate):
            return self.latitude == other.latitude and self.longitude == other.longitude
        return False
    
    def __str__(self):
        return f'{self.latitude},{self.longitude}'

@dataclass
class Waypoint:
    name: str
    coordinate: Coordinate

    def __eq__(self, other):
        if isinstance(other, Waypoint):
            return self.coordinate.latitude == other.coordinate.latitude and self.coordinate.longitude == other.coordinate.longitude
        return False


@dataclass 
class BaseRouteWaypoint:
    waypoint: Waypoint
    altitude: Union[int, str]
    speed: Optional[int] = None
    end: Optional[bool] = None

@dataclass
class ATCRouteWaypoint(BaseRouteWaypoint):
    end: bool = False

    def __post_init__(self):
        if isinstance(self.altitude, str):
            assert self.altitude[:2] == 'FL', 'altitude must be a flight level'

@dataclass
class DroneRouteWaypoint(BaseRouteWaypoint):
    pass

# @dataclass 
# class DroneServicePoint(BaseRouteWaypoint):
#     stationary_duration: int

class BoundingCircle:
    def __init__(self, corner_1: Union[Coordinate, Tuple[float, float]], corner_2: Union[Coordinate, Tuple[float, float]]):
        self.corner_1 = Coordinate(*corner_1)
        self.corner_2 = Coordinate(*corner_2)

    @cached_property
    def str_coord(self) -> str:
        return f'{self.corner_1.latitude},{self.corner_1.longitude} {self.corner_2.latitude},{self.corner_2.longitude}'

    @cached_property
    def lat_center(self) -> float:
        return (self.corner_1.latitude + self.corner_2.latitude) / 2

    @cached_property
    def lon_center(self) -> float:
        return (self.corner_1.longitude + self.corner_2.longitude) / 2

    @cached_property
    def min_lat(self) -> float:
        return min(self.corner_1.latitude, self.corner_2.latitude)

    @cached_property
    def min_lon(self) -> float:
        return min(self.corner_1.longitude, self.corner_2.longitude)

    @cached_property
    def max_lat(self) -> float:
        return max(self.corner_1.latitude, self.corner_2.latitude)

    @cached_property
    def max_lon(self) -> float:
        return max(self.corner_1.longitude, self.corner_2.longitude)
    
    @cached_property
    def radius_nm(self) -> float:
        center = (self.lat_center, self.lon_center)
        corner = (self.corner_1.latitude, self.corner_1.longitude)

        radius_nm = distance(center, corner).nautical  

        return radius_nm 



# region: Distribution
class StatsDist:
    def __init__(self, rv_dist: Union[rv_continuous, rv_discrete], minimum: float, lower_bound: float, upper_bound: float, scale: float):
        self.rv_dist = rv_dist 
        self.minimum = minimum

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.scale = scale
        self.random_state = None

    def set_random_state(self, np_rng):
        self.random_state = np_rng

    def sample(self):
        rv = self.rv_dist.rvs(random_state=self.random_state) * self.scale

        if self.minimum:
            rv = self.minimum + rv
        rv = max(rv, min(rv, self.upper_bound), self.lower_bound)
        return rv

    
class SpacingDistribution(StatsDist):
    def __init__(self, rv_dist: rv_continuous, minimum: float = 0, lower_bound: float = float('-inf'), upper_bound: float = float('inf'), scale: float = 1):
        super().__init__(rv_dist, minimum, lower_bound, upper_bound, scale)

class SpawnDistribution(StatsDist):
    def __init__(self, rv_dist: rv_discrete, minimum: float = 0, lower_bound: float = float('-inf'), upper_bound: float = float('inf'), scale: float = 1):
        super().__init__(rv_dist, minimum, lower_bound, upper_bound, scale)


class AircraftMetadataDistribution:
    def __init__(self, flight_prefix: Union[list[str], str], aircraft: Union[list[str], str], aircraft_color: Union[list[str], str], flight_prefix_probabilities: Optional[float] = None, aircraft_probabilities: Optional[float] = None):
        self.aircraft = aircraft
        self.flight_prefix = flight_prefix
        self.aircraft_color = aircraft_color
        self.aircraft_probabilities = aircraft_probabilities
        self.flight_prefix_probabilities = flight_prefix_probabilities

        assert type(aircraft) is type(aircraft_probabilities)
        assert type(flight_prefix_probabilities) is type(flight_prefix_probabilities)

        if flight_prefix_probabilities is not None and type(flight_prefix_probabilities) is list:
            assert sum(flight_prefix_probabilities) == 1, "Probabilities must sum to 1"
            assert len(flight_prefix) == len(flight_prefix_probabilities)
            
        if aircraft_probabilities is not None and type(aircraft_probabilities) is list:
            assert sum(aircraft_probabilities) == 1, "Probabilities must sum to 1"
            assert len(aircraft) == len(aircraft_probabilities)

        self.random_state = None

    def set_random_state(self, py_rng):
        self.random_state = py_rng

    def sample(self):
        rng = self.random_state or random
        if type(self.flight_prefix) is list:
            flight_prefix = rng.choices(self.flight_prefix, weights=self.flight_prefix_probabilities)[0]
        else:
            flight_prefix = self.flight_prefix
        if type(self.aircraft) is list:
            aircraft = rng.choices(self.aircraft, weights=self.aircraft_probabilities)[0]
        else:
            aircraft = self.aircraft
        if type(self.aircraft_color) is list:
            aircraft_color = rng.choices(self.aircraft_color)[0]
        else:
            aircraft_color = self.aircraft_color
        return flight_prefix, aircraft, aircraft_color


@dataclass
class BaseRouteDistribution:
    name: str
    spawn: SpawnDistribution
    aircraft_metadata: AircraftMetadataDistribution

@dataclass
class ATCRouteDistribution(BaseRouteDistribution):
    spacing: SpacingDistribution
    direction_forward_probability: float = 1
    path_probabilities: Optional[list[float]] = None

    def __post_init__(self):
        if self.direction_forward_probability is not None:
            assert 0 <= self.direction_forward_probability <= 1, 'TEST FAILED: direction_forward_probability must be between 0 and 1'
        if self.path_probabilities is not None:
            assert sum(self.path_probabilities) == 1, 'TEST FAILED: path_probabilities must sum to 1' 

# @dataclass
# class DroneRouteDistribution:
#     n_destinations: SpawnDistribution

# region: Route

@dataclass
class BaseRoute:
    flight_prefix: List[str]

@dataclass
class ATCRoute(BaseRoute): 
    route_distribution: List[ATCRouteDistribution]
    possible_paths: List[List[ATCRouteWaypoint]]

    def __post_init__(self):
        assert isinstance(self.possible_paths, List) or self.possible_paths is None, 'TEST FAILED: possible_intermediary_waypoints must be a list of list of RouteWaypoint or None'
        for distribution in self.route_distribution:
            assert len(distribution.path_probabilities) == len(self.possible_paths) or not distribution.path_probabilities, 'TEST FAILED: number of path_probabilities must match possible_paths'

        self.random_state: Optional[random.Random] = None

    def set_random_state(self, np_rng, py_rng):
        self.random_state = py_rng
        for dist in self.route_distribution:
            dist.spawn.set_random_state(np_rng)
            dist.spacing.set_random_state(np_rng)
            dist.aircraft_metadata.set_random_state(py_rng)

    def sample_route_distribution(self):
        rng = self.random_state or random
        if isinstance(self.route_distribution, list):
            return rng.choice(self.route_distribution)
        return self.route_distribution

    def sample_path(self, distribution: ATCRouteDistribution):
        rng = self.random_state or random
        if len(self.possible_paths) > 0:
            if distribution.path_probabilities is not None:
                intermediary_waypoints = rng.choices(self.possible_paths, distribution.path_probabilities)[0]
            else:
                intermediary_waypoints = rng.choice(self.possible_paths)
        else:
            intermediary_waypoints = []

        # return [self.end_waypoint_1] + intermediary_waypoints + [self.end_waypoint_2]
        return intermediary_waypoints


# @dataclass 
# class Drone(BaseRoute):
#     route_distribution: List[DroneRouteDistribution]
#     home_base: ATCRouteWaypoint
    
# region: Flight plan

@dataclass
class BaseFlightPlan:
    flight_number: str
    aircraft_type: str
    aircraft_color: str
    entry_time: float
    path: List[BaseRouteWaypoint]


@dataclass
class ATCFlightPlan(BaseFlightPlan):
    path: List[ATCRouteWaypoint]

    def __post_init__(self):
        assert len(self.path) >= 2, 'TEST FAILED: FlightPlan must have at least 2 waypoints'

# @dataclass 
# class DroneFlightPlan(BaseFlightPlan):
#     path: List[Union[DroneRouteWaypoint, Drone]]