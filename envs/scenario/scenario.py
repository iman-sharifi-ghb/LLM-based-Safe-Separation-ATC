import json
import random
from typing import List, Tuple

from .gui import calculate_bounding_circle, generate_route_line_commands
from .parsing import parse_flight_routes, parse_waypoints
from .structure import ATCFlightPlan, ATCRoute
from bluesky import stack

class BaseScenario:

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.bounding_cicle = calculate_bounding_circle(self.waypoints)

    def render_environment(self):
        stack.stack(f'CIRCLE env {self.bounding_cicle.lat_center},{self.bounding_cicle.lon_center},{self.bounding_cicle.radius_nm}')
        stack.stack('AREA env')
        # print the coordinates of the bounding circle center and radius
        # print(f'Bounding circle center: ({self.bounding_cicle.lat_center}, {self.bounding_cicle.lon_center}), radius: {self.bounding_cicle.radius_nm} NM')
    


class ATCScenario(BaseScenario):

    def __init__(self, route_file='routes/matc/routes.json'):
        Root_path = "/Users/imansharifi/Documents/VirtualEnvs/ReinforcementLearning/Code/ATC_LLM/"
        with open(Root_path+route_file, 'r') as f:
            data = json.load(f)
        
        waypoints = parse_waypoints(data.get('waypoints'))
        super().__init__(waypoints)

        self.flight_routes = parse_flight_routes(data.get('routes'), {waypoint.name: waypoint for waypoint in self.waypoints})
        self._line_route_commands = generate_route_line_commands(self.flight_routes)
    
    def sample_schedule(self):
        
        flight_schedule: List[ATCFlightPlan] = []
        n_airline_flights = {}
        
        for route in self.flight_routes:
            distribution = route.sample_route_distribution()
            
            n_spawn = distribution.spawn.sample()       
            prev_spawn_time = 0. # [minutes]

            direction_forward = True # random.random() < distribution.direction_forward_probability
            for _ in range(n_spawn):
                flight_prefix, aircraft, color = distribution.aircraft_metadata.sample()
                path = route.sample_path(distribution)

                path = path if direction_forward else path[::-1]

                n_airline_flights[flight_prefix] = n_airline_flights.get(flight_prefix, 0) + 1

                flight_number = f'{flight_prefix}{n_airline_flights[flight_prefix]:04d}'

                spacing = distribution.spacing.sample()
                # spacing = 100
                spawn_time = prev_spawn_time + spacing

                prev_spawn_time = spawn_time

                flight_plan = ATCFlightPlan(flight_number=flight_number, aircraft_type=aircraft, aircraft_color=color, path=path, entry_time=spawn_time)
                flight_schedule.append(flight_plan)

        for n_flights in n_airline_flights.values():
            assert n_flights <= 9999, 'TEST FAILED: Maximum number of flights exceeded'

        return flight_schedule
    
    # def sample_schedule(self):
    #     flight_schedule: List[ATCFlightPlan] = []
    #     n_airline_flights = {}

    #     # Define the number of aircraft per route
    #     n_spawn_per_route = 30  # Equal number of aircraft for each route

    #     # Define a base spawn time offset for each route
    #     route_spawn_offsets = [100, 150, 200]  # Adjust these offsets to create conflicts at intersections

    #     for route_index, route in enumerate(self.flight_routes):
    #         distribution = route.sample_route_distribution()

    #         prev_spawn_time = route_spawn_offsets[route_index]  # Start with the route-specific offset

    #         direction_forward = True  # Keep direction consistent for simplicity
    #         for _ in range(n_spawn_per_route):
    #             flight_prefix, aircraft = distribution.aircraft_metadata.sample()
    #             path = route.sample_path(distribution)

    #             path = path if direction_forward else path[::-1]

    #             n_airline_flights[flight_prefix] = n_airline_flights.get(flight_prefix, 0) + 1

    #             flight_number = f'{flight_prefix}{n_airline_flights[flight_prefix]:04d}'

    #             # Use the spacing distribution to control spawn intervals
    #             spacing = distribution.spacing.sample()
    #             spawn_time = prev_spawn_time + spacing

    #             prev_spawn_time = spawn_time

    #             flight_plan = ATCFlightPlan(
    #                 flight_number=flight_number,
    #                 aircraft_type=aircraft,
    #                 path=path,
    #                 entry_time=spawn_time
    #             )
    #             flight_schedule.append(flight_plan)

    #     for n_flights in n_airline_flights.values():
    #         assert n_flights <= 9999, 'TEST FAILED: Maximum number of flights exceeded'

    #     return flight_schedule

    def render_environment(self):
        super().render_environment()

        for line_command in self._line_route_commands:
            stack.stack(line_command)


# class DroneScenario:
#     def __init__(self):
        