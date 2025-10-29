
from fractions import Fraction
from typing import Dict, List, Union

from scipy import stats
from .structure import AircraftMetadataDistribution, Coordinate, ATCRoute, ATCRouteDistribution, ATCRouteWaypoint, SpacingDistribution, SpawnDistribution, Waypoint #, SpeedDeviationDistribution
import bluesky as bs

def parse_flight_routes(raw_routes: List[Dict[str, any]], dict_waypoints: Dict[str, Waypoint]) -> List[ATCRoute]:

    def parse_stat_distribution_data(data: Dict[str, Union[float, int]]):
        parsed_data = {'rv_dist': getattr(stats, data['type'])(**data['parameters'])}

        if 'minimum' in data:
            parsed_data.update({'minimum': data.get('minimum')})
        if 'lower_bound' in data:
            parsed_data.update({'lower_bound': data.get('lower_bound')})
        if 'upper_bound' in data:
            parsed_data.update({'upper_bound': data.get('upper_bound')})
        if 'scale' in data:
            parsed_data.update({'scale': data.get('scale')})

        return parsed_data
    
    def parse_aircraft_metadata_distribution_data(flight_prefix, aircraft, aircraft_color, data):
        flight_prefix_probabilities = data.get('flight_prefix_probabilities', None)
        aircraft_probabilities = data.get('aircraft_probabilities', None)

        parsed_data = {
            'flight_prefix': flight_prefix,
            'aircraft': aircraft,
            'aircraft_color': aircraft_color,
            'flight_prefix_probabilities': flight_prefix_probabilities,
            'aircraft_probabilities': aircraft_probabilities
        }

        return parsed_data

    flight_routes = []
    for route in raw_routes:
        flight_prefix = route['flight_prefix']
        # print(route['aircraft_color'])
    
        raw_route_distribution = route.get('distribution')
        raw_route_distribution = raw_route_distribution if isinstance(raw_route_distribution, list) else [raw_route_distribution]  
    
        route_distribution = []    
        for distribution in raw_route_distribution:
            distribution_name = distribution.get('name')
            distribution: Dict[str, Dict]
        
            spawn_distribution = SpawnDistribution(
                **parse_stat_distribution_data(
                    distribution.get('spawn')
                    )
                )
            spacing_distribution = SpacingDistribution(
                **parse_stat_distribution_data(
                    distribution.get('spacing')
                    )
                )
     
            aircraft_metadata_distribution = AircraftMetadataDistribution(
                **parse_aircraft_metadata_distribution_data(
                    route.get('flight_prefix'), 
                    route.get('aircraft'), 
                    route.get('aircraft_color'),
                    distribution
                    )
                )
            direction_forward_probability = distribution.get('direction_forward_probability')
            path_probabilities = (
                [
                    float(Fraction(probability)) if isinstance(probability, str) else probability
                    for probability in distribution.get('path_probabilities')
                ]
                if 'path_probabilities' in distribution else None
            )

            route_distribution.append(ATCRouteDistribution(
                name=distribution_name,
                spawn=spawn_distribution,
                spacing=spacing_distribution,
                aircraft_metadata=aircraft_metadata_distribution,
                direction_forward_probability=direction_forward_probability,
                path_probabilities=path_probabilities
            ))


        raw_end_waypoint_1 = route.get('end_waypoint_1')
        raw_end_waypoint_2 = route.get('end_waypoint_2')

        end_waypoint_1 = ATCRouteWaypoint(
            waypoint=dict_waypoints[raw_end_waypoint_1.get('name')],
            altitude=raw_end_waypoint_1.get('altitude', 0),
            speed=raw_end_waypoint_1.get('speed', 0),
            end=raw_end_waypoint_1.get('end', False)
        )

        end_waypoint_2 = ATCRouteWaypoint(
            waypoint=dict_waypoints[raw_end_waypoint_2['name']],
            altitude=raw_end_waypoint_2.get('altitude', 0),
            speed=raw_end_waypoint_2.get('speed', 0),
            end=raw_end_waypoint_2.get('end', False)
        )

        possible_paths = [
            [
                ATCRouteWaypoint(
                    waypoint=dict_waypoints[waypoint.get("name")], 
                    altitude=waypoint.get('altitude', 0),
                    speed=waypoint.get('speed', 0),
                    end=waypoint.get('end', False)
                ) 
                for waypoint in list(intermediary_waypoints)] for intermediary_waypoints in route.get('possible_paths')
        ]
        route = ATCRoute(
            flight_prefix=flight_prefix,
            route_distribution=route_distribution,
            possible_paths=possible_paths
        )

        flight_routes.append(route)
    return flight_routes



def parse_waypoints(raw_waypoints: list[Dict]):
    parsed_waypoints: list[Waypoint] = []
    set_waypoint_names = set()

    for i, waypoint in enumerate(raw_waypoints, 1):
        wp_name = waypoint.get("name", None)
        wp_type = waypoint.get("type", None)
        latitude = waypoint.get('latitude', None)
        longitude = waypoint.get('longitude', None)
        
        if wp_name:
            assert (wp_id := bs.navdb.getwpidx(wp_name)) == -1, f"Waypoint {wp_name} already defined in Bluesky database; id: {wp_id},"
            assert wp_name not in set_waypoint_names, f"Waypoint {wp_name} already defined"

    
        if 'latitude' not in waypoint and 'longitude' not in waypoint:
            wp_idx = bs.navdb.getwpidx(wp_name)
            if wp_idx == -1:
                raise ValueError(f'Waypoint {wp_name} not found')
            latitude, longitude = bs.navdb.wplat[wp_idx].item(), bs.navdb.wplon[wp_idx].item()
        else:
            latitude, longitude = waypoint.get('latitude'), waypoint.get('longitude')
        
        assert latitude and longitude, f'Waypoint {i} invalid'

        parsed_waypoints.append(Waypoint(
                name=wp_name, 
                type=wp_type,
                coordinate=Coordinate(latitude=latitude, longitude=longitude)
            )
        )
        set_waypoint_names.add(wp_name)

    return parsed_waypoints