from enum import Enum
from typing import  Dict, List, Literal, Optional, TypedDict, Union
import numpy as np
import bluesky as bs
from pettingzoo import ParallelEnv
from gymnasium.utils import seeding
from gymnasium import spaces
import functools
from geopy import distance
from bluesky.traffic.performance import openap
from bluesky import stack
from bluesky.tools import geo
import random
import json

from .scenario.utils import calculate_heading, metric_spd_to_knots_rounded
from .scenario.scenario import ATCScenario
from .scenario.utils import haversine_distance, normalize

DELTA_T = 3.0
ZOOM = 10.0
NOISE = False
SAVE_SCENARIO = False

MAX_MISSION_TIME_COA = 18 * 60 # seconds
MAX_MISSION_TIME_COB = 23 * 60 # seconds

# Company 1: Amazon
# Company 2: Google Wing 
ACC_RANGE = 1.0

# North is 0 degrees, east is 90 degrees, south is 180 degrees, and west is -90 degrees
HDG_RANGE = 180.0

MAX_NUM_AGENTS = 20
NUM_INTRUDERS = 3
NUM_ROUTES = 4
GOAL_RADIUS = 50 * 0.000539957 # 0.5 nautical miles

STATE_SIZE = 6
INTRUDER_STATE_SIZE = 5
ACTION_SIZE = 3

# Weights for the rewards
W_LOS_REW = 1.0
W_VEL_REW = 0.001
W_ACT_REW = 0.0001
W_TIME_REW = 1.0
W_MISSION_REW = 0.1

OFFSET_TIME = 55.00

class MATCEnv(ParallelEnv):
    metadata = {"render_modes": ['sim', 'server', 'export_scenario']}

    def __init__(self, flight_route_file: str, aircraft_config_file: str, render_mode: Literal['sim', 'export_scenario', 'server'] = 'sim'):

        self.render_mode = render_mode
        self.noise = NOISE
        self.dt = DELTA_T
        self.zoom = ZOOM

        # Initialize bluesky
        bs.init(mode='sim', detached=(self.render_mode != 'server'))
        if self.render_mode == 'server':
            print('Waiting to connect to server...')
            bs.net.connect() 
            bs.net.update() 
            print('Connected to server!')

        # Initialize the scenario
        self.scenario = ATCScenario(flight_route_file)
        with open(aircraft_config_file, 'r') as f:
            self.ac_configs = json.load(f)

        merging_waypoint1 = "WP3"
        merging_waypoint2 = "WP7"
        intersection_waypoint = "WP9"
        # Import waypoints from the scenario
        self.regular_wpts = []
        self.merging_wpts = []
        self.intersection_wpts = []
        for waypoint in self.scenario.waypoints:
            if waypoint.name == merging_waypoint1:
                self.merging_waypoint1_location = (waypoint.coordinate.latitude, waypoint.coordinate.longitude)
            if waypoint.name == merging_waypoint2:
                self.merging_waypoint2_location = (waypoint.coordinate.latitude, waypoint.coordinate.longitude)
            if waypoint.name == intersection_waypoint:
                self.intersection_waypoint_location = (waypoint.coordinate.latitude, waypoint.coordinate.longitude)

            if waypoint.type == 'regular':
                self.regular_wpts.append(waypoint.name)
            if waypoint.type == 'merging':
                self.merging_wpts.append(waypoint.name)
            if waypoint.type == 'intersection':
                self.intersection_wpts.append(waypoint.name)

        # self.merging_wpts = [merging_waypoint1, merging_waypoint2]
        # self.intersection_wpts = [intersection_waypoint]
        # call _seed to set the random seed
        # self._seed(seed=42)

        # Take a sample schedule from the scenario
        self.schedule = self.scenario.sample_schedule()

        self.time_step = 0
        self.lat_lon_accs = {}

        self.min_acc = -ACC_RANGE
        self.max_acc = ACC_RANGE

        self.distances_to_ownship = {}
        self.actions = {}
        self.agent_counter = {}
        self.truncated_agents = {}
        self.n_spawn = MAX_NUM_AGENTS
        self.all_possible_agents = []
        self.collided_agents = []
        self.each_agent_rewards = {}
        self.previous_actions = {}
        self.terminated_agents = {}
        self.agent_route_ids = {}
        self.previous_speeds = {}
        self.current_speeds = {}
        self.agent_goal_locations = {}
        self.sum_episode_rewards = 0
        self.dones = {}
        self.agent_phases = {}
        self.unsafe_agents = {}
        self.term_agent_list = []  
        self.trunc_agent_list = []
        self.episode_spds = {}

        self.collision_counter = {
            'All': 0,
            'AA': 0,
            'AB': 0,
            'BB': 0,
            'M1': 0,
            'M2': 0,
            'In': 0
        }

        self.action_counter_A = {
            'decelerate': 0,
            'hold': 0,
            'accelerate': 0
        }
        self.action_counter_B = {
            'decelerate': 0,
            'hold': 0,
            'accelerate': 0
        }

        self.cmd_history = []
        self.N_successful_agents = MAX_NUM_AGENTS

        self.rule_based_dataset = []
        self.all_state_action_pairs = []

    def _stack(self, cmd: str, log: bool = False):
        # if log:
        #     print(cmd)
        stack.stack(cmd)
        if self.render_mode == 'export_scenario':
            timestamp = bs.sim.utc.strftime('%H:%M:%S.%f')[:-4]  # type: ignore
            self.cmd_history.append(f'{timestamp}>{cmd}')

    def _agent_has_terminated(self, agent):
        return bs.traf.id2idx(agent) == -1

    def _create_ac(self, ac_id: Union[str, int], ac_type: str, ac_lat: float, ac_lon: float, ac_hdg: int, ac_alt: int, ac_spd: int):
        stack.stack(f'CRE {ac_id} {ac_type} {ac_lat},{ac_lon} {ac_hdg} {ac_alt} {ac_spd}')

    def _sim_step(self):
        bs.sim.step()  # type: ignore
        self.render()

    def render(self):
        bs.net.update()  # type: ignore

    def close(self):
        pass

    def _seed(self, seed: Optional[int] = None):
        if seed is not None:
            assert 0 <= seed <= 2**32 - 1, 'TEST FAILED: Seed must be between 0 and 2^32 - 1'
            self.np_random, seed = seeding.np_random(seed)
            random.seed(seed)
        else:
            seed = np.random.randint(0, 2**32)
            self.np_random, seed = seeding.np_random(seed)
            random.seed(seed)

        # Create local RNGs
        np_rng = np.random.RandomState(seed)
        py_rng = random.Random(seed)

        # Set random state for all routes and their distributions
        for route in self.scenario.flight_routes:
            route.set_random_state(np_rng, py_rng)  # This will propagate to all distributions

        # Enable BlueSky's internal seeding for full reproducibility
        stack.stack(f'SEED {seed}')
    
    def reset(self, seed=None):

        self._seed(seed)
        
        # Hard reset BlueSky simulation state for reproducibility
        stack.stack('RESET')
        
        # Update the scenario in each episode
        self.schedule = self.scenario.sample_schedule()

        # make sure there is no agent in the simulation
        if len(self.agents) > 0:
            # Remove all agents from the simulation
            print(f"Removing {len(self.agents)} agents from the simulation.")
            for agent in self.agents:
                stack.stack(f'DEL {agent}')
            self._sim_step()

        stack.stack('RESET')

        for waypoint in self.scenario.waypoints:
            stack.stack(f'DEFWPT {waypoint.name},{waypoint.coordinate}')

        stack.stack(f'PAN {self.scenario.bounding_cicle.lat_center},{self.scenario.bounding_cicle.lon_center}')
        stack.stack(f'ZOOM {self.zoom}')

        self.scenario.render_environment()

        # stack.stack('CDMET OD STATEBASED')
        stack.stack('CDMETHOD OFF')
        stack.stack(f"NOISE {'ON' if self.noise else 'OFF'}")

        stack.stack(f'DT {self.dt}')
        stack.stack('OP')

        self.agent_init_wpts = {}
        self.agent_route_ids = {}
        for flight in self.schedule:
            origin = flight.path[0]
            destination = flight.path[-1]
            initial_heading = calculate_heading(origin.waypoint.coordinate, destination.waypoint.coordinate)
            initial_speed = origin.speed 
            
            stack.stack(f'DELAY {flight.entry_time} CRE {flight.flight_number} {flight.aircraft_type} {origin.waypoint.coordinate} {initial_heading} {origin.altitude} {initial_speed if not origin.end else 60}')

            stack.stack(f'DELAY {flight.entry_time} COLOR {flight.flight_number} {flight.aircraft_color}')

            if origin.end:
                stack.stack(f'DELAY {flight.entry_time} ORIG {flight.flight_number} {origin.waypoint.name}')

            for wpt in flight.path[0:-1]:
                stack.stack(f'DELAY {flight.entry_time} ADDWPT {flight.flight_number} {wpt.waypoint.name} {wpt.altitude},{wpt.speed}')
            
            # Add the last waypoint as a destination
            if destination.end:
                stack.stack(f'DELAY {flight.entry_time} DEST {flight.flight_number} {destination.waypoint.name}')
            else:
                stack.stack(f'DELAY {flight.entry_time} ADDWPT {flight.flight_number} {destination.waypoint.name} {destination.altitude},{destination.speed}')

            stack.stack(f'DELAY {flight.entry_time} {flight.flight_number} AT {destination.waypoint.name} STACK DEL {flight.flight_number}')

            stack.stack(f'DELAY {flight.entry_time} LNAV {flight.flight_number} ON')
            stack.stack(f'DELAY {flight.entry_time} VNAV {flight.flight_number} ON')

            # Get initial waypoints
            self.agent_init_wpts[flight.flight_number] = flight.path[0].waypoint.name
            self.agent_route_ids[flight.flight_number] = self.get_route_id(flight.flight_number)  
            # print(f"Flight {flight.flight_number} initial waypoint: {self.agent_init_wpts[flight.flight_number]}")
            
            # Convert destination waypoint coordinates from string of numbers to tuple of floats
            destination_coordinates = (destination.waypoint.coordinate.latitude, destination.waypoint.coordinate.longitude)
            # Store the destination coordinates in the agent_goal_locations dictionary
            self.agent_goal_locations[flight.flight_number] = destination_coordinates

            # draw a circle around waypoints
            # waypoint #3
            # aux_dist = 3000 * 0.000539957
            # self.draw_circle(name='C1', lat=33.18389, lon=-96.88552, radius=self.ownship_collision_dist)
            # self.draw_circle(name='C2', lat=33.18389, lon=-96.88552, radius=aux_dist)

            # waypoint #7
            # aux_dist = 3000 * 0.000539957
            # self.draw_circle(name='C3', lat=33.13934, lon=-96.87712, radius=self.ownship_collision_dist)
            # self.draw_circle(name='C4', lat=33.13934, lon=-96.87712, radius=aux_dist)

            # waypoint #9
            # self.draw_circle(name='C5', lat=33.16198, lon=-96.84680, radius=self.ownship_collision_dist)
            # self.draw_circle(name='C6', lat=33.16198, lon=-96.84680, radius=self.safety_distance)

            # stack.stack('DTMULT 0.1')

        # self.agent_route_ids = {}
        # # Convert initial waypoints to route IDs for each agent
        # for agent, init_wpt in self.agent_init_wpts.items():
        #     if init_wpt == "WP1":
        #         route_id = 1
        #     elif init_wpt == "WP2":
        #         route_id = 2
        #     elif init_wpt == "WP5":
        #         route_id = 3
        #     elif init_wpt == "WP6":
        #         route_id = 4
        #     else:
        #         route_id = -1
        #     self.agent_route_ids[agent] = route_id
        
        # One step forward in the simulation
        while len(self.agents) < 1:
            self._sim_step()

        current_step_agents = self.agents
        observations = self.observations(current_step_agents)
        processed_observations = self.process_agents(observations)

        # Reset the parameters
        self.agent_counter = {}
        self.truncated_agents = {}
        self.each_agent_rewards = {}
        self.terminated_agents = {}
        self.previous_actions = {}
        self.sum_episode_rewards = 0
        self.collided_agents = []
        self.agent_phases = {}
        self.unsafe_agents = {}
        self.dones = {}
        self.all_possible_agents = []
        self.episode_spds = {}
        for flight in self.schedule:
            # Initialize the agent counter
            self.all_possible_agents.append(flight.flight_number)
            self.episode_spds[flight.flight_number] = []
            self.agent_counter[flight.flight_number] = 0
            self.truncated_agents[flight.flight_number] = False
            self.each_agent_rewards[flight.flight_number] = []
            self.terminated_agents[flight.flight_number] = False
            self.previous_actions[flight.flight_number] = 1
            self.dones[flight.flight_number] = False

        self.collision_counter = {
            'All': 0,
            'AA': 0,
            'AB': 0,
            'BB': 0,
            'M1': 0,
            'M2': 0,
            'In': 0
        }
        self.action_counter_A = {
            'decelerate': 0,
            'hold': 0,
            'accelerate': 0
        }
        self.action_counter_B = {
            'decelerate': 0,
            'hold': 0,
            'accelerate': 0
        }

        # Start scenario recording
        self.gather_cmd_history()

        return observations, processed_observations
    
    def gather_cmd_history(self):
        
        self.cmd_history = []
        # Append commands to the command history
        self.cmd_history.append("00:00:00.00>CDMETHOD OFF")
        self.cmd_history.append("00:00:00.00>TRAILS ON")
        self.cmd_history.append("00:00:00.00>TAXI OFF 4")
        self.cmd_history.append("00:00:00.00>PAN 33.16731, -96.84644")
        self.cmd_history.append(f"00:00:00.00>ZOOM 12")

        agent_command_times = {}
        for flight in self.schedule:
            # Initialize the agent command times
            agent_command_times[flight.flight_number] = -OFFSET_TIME

        self.agent_init_wpts = {}
        for flight in self.schedule:
            origin = flight.path[0]
            destination = flight.path[-1]
            initial_heading = calculate_heading(origin.waypoint.coordinate, destination.waypoint.coordinate)
            initial_speed = origin.speed 

            # Update agent command times
            agent_command_times[flight.flight_number] += flight.entry_time
            self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>CRE {flight.flight_number} {flight.aircraft_type} {origin.waypoint.coordinate} {initial_heading} {origin.altitude} {initial_speed if not origin.end else 60}')    
            if origin.end:
                self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>ORIG {flight.flight_number} {origin.waypoint.name}')  
            for wpt in flight.path[0:-1]:
                self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>ADDWPT {flight.flight_number} {wpt.waypoint.name} {wpt.altitude},{wpt.speed}')
        
            if destination.end:
                self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>DEST {flight.flight_number} {destination.waypoint.name}')     
            else:
                self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>ADDWPT {flight.flight_number} {destination.waypoint.name} {destination.altitude},{destination.speed}')
            self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>{flight.flight_number} AT {destination.waypoint.name} STACK DEL {flight.flight_number}')

            self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>LNAV {flight.flight_number} ON')      
            self.cmd_history.append(f'00:00:{agent_command_times[flight.flight_number]:.2f}>VNAV {flight.flight_number} ON')
            self.cmd_history.append('\n')

    def record_scenario(self, scenario_name='frisco.scn'):
        """
        Record the scenario to a file.
        """ 
        with open(scenario_name, 'w') as f:
            for cmd in self.cmd_history:
                f.write(cmd + '\n')
        f.close()
        print(f'Scenario recorded to {scenario_name}')

    @property
    def agents(self):
        return bs.traf.id

    def get_operation_phase_agents(self, observations):
        # We have in general 3 phases: before the merging waypoint, between the merging waypoint and the intersection waypoint, and after the intersection waypoint
        agent_phases = {}
        for agent in observations:
            # if agent longitude is less than the merging waypoint longitude, then the aircraft is in the first phase
            if observations[agent][0][1] < self.merging_waypoint1_location[1]:
                phase = 0
            # if agent longitude is between the merging waypoint and the intersection waypoint, then the aircraft is in the second phase
            elif (observations[agent][0][0] - self.intersection_waypoint_location[0]) > (observations[agent][0][1] - self.intersection_waypoint_location[1]):
                phase = 1
            # if agent longitude is after the intersection waypoint, then the aircraft is in the third phase
            else:
                phase = 2
            agent_phases[agent] = phase 
        return agent_phases
    
    def get_operation_phase_agent(self, agent):
        # We have in general 3 phases: before the merging waypoint, between the merging waypoint and the intersection waypoint, and after the intersection waypoint
        # find the latitude and longitude of the aircraft
        ownship_ac_idx = bs.traf.id2idx(agent)
        if ownship_ac_idx == -1:
            return -1
        agent_lat = bs.traf.lat[ownship_ac_idx]
        agent_lon = bs.traf.lon[ownship_ac_idx]

        if agent_lon < self.merging_waypoint1_location[1]:
            phase = 0
        # if agent longitude is between the merging waypoint and the intersection waypoint, then the aircraft is in the second phase
        elif (agent_lat - self.intersection_waypoint_location[0]) > (agent_lon - self.intersection_waypoint_location[1]):
            phase = 1
        # if agent longitude is after the intersection waypoint, then the aircraft is in the third phase
        else:
            phase = 2
        return phase
    
    def entering_dangerous_area(self, agent):
        ownship_ac_idx = bs.traf.id2idx(agent)
        if ownship_ac_idx == -1:
            return -1
        # Check the operation phase for each agent first
        agent_phase = self.get_operation_phase_agent(agent)

        # find the distance to the merging waypoint
        ownship_lat = bs.traf.lat[ownship_ac_idx]
        ownship_lon = bs.traf.lon[ownship_ac_idx]
        dist_to_merging_waypoint = haversine_distance(ownship_lat, ownship_lon, self.merging_waypoint1_location[0], self.merging_waypoint1_location[1])
        dist_to_intersection_waypoint = haversine_distance(ownship_lat, ownship_lon, self.intersection_waypoint_location[0], self.intersection_waypoint_location[1])

        # Check if the aircraft is entering the dangerous area
        # If the aircraft is in the first phase and the distance to the merging waypoint is less than the safety distance, then the aircraft is entering the dangerous area
        # If the aircraft is in the second phase and the distance to the intersection waypoint is less than the safety distance, then the aircraft is entering the dangerous area
        # If the aircraft is in the third phase, then the aircraft is not entering the dangerous area
        if agent_phase == 0 and dist_to_merging_waypoint < self.get_agent_safety_distance(agent):
            flag = 1
        elif agent_phase == 1 and dist_to_intersection_waypoint < self.get_agent_safety_distance(agent):
            flag = 1
        else:
            flag = 0
        
        # print(f"Agent: {agent}, Phase: {agent_phase}, Distance to merging waypoint: {dist_to_merging_waypoint}, Distance to intersection waypoint: {dist_to_intersection_waypoint}, Dangerous zone flag: {flag}")
        return flag

    def check_termination(self, agent):
        if self._agent_has_terminated(agent):
            self.terminated_agents[agent] = True
            terminated = True
        else:
            terminated = False
        return terminated

    def terminate_agents(self):
        for agent in self.terminated_agents.keys():
            if self.terminated_agents[agent]:
                stack.stack(f'DEL {agent}')
                if agent not in self.term_agent_list:
                    self.cmd_history.append(f'00:00:{self.get_agent_time(agent):.2f}>DEL {agent}')

    def truncate_agents(self):
        for agent in self.truncated_agents.keys():
            if self.truncated_agents[agent]:
                stack.stack(f'DEL {agent}')
                if agent not in self.trunc_agent_list:
                    self.cmd_history.append(f'00:00:{self.get_agent_time(agent):.2f}>DEL {agent}')

    def check_done(self, agent):
        if self.truncated_agents[agent] or self.terminated_agents[agent]:
            return True
        else:
            return False

    def check_done_agents(self, current_step_agents):
        for agent in current_step_agents:
            if self.check_done(agent):
                self.dones[agent] = True

    def update_observations(self, observations, current_step_agents):
        for agent in current_step_agents:
            if agent in self.truncated_agents.keys():
                del observations[agent]
        return observations
    
    def get_agent_time(self, agent):
        # Extract the agent counter
        # agent_cnt = self.agent_counter.get(agent, 0)
        # agent_time = agent_cnt * self.dt
        agent_time = bs.sim.simt - OFFSET_TIME # in seconds
        return agent_time

    def get_spd(self, agent):
        ac_id = bs.traf.id2idx(agent)
        if ac_id == -1:
            return 0
        # Get the current speed of the aircraft
        cur_ac_spd = metric_spd_to_knots_rounded(bs.traf.cas[ac_id])
        return cur_ac_spd
    
    def get_all_agent_spds(self):
        return {agent: self.get_agent_vels(agent)[0] for agent in self.agents}
    
    def get_acc(self, agent):
        ac_id = bs.traf.id2idx(agent)
        if ac_id == -1:
            return 0
        # Get the current acceleration of the aircraft
        cur_ac_acc = bs.traf.ax[ac_id]
        return cur_ac_acc

    def get_alt(self, agent):
        ac_id = bs.traf.id2idx(agent)
        if ac_id == -1:
            return 0
        # Get the current altitude of the aircraft
        cur_ac_alt = bs.traf.alt[ac_id]
        return cur_ac_alt
    
    def compute_avg_episode_spd(self, company: str):
        all_spds = self.episode_spds.copy()
        if company == 'CoA':
            relevant_agents = [agent for agent in all_spds.keys() if agent.startswith('AA') or agent.startswith('AC')]
        elif company == 'CoB':
            relevant_agents = [agent for agent in all_spds.keys() if agent.startswith('AB') or agent.startswith('AD')]
        else:
            relevant_agents = []

        if not relevant_agents:
            return 0

        avg_spd = np.mean([np.mean([spd for _, spd in all_spds[agent]]) for agent in relevant_agents])
        return avg_spd
    
    def get_avg_episode_spd(self):
        companies = ['CoA', 'CoB']
        avg_spds = {company: self.compute_avg_episode_spd(company) for company in companies}
        return avg_spds

    def record_agent_spds_per_episode(self):
        current_spds = self.current_speeds.copy()
        step = bs.sim.simt - OFFSET_TIME  # in seconds
        for agent in current_spds.keys():
            if agent not in self.episode_spds.keys():
                self.episode_spds[agent] = []
            self.episode_spds[agent].append((step, current_spds[agent]))

    def get_agent_vels(self, agent):
        ac_id = bs.traf.id2idx(agent)
        absolute_velocity = bs.traf.cas[ac_id]
        heading_radians = np.radians(bs.traf.hdg[ac_id])

        # Calculate longitudinal (eastward) and lateral (northward) velocity components
        lateral_velocity = absolute_velocity * np.cos(heading_radians)  # Northward
        longitudinal_velocity = absolute_velocity * np.sin(heading_radians)  # Eastward

        # Convert heading to 
        return absolute_velocity, (lateral_velocity, longitudinal_velocity)
    
    def get_agent_list(self):
        return self.agents
    
    def draw_circle(self, name='C1', lat=32.90437, lon=-97.15455, radius=1):
        # draw a circle around the aircraft
        stack.stack(f'CIRCLE {name}, {lat}, {lon}, {radius}') 

    def draw_circles(self, observations):
        """
        Draw circles around each agent to represent their sensory range.
        """
        for agent, obs in observations.items():
            agent_lat, agent_lon = obs[0][0], obs[0][1]
            radius_in_degrees = self.get_agent_detection_range(agent)
            self.draw_circle(agent, agent_lat, agent_lon, radius_in_degrees)

    def observation(self, agent):
        ownship_ac_idx = bs.traf.id2idx(agent)
        if ownship_ac_idx == -1:
            return np.zeros_like(8)

        ownship_lat = bs.traf.lat[ownship_ac_idx]
        ownship_lon = bs.traf.lon[ownship_ac_idx]
        ownship_alt = bs.traf.alt[ownship_ac_idx]

        lateral_distances = np.fromiter(
            (distance.geodesic((ownship_lat, ownship_lon), (ac_lat, ac_lon)).m for ac_lat, ac_lon in zip(bs.traf.lat, bs.traf.lon)),
            dtype=float
        )
        vertical_distances = ownship_alt - bs.traf.alt

        distances = np.hypot(lateral_distances, vertical_distances)
        sorted_ac_idxs = np.argsort(distances)

        desired_spd = metric_spd_to_knots_rounded(bs.traf.actwp.spd[sorted_ac_idxs])
        # replace unconstrained airspeed with current aircraft speed
        unconstrained_spd_ac = np.where(desired_spd < 0)
        desired_spd[unconstrained_spd_ac] = metric_spd_to_knots_rounded(bs.traf.selspd[unconstrained_spd_ac])

        # Get the current speed of the aircraft
        abs_vel = self.get_spd(agent)

        # accelerations
        self.current_speeds = self.get_all_agent_spds()
        if agent not in self.previous_speeds.keys():
            abs_acceleration = 0.0
        else:
            abs_acceleration = (self.current_speeds[agent] - self.previous_speeds[agent])/self.dt

        route_id = self.agent_route_ids[agent]

        if agent in self.previous_actions.keys():
            prev_action = self.previous_actions[agent]
        else:
            prev_action = 1

        # Classify important zones
        dangerous_zone_flag = self.entering_dangerous_area(agent)

        ownship_observations = np.array([[
            bs.traf.lat[ownship_ac_idx],
            bs.traf.lon[ownship_ac_idx],
            abs_vel, 
            bs.traf.hdg[ownship_ac_idx],
            abs_acceleration,
            route_id,
            prev_action,
            dangerous_zone_flag
        ]]).reshape(1, -1)
        return ownship_observations
    
    def observations(self, current_step_agents):
        return {agent: self.observation(agent) for agent in current_step_agents}
    
    def get_num_successful_agents(self):
        # Find the number of agents that have not been truncated
        num_successful_agents = 0
        for agent in self.truncated_agents.keys():
            if not self.truncated_agents[agent]:
                num_successful_agents += 1
        return num_successful_agents

    def distance_reward(self, agent, distance):
        if distance <= self.get_agent_collision_distance(agent):
            return -1, 1
        elif self.get_agent_collision_distance(agent) < distance < self.get_agent_safety_distance(agent):
            return 0.1 * (-1 + 1/(self.get_agent_safety_distance(agent) - self.get_agent_collision_distance(agent)) * (distance - self.get_agent_collision_distance(agent))), 2
            # return -(distance/self.safety_distance - 1) ** 2, False
            # return 0, False
        else:
            return 0, 3

    def velocity_reward(self, agent):
        ac_id = bs.traf.id2idx(agent)

        velocity_reward = 0
        if self.check_termination(agent):
            velocity_reward = 0
        else:
            current_spd = int(metric_spd_to_knots_rounded(bs.traf.cas[ac_id]))
            # velocity_reward = -(current_spd / desired_spd - 1) ** 2
            if current_spd < (self.get_min_spd(agent) + 10):
                velocity_reward = -1
            elif current_spd > (self.get_max_spd(agent) - 5):
                velocity_reward = -0.1
            else:
                velocity_reward = 0.0
        
        return velocity_reward
    
    def time_reward(self, agent):
        time_reward = 0
        # Find mission time for the agent based on the agent company
        agent_company = self.get_agent_company(agent)
        if agent_company == 'CoA':
            max_mission_time = MAX_MISSION_TIME_COA
        else:
            max_mission_time = MAX_MISSION_TIME_COB

        if self.truncated_agents[agent]:
            time_reward = 0
        else:
            time_reward = -1e-4
            agent_time = self.get_agent_time(agent)
            if agent_time >= max_mission_time:
                time_reward = -1
                print(f"Agent {agent} has reached the maximum mission time: {agent_time} seconds")
                self.truncated_agents[agent] = True
        return time_reward
    
    def action_reward(self, agent):
        action_reward = 0
        if agent in self.actions.keys():
            if agent in self.previous_actions.keys():
                if self.actions[agent] != self.previous_actions[agent]:
                    action_reward = -0.1
                else:
                    action_reward = 0
            else:
                action_reward = 0

        if agent in self.actions.keys():
            action = self.actions[agent]
            if action == 1:
                action_reward += 0
            else:
                action_reward += -1.0

        return action_reward

    def mission_accomplished_reward(self, agent):
        if self.check_mission_accomplished(agent):
            self.terminated_agents[agent] = True
            return 1.0
        return 0.0
    
    def check_mission_accomplished(self, agent):
        """
        Check if the agent has reached its destination waypoint.
        """
        ac_idx = bs.traf.id2idx(agent)
        if ac_idx == -1:
            return False
        agent_route = bs.traf.ap.route[ac_idx]
        final_wpt_lat, final_wpt_lon = agent_route.wplat[-1], agent_route.wplon[-1]
        # Calculate the distance to the destination waypoint
        distance_to_destination = haversine_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], final_wpt_lat, final_wpt_lon)
        # Check if the distance is less than the threshold
        return distance_to_destination < GOAL_RADIUS
    
    def reward(self, agent):

        if self.truncated_agents[agent] or self.check_termination(agent):
            return self.last_rewards[agent] if agent in self.last_rewards.keys() else 0.0
        else:
            # Velocity Reward = (current speed/desired speed - 1)^2
            velocity_reward = self.velocity_reward(agent)
            
            # Encouraging agents to finish the route
            time_reward = self.time_reward(agent)

            # action reward ******************************************************************
            # if action is not equal to the previous action receives -1
            action_reward = self.action_reward(agent)

            # Distance penalty
            # distance_reward = 0
            # if agent in self.distances_to_ownship.keys():
            #     for intruder, distance_to_ownship in self.distances_to_ownship[agent].items():
            #         dist_reward, truncation_flag = self.distance_reward(distance_to_ownship)
            #         distance_reward += dist_reward
            #         if truncation_flag:
            #             self.truncated_agents[agent] = True
            #             self.truncated_agents[intruder] = True
            #             if (agent, intruder) not in self.collided_agents and (intruder, agent) not in self.collided_agents:
            #                 self.collided_agents.append((agent, intruder))

            distance_reward = 0
            min_distance_to_agent = self.distances_to_ownship[agent]['distance']
            closest_intruder = self.distances_to_ownship[agent]['intruder']
            dist_reward, truncation_flag = self.distance_reward(agent, min_distance_to_agent)
            # print(f"Agent: {agent}, Closest intruder: {closest_intruder}, Distance: {min_distance_to_agent}, Distance reward: {dist_reward}")
            distance_reward += dist_reward

            if truncation_flag == 1:
                self.truncated_agents[agent] = True
                self.truncated_agents[closest_intruder] = True
                if (agent, closest_intruder) not in self.collided_agents and (closest_intruder, agent) not in self.collided_agents:
                    self.collided_agents.append((agent, closest_intruder))
                    self.collision_counter['All'] += 1
                    # Check the company of the agent and the intruder
                    agent_company = self.get_agent_company(agent)
                    intruder_company = self.get_agent_company(closest_intruder)
                    if agent_company != intruder_company:
                        self.collision_counter['AB'] += 1
                    else:
                        if agent_company == 'CoA':
                            self.collision_counter['AA'] += 1
                        else:
                            self.collision_counter['BB'] += 1

                    # Get agetn route ID
                    agent_route_id = self.agent_route_ids[agent]
                    intruder_route_id = self.agent_route_ids[closest_intruder]
                    if agent_route_id == 1 or agent_route_id == 2:
                        if intruder_route_id == 1 or intruder_route_id == 2:
                            self.collision_counter['M1'] += 1
                    if agent_route_id == 3 or agent_route_id == 4:
                        if intruder_route_id == 3 or intruder_route_id == 4:
                            self.collision_counter['M2'] += 1
                    if agent_route_id == 1 or agent_route_id ==2:
                        if intruder_route_id == 3 or intruder_route_id == 4:
                            self.collision_counter['In'] += 1
                    if agent_route_id == 3 or agent_route_id == 4:
                        if intruder_route_id == 1 or intruder_route_id == 2:
                            self.collision_counter['In'] += 1

            # Identify if the agent is entering a dangerous area (are)
            if truncation_flag == 2:
                self.unsafe_agents[agent] = True
            else:
                self.unsafe_agents[agent] = False

            # Mission Accomplished Reward
            mission_accomplished_reward = self.mission_accomplished_reward(agent)

            w1, w2, w3, w4, w5 = W_LOS_REW, W_ACT_REW, W_TIME_REW, W_MISSION_REW, W_VEL_REW
            total_reward = w1 * distance_reward + w2 * action_reward + w3 * time_reward + w4 * mission_accomplished_reward + w5 * velocity_reward
            self.sum_episode_rewards += total_reward

            # Add subrewards to the each step rewards
            if agent not in self.each_agent_rewards.keys():
                self.each_agent_rewards[agent] = [[w1*distance_reward, w2*action_reward, w3*time_reward, w4*mission_accomplished_reward, w5*velocity_reward]]
            else:
                self.each_agent_rewards[agent].append([w1*distance_reward, w2*action_reward, w3*time_reward, w4*mission_accomplished_reward, w5*velocity_reward])

            return np.float64(total_reward)
    
    def rewards(self, previous_step_agents):
        return {agent: self.reward(agent) for agent in previous_step_agents}
    
    def get_agent_company(self, agent):
        """
        Returns the company of the agent.
        """
        if agent.startswith('AA') or agent.startswith('AC'):
            return 'CoA'
        elif agent.startswith('AB') or agent.startswith('AD'):
            return 'CoB'
        else:
            return 'unknown'
        
    def get_agent_id_prefix(self, agent):
        """
        Returns the ID prefix of the agent.
        """
        if agent.startswith('AA'):
            return 'AA'
        elif agent.startswith('AB'):
            return 'AB'
        elif agent.startswith('AC'):
            return 'AC'
        elif agent.startswith('AD'):
            return 'AD'
        else:
            return 'unknown'
        
    def get_agent_safety_distance(self, agent):
        """
        Returns the safety distance of the agent based on its prefix.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, None)
        return agent_config['safety_radius(nm)']
    
    def get_agent_collision_distance(self, agent):
        """
        Returns the collision distance of the agent based on its prefix.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, None)
        return agent_config['collision_radius(nm)']
        
    def get_agent_detection_range(self, agent):
        """
        Returns the sensory range of the agent based on the agent prefix and its sensory range.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, None)
        return agent_config['detection_range(nm)']

    def get_latlon(self, agent):
        """
        Returns the latitude and longitude of the agent.
        """
        ac_idx = bs.traf.id2idx(agent)
        if ac_idx == -1:
            return np.inf, np.inf
        return bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]
    
    def get_nxt_wpt_id(self, agent):
        """
        Returns the ID of the next waypoint for the agent.
        """
        ac_idx = bs.traf.id2idx(agent)
        if ac_idx == -1:
            return None
        # find the next waypoint for the agent
        agent_route = bs.traf.ap.route[ac_idx]
        next_wp_id = agent_route.wpname[agent_route.iactwp]
        return next_wp_id
    
    def get_prvs_wpt_id(self, agent):
        """
        Returns the ID of the previous waypoint for the agent.
        """
        ac_idx = bs.traf.id2idx(agent)
        if ac_idx == -1:
            return None
        # find the previous waypoint for the agent
        agent_route = bs.traf.ap.route[ac_idx]
        prev_wp_id = agent_route.wpname[agent_route.iactwp - 1] if agent_route.iactwp > 0 else None
        return prev_wp_id

    def get_nxt_wpt_type(self, agent):
        """
        Returns the type of the next waypoint for the agent.
        If the waypoint is in merging_wpts, then return 'Merging',
        If the waypoint is in intersection_wpts, then return 'Intersection',
        Else return 'Regular'.
        """
        ac_idx = bs.traf.id2idx(agent)
        if ac_idx == -1:
            return None
        # find the next waypoint for the agent
        agent_route = bs.traf.ap.route[ac_idx]
        next_wp_id = agent_route.wpname[agent_route.iactwp]
        if next_wp_id in self.merging_wpts:
            return 'Merging'
        elif next_wp_id in self.intersection_wpts:
            return 'Intersection'
        else:
            return 'Regular'

    def get_latlon_nxt_wpt(self, agent):
        """
        Returns the latitude and longitude of the next waypoint for the agent.
        """
        ac_idx = bs.traf.id2idx(agent)
        if ac_idx == -1:
            return np.inf, np.inf
        # find the next waypoint for the agent
        agent_route = bs.traf.ap.route[ac_idx]

        # next_wpt_lat, next_wpt_lon = agent_route.wplat[-1], agent_route.wplon[-1]
        next_wpt_lat, next_wpt_lon = agent_route.wplat[agent_route.iactwp], agent_route.wplon[agent_route.iactwp]
        return next_wpt_lat, next_wpt_lon
        
    def distance_to_nxt_wpt(self, agent):
        """
        Returns the distance to the next waypoint for the agent. The scale is in nautical miles.
        """
        ac_idx = bs.traf.id2idx(agent)
        if ac_idx == -1:
            return np.inf
        # find the next waypoint for the agent
        next_wpt_lat, next_wpt_lon = self.get_latlon_nxt_wpt(agent)
        if next_wpt_lat == np.inf or next_wpt_lon == np.inf:
            return np.inf
        # Calculate the distance to the next waypoint
        return haversine_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], next_wpt_lat, next_wpt_lon)

    def get_front_intruders(self, ownship, observations):
        """
        Returns a list of intruders that are in front of the ownship agent. A vehicle is in front of the ownship 
        if its distance to the next waypoint is lower than the ownship's distance to the next waypoint. Moreover, 
        the intruder's distance to the ownship should be less than the ownship's sensory range. The intruder's 
        should be also ordered by their distance to the ownship, from lower to higher distance.
        Args:
            agent (str): The ID of the ownship agent.
            observations (dict): A dictionary of observations for all agents.
        """
        # Find the ownship latitude and longitude
        ownship_lat, ownship_lon = self.get_latlon(ownship)
        # get latitude and longitude of the next waypoint for the ownship agent
        ownship_nxt_wpt_lat, ownship_nxt_wpt_lon = self.get_latlon_nxt_wpt(ownship)
        # agent distance to the next waypoint
        ownship_distance_to_nxt_wpt = self.distance_to_nxt_wpt(ownship)
        front_intruders = []

        for intruder, intruder_obs in observations.items():
            if intruder == ownship:
                continue
            
            # get latitude and longitude of the next waypoint for the intruder agent
            intruder_nxt_wpt_lat, intruder_nxt_wpt_lon = self.get_latlon_nxt_wpt(intruder)
            # check to see if the intruder and ownship have the same next waypoint
            if intruder_nxt_wpt_lat == ownship_nxt_wpt_lat and intruder_nxt_wpt_lon == ownship_nxt_wpt_lon:
                # find the latitude and longitude of the intruder
                intruder_lat, intruder_lon = intruder_obs[0][0], intruder_obs[0][1]
                # find the distance of the intruder to the ownship
                intruder_distance_to_ownship = haversine_distance(ownship_lat, ownship_lon, intruder_lat, intruder_lon)
                # calculate the distance to the next waypoint for the intruder
                intruder_distance_to_nxt_wpt = self.distance_to_nxt_wpt(intruder)
                # find the distance between the ownship and the intruder
                if intruder_distance_to_nxt_wpt < ownship_distance_to_nxt_wpt and intruder_distance_to_ownship < self.get_agent_detection_range(ownship):
                    front_intruders.append(intruder)

        return front_intruders

    def get_front_intruders_ordered(self, ownship, observations):
        """
        Returns a list of intruders that are in front of the ownship agent. A vehicle is in front of the ownship 
        if its distance to the next waypoint is lower than the ownship's distance to the next waypoint. Moreover, 
        the intruder's distance to the ownship should be less than the ownship's sensory range. The intruder's 
        should be also ordered by their distance to the ownship, from lower to higher distance.
        """
        front_intruders = self.get_front_intruders(ownship, observations)
        # Order the intruders by their distance to the ownship
        # front_intruders.sort(key=lambda intruder: self.get_2agent_distance(ownship, intruder))

        ownship_dist_to_nxt_wpt = self.distance_to_nxt_wpt(ownship)
        front_intruders.sort(key=lambda intruder: 1/(abs(self.distance_to_nxt_wpt(intruder) - ownship_dist_to_nxt_wpt) + 1e-6))
        
        return front_intruders

    def get_all_intruders(self, ownship, observations):
        """
        Returns a list of all intruders relative to the ownship agent. An intruder is considered relevant if its
        distance to the next waypoint is within a certain threshold around the ownship's distance to the next waypoint. If the ownship distance
        is d_o, then the intruder's distance is in [d_o - epsilon, d_o + epsilon]. Moreover,
        the intruder's distance to the ownship should be less than the ownship's sensory range. The intruder's
        should be also ordered by their distance to the ownship, from lower to higher distance.
        Args:
            agent (str): The ID of the ownship agent.
            observations (dict): A dictionary of observations for all agents.
        """
        # Find the ownship latitude and longitude
        ownship_lat, ownship_lon = self.get_latlon(ownship)
        # get latitude and longitude of the next waypoint for the ownship agent
        ownship_nxt_wpt_lat, ownship_nxt_wpt_lon = self.get_latlon_nxt_wpt(ownship)
        # agent distance to the next waypoint
        all_intruders = []

        for intruder, intruder_obs in observations.items():
            if intruder == ownship:
                continue
            
            # get latitude and longitude of the next waypoint for the intruder agent
            intruder_nxt_wpt_lat, intruder_nxt_wpt_lon = self.get_latlon_nxt_wpt(intruder)
            # check to see if the intruder and ownship have the same next waypoint
            if intruder_nxt_wpt_lat == ownship_nxt_wpt_lat and intruder_nxt_wpt_lon == ownship_nxt_wpt_lon:
                # find the latitude and longitude of the intruder
                intruder_lat, intruder_lon = intruder_obs[0][0], intruder_obs[0][1]
                # find the distance of the intruder to the ownship
                intruder_distance_to_ownship = haversine_distance(ownship_lat, ownship_lon, intruder_lat, intruder_lon)
                # find the distance between the ownship and the intruder
                if intruder_distance_to_ownship < self.get_agent_detection_range(ownship):
                    all_intruders.append(intruder)

        return all_intruders

    def get_distances_to_ownship(self, observations):
        """
        Updates self.distances_to_ownship with the distances between ownship and intruders.
        """
        distances_to_ownship = {}

        for ownship_agent, ownship_obs in observations.items():
            ownship_lat, ownship_lon = ownship_obs[0][0], ownship_obs[0][1]
            intruder_distances = {}

            for intruder_agent, intruder_obs in observations.items():
                if ownship_agent == intruder_agent:
                    continue

                intruder_lat, intruder_lon = intruder_obs[0][0], intruder_obs[0][1]
                distance = haversine_distance(ownship_lat, ownship_lon, intruder_lat, intruder_lon)



                if distance > 0 and distance < self.get_agent_safety_distance(ownship_agent):
                    intruder_distances[intruder_agent] = distance

            distances_to_ownship[ownship_agent] = intruder_distances

        return distances_to_ownship

    def get_closest_distance_to_ownship(self, observations):
        """
        Finds the closest distance between ownship and intruders for each ownship agent.
        """
        closest_distances = {}

        for ownship_agent, ownship_obs in observations.items():
            ownship_lat, ownship_lon = ownship_obs[0][0], ownship_obs[0][1]
            min_distance = float('inf')  # Initialize with a very large value
            closest_intruder = None

            for intruder_agent, intruder_obs in observations.items():
                if ownship_agent == intruder_agent:
                    continue

                intruder_lat, intruder_lon = intruder_obs[0][0], intruder_obs[0][1]
                distance = haversine_distance(ownship_lat, ownship_lon, intruder_lat, intruder_lon)

                # Update the minimum distance and closest intruder
                if 0 < distance < min_distance:
                    min_distance = distance
                    closest_intruder = intruder_agent

            # Store the closest distance and intruder for the ownship agent
            if min_distance < self.get_agent_safety_distance(ownship_agent):
                closest_distances[ownship_agent] = {
                    'distance': min_distance,
                    'intruder': closest_intruder
                }
            else:
                closest_distances[ownship_agent] = {
                    'distance': np.inf,
                    'intruder': None
                }

        return closest_distances
        
    def get_max_spd(self, agent):
        """
        Returns the maximum speed based on the the agent's prefix and the configuration file.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, {})
        return agent_config.get('max_speed(knots)', None)
        
    def get_min_spd(self, agent):
        """
        Returns the minimum speed based on the the agent's prefix and the configuration file.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, {})
        return agent_config.get('min_speed(knots)', None)
    
    def get_spd_increment(self, agent):
        """
        Returns the speed increment per second based on the the agent's prefix and the configuration file.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, {})
        return agent_config.get('speed_increment_per_second(knots)', None)
    
    def get_agent_desired_spd(self, agent):
        """
        Returns the desired speed of the agent based on its prefix.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, {})
        desired_spd = agent_config.get('desired_speed(knots)', None)
        return desired_spd  
    
    def get_route_id(self, agent):
        """
        Returns the route ID of the agent based on its prefix.
        """
        agent_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_prefix, {})
        return agent_config.get('route_id', None)

    def update_co_action_history(self, agent, action):
        agent_company = self.get_agent_company(agent)
        if agent_company == 'CoA':
            if action == 0:
                self.action_counter_A['decelerate'] += 1
            elif action == 1:
                self.action_counter_A['hold'] += 1
            elif action == 2:
                self.action_counter_A['accelerate'] += 1
        elif agent_company == 'CoB':
            if action == 0:
                self.action_counter_B['decelerate'] += 1
            elif action == 1:
                self.action_counter_B['hold'] += 1
            elif action == 2:
                self.action_counter_B['accelerate'] += 1

    def step(self, actions):
        """
        Takes a step in the simulation with the given actions.
        Args:
            actions (dict): A dictionary where keys are agent IDs and values are actions (1 for no change, 2 for increase speed, 0 for decrease speed).
        Returns:
            processed_observations (dict): Processed observations for each agent.
            rewards (dict): Rewards for each agent.
            terminations (dict): Termination flags for each agent.
            truncations (dict): Truncation flags for each agent.
            dones (dict): Done flags for each agent.
        """
        if not actions:
            return {}, {}, {}, {}, {}
        
        # Count agent for each time step
        for agent in actions.keys():
            if agent not in self.agent_counter.keys():
                self.agent_counter[agent] = 0
            self.agent_counter[agent] += 1

        # update actions
        self.actions = actions

        for agent, action in actions.items():
            ac_id = bs.traf.id2idx(agent)
            if ac_id == -1:
                print(f"Warning: Agent {agent} not found in the simulation. Something went wrong.")
                continue
            
            flight_number = agent
            ac_phase = bs.traf.perf.phase[ac_id]
            
            if bs.traf.actwp.swlastwp[ac_id] and (ac_phase in (openap.ph.GD,)):
                # When landed the agent cannot make speed changes
                # Turn on vnav speed
                if not bs.traf.swvnavspd[ac_id]:
                    bs.traf.swvnavspd[ac_id] = True
                continue 

            if ac_phase == openap.ph.IC:
                if not bs.traf.swvnavspd[ac_id]:
                    bs.traf.swvnavspd[ac_id] = True
                # Cannot take action at initial climb
                continue

            # Collect taken actions
            # self.update_co_action_history(agent, action)
            
            if action != 1:               
                cur_ac_spd = metric_spd_to_knots_rounded(bs.traf.cas[ac_id])

                new_ac_spd = int(cur_ac_spd) + (action - 1) * self.get_spd_increment(agent) * self.dt
                new_ac_spd = min(max(new_ac_spd, self.get_min_spd(agent)), self.get_max_spd(agent))

                # print(f"Current speed: {cur_ac_spd} New speed: {new_ac_spd}")
                stack.stack(f'SPD {flight_number} {new_ac_spd}')
                self.cmd_history.append(f'00:00:{self.get_agent_time(agent):.2f}>SPD {flight_number} {new_ac_spd}')

        # Simulate the environment for one step
        self._sim_step()
        
        # Updating the agents in the scene
        old_agents = self.agents.copy()
        current_step_agents = set(self.agents).union(old_agents)

        # Updating the observations
        observations = self.observations(current_step_agents)
        processed_observations = self.process_agents(observations)

        # Get distances to ownship
        self.distances_to_ownship = self.get_closest_distance_to_ownship(observations)
        # self.agent_phases = self.get_operation_phase_agents(observations)

        # Collect rewards before sim step
        previous_step_agents = list(actions.keys())
        rewards = self.rewards(previous_step_agents)

        terminations = self.terminated_agents.copy()
        truncations = self.truncated_agents.copy()
        # infos = self.infos(current_step_agents)

        # Update previous actions
        self.previous_actions = actions
        self.previous_speeds = self.current_speeds.copy()
        # self.record_agent_spds_per_episode()

        # Truncate and terminate agents after receiving next observations and rewards
        self.truncate_agents()
        self.terminate_agents()
        self.check_done_agents(current_step_agents)

        self.last_rewards = rewards

        # Extract the list of terminated and truncated agents
        self.term_agent_list = [agent for agent, terminated in terminations.items() if terminated]
        self.trunc_agent_list = [agent for agent, truncated in truncations.items() if truncated]

        # Take one step in saving the state-action pairs
        self.ownship_intruder_info(observations)

        if len(self.all_state_action_pairs) > 0:
            # print(self.all_state_action_pairs[-1])
            last_prompt = self.get_ownship_prompt(self.all_state_action_pairs[-1])
            print(f"Ownship Prompt: {last_prompt}")

        # self.draw_circles(observations)
        
        return observations, processed_observations, rewards, terminations, truncations, self.dones

    def extract_states(self, agent, observations):
        """
        Extracts the state for a given agent from the observations.
        Args:
            agent (str): The agent ID.
            observations (dict): The observations for all agents.
        Returns:
            states (dict): A dictionary containing the normalized state for the agent.
        """

        # Define reasonable ranges for normalization

        # Calculate velocity components
        spd = observations[agent][0][2] # Velocity
        hdg = observations[agent][0][3] # Heading
        acc = observations[agent][0][4] # Acceleration

        # print(f"Agent: {agent}, v_x: {v_x}, v_y: {v_y}")
        # print speed, min speed, max speed
        # print(f"Agent: {agent}, Speed: {spd}, Min Speed: {self.get_min_spd(agent)}, Max Speed: {self.get_max_spd(agent)}")
        # Normalize velocity components
        spd_normalized = normalize(abs(spd), self.get_min_spd(agent), self.get_max_spd(agent))
        hdg_normalized = normalize(hdg, 0, HDG_RANGE)
        acc_normalized = normalize(acc, self.min_acc, self.max_acc)
        # print(f'v_x_normalized: {v_x_normalized}, v_y_normalized: {v_y_normalized}')

        # update min and max acceleration
        self.min_acc = min(self.min_acc, acc)
        self.max_acc = max(self.max_acc, acc)

        # Get route id
        normalized_route_id = observations[agent][0][5]/NUM_ROUTES

        # Get previous action
        normalized_prev_action = observations[agent][0][6]/(ACTION_SIZE - 1)

        # Find distance to next waypoint
        dist_to_nxt_wpt = self.distance_to_nxt_wpt(agent)

        # if agent == 'AB0002':
        #     print(f"Agent: {agent}, Distance to next waypoint: {dist_to_nxt_wpt}")

        # Normalize distance to next waypoint
        normalized_distance_to_nxt_wpt = normalize(dist_to_nxt_wpt, 0, self.get_agent_detection_range(agent))

        # Dangerous zone
        dangerous_zone_flag = observations[agent][0][7]

        # Store results for the agent
        states = {
            "spd": spd_normalized,
            "hdg": hdg_normalized,
            "acc": acc_normalized,
            "r_id": normalized_route_id,
            "prev_act": normalized_prev_action,
            "dist_to_goal": normalized_distance_to_nxt_wpt
        }
        return states

    def find_intruders(self, ownship_obs, observations):
        """
        Find intruders for the given ownship and calculate absolute relative distances (latitude and longitude).
        """

        intruders = []
        ownship_lat, ownship_lon = ownship_obs[0][0], ownship_obs[0][1]

        for agent, obs in observations.items():
            # Define sensory range for each agent

            intruder_lat, intruder_lon = obs[0][0], obs[0][1]
            distance = haversine_distance(ownship_lat, ownship_lon, intruder_lat, intruder_lon)
            if distance <= self.get_agent_detection_range(agent):
                # Calculate absolute latitude and longitude distances
                lat_distance = haversine_distance(ownship_lat, ownship_lon, intruder_lat, ownship_lon)
                lon_distance = haversine_distance(ownship_lat, ownship_lon, ownship_lat, intruder_lon)
                relative_distance = np.array([lat_distance, lon_distance])
                # Sum squared distances
                # print(f"Agent: {agent}, lat_distance: {lat_distance*max_lat_distance}, lon_distance: {lon_distance*max_lon_distance}")

                # Calculate and normalize relative velocity
                intruder_velocity = np.array([obs[0][2]])
                normalized_int_vel = normalize(intruder_velocity, self.get_min_spd(agent), self.get_max_spd(agent))

                # Relative heading
                intruder_heading = np.array([obs[0][3]])
                normalized_int_hdg = normalize(intruder_heading, 0, HDG_RANGE)

                # Calculate and normalize relative acceleration
                intruder_acceleration = np.array([obs[0][4]])
                normalized_int_acc = normalize(intruder_acceleration, self.min_acc, self.max_acc)

                # Relative route
                intruder_route = np.array([obs[0][5]])
                normalized_int_route = intruder_route / NUM_ROUTES

                # Combine absolute distances with normalized velocity and acceleration
                intruder_state = np.concatenate([relative_distance, intruder_velocity, intruder_heading, intruder_acceleration, intruder_route])
                intruder_state_normalized = np.concatenate([relative_distance/self.get_agent_detection_range(agent), normalized_int_vel, normalized_int_acc, normalized_int_hdg, normalized_int_route])

                intruders.append(intruder_state_normalized)

        # Sort intruders by total distance and limit to max_intruders
        intruders = sorted(intruders, key=lambda x: np.sqrt(x[0]**2 + x[1]**2))[:NUM_INTRUDERS]

        # Fill with zeros if fewer than max_intruders
        while len(intruders) < NUM_INTRUDERS:
            intruders.append(np.zeros(INTRUDER_STATE_SIZE))  # Fill with zeros (6 features: lat_distance, lon_distance, velocity, acceleration)

        return np.array(intruders)

    def find_intruders_ahead(self, ownship, ownship_obs, observations):
        """
        Find intruders for the given ownship and calculate absolute relative distances (latitude and longitude).
        """

        intruder_states = []
        ownship_lat, ownship_lon = ownship_obs[0][0], ownship_obs[0][1]
        front_intruders = self.get_front_intruders(ownship, observations)
        all_intruders = self.get_all_intruders(ownship, observations)
        intruders = front_intruders

        for agent, obs in observations.items():
            # find the ownship distance to the next waypoint
            ownship_dist_to_nxt_wpt = self.distance_to_nxt_wpt(ownship)
            # Find intruders that are ahead of the ownship
            if agent in intruders and agent != ownship:

                # Define sensory range for each agent
                intruder_lat, intruder_lon = obs[0][0], obs[0][1]

                # Find distance to the next waypoint for the intruder
                intruder_dist_to_nxt_wpt = self.distance_to_nxt_wpt(agent)
                intruder_dist = np.array([intruder_dist_to_nxt_wpt])
                normalized_intruder_dist = normalize(intruder_dist, 0, self.get_agent_detection_range(agent))

                # Calculate absolute latitude and longitude distances
                lat_distance = haversine_distance(ownship_lat, ownship_lon, intruder_lat, ownship_lon)
                lon_distance = haversine_distance(ownship_lat, ownship_lon, ownship_lat, intruder_lon)
                relative_distance = np.array([lat_distance, lon_distance])
                # Sum squared distances
                # print(f"Agent: {agent}, lat_distance: {lat_distance*max_lat_distance}, lon_distance: {lon_distance*max_lon_distance}")

                # Calculate and normalize relative velocity
                intruder_velocity = np.array([obs[0][2]])
                normalized_int_vel = normalize(intruder_velocity, self.get_min_spd(agent), self.get_max_spd(agent))

                # Relative heading
                intruder_heading = np.array([obs[0][3]])
                normalized_int_hdg = normalize(intruder_heading, 0, HDG_RANGE)

                # Calculate and normalize relative acceleration
                intruder_acceleration = np.array([obs[0][4]])
                normalized_int_acc = normalize(intruder_acceleration, self.min_acc, self.max_acc)

                # Relative route
                intruder_route = np.array([obs[0][5]])
                normalized_int_route = intruder_route / NUM_ROUTES

                # Combine absolute distances with normalized velocity and acceleration
                intruder_state = np.concatenate([relative_distance, intruder_velocity, intruder_heading, intruder_acceleration, intruder_route])
                intruder_state_normalized = np.concatenate([normalized_intruder_dist, normalized_int_vel, normalized_int_acc, normalized_int_hdg, normalized_int_route])

                intruder_states.append(intruder_state_normalized)

        # Sort intruders by total distance and limit to max_intruders
        intruder_states = sorted(intruder_states, key=lambda x: 1/(abs(x[0]-ownship_dist_to_nxt_wpt) + 1e-6))[:NUM_INTRUDERS]

        # Fill with zeros if fewer than max_intruders
        while len(intruder_states) < NUM_INTRUDERS:
            intruder_states.append(np.zeros(INTRUDER_STATE_SIZE))  # Fill with zeros (6 features: lat_distance, lon_distance, velocity, acceleration)

        return np.array(intruder_states)

    def process_agents(self, observations):
        """
        Process agents to calculate ownship state, intruder states, and absolute distances.
        """
        processed_states = {}

        for ownship_agent, ownship_obs in observations.items():
            # Calculate normalized velocities and accelerations for the ownship
            ownship_state = self.extract_states(ownship_agent, observations)

            # Find intruders for the ownship
            # intruder_states = self.find_intruders(ownship_obs, {k: v for k, v in observations.items() if k != ownship_agent})
            intruder_states = self.find_intruders_ahead(ownship_agent, ownship_obs, {k: v for k, v in observations.items() if k != ownship_agent})

            # Combine ownship state with intruder states and distances
            processed_states[ownship_agent] = {
                "ownship_state": ownship_state,
                "intruder_states": intruder_states
            }

        return processed_states
    
    def get_rb_actions_v1(self, observations):
        """
        This function gets the observation and decide which action to take given some rules.
        Inputs: observations
        Outputs: actions
        """

        closest_intruder_with_distances = self.get_closest_distance_to_ownship(observations)

        actions = {}
        reasoning_logs = {}
        for ownship_agent, ownship_obs in observations.items():

            # If actions[ownship_agent] is already defined, skip to the next agent
            if ownship_agent in actions.keys():
                continue

            closest_intruder = closest_intruder_with_distances[ownship_agent]['intruder']
            closest_distance = closest_intruder_with_distances[ownship_agent]['distance']

            # Closest intruder should be the front intruder with closest distance to the next waypoint as the ownship

            if closest_intruder is None or closest_distance > self.get_agent_safety_distance(ownship_agent):
                # Find the maximum speed for the ownship agent
                ownship_agent_max_spd = self.get_max_spd(ownship_agent)
                ownship_agent_spd = self.get_spd(ownship_agent)
                # If the ownship agent is below the maximum speed, accelerate
                if ownship_agent_spd < 0.75 * ownship_agent_max_spd:
                    actions[ownship_agent] = 2
                    reasoning_logs[ownship_agent] = "There is no intruder ahead within the safety distance. " + \
                    "Since the current speed is low, accelerate to reach desired speed, helping to avoid occlusion."
                else:
                    actions[ownship_agent] = 1
                    reasoning_logs[ownship_agent] = "There is no intruder ahead within the safety distance. " + \
                    "Since the current speed is high, maintain current speed."
                continue

            # Find intruders that are ahead of the ownship
            front_intruders = self.get_front_intruders(ownship_agent, observations)
            
            # If there is an intruder within the safety distance, check if it is ahead
            # If the intruder is ahead, decide to decelerate or accelerate based on the distance
            if closest_intruder is not None and closest_distance <= self.get_agent_safety_distance(ownship_agent):
                # check if the intruder is ahead
                if closest_intruder in front_intruders:
                    # Intruder is ahead, decide to decelerate
                    actions[ownship_agent] = 0
                    reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, the closest intruder is ahead of the ownship. Decelerate to avoid collision."
                else:
                    # Intruder is behind, decide to accelerate
                    actions[ownship_agent] = 2
                    reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, the closest intruder is behind the ownship. Accelerate to maintain distance."

                    # Find distance to next waypoint for ownship
                    ownship_dist_to_nxt_wpt = self.distance_to_nxt_wpt(ownship_agent)
                    # Find distance to next waypoint for intruder
                    intruder_dist_to_nxt_wpt = self.distance_to_nxt_wpt(closest_intruder)

                    ownship_spd =  self.get_spd(ownship_agent)
                    intruder_spd = self.get_spd(closest_intruder)

                    # random_action = random.choice([0, 2])
                    if abs(ownship_dist_to_nxt_wpt - intruder_dist_to_nxt_wpt) < 0.1 * self.get_agent_collision_distance(ownship_agent):
                        # Take 0 or 2 randomly for acceleration/deceleration
                        if ownship_spd > intruder_spd:
                            actions[ownship_agent] = 2  # Accelerate
                            reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, " + \
                                                            "the closest intruder distance to the next waypoint is nearly equal to that of the ownship, " + \
                                                            "and the ownship speed is higher than the intruder speed, " + \
                                                            "ownship should accelerate since it has a speed advantage."
                        elif ownship_spd < intruder_spd:
                            actions[ownship_agent] = 0  # Decelerate
                            reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, " + \
                                                            "the closest intruder distance to the next waypoint is nearly equal to that of the ownship, " + \
                                                            "and the ownship speed is lower than the intruder speed, " + \
                                                            "ownship should decelerate since it has a speed disadvantage."
                        else:
                            actions[ownship_agent] = random.choice([0, 2])
                            reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, " + \
                                                            "the closest intruder distance to the next waypoint is nearly equal to that of the ownship, " + \
                                                            "and both ownship and intruder have the same speed. " + \
                                                            "ownship randomly chooses to accelerate or decelerate."
                        actions[closest_intruder] = 2 - actions[ownship_agent]  # Take the opposite action
                        reasoning_logs[closest_intruder] = "Taking the opposite action of the ownship to maintain safe distance."

            ownship_id_prefix = self.get_agent_id_prefix(ownship_agent)
            ownship_config = self.ac_configs.get(ownship_id_prefix, {})
            ownship_agent_min_spd = ownship_config.get('min_speed(knots)', None)
            ownship_agent_min_spd = metric_spd_to_knots_rounded(ownship_agent_min_spd)
            ownship_agent_max_spd = ownship_config.get('max_speed(knots)', None)
            ownship_agent_max_spd = metric_spd_to_knots_rounded(ownship_agent_max_spd)

            ownship_agent_spd = self.get_spd(ownship_agent)
            cond1 = (ownship_agent_spd <= ownship_agent_min_spd and actions[ownship_agent] == 0)
            cond2 = (ownship_agent_spd >= ownship_agent_max_spd and actions[ownship_agent] == 2)
            if cond1 or cond2:
                actions[ownship_agent] = 1  # Maintain current speed
                reasoning_logs[ownship_agent] = "The decided action would violate speed limits. Maintain current speed instead."

        return actions, reasoning_logs
    
        
    def get_rb_actions(self, observations):
        """
        This function gets the observation and decide which action to take given some rules.
        Inputs: observations
        Outputs: actions
        """
        closest_intruder_with_distances = self.get_closest_distance_to_ownship(observations)

        actions = {}
        reasoning_logs = {}
        for ownship_agent, ownship_obs in observations.items():

            # If actions[ownship_agent] is already defined, skip to the next agent
            if ownship_agent in actions.keys():
                continue

            ownship_dist_to_nxt_wpt = self.distance_to_nxt_wpt(ownship_agent)

            if ownship_dist_to_nxt_wpt > self.get_agent_safety_distance(ownship_agent):
                flag = "far_from_waypoint"
                closest_intruder = closest_intruder_with_distances[ownship_agent]['intruder']
                closest_distance = closest_intruder_with_distances[ownship_agent]['distance']

                # Closest intruder should be the front intruder with closest distance to the next waypoint as the ownship
                if closest_intruder is None or closest_distance > self.get_agent_safety_distance(ownship_agent):
                    # Find the maximum speed for the ownship agent
                    ownship_agent_max_spd = self.get_max_spd(ownship_agent)
                    ownship_agent_spd = self.get_spd(ownship_agent)
                    # If the ownship agent is below the maximum speed, accelerate
                    if ownship_agent_spd < self.get_agent_desired_spd(ownship_agent):
                        actions[ownship_agent] = 2
                        reasoning_logs[ownship_agent] = "There is no intruder ahead within the safety distance. " + \
                        "Since the current speed is low, accelerate to reach desired speed, helping to avoid occlusion."
                    else:
                        actions[ownship_agent] = 1
                        reasoning_logs[ownship_agent] = "There is no intruder ahead within the safety distance. " + \
                        "Since the current speed is high, maintain current speed."
                    continue

                # Find intruders that are ahead of the ownship
                front_intruders = self.get_front_intruders(ownship_agent, observations)
                
                # If there is an intruder within the safety distance, check if it is ahead
                # If the intruder is ahead, decide to decelerate or accelerate based on the distance
                if closest_intruder is not None and closest_distance <= self.get_agent_safety_distance(ownship_agent):
                    # check if the intruder is ahead
                    if closest_intruder in front_intruders:
                        # Intruder is ahead, decide to decelerate
                        actions[ownship_agent] = 0
                        reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, the closest intruder is ahead of the ownship. Decelerate to avoid collision."
                    else:
                        # Intruder is behind, decide to accelerate
                        actions[ownship_agent] = 2
                        reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, the closest intruder is behind the ownship. Accelerate to maintain distance."

                ownship_id_prefix = self.get_agent_id_prefix(ownship_agent)
                ownship_config = self.ac_configs.get(ownship_id_prefix, {})
                ownship_agent_min_spd = ownship_config.get('min_speed(knots)', None)
                ownship_agent_min_spd = metric_spd_to_knots_rounded(ownship_agent_min_spd)
                ownship_agent_max_spd = ownship_config.get('max_speed(knots)', None)
                ownship_agent_max_spd = metric_spd_to_knots_rounded(ownship_agent_max_spd)

                ownship_agent_spd = self.get_spd(ownship_agent)
                cond1 = (ownship_agent_spd <= ownship_agent_min_spd and actions[ownship_agent] == 0)
                cond2 = (ownship_agent_spd >= ownship_agent_max_spd and actions[ownship_agent] == 2)
                if cond1 or cond2:
                    actions[ownship_agent] = 1  # Maintain current speed
                    reasoning_logs[ownship_agent] = "The decided action would violate speed limits. Maintain current speed instead."
            else:
                flag = "near_waypoint"
                # Find intruders that are ahead of the ownship
                front_intruders = self.get_front_intruders_ordered(ownship_agent, observations)

                if len(front_intruders) > 0:
                    closest_intruder = front_intruders[0]
                    closest_distance = abs(ownship_dist_to_nxt_wpt - self.distance_to_nxt_wpt(closest_intruder))
                else:
                    closest_intruder = None
                    closest_distance = float('inf')

                if closest_intruder is not None:

                    if self.check_2agents_on_same_route(ownship_agent, closest_intruder):
                        actions[ownship_agent] = 0
                        reasoning_logs[ownship_agent] = "The ownship is near the waypoint, and there is an intruder ahead " + \
                        "within the safety distance and on the same route. Decelerate to avoid collision while approaching the waypoint."
                    else:
                        actions[ownship_agent] = 0
                        reasoning_logs[ownship_agent] = "The ownship is near the waypoint, and there is an intruder ahead " + \
                        "within the safety distance and on a different route. Decelerate to avoid collision while approaching the waypoint."

                        ownship_spd =  self.get_spd(ownship_agent)
                        intruder_spd = self.get_spd(closest_intruder)

                        if closest_distance < self.get_agent_collision_distance(ownship_agent):
                            # Take 0 or 2 randomly for acceleration/deceleration
                            if ownship_spd > intruder_spd:
                                actions[ownship_agent] = 2  # Accelerate
                                reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, " + \
                                                                "the closest intruder distance to the next waypoint is nearly equal to that of the ownship, " + \
                                                                "and the ownship speed is higher than the intruder speed, " + \
                                                                "ownship should accelerate since it has a speed advantage."
                            elif ownship_spd < intruder_spd:
                                actions[ownship_agent] = 0  # Decelerate
                                reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, " + \
                                                                "the closest intruder distance to the next waypoint is nearly equal to that of the ownship, " + \
                                                                "and the ownship speed is lower than the intruder speed, " + \
                                                                "ownship should decelerate since it has a speed disadvantage."
                            else:
                                actions[ownship_agent] = random.choice([0, 2])
                                reasoning_logs[ownship_agent] = "Within the unsafe zone around the ownship, " + \
                                                                "the closest intruder distance to the next waypoint is nearly equal to that of the ownship, " + \
                                                                "and both ownship and intruder have the same speed. " + \
                                                                "ownship randomly chooses to accelerate or decelerate."
                        
                            actions[closest_intruder] = 2 - actions[ownship_agent]  # Take the opposite action
                            reasoning_logs[closest_intruder] = "Taking the opposite action of the ownship to maintain safe distance."
                else:
                    actions[ownship_agent] = 2
                    reasoning_logs[ownship_agent] = "The ownship is near the waypoint, and there is no intruder ahead within the safety distance. Accelerate to reach the waypoint faster."

                ownship_id_prefix = self.get_agent_id_prefix(ownship_agent)
                ownship_config = self.ac_configs.get(ownship_id_prefix, {})
                ownship_agent_min_spd = ownship_config.get('min_speed(knots)', None)
                ownship_agent_min_spd = metric_spd_to_knots_rounded(ownship_agent_min_spd)
                ownship_agent_max_spd = ownship_config.get('max_speed(knots)', None)
                ownship_agent_max_spd = metric_spd_to_knots_rounded(ownship_agent_max_spd)

                ownship_agent_spd = self.get_spd(ownship_agent)
                cond1 = (ownship_agent_spd <= ownship_agent_min_spd and actions[ownship_agent] == 0)
                cond2 = (ownship_agent_spd >= ownship_agent_max_spd and actions[ownship_agent] == 2)
                if cond1 or cond2:
                    actions[ownship_agent] = 1  # Maintain current speed
                    reasoning_logs[ownship_agent] = "The decided action would violate speed limits. Maintain current speed instead."

        return actions, reasoning_logs

    def get_2agent_distance(self, agent1, agent2):
        """
        Returns the distance between two agents.
        """
        lat1, lon1 = self.get_latlon(agent1)
        lat2, lon2 = self.get_latlon(agent2)
        if lat1 == np.inf or lon1 == np.inf or lat2 == np.inf or lon2 == np.inf:
            return np.inf
        return haversine_distance(lat1, lon1, lat2, lon2)
    
    def translate_action(self, action):
        """
        Translates the action from the RL agent to the action used in the environment.
        0 -> 0 (decelerate)
        1 -> 1 (hold)
        2 -> 2 (accelerate)
        """
        if action == 0:
            return 'decelerate'
        elif action == 1:
            return 'hold'
        elif action == 2:
            return 'accelerate'
        elif action == 'decelerate':
            return 0
        elif action == 'hold':
            return 1
        elif action == 'accelerate':
            return 2
        else:
            return 1
        
    def check_2agents_on_same_route(self, agent1, agent2):
        """
        Returns True if the previous and next waypoints of the two agents are the same.
        """
        agent1_nxt_wpt = self.get_nxt_wpt_id(agent1)
        agent1_prvs_wpt = self.get_prvs_wpt_id(agent1)
        agent2_nxt_wpt = self.get_nxt_wpt_id(agent2)
        agent2_prvs_wpt = self.get_prvs_wpt_id(agent2)

        return agent1_nxt_wpt == agent2_nxt_wpt and agent1_prvs_wpt == agent2_prvs_wpt
    
    def get_agent_type(self, agent):
        """
        Returns the type of the agent based on its prefix.
        """
        agent_id_prefix = self.get_agent_id_prefix(agent)
        agent_config = self.ac_configs.get(agent_id_prefix, {})
        return agent_config.get('type', None)

    def get_agent_info(self, agent, intruders, agent_flag='ownship'):
        """
        Returns a dictionary with information about the agent.
        The information includes:
            agent_type = "multirotor" if agent_id.startswith("AA") or agent_id.startswith("AC") else "fixed-wing"
            next_wpt_lat, next_wpt_lon,
            next_wpt_name,
            waypoint_type,
            distance_to_waypoint,
            speed,
            acceleration,
            heading,
            altitude
        """
        agent_info = {}
        agent_obs = self.observation(agent)
        agent_info["agent_id"] = agent
        agent_info["ac_type"] = self.get_agent_type(agent)
        agent_lat, agent_lon = self.get_latlon(agent)
        agent_lat = float(f"{agent_lat:.6f}")
        agent_lon = float(f"{agent_lon:.6f}")
        agent_info["ac_lat"], agent_info["ac_lon"] = agent_lat, agent_lon
        agent_info["next_wpt_id"] = self.get_nxt_wpt_id(agent)
        agent_info["next_wpt_type"] = self.get_nxt_wpt_type(agent)
        ac_dist_to_nxt_wpt = self.distance_to_nxt_wpt(agent) * 1852  # convert from nautical miles to meters
        ac_dist_to_nxt_wpt = float(f"{ac_dist_to_nxt_wpt:.2f}")
        agent_info["ac_dist_to_nxt_wpt(m)"] = ac_dist_to_nxt_wpt

        ac_spd = self.get_spd(agent) * 0.514444  # convert from knots to m/s
        ac_spd = float(f"{ac_spd:.2f}")
        agent_info["speed(m/s)"] = ac_spd

        # agent minimum and maximum speed
        ac_min_spd = self.get_min_spd(agent) * 0.514444  # convert from knots to m/s
        ac_min_spd = float(f"{ac_min_spd:.2f}")
        ac_max_spd = self.get_max_spd(agent) * 0.514444  # convert from knots to m/s
        ac_max_spd = float(f"{ac_max_spd:.2f}")
        agent_info["ac_min_spd(m/s)"] = ac_min_spd
        agent_info["ac_max_spd(m/s)"] = ac_max_spd

        ac_acc = self.get_spd_increment(agent) * 0.514444  # convert from knots to m/s
        ac_acc = float(f"{ac_acc:.2f}")
        agent_info["speed_change_per_second(m/s2)"] = ac_acc
        
        ac_hdg = agent_obs[0][3]
        ac_hdg = float(f"{ac_hdg:.2f}")
        agent_info["heading(deg)"] = ac_hdg

        agent_info["heading_scale"] = "{North=0, East=90, South=180, West=270}"

        ac_alt = self.get_alt(agent) / 3.28084
        ac_alt = float(f"{ac_alt:.2f}")
        agent_info["altitude(m)"] = ac_alt

        # Get route id
        agent_info["ac_route_id"] = f'R_{self.agent_route_ids[agent]}'

        # Get agent's last action
        ac_last_action = self.previous_actions.get(agent, None)
        ac_last_action = self.translate_action(ac_last_action)
        agent_info["ac_last_action"] = ac_last_action

        if agent_flag == 'ownship':

            agent_info["num_intruders_ahead"] = len(intruders)

            # Check if the agent and the first intruder are on a similar route
            if len(intruders) == 1:
                first_intruder = intruders[0]

                if self.check_2agents_on_same_route(agent, first_intruder):
                    agent_info["intruder1_on_same_route"] = True
                else:
                    agent_info["intruder1_on_same_route"] = False
                
                agent_info["intruder2_on_same_route"] = None

                ownship_to_intruder1_distance = self.get_2agent_distance(agent, first_intruder) * 1852  # convert from nautical miles to meters
                agent_info["distance_to_intruder1(m)"] = float(f"{ownship_to_intruder1_distance:.2f}")
                agent_info["distance_to_intruder2(m)"] = None
                
            elif len(intruders) == 2:
                first_intruder = intruders[0]

                if self.check_2agents_on_same_route(agent, first_intruder):
                    agent_info["intruder1_on_same_route"] = True
                else:
                    agent_info["intruder1_on_same_route"] = False

                second_intruder = intruders[1]

                if self.check_2agents_on_same_route(agent, second_intruder):
                    agent_info["intruder2_on_same_route"] = True
                else:
                    agent_info["intruder2_on_same_route"] = False

                ownship_to_intruder1_distance = self.get_2agent_distance(agent, first_intruder) * 1852  # convert from nautical miles to meters
                ownship_to_intruder2_distance = self.get_2agent_distance(agent, second_intruder) * 1852  # convert from nautical miles to meters

                agent_info["distance_to_intruder1(m)"] = float(f"{ownship_to_intruder1_distance:.2f}")
                agent_info["distance_to_intruder2(m)"] = float(f"{ownship_to_intruder2_distance:.2f}")
            else:
                agent_info["intruder1_on_same_route"] = None
                agent_info["intruder2_on_same_route"] = None
                agent_info["distance_to_intruder1(m)"] = None
                agent_info["distance_to_intruder2(m)"] = None

        return agent_info

    def ownship_intruder_info(self, observations):
        """
        Returns a list of dictionaries with information about the ownship and the first two intruders.
        """

        Actions, Reasons = self.get_rb_actions(observations)

        self.step_state_action_pairs = []
        
        for ownship_id, _ in observations.items():

            # Find attentive intruders (use your processed_states or find_intruders_ahead)
            intruders = self.get_front_intruders_ordered(ownship_id, observations)

            # find the fist two intruders
            intruders = intruders[:2]

            # Get ownship info
            ownship_info = self.get_agent_info(ownship_id, intruders, agent_flag='ownship')

            # Get intruder info
            intruders_info = [self.get_agent_info(intruder_id, intruders=[], agent_flag='intruder') for intruder_id in intruders]
            
            # Get rule action
            rule_action = "Hold" if Actions.get(ownship_id, 1) == 1 else ("Accelerate" if Actions.get(ownship_id, 1) == 2 else "Decelerate")

            if len(intruders_info) == 0:
                intruders_info = [[], []]
            elif len(intruders_info) == 1:
                intruders_info = [intruders_info[0], []]
            # ownship_info = [ownship_info, intruders_info[0], intruders_info[1]]
            state_action_pair = {"ownship_info": ownship_info, "first_closest_front_intruder_info": intruders_info[0], 
                                 "second_closest_front_intruder_info": intruders_info[1], "ownship_action": rule_action}
            self.step_state_action_pairs.append(state_action_pair)
            self.all_state_action_pairs.append(state_action_pair)

    def get_ownship_prompt(self, ownship_intruder_info):
        """
        Returns a formatted string prompt for the ownship information.
        """
        ownship_info = ownship_intruder_info['ownship_info']
        first_intruder_info = ownship_intruder_info['first_closest_front_intruder_info']
        second_intruder_info = ownship_intruder_info['second_closest_front_intruder_info']
        ownship_action = ownship_intruder_info['ownship_action']

        prompt = "You are an airspace tactical deconfliction assistant. " + \
                "Your task is to help the ownship aircraft avoid collisions with front intruder aircraft by suggesting appropriate speed adjustments. " + \
                "This task is a safety-critical task, requiring careful consideration of the situation. " + \
                "Front intruders are aircraft that are ahead of the ownship and closer to the next waypoint, no matter on the same route or a different route. " + \
                f"At each time step, the ownship can choose one of the following actions: 'Accelerate' (increase speed by {ownship_info['speed_change_per_second(m/s2)']} m/s per second), " + \
                f"'Decelerate' (decrease speed by -{ownship_info['speed_change_per_second(m/s2)']} m/s per second), or 'Hold' its current speed. " + \
                f"The simulation time step is {self.dt} seconds. \n" + \
                "Based on the following information about the ownship and its intruders, recommend one action to adjust " + \
                "its current speed to maintain safe separation. \nHere is the ownship information: \n"

        # prompt += f"Ownship ID: {ownship_info['agent_id']}, "
        prompt += f"Aircraft Type: {ownship_info['ac_type']}, "
        # prompt += f"Latitude: {ownship_info['ac_lat']}, Longitude: {ownship_info['ac_lon']}, "
        prompt += f"Next Waypoint ID: {ownship_info['next_wpt_id']}, Next Waypoint Type: {ownship_info['next_wpt_type']}, "
        prompt += f"Distance to Next Waypoint (m): {ownship_info['ac_dist_to_nxt_wpt(m)']}, "
        prompt += f"Speed (m/s): {ownship_info['speed(m/s)']}, "
        prompt += f"Speed Range (m/s): [{ownship_info['ac_min_spd(m/s)']}, {ownship_info['ac_max_spd(m/s)']}], "
        prompt += f"Speed Change per Second (m/s²): {ownship_info['speed_change_per_second(m/s2)']}, "
        # prompt += f"Acceleration (knot/s): {ownship_info['acceleration(knot/s)']}, "
        prompt += f"Heading (deg): {ownship_info['heading(deg)']} with heading scale {ownship_info['heading_scale']}, "
        # prompt += f"Altitude (ft): {ownship_info['altitude(ft)']}, "
        # prompt += f"Route ID: {ownship_info['ac_route_id']}, "
        prompt += f"Last Action: {ownship_info['ac_last_action']}, "
        
        safety_distance = self.get_agent_safety_distance(ownship_info['agent_id']) * 1852  # convert from nautical miles to meters
        ownship_collision_dist = self.get_agent_collision_distance(ownship_info['agent_id']) * 1852  # convert from nautical miles to meters
        prompt += f"The safety distance to maintain from intruders is {safety_distance:.2f} (m), which is a safe boundary around the ownship when there is no intruder within it. "
        prompt += f"The near mid-air collision distance is {ownship_collision_dist:.2f} (m), which is the most critical distance between the ownship and the intruder. \n"
        prompt += "Here are the intruder details: \n"
        prompt += f"Number of Intruders Ahead: {ownship_info['num_intruders_ahead']}. "
        if ownship_info['num_intruders_ahead'] == 1:
            prompt += "The first closest front intruder information: \n"
            # prompt += f"Intruder ID: {first_intruder_info['agent_id']}, "
            prompt += f"Type: {first_intruder_info['ac_type']}, "
            # prompt += f"Latitude: {first_intruder_info['ac_lat']}, Longitude: {first_intruder_info['ac_lon']}, "
            prompt += f"Next Waypoint ID: {first_intruder_info['next_wpt_id']}, Next Waypoint Type: {first_intruder_info['next_wpt_type']}, "
            prompt += f"Distance to Next Waypoint (m): {first_intruder_info['ac_dist_to_nxt_wpt(m)']}, "
            prompt += f"Speed (m/s): {first_intruder_info['speed(m/s)']}, "
            # prompt += f"Acceleration (knot/s): {first_intruder_info['acceleration(knot/s)']}, "
            prompt += f"Heading (deg): {first_intruder_info['heading(deg)']}, "
            # prompt += f"Altitude (ft): {first_intruder_info['altitude(m)']}, "
            # prompt += f"Route ID: {first_intruder_info['ac_route_id']}, "
            prompt += f"Distance to the ownship (m): {ownship_info['distance_to_intruder1(m)']}. "
            if ownship_info['intruder1_on_same_route']:
                prompt += "The first intruder and the ownship are on the same route. \n"
            else:
                prompt += "The first intruder and the ownship are not on the same route. \n"
            
        elif ownship_info['num_intruders_ahead'] == 2:
            prompt += "The first closest front intruder information: \n"
            # prompt += f"Intruder ID: {first_intruder_info['agent_id']}, "
            prompt += f"Type: {first_intruder_info['ac_type']}, "
            # prompt += f"Latitude: {first_intruder_info['ac_lat']}, Longitude: {first_intruder_info['ac_lon']}, "
            prompt += f"Next Waypoint ID: {first_intruder_info['next_wpt_id']}, Next Waypoint Type: {first_intruder_info['next_wpt_type']}, "
            prompt += f"Distance to Next Waypoint (m): {first_intruder_info['ac_dist_to_nxt_wpt(m)']}, "
            prompt += f"Speed (m/s): {first_intruder_info['speed(m/s)']}, "
            # prompt += f"Acceleration (knot/s): {first_intruder_info['acceleration(knot/s)']}, "
            prompt += f"Heading (deg): {first_intruder_info['heading(deg)']}, "
            # prompt += f"Altitude (ft): {first_intruder_info['altitude(m)']}, "
            # prompt += f"Route ID: {first_intruder_info['ac_route_id']}, "
            prompt += f"Distance to the ownship (m): {ownship_info['distance_to_intruder1(m)']}. "
            if ownship_info['intruder1_on_same_route']:
                prompt += "The first intruder and the ownship are on the same route. \n"
            else:
                prompt += "The first intruder and the ownship are not on the same route. \n"
            # prompt += f"Are the first intruder and the ownship on the same route? {ownship_info['intruder1_on_same_route']}. \n"
            
            prompt += "Tthe second closest front intruder information: \n"
            # prompt += f"Intruder ID: {second_intruder_info['agent_id']}, "
            prompt += f"Type: {second_intruder_info['ac_type']}, "
            # prompt += f"Latitude: {second_intruder_info['ac_lat']}, Longitude: {second_intruder_info['ac_lon']}, "
            prompt += f"Next Waypoint ID: {second_intruder_info['next_wpt_id']}, Next Waypoint Type: {second_intruder_info['next_wpt_type']}, "
            prompt += f"Distance to Next Waypoint (m): {second_intruder_info['ac_dist_to_nxt_wpt(m)']}, "
            prompt += f"Speed (m/s): {second_intruder_info['speed(m/s)']}, "
            # prompt += f"Acceleration (knot/s): {second_intruder_info['acceleration(knot/s)']}, "
            prompt += f"Heading (deg): {second_intruder_info['heading(deg)']}, "
            # prompt += f"Altitude (ft): {second_intruder_info['altitude(ft)']}, "
            # prompt += f"Route ID: {second_intruder_info['ac_route_id']}, "
            prompt += f"Distance to the ownship (m): {ownship_info['distance_to_intruder2(m)']}. "
            if ownship_info['intruder2_on_same_route']:
                prompt += "The second intruder and the ownship are on the same route. \n"
            else:
                prompt += "The second intruder and the ownship are not on the same route. \n"

        else:
            prompt += "There are no front intruders detected, so try to avoid occlusion. \n"

        prompt += f"Based on the above information, what actions should the ownship take? (Decelerate/Hold/Accelerate) \n"
        return prompt