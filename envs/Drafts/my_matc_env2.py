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

from .scenario.utils import calculate_heading, metric_spd_to_knots_rounded
from .scenario.scenario import ATCScenario

class ObservationTypes(Enum):
    LAT = 0
    LON = 1
    CAS = 2
    HDG = 3
    ACC = 4
    ROUTE = 5
    ACT = 6

# Define the min and max values for each observation type
OBSERVATION_BOUNDS = {
    ObservationTypes.LAT: (lambda self: self.scenario.bounding_cicle.min_lat, lambda self: self.scenario.bounding_cicle.max_lat),
    ObservationTypes.LON: (lambda self: self.scenario.bounding_cicle.min_lon, lambda self: self.scenario.bounding_cicle.max_lon),
    ObservationTypes.CAS: (0, 400),
    ObservationTypes.HDG: (0, 360),
    ObservationTypes.ACC: (-np.inf, np.inf),
    ObservationTypes.ROUTE: (0, 3),
    ObservationTypes.ACT: (0, 5)
}


class MATCEnv(ParallelEnv):
    metadata = {"render_modes": ['sim', 'server', 'export_scenario']}

    def __init__(self, flight_route_file: str, max_spd_shift: int = 1, spd_increment: int = 10,
                 render_mode: Literal['sim', 'export_scenario', 'server'] = 'sim', noise: bool = True,
                 dt: float = 1.0, zoom: float = 0.4):

        self.render_mode = render_mode
        self.noise = noise
        self.spd_increment = spd_increment
        self.dt = dt
        self.zoom = zoom
        self.max_spd_shift = max_spd_shift

        # Initialize bluesky
        bs.init(
            mode='sim',
            detached=(self.render_mode != 'server')
        )
        if self.render_mode == 'server':
            print('Waiting to connect to server...')
            bs.net.connect() 
            bs.net.update() 
            print('Connected to server!')


        self.scenario = ATCScenario(flight_route_file)
        self.schedule = self.scenario.sample_schedule()
        self.time_step = 0
        self.lat_lon_accs = {}

        # Ownship and intruders distance threshold
        self.ownship_distance_threshold = 3 # nautical miles
        self.safety_distance = self.ownship_distance_threshold * 5

        # Speed limits
        self.min_speed = 50
        self.avg_speed = 100
        self.max_speed = 200
        
        self.distances_to_ownship = {}
        self.actions = {}
        self.agent_counter = {}
        self.truncated_agents = {}
        self.n_spawn = 10
        self.out_of_energy_time = 1600
        self.collided_agents = []
        self.each_agent_rewards = {}
        self.previous_actions = {}
        self.terminated_agents = {}
        self.agent_route_ids = {}
        self.previous_speeds = {}
        self.current_speeds = {}
        self.agent_goal_locations = {}
        self.sum_episode_rewards = 0
        self.num_of_collisions = 0
        self.dones = {}


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        low = np.array([OBSERVATION_BOUNDS[obs_type][0](self) if callable(OBSERVATION_BOUNDS[obs_type][0]) else OBSERVATION_BOUNDS[obs_type][0] for obs_type in ObservationTypes])
        high = np.array([OBSERVATION_BOUNDS[obs_type][1](self) if callable(OBSERVATION_BOUNDS[obs_type][1]) else OBSERVATION_BOUNDS[obs_type][1] for obs_type in ObservationTypes])
        return spaces.Sequence(spaces.Box(low, high), stack=True)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.max_spd_shift * 2 + 1, start=-self.max_spd_shift)

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
        
        if seed:
            assert 0 <= seed <= 2*32 - 1, 'TEST FAILED: Seed must be between 0 and 2^32 - 1'
            self.np_random, seed = seeding.np_random(seed)
        else: 
            seed = np.random.randint(0, 2**32)
            self.np_random, seed = seeding.np_random(seed)

        for route in self.scenario.flight_routes:
            route_distribution = route.route_distribution
            for distribution in route_distribution:
                # Set random state for distributions
                distribution.spawn.random_state = self.np_random
                distribution.spacing.random_state = self.np_random
           
        stack.stack(f'SEED {seed}') # FIXME: LEGACY SEEDING

    
    def reset(self, seed=None, options=None):
        # schedule = []

        # while len(schedule) < 1:
        #     # Ensure at least 1 aircraft is spawned
        #     schedule = self.scenario.sample_schedule()

        stack.stack('RESET')
        self._seed(seed)
        
        for waypoint in self.scenario.waypoints:
            stack.stack(f'DEFWPT {waypoint.name},{waypoint.coordinate}')

        stack.stack(f'PAN {self.scenario.bounding_cicle.lat_center},{self.scenario.bounding_cicle.lon_center}')
        stack.stack(f'ZOOM {self.zoom}')

        self.scenario.render_environment()

        stack.stack('CDMETHOD STATEBASED')
        stack.stack(f"NOISE {'ON' if self.noise else 'OFF'}")

        stack.stack(f'DT {self.dt}')
        stack.stack('OP')

        self.agent_init_wpts = {}
        for flight in self.schedule:
            origin = flight.path[0]
            destination = flight.path[-1]
            initial_heading = calculate_heading(origin.waypoint.coordinate, destination.waypoint.coordinate)
            initial_speed = origin.speed 
            
            stack.stack(f'DELAY {flight.entry_time} CRE {flight.flight_number} {flight.aircraft_type} {origin.waypoint.coordinate} {initial_heading} {origin.altitude} {initial_speed if not origin.end else 60}')

            if origin.end:
                stack.stack(f'DELAY {flight.entry_time} ORIG {flight.flight_number} {origin.waypoint.name}')

            for wpt in flight.path[0:-1]:
                stack.stack(f'DELAY {flight.entry_time} ADDWPT {flight.flight_number} {wpt.waypoint.name} {wpt.altitude},{wpt.speed}')
            
            
            if destination.end:
                stack.stack(f'DELAY {flight.entry_time} DEST {flight.flight_number} {destination.waypoint.name}')
            else:
                stack.stack(f'DELAY {flight.entry_time} ADDWPT {flight.flight_number} {destination.waypoint.name} {destination.altitude},{destination.speed}')

            stack.stack(f'DELAY {flight.entry_time} {flight.flight_number} AT {destination.waypoint.name} STACK DEL {flight.flight_number}')


            stack.stack(f'DELAY {flight.entry_time} LNAV {flight.flight_number} ON')
            stack.stack(f'DELAY {flight.entry_time} VNAV {flight.flight_number} ON')

            # draw a circle around waypoints
            # waypoint #3
            stack.stack(f'DELAY {flight.entry_time} CIRCLE C1, 32.90437, -97.15455, 8')
            stack.stack(f'DELAY {flight.entry_time} CIRCLE C2, 32.90437, -97.15455, 3')

            # waypoint #5
            stack.stack(f'DELAY {flight.entry_time} CIRCLE C3, 32.92050, -96.80013, 8')
            stack.stack(f'DELAY {flight.entry_time} CIRCLE C4, 32.92050, -96.80013, 3')

            # waypoint #7
            stack.stack(f'DELAY {flight.entry_time} CIRCLE C5, 32.90437, -96.43877, 8')
            stack.stack(f'DELAY {flight.entry_time} CIRCLE C6, 32.90437, -96.43877, 3')

            # Get initial waypoints
            self.agent_init_wpts[flight.flight_number] = flight.path[0].waypoint.name
            
            # Convert destination waypoint coordinates from string of numbers to tuple of floats
            destination_coordinates = (destination.waypoint.coordinate.latitude, destination.waypoint.coordinate.longitude)
            # Store the destination coordinates in the agent_goal_locations dictionary
            self.agent_goal_locations[flight.flight_number] = destination_coordinates

        self.agent_route_ids = {}
        # Convert initial waypoints to route IDs for each agent
        for agent, init_wpt in self.agent_init_wpts.items():
            if init_wpt == "WP01":
                route_id = 1
            elif init_wpt == "WP02":
                route_id = 2
            elif init_wpt == "WP4":
                route_id = 3
            else:
                route_id = None
            self.agent_route_ids[agent] = route_id

        while len(self.agents) < 1:
            self._sim_step()

        current_step_agents = self.agents
        observations = self.observations(current_step_agents)
        infos = self.infos(current_step_agents)

        # Reset the parameters
        self.agent_counter = {}
        self.truncated_agents = {}
        self.each_agent_rewards = {}
        self.terminated_agents = {}
        self.previous_actions = {}
        self.sum_episode_rewards = 0
        self.num_of_collisions = 0
        self.collided_agents = []
        self.dones = {}

        return observations, infos

    @property
    def agents(self):
        return bs.traf.id

    AgentInfo = TypedDict('AgentInfo', {
        'CONFLICT': bool,
        'MAXIMUM_CLOSEST_POINT_OF_APPROACH_TIME': str,
        'AIRCRAFT_MINIMUM_HORIZONTAL_SEPARATION': float,
        'AIRCRAFT_MINIMUM_VERTICAL_SEPARATION': float,
        'AIRCRAFT_TYPE': str,
        'PHASE': float,
        'THRUST': float,
        'FUEL_FLOW': float,
        'DRAG': float,
        'DISTANCE_FLOWN': float
    })
    def info(self, agent) -> AgentInfo:
        ac_id = bs.traf.id2idx(agent)
        if ac_id == -1:
            return {}

        return {
            'CONFLICT': bs.traf.cd.inconf[ac_id].item(),
            'MAXIMUM_CLOSEST_POINT_OF_APPROACH_TIME': bs.traf.cd.tcpamax[ac_id].item(),
            'AIRCRAFT_MINIMUM_HORIZONTAL_SEPARATION': bs.traf.cd.rpz[ac_id].item(),
            'AIRCRAFT_MINIMUM_VERTICAL_SEPARATION': bs.traf.cd.rpz[ac_id].item(),
            'AIRCRAFT_TYPE': bs.traf.type[ac_id],
            'PHASE': bs.traf.perf.phase[ac_id].item(),
            'THRUST': bs.traf.perf.thrust[ac_id].item(),
            'FUEL_FLOW': bs.traf.perf.fuelflow[ac_id].item(),
            'DRAG': bs.traf.perf.drag[ac_id].item(),
            'DISTANCE_FLOWN': bs.traf.distflown[ac_id].item()
        }
    
    def infos(self, current_step_agents):
        return {agent: self.info(agent) for agent in current_step_agents}

    def check_termination(self, agent):
        if agent not in self.terminated_agents.keys():
            self.terminated_agents[agent] = False
            terminated = False
        if self._agent_has_terminated(agent):
            self.terminated_agents[agent] = True
            terminated = True
        else:
            terminated = False
        return terminated

    def terminate_agents(self):
        for agent in self.terminated_agents.keys():
            if self._agent_has_terminated(agent):
                stack.stack(f'DEL {agent}')

    def truncate_agents(self):
        for agent in self.truncated_agents.keys():
            if self.truncated_agents[agent]:
                stack.stack(f'DEL {agent}')

    def check_done(self, agent):
        if agent in self.truncated_agents.keys() or agent in self.terminated_agents:
            if agent in self.truncated_agents.keys():
                return self.truncated_agents[agent]
            elif agent in self.terminated_agents:
                return True
        else:
            return False

    def check_done_agents(self, current_step_agents):
        for agent in current_step_agents:
            if agent not in self.dones.keys():
                self.dones[agent] = False

            if self.check_done(agent):
                self.dones[agent] = True

    def update_observations(self, observations, current_step_agents):
        for agent in current_step_agents:
            if agent in self.truncated_agents.keys():
                del observations[agent]
        return observations
     
    def get_spd(self, agent):
        ac_id = bs.traf.id2idx(agent)
        if ac_id == -1:
            return 0
        # Get the current speed of the aircraft
        cur_ac_spd = metric_spd_to_knots_rounded(bs.traf.cas[ac_id])
        return cur_ac_spd

    def get_all_agent_spds(self):
        return {agent: self.get_agent_spds(agent)[0] for agent in self.agents}
    
    def get_agent_spds(self, agent):
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
    
    def draw_circle(self):
        # draw a circle around the aircraft
        stack.stack(f'CIRCLE C1, 32.90437, -97.15455, 1000') 

    def observation(self, agent):
        ownship_ac_idx = bs.traf.id2idx(agent)
        if ownship_ac_idx == -1:
            return np.zeros_like(len(ObservationTypes))

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
        abs_vel, _ = self.get_agent_spds(agent)

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

        ownship_observations = np.array([[
            bs.traf.lat[ownship_ac_idx],
            bs.traf.lon[ownship_ac_idx],
            abs_vel, 
            bs.traf.hdg[ownship_ac_idx],
            abs_acceleration,
            route_id,
            prev_action
        ]]).reshape(1, -1)
        return ownship_observations
    
    def observations(self, current_step_agents):
        return {agent: self.observation(agent) for agent in current_step_agents}
    
    def distance_reward(self, distance):
        if distance <= self.ownship_distance_threshold:
            return -1, True
        elif self.ownship_distance_threshold < distance < self.safety_distance:
            return -(distance/self.safety_distance - 1) ** 2, False
        else:
            return 0, False

    def velocity_reward(self, agent):
        desired_spd = self.avg_speed
        if self.check_termination(agent):
            velocity_reward = 0
        else:
            current_spd = metric_spd_to_knots_rounded(bs.traf.cas[bs.traf.id2idx(agent)])
            velocity_reward = -(current_spd / desired_spd - 1) ** 2
        
        return velocity_reward

    def reward(self, agent):

        if agent not in self.truncated_agents.keys():
            self.truncated_agents[agent] = False

        # Velocity Reward = (current speed/desired speed - 1)^2
        velocity_reward = self.velocity_reward(agent)
        
        # Count agent for each time step
        if agent not in self.agent_counter.keys():
            self.agent_counter[agent] = 0
        self.agent_counter[agent] += 1

        # Encouraging agents to finish the route
        truncated_flag = False
        if self.agent_counter[agent] == self.out_of_energy_time:
            time_reward = -1
            self.truncated_agents[agent] = True
        elif self.agent_counter[agent] > self.out_of_energy_time:
            time_reward = -1
            self.truncated_agents[agent] = True
        else:
            if agent in self.terminated_agents:
                time_reward = 1
                truncated_flag = True
            else:
                if not truncated_flag:
                    time_reward = -0.001
                    self.truncated_agents[agent] = False
                else:
                    time_reward = 0

        # action reward ******************************************************************
        # if action is not equal to the previous action receives -1
        action_reward = 0
        if agent in self.actions.keys():
            if agent in self.previous_actions.keys():
                if self.actions[agent] != self.previous_actions[agent]:
                    action_reward = -1
                else:
                    action_reward = 0
            else:
                action_reward = 0

        # action_reward = 0
        # if agent in self.actions.keys():
        #     action = self.actions[agent]
        #     if action == 1:
        #         action_reward = 0
        #     else:
        #         action_reward = -1

        # Distance penalty
        distance_reward = 0
        if agent in self.distances_to_ownship.keys():
            for intruder, distance_to_ownship in self.distances_to_ownship[agent].items():
                dist_reward, truncation_flag = self.distance_reward(distance_to_ownship)
                distance_reward += dist_reward
                if truncation_flag:
                    self.truncated_agents[agent] = True
                    self.truncated_agents[intruder] = True
                    self.collided_agents.append((agent, intruder))
                    self.num_of_collisions += 0.5

        # print(distance_reward)

        # # Terminate the agent
        # if time_reward == -1:
        #     self.truncated_agents[agent] = True
        # else:
        #     self.truncated_agents[agent] = False

        w1, w2, w3, w4 = 1.0, 0.01, 0.01, 0.1
        total_reward = w1 * distance_reward #+ w2 * velocity_reward #+ w4 * time_reward #+ w3 * action_reward 
        self.sum_episode_rewards += total_reward
        # total_reward =  w1 * distance_reward

        # Add subrewards to the each step rewards
        if agent not in self.each_agent_rewards.keys():
            self.each_agent_rewards[agent] = [[w1*distance_reward, w2*velocity_reward, w3*action_reward, w4*time_reward]]
        else:
            self.each_agent_rewards[agent].append([w1*distance_reward, w2*velocity_reward, w3*action_reward, w4*time_reward])

        return np.float64(total_reward)
    

    def rewards(self, previous_step_agents):
        return {agent: self.reward(agent) for agent in previous_step_agents}

    def step(self, actions):
        # self.draw_circle()
        if not actions:
            return {}, {}, {}, {}, {}
        
        # update actions
        self.actions = actions

        for agent, action in actions.items():
            ac_id = bs.traf.id2idx(agent)
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
            
            if action != 1:
                # No action taken                 
                cur_ac_spd = metric_spd_to_knots_rounded(bs.traf.cas[ac_id])
                new_ac_spd = cur_ac_spd + (action - 1) * self.spd_increment
                new_ac_spd = round(new_ac_spd / self.spd_increment) * self.spd_increment

                # if action == 0:
                #     new_ac_spd = self.min_speed
                # else:
                #     new_ac_spd = self.max_speed

                # new_ac_spd = max(self.min_speed, min(new_ac_spd, self.max_speed))
                # print(f"Current speed: {cur_ac_spd} New speed: {new_ac_spd}")                
                stack.stack(f'{flight_number} SPD {new_ac_spd}')

        # Simulate the environment for one step
        self._sim_step()
        
        old_agents = self.agents.copy()
        current_step_agents = set(self.agents).union(old_agents)
        previous_step_agents = list(actions.keys())

        observations = self.observations(current_step_agents)
        rewards = self.rewards(previous_step_agents)
        terminations = self.terminated_agents.copy()
        truncations = self.truncated_agents.copy()
        infos = self.infos(current_step_agents)

        # Update previous actions
        self.previous_actions = actions
        self.previous_speeds = self.current_speeds

        # Truncate and terminate agents after receiving next observations and rewards
        self.truncate_agents()
        self.terminate_agents()
        self.check_done_agents(current_step_agents)

        return observations, rewards, terminations, truncations, self.dones

