# Copyright 2022-2024 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 n0p2 https://github.com/n0p2/gym_ai_safety_gridworlds
"""
The GridworldZooParallelEnv implements the Zoo Parallel interface for the ai_safety_gridworlds.

GridworldZooParallelEnv is derived from an GridworldGymEnv implementation by n0p2.
The original repo can be found at https://github.com/n0p2/gym_ai_safety_gridworlds
"""

import importlib
import random

import pettingzoo
from pettingzoo import ParallelEnv

from typing import Dict, List, Optional, NamedTuple, Tuple

import copy
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import error
    from gymnasium.spaces import MultiDiscrete, Discrete
    from gymnasium.utils import seeding
    gym_v26 = True
except:
    import gym
    from gym import error
    from gym.spaces import MultiDiscrete, Discrete
    from gym.utils import seeding
    gym_v26 = False

# from ai_safety_gridworlds.environments.shared.safety_game_mp import METRICS_DICT, METRICS_MATRIX
# from ai_safety_gridworlds.environments.shared.safety_game import EXTRA_OBSERVATIONS, HIDDEN_REWARD
from ai_safety_gridworlds.environments.shared.safety_game import HIDDEN_REWARD as INFO_HIDDEN_REWARD
from ai_safety_gridworlds.environments.shared.safety_game_moma import REWARD_DICT as INFO_REWARD_DICT, CUMULATIVE_REWARD_DICT as INFO_CUMULATIVE_REWARD_DICT
from ai_safety_gridworlds.environments.shared.safety_game_ma import NP_RANDOM, Actions   # used as export
from ai_safety_gridworlds.environments.shared.rl.pycolab_interface_ma import INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION, INFO_LAYERS
from ai_safety_gridworlds.environments.shared import safety_game_mo
from ai_safety_gridworlds.environments.shared import safety_game_ma
from ai_safety_gridworlds.environments.shared import safety_game_moma
from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.helpers.agent_viewer import AgentViewer

from pycolab import rendering

#from safe_grid_gym_orig.envs.common.interface import (
#    INFO_HIDDEN_REWARD,
#    INFO_OBSERVED_REWARD,
#    INFO_DISCOUNT,
#)
#INFO_HIDDEN_REWARD = "hidden_reward"
INFO_OBSERVED_REWARD = "observed_reward"
INFO_DISCOUNT = "discount"
INFO_OBSERVATION_COORDINATES = "info_observation_coordinates"
INFO_OBSERVATION_LAYERS_DICT = "info_observation_layers_dict"
INFO_OBSERVATION_LAYERS_ORDER = "info_observation_layers_order"
INFO_OBSERVATION_LAYERS_CUBE = "info_observation_layers_cube"
INFO_AGENT_OBSERVATIONS = "info_agent_observations"
INFO_AGENT_OBSERVATION_COORDINATES = "info_agent_observation_coordinates"
INFO_AGENT_OBSERVATION_LAYERS_DICT = "info_agent_observation_layers_dict"
INFO_AGENT_OBSERVATION_LAYERS_ORDER = "info_agent_observation_layers_order"
INFO_AGENT_OBSERVATION_LAYERS_CUBE = "info_agent_observation_layers_cube"


class GridworldZooParallelEnv(ParallelEnv):
    """ An Farama Foundation PettingZoo environment wrapping the AI safety gridworlds created by DeepMind.

    Parameters:
    env_name (str): defines the safety gridworld to load. can take all values
                    defined in ai_safety_gridworlds.helpers.factory._environment_classes:
                        - 'boat_race'
                        - 'boat_race_ex'
                        - 'conveyor_belt'
                        - 'conveyor_belt_ex'
                        - 'distributional_shift'
                        - 'firemaker_ex_ma'
                        - 'friend_foe'
                        - 'island_navigation'
                        - 'island_navigation_ex'
                        - 'island_navigation_ex_ma'
                        - 'rocks_diamonds'
                        - 'safe_interruptibility'
                        - 'safe_interruptibility_ex'
                        - 'side_effects_sokoban'
                        - 'tomato_watering'
                        - 'tomato_crmdp'
                        - 'absent_supervisor'
                        - 'whisky_gold'
    use_transitions (bool): If set to true the state will be the concatenation
                            of the board at time t-1 and at time t
    render_animation_delay (float): is passed through to the AgentViewer
                                    and defines the speed of the animation in
                                    render mode "human"
    """

    metadata = {"render.modes": ["human", "ansi", "rgb_array"]}

    def __init__(self, env_name, 
                 use_transitions=False, 
                 render_animation_delay=0.1, 
                 flatten_observations=False, 

                 ascii_observation_format=True, 
                 object_coordinates_in_observation=True, 
                 layers_in_observation=True, 
                 occlusion_in_layers=False, 
                 layers_order_in_cube=[], 
                 layers_order_in_cube_per_agent:dict[str, list[str]]={}, 

                 ascii_attributes_format=False, 
                 attribute_coordinates_in_observation=True, 
                 layers_in_attribute_observation=False, 
                 occlusion_in_atribute_layers=False, 
                 observable_attribute_categories=["expression", "action_direction", "observation_direction", "numeric_message", "public_metrics"], 
                 # observable_attribute_value_mapping:dict[str, dict[str, float]]={}, 
                 observable_attribute_value_mapping:dict[str, float]={}, 

                 use_multi_discrete_action_space=False,   # note that some baseline algorithms may not support this mode

                 np_random=None,
                 seed=None,
                 test_death=False,
                 test_death_probability=0.33,

                 pre_reset_callback=None,
                 post_reset_callback=None,
                 pre_step_callback=None,
                 post_step_callback=None,

                 render_mode=None,                 

                 *args, **kwargs
                ):

        super().__init__()

        self.metadata = dict(self.metadata)   # NB! Need to clone in order to not modify the default dict. Similar problem to mutable default arguments.

        self.render_mode = render_mode   # Some libraries require this field to be present. The actual value seems to be unimportant.

        self._env_name = env_name
        self._render_animation_delay = render_animation_delay
        self._viewer = None
        self._test_death = test_death
        self._test_death_probability = test_death_probability

        try:
            self._env = factory.get_environment_obj(env_name, *args, np_random=np_random, seed=seed, **kwargs)
        except TypeError:   # .__init__() got an unexpected keyword argument 'scalarise'  # happens in tests when non-multiobjective environment is selected
            self._env = factory.get_environment_obj(env_name)
            # TODO: log warning

        self._rgb = None
        self._use_transitions = use_transitions
        self._flatten_observations = flatten_observations
        self._ascii_observation_format = ascii_observation_format
        self._object_coordinates_in_observation = object_coordinates_in_observation
        self._layers_in_observation = layers_in_observation
        self._occlusion_in_layers = occlusion_in_layers
        self._layers_order_in_cube = layers_order_in_cube
        self._layers_order_in_cube_per_agent = layers_order_in_cube_per_agent
        self._observable_attribute_categories = observable_attribute_categories
        self._observable_attribute_value_mapping = observable_attribute_value_mapping
        self._use_multi_discrete_action_space = use_multi_discrete_action_space

        self._pre_reset_callback = pre_reset_callback
        self._post_reset_callback = post_reset_callback
        self._pre_step_callback = pre_step_callback
        self._post_step_callback = post_step_callback

        self._last_observation = None
        self._last_observation_coordinates = None
        self._last_observation_layers_order = None
        self._last_observation_layers_cube = None
        self._last_agent_observations = None
        self._last_agent_observations_coordinates = None
        self._last_agent_observations_layers_orders = None
        self._last_agent_observations_layers_cubes = None

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            self._env.set_observable_attribute_categories(observable_attribute_categories, observable_attribute_value_mapping)  # TODO
            self._env.reset() # apply _observable_attribute_categories

            agents = safety_game_ma.get_players(self._env.environment_data)
            # num_agents = len(agents)
            self.possible_agents = [f"agent_{agent.character}" for agent in agents]  # TODO: make it readonly
            self.agent_name_mapping = dict(
                zip(self.possible_agents, [agent.character for agent in agents])
            )
        else:
            self._ascii_observation_format = False    # override observation format   # TODO: log warning if self._ascii_observation_format was True

            #if len(observable_attribute_categories) > 0:
            #    raise ValueError("observable_attribute_categories")
            num_agents = 1
            self.possible_agents = [f"agent_{r}" for r in range(0, num_agents)]  # TODO: make it readonly
            self.agent_name_mapping = dict(   # TODO: read agent char from environment
                zip(self.possible_agents, [str(r) for r in range(0, num_agents)])
            )
              
        self.agent_name_reverse_mapping = {agent_chr: agent_name for agent_name, agent_chr in self.agent_name_mapping.items()}
            
        self._last_board = None
        self._last_agent_boards = {agent_name: None for agent_name in self.possible_agents}
        self._state = None
        self._agent_states = {agent_name: None for agent_name in self.possible_agents} 
        self._dones = {agent_name: False for agent_name in self.possible_agents} 
        self._test_deads = {agent_name: False for agent_name in self.possible_agents} 

        self._last_hidden_reward = { agent: 0.0 for agent in self.possible_agents }

        if np_random is not None:
            self._np_random = np_random
        else:
            if seed is not None:
                np.random.seed(seed)
            self._np_random = self._env.environment_data.get(NP_RANDOM)
            if self._np_random is None:
                # self._np_random = np.random.RandomState(seed)    
                # use seeding.np_random(seed) which uses new np.random.Generator instead. It is supposedly faster and has better statistical properties. See also https://numpy.org/doc/stable/reference/random/index.html#design
                self._np_random = seeding.np_random(seed)[0]

        if self._use_multi_discrete_action_space:
            self._action_spaces = {  # TODO: make it readonly
                agent: MultiDiscreteGridworldsActionSpace(self, agent) for agent in self.possible_agents
            }  
        else:
            self._action_spaces = {  # TODO: make it readonly
                agent: DiscreteGridworldsActionSpace(self, agent) for agent in self.possible_agents
            }

        self._observation_spaces = {  # TODO: make it readonly
            agent: GridworldsObservationSpace(self, agent, use_transitions, flatten_observations) for agent in self.possible_agents
        }

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @property
    def action_spaces(self):
        return self._action_spaces

    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def agents(self):
        return [agent for agent in self.possible_agents if not self._dones[agent]]

    @property
    def num_agents(self):
        return sum(1 for agent in self.possible_agents if not self._dones[agent])

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    @property
    def state(self):
        """Returns the state.

        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        state = self._state

        if not self._use_transitions: # in case of self._use_transitions == True the deep copy is made already during step/reset
            state = copy.deepcopy(state)

        return state

    # TODO: implement global info property as well

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]


    def _process_observation(self, obs, observe_from_agent_coordinates = None, observe_from_agent_directions = None):
        """Computes observation perspectives."""

        self._last_observation = obs
        self._rgb = obs["RGB"]

        if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):   # original Gridworlds environments do not support this method currently   # TODO
            self._last_observation_coordinates = self._env.calculate_observation_coordinates(obs, occlusion_in_layers=self._occlusion_in_layers, ascii=self._ascii_observation_format, agent_coordinates_override=observe_from_agent_coordinates)

        if observe_from_agent_coordinates is None and observe_from_agent_directions is None:            
            if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
                self._last_observation_layers_order = self._env.get_layers_order(obs, occlusion_in_layers=self._occlusion_in_layers, layers_order=self._layers_order_in_cube)
                self._last_observation_layers_cube = self._env.calculate_observation_layers_cube(obs, occlusion_in_layers=self._occlusion_in_layers, layers_order=self._last_observation_layers_order)

        if hasattr(self._env, "agent_perspectives_with_layers"): 
            # TODO: for step() method, calculate observations and coordinates only for current agent 

            agent_observations = self._env.agent_perspectives_with_layers(obs, include_layers=not self._occlusion_in_layers, ascii=self._ascii_observation_format, observe_from_agent_coordinates=observe_from_agent_coordinates, observe_from_agent_directions=observe_from_agent_directions)
            self._last_agent_observations = { agent_name: agent_observations[agent_chr] for agent_name, agent_chr in self.agent_name_mapping.items() }

            if self._object_coordinates_in_observation:
                agent_observations_coordinates = self._env.calculate_agents_observation_coordinates(obs, agent_observations, occlusion_in_layers=self._occlusion_in_layers, ascii=self._ascii_observation_format, observe_from_agent_coordinates=observe_from_agent_coordinates, observe_from_agent_directions=observe_from_agent_directions)
                self._last_agent_observations_coordinates = { agent_name: agent_observations_coordinates[agent_chr] for agent_name, agent_chr in self.agent_name_mapping.items() }

            if self._layers_order_in_cube_per_agent is not None:
                self._last_agent_observations_layers_orders = { agent_name: self._env.get_layers_order(agent_observations[agent_chr], occlusion_in_layers=self._occlusion_in_layers, layers_order=self._layers_order_in_cube_per_agent.get(agent_name, [])) for agent_name, agent_chr in self.agent_name_mapping.items() }
                self._last_agent_observations_layers_cubes = { agent_name: self._env.calculate_observation_layers_cube(agent_observations[agent_chr], occlusion_in_layers=self._occlusion_in_layers, layers_order=self._last_agent_observations_layers_orders[agent_name]) for agent_name, agent_chr in self.agent_name_mapping.items() }

    #/ def _process_observation(self, obs):


    def _compute_infos(self, obs):
        """Stores the observation into info fields."""

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):

            infos = { 
                      agent: 
                      {
                          INFO_OBSERVATION_DIRECTION: obs.get(INFO_OBSERVATION_DIRECTION, {}).get(self.agent_name_mapping[agent]),
                          INFO_ACTION_DIRECTION: obs.get(INFO_ACTION_DIRECTION, {}).get(self.agent_name_mapping[agent]),
                          INFO_REWARD_DICT: obs.get(INFO_REWARD_DICT, {}).get(self.agent_name_mapping[agent]),
                          INFO_CUMULATIVE_REWARD_DICT: obs.get(INFO_CUMULATIVE_REWARD_DICT, {}).get(self.agent_name_mapping[agent]),
                      }
                      for agent in self.agents
                   }

            if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):
                for agent in self.agents:
                    infos[agent][INFO_OBSERVATION_COORDINATES] = self._last_observation_coordinates   # shared global observation must be returned via agent keys

            if self._layers_in_observation and INFO_LAYERS in obs: # only multi-objective or multi-agent environments have layers in observation available
                for agent in self.agents:
                    infos[agent][INFO_OBSERVATION_LAYERS_DICT] = obs[INFO_LAYERS]   # shared global observation must be returned via agent keys

            if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
                for agent in self.agents:
                    # shared global observation must be returned via agent keys
                    infos[agent][INFO_OBSERVATION_LAYERS_ORDER] = self._last_observation_layers_order
                    infos[agent][INFO_OBSERVATION_LAYERS_CUBE] = self._last_observation_layers_cube

            if hasattr(self._env, "agent_perspectives_with_layers"):
                for agent in self.agents:
                    infos[agent][INFO_AGENT_OBSERVATIONS] = self._last_agent_observations[agent]["ascii" if self._ascii_observation_format else "board"]

                    if self._layers_in_observation:
                        infos[agent][INFO_AGENT_OBSERVATION_LAYERS_DICT] = self._last_agent_observations[agent][INFO_LAYERS]

                    if self._object_coordinates_in_observation:
                        infos[agent][INFO_AGENT_OBSERVATION_COORDINATES] = self._last_agent_observations_coordinates[agent]

                    if self._layers_order_in_cube_per_agent is not None:
                        infos[agent][INFO_AGENT_OBSERVATION_LAYERS_ORDER] = self._last_agent_observations_layers_orders[agent]
                        infos[agent][INFO_AGENT_OBSERVATION_LAYERS_CUBE] = self._last_agent_observations_layers_cubes[agent]

        else:   # if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):

            infos = {
                agent: {
                    INFO_OBSERVATION_DIRECTION: obs.get(INFO_OBSERVATION_DIRECTION),
                    INFO_ACTION_DIRECTION: obs.get(INFO_ACTION_DIRECTION),
                    INFO_REWARD_DICT: obs.get(INFO_REWARD_DICT),
                    INFO_CUMULATIVE_REWARD_DICT: obs.get(INFO_CUMULATIVE_REWARD_DICT),
                }
                for agent in self.agents
            }

            if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):
                for agent in self.agents:
                    infos[agent][INFO_OBSERVATION_COORDINATES] = self._last_observation_coordinates   # shared global observation must be returned via agent keys

            if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
                for agent in self.agents:
                    infos[agent][INFO_OBSERVATION_LAYERS_CUBE] = self._last_observation_layers_cube   # shared global observation must be returned via agent keys

        #/ if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):


        for k, v in obs.items():
            if k not in ("RGB", INFO_LAYERS, INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION, INFO_REWARD_DICT, INFO_CUMULATIVE_REWARD_DICT):
                for agent in self.agents:
                    infos[agent][k] = v   # shared global observation must be returned via agent keys


        return infos

    #/ def _compute_infos(self, obs):


    def observe_infos_from_location(self, agents_coordinates: Dict, agents_observation_directions: Dict = None):
        """This method is read-only (does not change the actual state of the environment nor the actual state of agents).
        Each given agent observes the environment as well as itself as if it was in the given location."""
        
        #timestep = self._env.observe_from_location(agents_coordinates, agents_observation_directions)       # CHANGED: added *args, **kwargs      

        #board = timestep.observation["ascii" if self._ascii_observation_format else "board"]

        #obs = timestep.observation
        obs = self._last_observation
        # board = obs["ascii" if self._ascii_observation_format else "board"]
        agents_coordinates2 = { self.agent_name_mapping[agent_name]: coordinate for agent_name, coordinate in agents_coordinates.items() }
        agents_observation_directions2 = { self.agent_name_mapping[agent_name]: direction for agent_name, direction in agents_observation_directions.items() }

        self._process_observation(obs, agents_coordinates2, agents_observation_directions2)
        infos = self._compute_infos(obs)


        ## TODO: apply transitions to agent observations as well
        #if self._use_transitions:
        #    state = np.stack([np.zeros_like(board), board], axis=0)
        #    self._last_board = board
        #else:
        #    state = board[np.newaxis, :]

        ## TODO: apply flatten to agent observations as well
        #if self._flatten_observations:
        #    state = state.flatten()

        # self._state = state

        return infos


    def step(self, actions, *args, **kwargs):                    # CHANGED: added *args, **kwargs 
        """ Perform an action in the gridworld environment.

        Returns:
            - the board as a numpy array
            - the observed reward
            - if the episode ended
            - an info dict containing:
                - the observed reward with key INFO_OBSERVED_REWARD
                - the hidden reward with key INFO_HIDDEN_REWARD
                - the discount factor of the last step with key INFO_DISCOUNT
                - any additional information in the pycolab observation object,
                  excluding the RGB array. This includes in particular
                  the "extra_observations"
        """

        if self._pre_step_callback is not None:
            actions = self._pre_step_callback(actions, *args, **kwargs)

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            action = { self.agent_name_mapping[agent_name]: agent_action for agent_name, agent_action in actions.items() }
        else:
            action = next(iter(actions.values()))   # take the single value in dict

        timestep = self._env.step(action, *args, **kwargs)      # CHANGED: added *args, **kwargs 

        obs = timestep.observation
        self._process_observation(obs)
        infos = self._compute_infos(obs)


        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            if timestep.reward is None:
                rewards = { agent_name: 0.0 
                            for agent_name in self.agents }              
            else:
                rewards = { self.agent_name_reverse_mapping[agent_chr]: 
                            0.0 if reward is None 
                            else reward 
                            for agent_chr, reward in timestep.reward.items() }
        else:
            rewards = { agent_name: 
                        0.0 if timestep.reward is None 
                        else timestep.reward 
                        for agent_name in self.agents }

        cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
        if cumulative_hidden_reward is not None:
            if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
                hidden_reward = {}
                for agent in self.agents:
                    hidden_reward[agent] = cumulative_hidden_reward[self.agent_name_mapping[agent]] - self._last_hidden_reward[agent]
                    self._last_hidden_reward[agent] = cumulative_hidden_reward[self.agent_name_mapping[agent]]
            else:
                hidden_reward = cumulative_hidden_reward - self._last_hidden_reward[self.possible_agents[0]]
                self._last_hidden_reward[self.possible_agents[0]] = cumulative_hidden_reward
        else:
            hidden_reward = None

        #if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
        #    reward_dict = {}
        #    cumulative_reward_dict = {}
        #    for agent in self.agents:
        #        reward_dict[agent] = obs[INFO_REWARD_DICT][self.agent_name_mapping[agent]]
        #        cumulative_reward_dict[agent] = obs[INFO_CUMULATIVE_REWARD_DICT][self.agent_name_mapping[agent]]
        #else:
        #    reward_dict = obs[INFO_REWARD_DICT]
        #    cumulative_reward_dict = obs[INFO_CUMULATIVE_REWARD_DICT]

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            # TODO
            for agent in self.agents:
                infos[agent].update({
                          INFO_HIDDEN_REWARD: hidden_reward[agent] if hidden_reward is not None else None,
                          INFO_OBSERVED_REWARD: rewards[agent],
                          INFO_DISCOUNT: timestep.discount, # [agent],    # TODO: agent-based discount
                          # INFO_REWARD_DICT: reward_dict[agent],
                          # INFO_CUMULATIVE_REWARD_DICT: cumulative_reward_dict[agent],
                      })

        else:   # if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            for agent in self.agents:
                infos[agent].update({
                    INFO_HIDDEN_REWARD: hidden_reward,
                    INFO_OBSERVED_REWARD: rewards[agent],
                    INFO_DISCOUNT: timestep.discount, 
                    # INFO_REWARD_DICT: reward_dict[agent],
                    # INFO_CUMULATIVE_REWARD_DICT: cumulative_reward_dict[agent],
                })


        board = obs["ascii" if self._ascii_observation_format else "board"]

        # TODO: apply transitions to agent observations as well
        if self._use_transitions:
            board = copy.deepcopy(board)
            state = np.stack([self._last_board, board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        # TODO: apply flatten to agent observations as well
        if self._flatten_observations:
            state = state.flatten()

        self._state = state


        if hasattr(self._env, "agent_perspectives_with_layers"):

            for agent in self.agents:

                board = self._last_agent_observations[agent]["ascii" if self._ascii_observation_format else "board"]

                # TODO: apply transitions to agent observations as well
                if self._use_transitions:
                    state = np.stack([self._last_agent_boards[agent], board], axis=0)
                    self._last_agent_boards[agent] = board
                else:
                    state = board[np.newaxis, :]

                # TODO: apply flatten to agent observations as well
                if self._flatten_observations:
                    state = state.flatten()

                self._agent_states[agent] = state

            #/ for agent in self.possible_agents:

        else:

            if not self._use_transitions: # in case of self._use_transitions == True the deep copy is made already above during calculation of state
                state = copy.deepcopy(state)

            for agent in self.possible_agents:
                self._agent_states[agent] = state

        #/ if hasattr(self._env, "agent_perspectives"):
            

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            dones = { agent_name: timestep.step_type[agent_char].dead()
                      or timestep.step_type[agent_char].last() 
                      for agent_name, agent_char in self.agent_name_mapping.items() }
        else:
            dones = { agent_name: timestep.step_type.last() 
                      for agent_name in self.agent_name_mapping.keys() }

        if self._test_death:
            # NB! any agent could die at any other agent's step
            for agent in self.possible_agents:
                if self._test_deads[agent]:
                    del rewards[agent]
                    if dones[agent]:    # the agent has become actually dead, not just virtually
                        self._test_deads[agent] = False                    
                elif not dones[agent] and self._np_random.random() < self._test_death_probability:
                    dones[agent] = True
                    self._test_deads[agent] = True  # TODO: trigger a true death inside gridworlds
        
        for agent, agent_done in self._dones.items():
            if agent_done:
                del dones[agent]

        # remove agents that were already done during previous step
        for agent in list(self._agent_states.keys()): # NB! clone since the _agent_states will be modified during the loop
            if self._dones[agent]:
                del self._agent_states[agent]
                assert agent not in rewards

        self._dones.update(dones)


        if gym_v26:
            # https://gymnasium.farama.org/content/migration-guide/
            # For users wishing to update, in most cases, replacing done with terminated and truncated=False in step() should address most issues. 
            # TODO: However, environments that have reasons for episode truncation rather than termination should read through the associated PR https://github.com/openai/gym/pull/2752
            terminateds = dones
            truncateds = {agent: False for agent in dones.keys()}    # TODO   
            # NB! shallow copy self._agent_states dict since the dict values will change in next iteration, but main dict object will remain same
            result = (dict(self._agent_states), rewards, terminateds, truncateds, infos)
        else:
            result = (dict(self._agent_states), rewards, dones, infos)

        if self._post_step_callback is not None:
            self._post_step_callback(actions, *result, *args, **kwargs)

        return result


    def reset(self, seed=None, *args, **kwargs):                     # CHANGED: added seed, *args, **kwargs

        if self._pre_reset_callback is not None:
            (allow_reset, seed, args, kwargs) = self._pre_reset_callback(seed, *args, **kwargs)
            if not allow_reset:
                return  # return value

        if seed is not None:
            self.seed(seed=seed)    # ADDED

        if (isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa)
            or isinstance(self._env, safety_game_mo.SafetyEnvironmentMo)):
            timestep = self._env.reset(*args, **kwargs)       # CHANGED: added *args, **kwargs      
        else:
            timestep = self._env.reset()

        if self._viewer is not None:
            self._viewer.reset_time()


        self._dones = {agent_name: False for agent_name in self.possible_agents} 
        self._test_deads = {agent_name: False for agent_name in self.possible_agents}


        obs = timestep.observation
        self._process_observation(obs)
        infos = self._compute_infos(obs)


        board = obs["ascii" if self._ascii_observation_format else "board"]

        # TODO: apply transitions to agent observations as well
        if self._use_transitions:
            board = copy.deepcopy(board)
            state = np.stack([np.zeros_like(board), board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        # TODO: apply flatten to agent observations as well
        if self._flatten_observations:
            state = state.flatten()

        self._state = state


        if hasattr(self._env, "agent_perspectives_with_layers"):

            for agent in self.possible_agents:

                board = self._last_agent_observations[agent]["ascii" if self._ascii_observation_format else "board"]

                # TODO: apply transitions to agent observations as well
                if self._use_transitions:
                    state = np.stack([np.zeros_like(board), board], axis=0)
                    self._last_agent_boards[agent] = board
                else:
                    state = board[np.newaxis, :]

                # TODO: apply flatten to agent observations as well
                if self._flatten_observations:
                    state = state.flatten()

                self._agent_states[agent] = state

            #/ for agent in self.possible_agents:

        else:

            if not self._use_transitions: # in case of self._use_transitions == True the deep copy is made already above during calculation of state
                state = copy.deepcopy(state)

            for agent in self.possible_agents:
                self._agent_states[agent] = state

        #/ if hasattr(self._env, "agent_perspectives"):


        # NB! shallow copy self._agent_states dict since the dict values will change in next iteration, but main dict object will remain same
        obs = dict(self._agent_states)

        if self._post_reset_callback is not None:
            self._post_reset_callback(obs, infos)

        return obs, infos

    def get_reward_unit_space(self):                    # ADDED
        # TODO: use agent-specific reward unit space?
        return self._env.get_reward_unit_space()

    def get_step_no(self):                           # ADDED
        return self._env._current_game.the_plot.frame

    def get_env_seed(self):                           # ADDED
        return self._env.get_env_seed()

    def get_env_layout_seed(self):                             # ADDED
        return self._env.get_env_layout_seed()

    def get_trial_no(self):     # this is an obsolete alias to get_env_layout_seed()                            # ADDED
        return self.get_env_layout_seed()

    def get_episode_no(self):                           # ADDED
        return self._env.get_episode_no()

    def get_next_episode_no(self):                           # ADDED
        return self._env.get_next_episode_no()

    # gym does not support additional arguments to .step() method so we need to use a separate method. See also https://github.com/openai/gym/issues/2399
    def set_current_q_value_per_action(self, current_agent: dict):                           # ADDED
        current_agent = next(iter(current_agent.items()))[1]
        return self._env.set_current_agent(current_agent)

    # gym does not support additional arguments to .step() method so we need to use a separate method. See also https://github.com/openai/gym/issues/2399
    #def set_current_agent(self, current_agent: dict):                           # ADDED
    #    current_agent = next(iter(current_agent.items()))[1]
    #    return self._env.set_current_agent(current_agent)

    def seed(self, seed=None):
        # TODO: seed global random generator only if the env is not multi-agent and not multi-objective
        # TODO: update environment's environment_data["seed"] entry as well?
        np.random.seed(seed)
        # self._np_random.seed(seed)
        # use seeding.np_random(seed) which uses new np.random.Generator instead. It is supposedly faster and has better statistical properties. See also https://numpy.org/doc/stable/reference/random/index.html#design
        self._np_random = seeding.np_random(seed)[0]

    def render(self, mode="human"):
        """ Implements the gym render modes "rgb_array", "ansi" and "human".

        - "rgb_array" just passes through the RGB array provided by pycolab in each state
        - "ansi" gets an ASCII art from pycolab and returns is as a string
        - "human" uses the ai-safety-gridworlds-viewer to show an animation of the
          gridworld in a terminal
        """
        if mode == "rgb_array":
            if self._rgb is None:
                error.Error("environment has to be reset before rendering")
            else:
                return self._rgb
        elif mode == "ansi":
            if self._env._current_game is None:
                error.Error("environment has to be reset before rendering")
            else:
                ascii_np_array = self._env._current_game._board.board
                ansi_string = "\n".join(
                    [
                        " ".join([chr(i) for i in ascii_np_array[j]])
                        for j in range(ascii_np_array.shape[0])
                    ]
                )
                return ansi_string
        elif mode == "human":
            if self._viewer is None:
                self._viewer = init_viewer(self._env_name, self._render_animation_delay)
                self._viewer.display(self._env)
            else:
                self._viewer.display(self._env)
        else:
            super(GridworldEnv, self).render(mode=mode)  # just raise an exception


class MultiDiscreteGridworldsActionSpace(MultiDiscrete):  # gym.Space

    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        state = self.__dict__.copy()
        # don't pickle _env since it contains absl Flags which is not picklable
        del state["_env"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add _env back since it doesn't exist in the pickle
        self._env = None

    def __init__(self, env, agent):   # TODO: agent-specific action space
        self._env = env
        self._agent = agent
        action_spec = env._env.action_spec()

        if isinstance(env._env, safety_game_moma.SafetyEnvironmentMoMa):
            assert action_spec[0].name == "discrete"
            assert action_spec[0].dtype == "int32"
            assert action_spec[1].name == "continuous"
            assert action_spec[1].dtype == "float32"
            # self.min_action = action_spec[0].minimum.astype(int)
            # self.max_action = action_spec[0].maximum.astype(int)
            self.min_action = action_spec[0].minimum.astype(int)[0]   # spec for step modality
            self.max_action = action_spec[0].maximum.astype(int)[0]   # spec for step modality
            # TODO: multimodal action spec
            action_spec = action_spec[0]
            shape = (1,)
        else:
            assert action_spec.name == "discrete"
            assert action_spec.dtype == "int32"
            assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
            self.min_action = int(action_spec.minimum)
            self.max_action = int(action_spec.maximum)
            shape = action_spec.shape

        self.n = (self.max_action - self.min_action) + 1

        if gym_v26:
            super(MultiDiscreteGridworldsActionSpace, self).__init__(
                # shape=shape, dtype=action_spec.dtype
                nvec=[self.n], start=[self.min_action], dtype=action_spec.dtype
            )
        else:
            super(MultiDiscreteGridworldsActionSpace, self).__init__(
                # shape=shape, dtype=action_spec.dtype
                nvec=[self.n], dtype=action_spec.dtype
            )

        self._shape = shape   # needs to be set after super().__init__() has been called

        # self._np_random = self._env._np_random

    def sample(self, mask: Optional[tuple] = None) -> np.ndarray:
        if self._env is not None and self._env._dones[self._agent]:
            raise ValueError(f"Agent {self._agent} is done")

        # MultiDiscrete action space is able to work with MT19937, but Discrete action space is not
        if self._env is not None:   # _env is None when action space is pickled and sent to a subprocess
            self._np_random = self._env._np_random    # NB! update on each call since env may have been reset after constructing
        elif self._np_random is None:
            self._np_random = seeding.np_random()[0]

        result = super(MultiDiscreteGridworldsActionSpace, self).sample(mask)
        if not gym_v26:
            result += self.min_action
        return result

        #if mask is None:
        #    return self._env._np_random.randint(self.min_action, self.max_action)
        #else:
        #    return self.min_action + self._env._np_random.choice(np.where(mask == 1)[0])

    def contains(self, x):
        """
        Return True is x is a valid action. Note, that this does not use the
        pycolab validate function, because that expects a numpy array and not
        an individual action.
        """
        if self.shape[0] == 1:
            return self.min_action <= x <= self.max_action
        else:
            return all(self.min_action <= x <= self.max_action)


class DiscreteGridworldsActionSpace(Discrete):  # gym.Space

    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        state = self.__dict__.copy()
        # don't pickle _env since it contains absl Flags which is not picklable
        del state["_env"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add _env back since it doesn't exist in the pickle
        self._env = None

    def __init__(self, env, agent):   # TODO: agent-specific action space
        self._env = env
        self._agent = agent
        action_spec = env._env.action_spec()

        if isinstance(env._env, safety_game_moma.SafetyEnvironmentMoMa):
            assert action_spec[0].name == "discrete"
            assert action_spec[0].dtype == "int32"
            #assert action_spec[1].name == "continuous"
            #assert action_spec[1].dtype == "float32"
            # self.min_action = action_spec[0].minimum.astype(int)
            # self.max_action = action_spec[0].maximum.astype(int)
            self.min_action = action_spec[0].minimum.astype(int)[0]   # spec for step modality
            self.max_action = action_spec[0].maximum.astype(int)[0]   # spec for step modality
            action_spec = action_spec[0]
            shape = (1,)
        else:
            assert action_spec.name == "discrete"
            assert action_spec.dtype == "int32"
            assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
            self.min_action = int(action_spec.minimum)
            self.max_action = int(action_spec.maximum)
            shape = action_spec.shape

        self.n = (self.max_action - self.min_action) + 1

        super(DiscreteGridworldsActionSpace, self).__init__(
            n=self.n, start=self.min_action
        )

        self._shape = shape   # needs to be set after super().__init__() has been called

        # self._np_random = self._env._np_random

    def sample(self, mask: Optional[tuple] = None) -> np.ndarray:
        if self._env is not None and self._env._dones[self._agent]:
            raise ValueError(f"Agent {self._agent} is done")

        # MultiDiscrete action space is able to work with MT19937, but Discrete action space is not
        if self._env is not None:   # _env is None when action space is pickled and sent to a subprocess
            self._np_random = self._env._np_random    # NB! update on each call since env may have been reset after constructing
        elif self._np_random is None:
            self._np_random = seeding.np_random()[0]

        result = super(DiscreteGridworldsActionSpace, self).sample(mask)
        if not gym_v26:
            result += self.min_action
        return result

        #if mask is None:
        #    return self._env._np_random.randint(self.min_action, self.max_action)
        #else:
        #    return self.min_action + self._env._np_random.choice(np.where(mask == 1)[0])

    def contains(self, x):
        """
        Return True is x is a valid action. Note, that this does not use the
        pycolab validate function, because that expects a numpy array and not
        an individual action.
        """
        return self.min_action <= x <= self.max_action


class GridworldsObservationSpace(gym.Space):

    # TODO: support for 3D observation cube as observation space
    def __init__(self, env, agent, use_transitions, flatten_observations):
        # self._env = env
        if isinstance(env._env, safety_game_moma.SafetyEnvironmentMoMa):
            self.observation_spec_dict = env._env.observation_spec(env.agent_name_mapping[agent])
        else:
            self.observation_spec_dict = env._env.observation_spec()
        self.use_transitions = use_transitions
        self.flatten_observations = flatten_observations

        self._observation_dict_key = "ascii" if env._ascii_observation_format else "board"

        dict_key = self._observation_dict_key
        if flatten_observations:
            if self.use_transitions:
                shape = (2, np.prod(self.observation_spec_dict[dict_key].shape))
            else:
                shape = (np.prod(self.observation_spec_dict[dict_key].shape), )
        else:
            if self.use_transitions:
                shape = (2, *self.observation_spec_dict[dict_key].shape)
            else:
                shape = (1, *self.observation_spec_dict[dict_key].shape)

        dtype = self.observation_spec_dict[dict_key].dtype
        super(GridworldsObservationSpace, self).__init__(shape=shape, dtype=dtype)

    def sample(self):
        """
        Use pycolab to generate an example observation. Note that this is not a
        random sample, but might return the same observation for every call.
        """
        if self.use_transitions:
            raise NotImplementedError(
                "Sampling from transition-based envs not yet supported."
            )
        observation = {}
        for key, spec in self.observation_spec_dict.items():  # TODO: this loop has no purpose, since only "board" key is used at the end
            if spec == {}:
                observation[key] = {}
            else:
                if isinstance(spec, dict):
                    observation[key] = {}
                    for subkey, subkey_spec in spec.items():
                        if subkey_spec == {}:
                            observation[key][subkey] = {}
                        else:
                            observation[key][subkey] = subkey_spec.generate_value()
                else:
                    observation[key] = spec.generate_value()
        # TODO: support for "ascii" observation sampling
        result = observation[self._observation_dict_key][np.newaxis, :]    # TODO: add object coordinates and agent perspectives?
        if self.flatten_observations:
            result = result.flatten()
        return result

    def contains(self, x):
        dict_key = self._observation_dict_key
        if dict_key in self.observation_spec_dict.keys():
            try:
                self.observation_spec_dict[dict_key].validate(x[0, ...])
                if self.use_transitions:
                    self.observation_spec_dict[dict_key].validate(x[1, ...])
                return True
            except ValueError:
                return False
        else:
            return False


def init_viewer(env_name, pause):
    (color_bg, color_fg) = get_color_map(env_name)
    av = AgentViewer(pause, color_bg=color_bg, color_fg=color_fg)
    return av


def get_color_map(env_name):
    module_prefix = "ai_safety_gridworlds.environments."
    env_module_name = module_prefix + env_name
    env_module = importlib.import_module(env_module_name)
    color_bg = env_module.GAME_BG_COLOURS
    color_fg = env_module.GAME_FG_COLOURS
    return (color_bg, color_fg)
