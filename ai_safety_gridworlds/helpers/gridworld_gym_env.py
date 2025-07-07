# Copyright 2022-2024 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 n0p2 https://github.com/n0p2/gym_ai_safety_gridworlds
"""
The GridworldGymEnv implements the gym interface for the ai_safety_gridworlds.

GridworldGymEnv is based on an implementation by n0p2.
The original repo can be found at https://github.com/n0p2/gym_ai_safety_gridworlds
"""

import importlib
import random

from typing import Dict, List, Optional, NamedTuple, Tuple

import copy
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import error
    from gymnasium.spaces import MultiDiscrete, Discrete
    from gymnasium.utils import seeding
    from gymnasium.utils.env_checker import data_equivalence
    gym_v26 = True
except:
    import gym
    from gym import error
    from gym.spaces import MultiDiscrete, Discrete
    from gym.utils import seeding
    from gym.utils.env_checker import data_equivalence
    gym_v26 = False

# from ai_safety_gridworlds.environments.shared.safety_game_mp import METRICS_DICT, METRICS_MATRIX
# from ai_safety_gridworlds.environments.shared.safety_game import EXTRA_OBSERVATIONS, HIDDEN_REWARD
from ai_safety_gridworlds.environments.shared.safety_game import HIDDEN_REWARD as INFO_HIDDEN_REWARD
from ai_safety_gridworlds.environments.shared.safety_game_mo import REWARD_DICT as INFO_REWARD_DICT, CUMULATIVE_REWARD_DICT as INFO_CUMULATIVE_REWARD_DICT
from ai_safety_gridworlds.environments.shared.safety_game_mo_base import NP_RANDOM, Actions   # used as export
from ai_safety_gridworlds.environments.shared.rl import pycolab_interface_ma
from ai_safety_gridworlds.environments.shared.rl.pycolab_interface_mo import INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION, INFO_LAYERS
from ai_safety_gridworlds.environments.shared import safety_game_ma
from ai_safety_gridworlds.environments.shared import safety_game_moma
from ai_safety_gridworlds.environments.shared import safety_game_mo
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


class GridworldGymEnv(gym.Env):
    """ An OpenAI Gym environment wrapping the AI safety gridworlds created by DeepMind.

    Parameters:
    env_name (str): defines the safety gridworld to load. can take all values
                    defined in ai_safety_gridworlds.helpers.factory._environment_classes:
                        - 'boat_race'
                        - 'boat_race_ex'
                        - 'conveyor_belt'
                        - 'conveyor_belt_ex'
                        - 'distributional_shift'
                        - 'friend_foe'
                        - 'island_navigation'
                        - 'island_navigation_ex'
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
    reward_range = (-float("inf"), float("inf"))    # TODO: multi-objective reward support and using unit space

    def __init__(self, env_name, 
                 use_transitions=False, 
                 render_animation_delay=0.1, 
                 flatten_observations=False, 

                 ascii_observation_format=True, 
                 object_coordinates_in_observation=True, 
                 layers_in_observation=True, 
                 occlusion_in_layers=False, 
                 layers_order_in_cube=[],  

                 ascii_attributes_format=False, 
                 attribute_coordinates_in_observation=True, 
                 layers_in_attribute_observation=False, 
                 occlusion_in_atribute_layers=False, 
                 observable_attribute_categories=["expression", "action_direction", "observation_direction", "numeric_message", "public_metrics"], 
                 # observable_attribute_value_mapping:dict[str, dict[str, float]]={},  
                 observable_attribute_value_mapping:dict[str, float]={}, 

                 use_multi_discrete_action_space=False,   # note that some baseline algorithms may not support this mode

                 agent_character=None,    # used in case of multi-agent environments

                 np_random=None,
                 seed=None,

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
        self._observable_attribute_categories = observable_attribute_categories
        self._observable_attribute_value_mapping = observable_attribute_value_mapping
        self._use_multi_discrete_action_space = use_multi_discrete_action_space

        self._pre_reset_callback = pre_reset_callback
        self._post_reset_callback = post_reset_callback
        self._pre_step_callback = pre_step_callback
        self._post_step_callback = post_step_callback

        self._last_board = None
        self._last_agent_board = None
        self._state = None
        self._agent_state = None
        # self._last_observed_agent_board = None
        self._last_observation = None
        self._last_observation_coordinates = None
        self._last_observation_layers_order = None
        self._last_observation_layers_cube = None
        self._last_agent_observation = None
        self._last_agent_observation_coordinates = None
        self._last_agent_observation_layers_order = None
        self._last_agent_observation_layers_cube = None

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):   # TODO: ascii support for multi-objective environments
            self._env.set_observable_attribute_categories(observable_attribute_categories, observable_attribute_value_mapping)
            self._env.reset() # apply _observable_attribute_categories
            if agent_character is not None:
                self._agent_chr = agent_character
            else:
                agents = list(safety_game_ma.get_players(self._env.environment_data))
                self._agent_chr = agents[0].character   
        else:
            self._ascii_observation_format = False    # override observation format  # TODO: log warning if self._ascii_observation_format was True
            self._agent_chr = None  # unused

        self._last_hidden_reward = 0.0

        # state = None
        # info = None

        self._cumulative_reward = 0.0
        # self._info = info
        # self._agents_prev_step_result = None

        #if gym_v26:
        #    self._agents_last_step_result = (state, 0.0, False, False, info)
        #else:
        #    self._agents_last_step_result = (state, 0.0, False, info)

        if np_random is not None:
            self._internal_np_random = np_random
        else:
            if seed is not None:
                np.random.seed(seed)
            self._internal_np_random = self._env.environment_data.get(NP_RANDOM)
            if self._internal_np_random is None:
                # self._internal_np_random = np.random.RandomState(seed)
                # use seeding.np_random(seed) which uses new np.random.Generator instead. It is supposedly faster and has better statistical properties. See also https://numpy.org/doc/stable/reference/random/index.html#design
                self._internal_np_random = seeding.np_random(seed)[0]

        # TODO: make these fields readonly
        if self._use_multi_discrete_action_space:
            self._action_space = MultiDiscreteGridworldsActionSpace(self)
        else:
            self._action_space = DiscreteGridworldsActionSpace(self)

        # TODO: agent-centric observation space support
        self._observation_space = GridworldsObservationSpace(self, use_transitions, flatten_observations)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


    class _bit_generator_state_wrapper(): # hack: needed for Gym env_checker test

        def __init__(self, state):
            self.__dict__["__state"] = state

        def __getattr__(self, key):
           return getattr(self.__dict__["__state"], key)

        def __eq__(self, other):
           return data_equivalence(self.__dict__["__state"], other.__dict__["__state"])


    class _bit_generator_wrapper(): # hack: needed for Gym env_checker test

        def __init__(self, bit_generator):
            self.__dict__["__bit_generator"] = bit_generator

        def __getattr__(self, key):
            if key == "state":
                return GridworldGymEnv._bit_generator_state_wrapper(self.__dict__["__bit_generator"].state)
            else:
                return getattr(self.__dict__["__bit_generator"], key)


    class _np_random_wrapper(): # hack: needed for Gym env_checker test

        def __init__(self, np_random):
            self.__dict__["__np_random"] = np_random

        # https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object
        def __deepcopy__(self, memo):   # called by Gym env_checker test
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for key, value in self.__dict__.items():
                result.__dict__[key] = copy.deepcopy(value, memo)
            return result

        def __getattr__(self, key):
            if key == "bit_generator":
                return GridworldGymEnv._bit_generator_wrapper(self.__dict__["__np_random"]._bit_generator)
            else:
                return getattr(self.__dict__["__np_random"], key)


    @property
    def _np_random(self):   # hack: needed for Gym env_checker test
        return GridworldGymEnv._np_random_wrapper(self._internal_np_random)


    #@property
    #def info(self):
    #    return self._info

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state(self):
        """State returns a global view of the environment."""
        state = self._state
        return state 

    #def observe(self, transition_from_agents_prev_step_result=False):

    #    # get board observation from latest step, regardless whether the latest step was made by current agent or some other. If agent perspectives are available, we get current agent's perspective computed after that latest step made by any agent.
    #    if hasattr(self._env, "agent_perspectives_with_layers"): # are agent perspectives enabled and available?
    #        board = self._last_agent_observation["ascii" if self._ascii_observation_format else "board"]
    #    else:
    #        board = copy.deepcopy(self._last_observation["ascii" if self._ascii_observation_format else "board"])

    #    if self._use_transitions:
    #        if not transition_from_agents_prev_step_result:
    #            # transition from previous observation by current agent
    #            last_agent_board = self._last_observed_agent_board
    #            if last_agent_board is None:
    #                last_agent_board = np.zeros_like(board)
    #            state = np.stack([last_agent_board, board], axis=0)
    #            self._last_observed_agent_board = board
    #        else:
    #            # transition from last step result by current agent
    #            agents_prev_step_result = self._agents_prev_step_result
    #            if agents_prev_step_result is None:
    #                last_agent_board = np.zeros_like(board)
    #            else:
    #                last_agent_board = agents_prev_step_result[0][-1] # -1 takes last element from agent state. State may contain two elements in case self._use_transitions == True
    #            state = np.stack([last_agent_board, board], axis=0)
    #            self._last_observed_agent_board = board
    #    else:
    #        state = board[np.newaxis, :]

    #    if self._flatten_observations:
    #        state = state.flatten()

    #    return state


    # def observe_from_location(self, agent_name, agent_coordinates, agent_observation_direction):
    def observe_info_from_location(self, agent_coordinates, agents_observation_direction = None):
        """This method is read-only (does not change the actual state of the environment nor the actual state of agents).
        Each given agent observes the environment as well as itself as if it was in the given location."""
        
        #timestep = self._env.observe_from_location({ self._agent_chr: agent_coordinates }, { self._agent_chr: agent_observation_direction })       # CHANGED: added *args, **kwargs      

        #board = timestep.observation["ascii" if self._ascii_observation_format else "board"]

        #obs = timestep.observation
        obs = self._last_observation
        # board = obs["ascii" if self._ascii_observation_format else "board"]
        self._process_observation(obs, agent_coordinate, agent_observation_direction)
        info = self._compute_info(obs)

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

        #return (state, info)
        return info

    def _process_observation(self, obs, observe_from_agent_coordinates = None, observe_from_agent_directions = None):
        """Computes observation perspectives."""

        self._last_observation = obs
        self._rgb = obs["RGB"]

        if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):   # original Gridworlds environments do not support this method currently   # TODO
            self._last_observation_coordinates = self._env.calculate_observation_coordinates(obs, occlusion_in_layers=self._occlusion_in_layers, ascii=self._ascii_observation_format, agent_coordinates_override={ self._agent_chr: observe_from_agent_coordinates } if observe_from_agent_coordinates else None)

        if observe_from_agent_coordinates is None and observe_from_agent_directions is None:
            if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
                self._last_observation_layers_order = self._env.get_layers_order(obs, occlusion_in_layers=self._occlusion_in_layers, layers_order=self._layers_order_in_cube)
                self._last_observation_layers_cube = self._env.calculate_observation_layers_cube(obs, occlusion_in_layers=self._occlusion_in_layers, layers_order=self._last_observation_layers_order)

        if hasattr(self._env, "agent_perspectives_with_layers"): 

            agent_observations = self._env.agent_perspectives_with_layers(obs, include_layers=not self._occlusion_in_layers, ascii=self._ascii_observation_format, observe_from_agent_coordinates={ self._agent_chr: observe_from_agent_coordinates } if observe_from_agent_coordinates else None, observe_from_agent_directions={ self._agent_chr: observe_from_agent_directions } if observe_from_agent_directions else None)
            self._last_agent_observation = agent_observations[self._agent_chr]

            if self._object_coordinates_in_observation:
                agent_observations_coordinates = self._env.calculate_agents_observation_coordinates(obs, agent_observations, occlusion_in_layers=self._occlusion_in_layers, ascii=self._ascii_observation_format, observe_from_agent_coordinates={ self._agent_chr: observe_from_agent_coordinates } if observe_from_agent_coordinates else None, observe_from_agent_directions={ self._agent_chr: observe_from_agent_directions } if observe_from_agent_directions else None)
                self._last_agent_observation_coordinates = agent_observations_coordinates[self._agent_chr]

            if self._layers_order_in_cube is not None:
                self._last_agent_observation_layers_order = self._env.get_layers_order(agent_observations[self._agent_chr], occlusion_in_layers=self._occlusion_in_layers, layers_order=self._layers_order_in_cube)
                self._last_agent_observation_layers_cube = self._env.calculate_observation_layers_cube(agent_observations[self._agent_chr], occlusion_in_layers=self._occlusion_in_layers, layers_order=self._last_agent_observation_layers_order)

    #/ def _process_observation(self, obs):


    def _compute_info(self, obs):
        """Stores the observation into info fields."""
      
        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            observation_direction = obs.get(INFO_OBSERVATION_DIRECTION, {}).get(self._agent_chr)
            action_direction = obs.get(INFO_ACTION_DIRECTION, {}).get(self._agent_chr)
            reward_dict = obs.get(INFO_REWARD_DICT, {}).get(self._agent_chr)
            cumulative_reward_dict = obs.get(INFO_CUMULATIVE_REWARD_DICT, {}).get(self._agent_chr)
        else:
            observation_direction = obs.get(INFO_OBSERVATION_DIRECTION)
            action_direction = obs.get(INFO_ACTION_DIRECTION)
            reward_dict = obs.get(INFO_REWARD_DICT)
            cumulative_reward_dict = obs.get(INFO_CUMULATIVE_REWARD_DICT)

        info = {
            INFO_OBSERVATION_DIRECTION: observation_direction,
            INFO_ACTION_DIRECTION: action_direction,
            INFO_REWARD_DICT: reward_dict,
            INFO_CUMULATIVE_REWARD_DICT: cumulative_reward_dict,
        }

        if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):
            info[INFO_OBSERVATION_COORDINATES] = self._last_observation_coordinates

        if self._layers_in_observation and INFO_LAYERS in obs: # only multi-objective or multi-agent environments have layers in observation available
            info[INFO_OBSERVATION_LAYERS_DICT] = obs[INFO_LAYERS]

        if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
            info[INFO_OBSERVATION_LAYERS_ORDER] = self._last_observation_layers_order
            info[INFO_OBSERVATION_LAYERS_CUBE] = self._last_observation_layers_cube

        if hasattr(self._env, "agent_perspectives_with_layers"):
            info[INFO_AGENT_OBSERVATIONS] = self._last_agent_observation["ascii" if self._ascii_observation_format else "board"]

            if self._layers_in_observation:
                info[INFO_AGENT_OBSERVATION_LAYERS_DICT] = self._last_agent_observation[INFO_LAYERS]

            if self._object_coordinates_in_observation:
                info[INFO_AGENT_OBSERVATION_COORDINATES] = self._last_agent_observation_coordinates

            if self._layers_order_in_cube is not None:
                info[INFO_AGENT_OBSERVATION_LAYERS_ORDER] = self._last_agent_observation_layers_order
                info[INFO_AGENT_OBSERVATION_LAYERS_CUBE] = self._last_agent_observation_layers_cube


        for k, v in obs.items():
            if k not in ("RGB", INFO_LAYERS, INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION, INFO_REWARD_DICT, INFO_CUMULATIVE_REWARD_DICT):
                info[k] = v


        # self._info = info


        return info

    #/ def _compute_info(self, obs, agent_name):


    def step(self, action, *args, **kwargs):                    # CHANGED: added *args, **kwargs 
        """ Perform an action in the gridworld environment.

        Returns:
            - the board as a numpy array
            - the observed reward
            - terminated
            - truncated
            - an info dict containing:
                - the observed reward with key INFO_OBSERVED_REWARD
                - the hidden reward with key INFO_HIDDEN_REWARD
                - the discount factor of the last step with key INFO_DISCOUNT
                - any additional information in the pycolab observation object,
                  excluding the RGB array. This includes in particular
                  the "extra_observations"
        """

        if self._pre_step_callback is not None:
            action = self._pre_step_callback(action, *args, **kwargs)

        # in case of multi-agent environment, step only one specific agent. Other agents should then be controlled by the environment code as NPC-s.

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            timestep = self._env.step({ self._agent_chr: action }, *args, **kwargs)      # CHANGED: added *args, **kwargs 
        else:
            timestep = self._env.step(action, *args, **kwargs)      # CHANGED: added *args, **kwargs 

        obs = timestep.observation
        self._process_observation(obs)
        info = self._compute_info(obs)
        # self._info = info


        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            reward = (0.0 
                        if timestep.reward is None 
                        else timestep.reward[self._agent_chr])
        else:
            reward = (0.0 
                      if timestep.reward is None 
                      else timestep.reward)
            

        cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
        if cumulative_hidden_reward is not None:
            if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
                hidden_reward = cumulative_hidden_reward[self._agent_chr] - self._last_hidden_reward
                self._last_hidden_reward = cumulative_hidden_reward[self._agent_chr]
            else:
                hidden_reward = cumulative_hidden_reward - self._last_hidden_reward
                self._last_hidden_reward = cumulative_hidden_reward
        else:
            hidden_reward = None

        #if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
        #    reward_dict = obs[INFO_REWARD_DICT][self._agent_chr]
        #    cumulative_reward_dict = obs[INFO_CUMULATIVE_REWARD_DICT][self._agent_chr]
        #else:
        #    reward_dict = obs[INFO_REWARD_DICT]
        #    cumulative_reward_dict = obs[INFO_CUMULATIVE_REWARD_DICT]

        info.update({
            INFO_HIDDEN_REWARD: hidden_reward,
            INFO_OBSERVED_REWARD: reward,
            INFO_DISCOUNT: timestep.discount,
            # INFO_REWARD_DICT: reward_dict,
            # INFO_CUMULATIVE_REWARD_DICT: cumulative_reward_dict,
        })


        board = copy.deepcopy(obs["ascii" if self._ascii_observation_format else "board"])

        if self._use_transitions:
            state = np.stack([self._last_board, board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        if self._flatten_observations:
            state = state.flatten()

        self._state = state


        if hasattr(self._env, "agent_perspectives_with_layers"):

            agent_board = self._last_agent_observation["ascii" if self._ascii_observation_format else "board"]

            # TODO: apply transitions to agent observations as well
            if self._use_transitions:
                agent_state = np.stack([self._last_agent_board, agent_board], axis=0)
                self._last_agent_board = agent_board
            else:
                agent_state = agent_board[np.newaxis, :]

            # TODO: apply flatten to agent observations as well
            if self._flatten_observations:
                agent_state = agent_state.flatten()

            self._agent_state = agent_state

        else:

            self._agent_state = state

        #/ if hasattr(self._env, "agent_perspectives"):


        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            done = timestep.step_type[self._agent_chr].last()
        else:
            done = timestep.step_type.last()

        self._cumulative_reward += reward

        # self._agents_prev_step_result = self._agents_last_step_result

        if gym_v26:
            # https://gymnasium.farama.org/content/migration-guide/
            # For users wishing to update, in most cases, replacing done with terminated and truncated=False in step() should address most issues. 
            # TODO: However, environments that have reasons for episode truncation rather than termination should read through the associated PR https://github.com/openai/gym/pull/2752
            terminated = done
            truncated = False
            result = (self._agent_state, reward, terminated, truncated, info) 
        else:
            result = (self._agent_state, reward, done, info) 

        if self._post_step_callback is not None:
            self._post_step_callback(action, *result, *args, **kwargs)

        return result


    def reset(self, seed=None, return_info=False, *args, **kwargs):                     # CHANGED: added seed, *args, **kwargs

        if self._pre_reset_callback is not None:
            (allow_reset, seed, args, kwargs) = self._pre_reset_callback(seed, *args, **kwargs)
            if not allow_reset:
                return  # TODO return value

        if seed is not None:
            self.seed(seed=seed)    # ADDED

        if (isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa)
            or isinstance(self._env, safety_game_mo.SafetyEnvironmentMo)):
            timestep = self._env.reset(*args, **kwargs)       # CHANGED: added *args, **kwargs      
        else:
            timestep = self._env.reset()

        if self._viewer is not None:
            self._viewer.reset_time()

        # self._last_observed_agent_board = None

        obs = timestep.observation
        self._process_observation(obs)
        if gym_v26 or return_info:
            # self._info = self._compute_info(obs)
            info = self._compute_info(obs)


        board = copy.deepcopy(obs["ascii" if self._ascii_observation_format else "board"])

        if self._use_transitions:
            state = np.stack([np.zeros_like(board), board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        if self._flatten_observations:
            state = state.flatten()

        # self._agents_prev_step_result = None

        self._state = state


        if hasattr(self._env, "agent_perspectives_with_layers"):

            agent_board = self._last_agent_observation["ascii" if self._ascii_observation_format else "board"]

            # TODO: apply transitions to agent observations as well
            if self._use_transitions:
                agent_state = np.stack([np.zeros_like(agent_board), agent_board], axis=0)
                self._last_agent_board = agent_board
            else:
                agent_state = agent_board[np.newaxis, :]

            # TODO: apply flatten to agent observations as well
            if self._flatten_observations:
                agent_state = agent_state.flatten()

            self._agent_state = agent_state

        else:

            self._agent_state = state

        #/ if hasattr(self._env, "agent_perspectives"):


        reward = 0.0
        self._cumulative_reward = reward

        #if gym_v26:
        #    self._agents_last_step_result = (state, reward, False, False, self._info)
        #else:
        #    self._agents_last_step_result = (state, reward, False, self._info)

        if gym_v26 or return_info:
            # return state, self._info
            result = (self._agent_state, info)
            if self._post_reset_callback is not None:
                self._post_reset_callback(*result)
        else:
            result = self._agent_state
            if self._post_reset_callback is not None:
                self._post_reset_callback(result)
            
        return result


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
    def set_current_q_value_per_action(self, q_value_per_action):                           # ADDED
        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
              return self._env.set_current_q_value_per_action({ self._agent_chr: q_value_per_action })
        else:
              return self._env.set_current_q_value_per_action(q_value_per_action)

    # gym does not support additional arguments to .step() method so we need to use a separate method. See also https://github.com/openai/gym/issues/2399
    #def set_current_agent(self, current_agent):                           # ADDED
    #    return self._env.set_current_agent(current_agent)

    def seed(self, seed=None):
        # TODO: seed global random generator only if the env is not multi-agent and not multi-objective
        # TODO: update environment's environment_data["seed"] entry as well?
        np.random.seed(seed)
        # self._internal_np_random.seed(seed)
        # use seeding.np_random(seed) which uses new np.random.Generator instead. It is supposedly faster and has better statistical properties. See also https://numpy.org/doc/stable/reference/random/index.html#design
        self._internal_np_random = seeding.np_random(seed)[0]

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

    def __init__(self, env):
        self._env = env
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

        # self._internal_np_random = self._env._internal_np_random

    def sample(self, mask: Optional[tuple] = None) -> np.ndarray:

        # MultiDiscrete action space is able to work with MT19937, but Discrete action space requires using the newer Generator class
        if self._env is not None:   # _env is None when action space is pickled and sent to a subprocess
            self._np_random = self._env._np_random    # NB! update on each call since env may have been reset after constructing
        elif self._np_random is None:
            self._np_random = seeding.np_random()[0]

        result = super(MultiDiscreteGridworldsActionSpace, self).sample(mask)
        if not gym_v26:
            result += self.min_action
        return result

        #if mask is None:
        #    return self._env._internal_np_random.randint(self.min_action, self.max_action)
        #else:
        #    return self.min_action + self._env._internal_np_random.choice(np.where(mask == 1)[0])

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

    def __init__(self, env):
        self._env = env
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

        # self._internal_np_random = self._env._internal_np_random

    def sample(self, mask: Optional[tuple] = None) -> np.ndarray:

        # MultiDiscrete action space is able to work with MT19937, but Discrete action space requires using the newer Generator class
        if self._env is not None:   # _env is None when action space is pickled and sent to a subprocess
            self._np_random = self._env._np_random    # NB! update on each call since env may have been reset after constructing
        elif self._np_random is None:
            self._np_random = seeding.np_random()[0]

        result = super(DiscreteGridworldsActionSpace, self).sample(mask)
        if not gym_v26:
            result += self.min_action
        return result

        #if mask is None:
        #    return self._env._internal_np_random.randint(self.min_action, self.max_action)
        #else:
        #    return self.min_action + self._env._internal_np_random.choice(np.where(mask == 1)[0])

    def contains(self, x):
        """
        Return True is x is a valid action. Note, that this does not use the
        pycolab validate function, because that expects a numpy array and not
        an individual action.
        """
        return self.min_action <= x <= self.max_action


class GridworldsObservationSpace(gym.Space):  # TODO: support for agent-centric observations

    def __init__(self, env, use_transitions, flatten_observations):
        # self._env = env
        if isinstance(env._env, safety_game_moma.SafetyEnvironmentMoMa):
            self.observation_spec_dict = env._env.observation_spec(env._agent_chr)
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
