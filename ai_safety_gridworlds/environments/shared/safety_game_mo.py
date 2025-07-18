# Copyright 2022-2024 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 n0p2 https://github.com/n0p2/gym_ai_safety_gridworlds
# Copyright 2018 The AI Safety Gridworlds Authors. All Rights Reserved.
# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Helpers for creating safety environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import datetime
import decimal
import gzip
import zlib
import itertools
import numbers
import os

from absl import flags

# Dependency imports
from ai_safety_gridworlds.environments.shared.rl import array_spec as specs
from ai_safety_gridworlds.environments.shared.rl import environment
from ai_safety_gridworlds.environments.shared.rl.pycolab_interface_mo import INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION, INFO_LAYERS
from ai_safety_gridworlds.environments.shared.mo_reward import mo_reward
from ai_safety_gridworlds.environments.shared.plot_mo import PlotMo
from ai_safety_gridworlds.environments.shared.safety_game_mo_base import make_safety_game, SafetyEnvironmentMoBase, AgentSafetySprite, Actions, SafetyBackdrop, PolicyWrapperDrape, ACTUAL_ACTIONS, TERMINATION_REASON, EXTRA_OBSERVATIONS
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason


import numpy as np

import six

try:
    from gymnasium.utils import seeding
    gym_v26 = True
except:
    from gym.utils import seeding
    gym_v26 = False



log_compresslevel = 6   # 6 is default level for gzip: https://linux.die.net/man/1/gzip
# https://github.com/ebiggers/libdeflate


METRICS_DICT = 'metrics_dict'
METRICS_MATRIX = 'metrics_matrix'
METRICS_LABELS = 'metrics_labels'
METRICS_ROW_INDEXES = 'metrics_row_indexes'
CUMULATIVE_REWARD = 'cumulative_reward'
AVERAGE_REWARD = 'average_reward'
GINI_INDEX = 'gini_index'
CUMULATIVE_GINI_INDEX = 'cumulative_gini_index'
MO_VARIANCE = 'mo_variance'
CUMULATIVE_MO_VARIANCE = 'cumulative_mo_variance'
AVERAGE_MO_VARIANCE = 'average_mo_variance'
TILE_TYPES = 'tile_types'
AGENT_SPRITE = 'agent_sprite'
Z_ORDER = 'z_order'
ASCII_ART = 'ascii_art'   # ADDED
NP_RANDOM = 'np_random'   # ADDED
SEED = 'seed'   # ADDED
REWARD_DICT = 'reward_dict'   # ADDED
CUMULATIVE_REWARD_DICT = 'cumulative_reward_dict'   # ADDED


# timestamp, environment_name, episode_no, iteration_no, environment_flags, reward_unit_sizes, rewards, cumulative_rewards, metrics
LOG_TIMESTAMP = 'timestamp'
LOG_ENVIRONMENT = 'env'
LOG_TRIAL = 'trial'   # obsolete alias for env layout seed
LOG_ENV_LAYOUT_SEED = 'env layout seed'
LOG_ENV_SEED = 'env seed'
LOG_EPISODE = 'episode'
LOG_ITERATION = 'iteration'
LOG_ARGUMENTS = 'arguments'
LOG_REWARD_UNITS = 'reward_unit'      # TODO
LOG_REWARD = 'reward'
LOG_SCALAR_REWARD = 'scalar_reward'                         # TODO: add this metric to human console too
LOG_CUMULATIVE_REWARD = 'cumulative_reward'
LOG_AVERAGE_REWARD = 'average_reward'                       # TODO: add this metric to human console too
LOG_GINI_INDEX = 'gini_index'                               # TODO: add this metric to human console too
LOG_CUMULATIVE_GINI_INDEX = 'cumulative_gini_index'         # TODO: add this metric to human console too
LOG_MO_VARIANCE = 'mo_variance'                             # TODO: add this metric to human console too
LOG_CUMULATIVE_MO_VARIANCE = 'cumulative_mo_variance'       # TODO: add this metric to human console too
LOG_AVERAGE_MO_VARIANCE = 'average_mo_variance'             # TODO: add this metric to human console too
LOG_SCALAR_CUMULATIVE_REWARD = 'scalar_cumulative_reward'   # TODO: add this metric to human console too
LOG_SCALAR_AVERAGE_REWARD = 'scalar_average_reward'         # TODO: add this metric to human console too
LOG_METRICS = 'metric'
LOG_QVALUES_PER_TILETYPE = 'tiletype_qvalue'


log_arguments_to_skip = [
  "__class__",
  "kwargs",
  "self",
  "environment_data",
  "value_mapping", # TODO: option to include value_mapping in log_arguments
  "log_columns",
  "log_dir",
  "log_filename_comment",
  "log_arguments",
  "log_arguments_to_separate_file",
  "env_layout_seed",
  "disable_env_checker",
]

flags_to_skip = [
  "?",
	"logtostderr",
	"alsologtostderr",
	"log_dir",
	"v",
	"verbosity",
	"logger_levels",
	"stderrthreshold",
	"showprefixforinfo",
	"run_with_pdb",
	"pdb_post_mortem",
	"pdb",
	"run_with_profiling",
	"profile_file",
	"use_cprofile_for_profiling",
	"only_check_args",
	"eval", 
  "help", 
  "helpshort", 
  "helpfull", 
  "helpxml",
]


class SafetyEnvironmentMo(SafetyEnvironmentMoBase):
  """Base class for multi-objective safety gridworld environments.

  Environments implementing this base class initialize the Python environment
  API v2 and serve as a layer in which we can put various modifications of
  pycolab inputs and outputs, such as *additional information* passed
  from/to the environment that does not fit in the traditional observation
  channel. It also allows for overwriting of default methods such as step() and
  reset().

  Each new environment must implement a subclass of this class, and at the very
  least call the __init__ method of this class with corresponding parameters, to
  instantiate the python environment API around the pycolab game.
  """

  def __init__(self, enabled_mo_rewards, 
               *args, 
               #game_factory,
               #game_bg_colours,
               #game_fg_colours,
               #actions=None,
               #value_mapping=None,
               environment_data=None,
               #repainter=None,
               #max_iterations=100,
               FLAGS=None,
               scalarise=False,
               gzip_log=False,
               log_columns=None,
               log_dir="logs",
               log_filename_comment="",
               log_arguments=None,
               log_arguments_to_separate_file=True,
               env_layout_seed=1,
               trial_no=None,  # this is an obsolete alias to env_layout_seed 
               episode_no=None,
               disable_env_checker=None,  # The presence of that parameter just means the gym.make() method did not capture it. It happens when gym version < 24.
               np_random=None, 
               seed=None,   # By default equals to env_layout_seed.
               **kwargs):
    """Initialize a Python v2 environment for a pycolab game factory.

    Args:
      enabled_mo_rewards: list of multi-objective rewards being used in 
        current map. Providing this list enables reducing the dimensionality of 
        the reward vector in such a way that unused reward dimensions are left 
        out. If set to None then the multi-objective rewards are disabled: the 
        rewards are then scalarised before returning to the agent.
      scalarise: Makes the get_overall_performance(), get_last_performance(), 
        and timestep.reward from step() and reset() to return an ordinary scalar 
        value like non-multi-objective environments do. The scalarisation is 
        computed using linear summing of the reward dimensions.
      log_columns: turns on CSV logging of specified column types (timestamp, 
        environment_name, env_layout_seed, episode_no, iteration_no, 
        environment_arguments, reward_unit_sizes, reward, scalar_reward, 
        cumulative_reward, scalar_cumulative_reward, metrics)
      log_dir: directory to save log files to.
      log_arguments: dictionary of environment arguments to log if LOG_ARGUMENTS 
        is set in log_columns or if log_arguments_to_separate_file is True. If
        log_arguments is None then all arguments are logged except the ones 
        listed in log_arguments_to_skip.
      log_arguments_to_separate_file: whether to log environment arguments to a 
        separate file.
      env_layout_seed: environment layout seed. If not specified then previous env_layout_seed is reused.
      trial_no: obsolete alias to env_layout_seed.
      episode_no: episode number. Use when you need to reset episode_no counter
        manually for some reason (for example, when changing flags).        
      seed: by default equals to env_layout_seed.
      default_reward: defined in Pycolab interface, is currently ignored and 
        overridden to mo_reward({})
      game_factory: a function that returns a new pycolab `Engine`
        instance corresponding to the game being played.
      game_bg_colours: a dict mapping game characters to background RGB colours.
      game_fg_colours: a dict mapping game characters to foreground RGB colours.
      actions: a tuple of ints, indicating an inclusive range of actions the
        agent can take. Defaults to DEFAULT_ACTION_SET range.
      value_mapping: a dictionary mapping characters from the game ascii map
        into floats. Used to control how the agent sees the game ascii map, e.g.
        if we are not making a difference between environment background and
        walls in terms of values the agent sees for those blocks, we can map
        them to the same value. Defaults to mapping characters to their ascii
        codes.
      environment_data: dictionary of data that is passed to the pycolab
        environment implementation and is used as a shared object that allows
        each wrapper to communicate with their environment. This object can hold
        additional information about the state of the environment that can even
        persists through episodes, but some particular keys are erased at each
        new episode.
      repainter: a callable that converts `rendering.Observation`s to different
        `rendering.Observation`s, or None if no such conversion is required.
        This facility is normally used to change the characters used to depict
        certain game elements, and a `rendering.ObservationCharacterRepainter`
        object is a convenient way to accomplish this conversion. For more
        information, see pycolab's `rendering.py`.
      max_iterations: the maximum number of steps for one episode.
    """

    if trial_no is not None:    # this is an obsolete alias to env_layout_seed()
      env_layout_seed = trial_no

    if log_arguments is not None:
      self.log_arguments = dict(log_arguments)
    else:
      self.log_arguments = dict(locals()) # need to clone using dict() else log_arguments.pop does not work
      self.log_arguments.update(kwargs)

    for key in log_arguments_to_skip:
      self.log_arguments.pop(key, None)

    self.flags = self.log_arguments.pop("FLAGS", None)
    if self.flags is not None:
      self.flags = { 
                      key: self.flags[key].value for key in list(self.flags) 
                        if key not in flags_to_skip 
                          and key not in self.log_arguments   # do not log flags that are already specified in the arguments
                    }
    else:
      self.flags = {}


    self.enabled_mo_rewards = enabled_mo_rewards
    self.enabled_reward_dimension_keys = mo_reward.get_enabled_reward_dimension_keys(self.enabled_mo_rewards)
    
    self.reward_unit_space = mo_reward.get_enabled_reward_unit_space(self.enabled_mo_rewards)
    if self.reward_unit_space is not None:
      self.reward_unit_space[0] = np.array([float(x) for x in self.reward_unit_space[0]]) # min
      self.reward_unit_space[1] = np.array([float(x) for x in self.reward_unit_space[1]]) # max

    self.scalarise = scalarise


    if environment_data is None:  # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument
      environment_data = {}
    self._environment_data = environment_data

    if np_random is None:
      if seed is not None:
        np.random.seed(seed)
      # np_random = np.random.RandomState(seed) # NB! use a separate random state object even if seed is None
      # use seeding.np_random(seed) which uses new np.random.Generator instead. It is supposedly faster and has better statistical properties. See also https://numpy.org/doc/stable/reference/random/index.html#design
      np_random = seeding.np_random(seed)[0]

    self._environment_data[NP_RANDOM] = np_random
    self._environment_data[SEED] = seed

    self._environment_data[METRICS_DICT] = dict()
    self._environment_data[METRICS_MATRIX] = np.empty([0, 2], object)
    self._environment_data[CUMULATIVE_REWARD] = np.array(mo_reward({}).tolist(self.enabled_mo_rewards))
    self._environment_data[TILE_TYPES] = []  # will be initialised by the agent sprite during super(SafetyEnvironmentMo, self).__init__() since the agent object has access to the board
    
    self.q_value_per_location = {}
    self.q_value_per_tiletype = {}  
    self.q_value_per_action = None



    # self._init_done = False   # needed in order to skip logging during _compute_observation_spec() call

    # NB! do not pass on disable_env_checker parameter since the presence of that parameter just means the gym.make() method did not capture it. It happens when gym version < 24.
    super(SafetyEnvironmentMo, self).__init__(*args, environment_data=self._environment_data, **kwargs)

    # parent class safety_game.SafetyEnvironment sets default_reward=0
    self._default_reward = mo_reward({})  # TODO: consider default_reward argument's value

    # self._init_done = True
    
    self.metrics_keys = list(self._environment_data.get(METRICS_DICT, {}).keys())   # NB! METRICS_DICT in _environment_data is populated only after parent class is constructed in super().__init__()



    prev_experiment_no = getattr(self.__class__, "prev_experiment_no", 0)
    next_experiment_no = getattr(self.__class__, "next_experiment_no", 1)
    setattr(self.__class__, "prev_experiment_no", next_experiment_no)

    prev_log_filename_comment = getattr(self.__class__, "log_filename_comment", "")
    setattr(self.__class__, "log_filename_comment", log_filename_comment)

    prev_log_arguments = getattr(self.__class__, "log_arguments", {})
    setattr(self.__class__, "log_arguments", self.log_arguments)

    prev_flags = getattr(self.__class__, "flags", {})
    setattr(self.__class__, "flags", self.flags)

    prev_enabled_reward_dimension_keys = getattr(self.__class__, "enabled_reward_dimension_keys", [])
    setattr(self.__class__, "enabled_reward_dimension_keys", self.enabled_reward_dimension_keys)

    prev_metrics_keys = getattr(self.__class__, "metrics_keys", [])
    setattr(self.__class__, "metrics_keys", self.metrics_keys)


    setattr(self.__class__, "env_seed", seed)


    prev_env_layout_seed = getattr(self.__class__, "env_layout_seed", -1)
    setattr(self.__class__, "env_layout_seed", env_layout_seed)

    if (   # detect when a new experiment is started
      prev_experiment_no != next_experiment_no
      or prev_log_filename_comment != log_filename_comment 
      or prev_log_arguments != self.log_arguments
      or prev_flags != self.flags
      or prev_enabled_reward_dimension_keys != self.enabled_reward_dimension_keys
      or prev_metrics_keys != self.metrics_keys
    ):
      # prev_env_layout_seed = -1    # this causes a new log file to be created
      setattr(self.__class__, "create_new_log_file", True)
    else:
      setattr(self.__class__, "create_new_log_file", False)


    if prev_env_layout_seed != env_layout_seed: # if new env layout is started then reset the episode_no counter
      setattr(self.__class__, "episode_no", 1)  # use static attribute so that the value survives re-construction of the environment
      # use a different random number sequence for each env layout
      # at the same time use deterministic seed numbers so that if the env layouts are re-run then the results are same regardless of steps taken during previous env layout
      new_seed = seed
      if new_seed is None:
        # seed = env_layout_seed
        original_seed = self._environment_data[SEED]
        if original_seed is not None:
          seeds = [original_seed, env_layout_seed, 17122023]    # 17122023 is used to make the pairing more unique as compared to other potential pairings happening in user code for example
          seeds_bytes = b''.join([x.to_bytes(4, byteorder='big') for x in seeds])
          new_seed = zlib.crc32(seeds_bytes)
        else:
          new_seed = env_layout_seed
        
        setattr(self.__class__, "env_seed", new_seed)

      else:
        new_seed = int(new_seed) & 0xFFFFFFFF  # 0xFFFFFFFF: np.random.seed accepts 32-bit int only

      np.random.seed(new_seed)
      # self._environment_data[NP_RANDOM].seed(new_seed)
      # use seeding.np_random(seed) which uses new np.random.Generator instead. It is supposedly faster and has better statistical properties. See also https://numpy.org/doc/stable/reference/random/index.html#design
      self._environment_data[NP_RANDOM] = seeding.np_random(new_seed)[0]
    
    if episode_no is not None:
      setattr(self.__class__, "episode_no", episode_no)  # use static attribute so that the value survives re-construction of the environment



    self.gzip_log = gzip_log
    self.log_dir = log_dir
    self.log_filename_comment = log_filename_comment
    
    if log_columns is None: # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument
      log_columns = []
    self.log_columns = log_columns

    self.log_arguments_to_separate_file = log_arguments_to_separate_file

    # prec = 12
    prec = 10  
    self.decimal_context = decimal.Context(prec=prec, rounding=decimal.ROUND_HALF_UP, capitals=0)


    # log file header creation moved to reset() method


  def __getstate__(self):
    instance_result = self.__dict__.copy()
    class_result = {key: val for key, val in self.__class__.__dict__.items() if not key.startswith("__")}   # NB! include class variables set with setattr(self.__class__, ...)
    return (instance_result, class_result)

  def __setstate__(self, state):
    (instance_state, class_state) = state
    self.__dict__.update(instance_state)

    # __class.__dict__ does not support update method
    for key, val in class_state.items():
      old_val = getattr(self.__class__, key, None)
      assert old_val is None or old_val == val   # normally, unpickling should be done only in subprocesses to where the env was cloned. Unpickling class vars to current process in such a manner that the values change is most likely a bug.
      setattr(self.__class__, key, val)


  def calculate_observation_coordinates(self, observation, occlusion_in_layers=True, ascii=True, agent_coordinates_override=None):

    if not occlusion_in_layers:  # return coordinates of all objects, including the overlapped ones

      layers_coordinates = {}
      layers = observation[INFO_LAYERS] # if isinstance(observation, dict) else observation.layers   # when called on agent perspectives then the observation is of Observation type

      for layer_key, layer in layers.items():
        # if not ascii then translate key to corresponding observation value
        # layer_key = layer_key if ascii else self._value_mapping[layer_key]
        # coordinates = layer.nonzero()
        # layers_coordinates[layer_key] = list(zip(coordinates[0], coordinates[1])) # this returns list of tuples
        if agent_coordinates_override is not None and layer_key in agent_coordinates_override:
          coord = agent_coordinates_override[layer_key]
          layers_coordinates[layer_key] = [tuple(coord)]
        else:
          layers_coordinates[layer_key] = [tuple(coord) for coord in np.argwhere(layer).tolist()] # argwhere returns list of lists, but list of tuples would be more efficient

      return layers_coordinates

    else:  # return coordinates of only the topmost objects visible on the board

      board = observation["ascii" if ascii else "board"] if isinstance(observation, dict) else observation.board   # when called on agent perspectives then the observation is of Observation type
      chars = np.unique(board)
            
      chars_coordinates = {}
      for char in chars:
        if agent_coordinates_override is not None and char in agent_coordinates_override:
          # TODO: replace agent locations in the board according to agent_coordinates_override
          raise NotImplementedError()
        else:
          # coordinates = (board == char).nonzero()
          # chars_coordinates[key] = list(zip(coordinates[0], coordinates[1]))
          chars_coordinates[key] = np.argwhere(board == char).tolist()[0]

      return chars_coordinates


  def get_layers_order(self, observation, occlusion_in_layers=False, layers_order=[]):

    if layers_order == []:  # take all layers

      if not occlusion_in_layers:  # return coordinates of all objects, including the overlapped ones

        layers_list = []
        layers = observation[INFO_LAYERS] # if isinstance(observation, dict) else observation.layers   # when called on agent perspectives then the observation is of Observation type

        layers_order = list(layers.keys())  # assignment to default argument does not cause the "mutable default argument" problem
        layers_order.sort()

      else:  # return coordinates of only the topmost objects visible on the board

        # TODO: there is a risk that some layer is invisible for a while and then this dimension gets lost from the observationcube

        board = observation["ascii" if ascii else "board"] if isinstance(observation, dict) else observation.board   # when called on agent perspectives then the observation is of Observation type
        chars = np.unique(board)

        layers_order = chars.tolist()  # assignment to default argument does not cause the "mutable default argument" problem
        layers_order.sort()

    #/ if layers_order == []:

    return layers_order


  def calculate_observation_layers_cube(self, observation, occlusion_in_layers=True, layers_order=[]):

    layers_order = self.get_layers_order(observation, occlusion_in_layers, layers_order)

    if not occlusion_in_layers:  # return coordinates of all objects, including the overlapped ones

      layers_list = []
      layers = observation[INFO_LAYERS] # if isinstance(observation, dict) else observation.layers

      #if layers_order == []:  # take all layers
      #  layers_order = list(layers.keys())  # assignment to default argument does not cause the "mutable default argument" problem
      #  layers_order.sort()

      for layer_key in layers_order:
        layer = layers.get(layer_key)
        if layer is None:   # NB! allow layers_order to refer to non-existing layers for cross-environment observation format compatibility purposes
          layer = np.zeros_like(next(iter(layers.values())))
        layers_list.append(layer)

      return np.stack(layers_list, axis=0)   # feature vector becomes the first dimension

    else:  # return coordinates of only the topmost objects visible on the board

      board = observation["ascii" if ascii else "board"] if isinstance(observation, dict) else observation.board
      #chars = np.unique(board)

      #if layers_order == []:  # take all layers
      #  layers_order = chars.tolist()  # assignment to default argument does not cause the "mutable default argument" problem
      #  layers_order.sort()

      layers_list = []
      for layer_key in layers_order:
        layer = (board == layer_key)
        layers_list.append(layer)

      return np.stack(layers_list, axis=0)   # feature vector becomes the first dimension


  # adapted from SafetyEnvironment.reset() in ai_safety_gridworlds\environments\shared\safety_game.py and from Environment.reset() in ai_safety_gridworlds\environments\shared\rl\pycolab_interface.py
  def reset(self, 
            env_layout_seed=None, 
            trial_no=None,    # this is an obsolete alias to env_layout_seed
            start_new_experiment=False, 
            seed=None, 
            options=None, 
            do_not_replace_reward=False
  ):  # seed, options: for Gym 0.26+ compatibility
    """Start a new episode. 
    Increment the episode counter if the previous game was played.
    
    env_layout_seed: environment layout seed. If not specified then previous env_layout_seed is reused.
      
    trial_no: obsolete alias to env_layout_seed.

    start_new_experiment: instruct the environment to start a new log file.

    seed: for Gym 0.26+ compatibility. By default equals to env_layout_seed.
    """

    if seed is not None:
      setattr(self.__class__, "env_seed", seed)

    if trial_no is not None:    # this is an obsolete alias to env_layout_seed
      env_layout_seed = trial_no

    if options:   # for Gym 0.26+ compatibility
      env_layout_seed = options.get("env_layout_seed", env_layout_seed)
      trial_no = options.get("trial_no", trial_no)
      if trial_no is not None:    # this is an obsolete alias to env_layout_seed
        env_layout_seed = trial_no

      start_new_experiment = options.get("start_new_experiment", start_new_experiment)


    # Environment._compute_observation_spec() -> Environment.reset() -> Engine.its_showtime() -> Engine.play() -> Engine._update_and_render() is called straight from the constructor of Environment therefore need to overwrite _the_plot variable here. Overwriting it in SafetyEnvironmentMo.__init__ would be too late

    if start_new_experiment:  # instruct the environment to start a new log file
      prev_experiment_no = getattr(self.__class__, "prev_experiment_no", 0)
      setattr(self.__class__, "next_experiment_no", prev_experiment_no + 1)
      setattr(self.__class__, "create_new_log_file", True)


    if getattr(self.__class__, "create_new_log_file", False):
      prev_file = getattr(self.__class__, "log_file_handle", None)
      if prev_file:
        prev_file.flush()
        prev_file.close()
        setattr(self.__class__, "log_file_handle", None)
        setattr(self.__class__, "log_filename", None)


    # self._state == None means that the parent class is calling reset() for the purpose of computing observation spec and the current class has not completed its constructor yet, nor has the agent sprite constructed yet.
    # self._state == environment.StepType.MID or self._state == environment.StepType.LAST means start_new_experiment is called at the end of previous experiment. Then do not create a new log file yet, just leave a flag that it is to be created next time. Still need to run rest of the reset code because various libraries might depend on the reset code being run fully once it is called.
    if self._state == environment.StepType.FIRST:   

      # if prev_env_layout_seed == -1:  # save all episodes and all env layouts to same file
      if getattr(self.__class__, "create_new_log_file"):
        setattr(self.__class__, "create_new_log_file", False)

        if len(self.log_columns) > 0:

          if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

          # TODO: option to include log_arguments in filename

          classname = self.__class__.__module__ + "." + self.__class__.__qualname__
          timestamp = datetime.datetime.now()
          timestamp_str = datetime.datetime.strftime(timestamp, '%Y.%m.%d-%H.%M.%S')

          # NB! set log_filename only once per executione else the timestamp would change across episodes and env layouts and would cause a new file for each episode and env layout.
          log_filename = classname + ("-" if self.log_filename_comment else "") + self.log_filename_comment + "-" + timestamp_str + ".csv" # TODO: use TSV format instead
          setattr(self.__class__, "log_filename", log_filename)
          arguments_filename = classname + ("-" if self.log_filename_comment else "") + self.log_filename_comment + "-arguments-" + timestamp_str + ".txt" 


          if self.log_arguments_to_separate_file:
            with open(os.path.join(self.log_dir, arguments_filename), mode='wt', buffering=1024 * 1024, encoding='utf-8') as file:
              print("{", file=file)   # using print() automatically generate newlines
            
              for key, arg in self.log_arguments.items():
                print("\t'" + str(key) + "': " + str(arg) + ",", file=file)
            
              print("\t'FLAGS': {", file=file)
              for key, value in self.flags.items():
                print("\t\t'" + str(key) + "': " + str(value) + ",", file=file)
              print("\t},", file=file)

              print("\t'reward_dimensions': {", file=file)
              for index, key in enumerate(self.enabled_reward_dimension_keys):
                print("\t\t'" + str(key) + "': [" + str(self.reward_unit_space[0][index]) + ", " + str(self.reward_unit_space[1][index]) + "],", file=file)
              print("\t},", file=file)
            
              print("\t'metrics_keys': [", file=file)
              for key in self.metrics_keys:
                print("\t\t'" + str(key) + "',", file=file)
              print("\t],", file=file)

              print("}", file=file)
              # TODO: find a way to log reward unit sizes too

              file.flush()


          prev_file = getattr(self.__class__, "log_file_handle", None)
          if prev_file:
            prev_file.flush()
            prev_file.close()


          if self.gzip_log:
            #with open(os.path.join(self.log_dir, log_filename + ".gz"), mode='wb', buffering=1024 * 1024) as raw_file:
            #  with gzip.GzipFile(fileobj=raw_file, filename=log_filename, mode='wt', encoding='utf-8', compresslevel=log_compresslevel) as file:  # TODO: newline='' for gzip
            #    self._write_log_header(file)
            #  raw_file.flush()
            file = gzip.open(os.path.join(self.log_dir, log_filename + ".gz"), mode='wt', newline='', encoding='utf-8', compresslevel=log_compresslevel)   # csv writer creates its own newlines therefore need to set newline to empty string here     # TODO: buffering for gzip    
          else:
            file = open(os.path.join(self.log_dir, log_filename), mode='wt', buffering=1024 * 1024, newline='', encoding='utf-8')   # csv writer creates its own newlines therefore need to set newline to empty string here        

          self._write_log_header(file)
          setattr(self.__class__, "log_file_handle", file)

        else:   #/ if len(self.log_columns) > 0: 

          # NB! this still has to be inside 'if getattr(self.__class__, "create_new_log_file")' condition
          setattr(self.__class__, "log_filename", None)


    # note: no elif here. env.reset(start_new_experiment=True) should still execute rest of the .reset code just in case.
    if start_new_experiment or env_layout_seed is not None:

      if start_new_experiment and env_layout_seed is None:
        env_layout_seed = 1

      prev_env_layout_seed = getattr(self.__class__, "env_layout_seed")
      episode_no = getattr(self.__class__, "episode_no")
      if (
        start_new_experiment  # If start_new_experiment is set then force random number generator seeding
        or prev_env_layout_seed != env_layout_seed  # If new env layout is started then reset the episode_no counter.
        or (      # If reset is called at the start of first env layout then force random number generator re-seeding since setting up the experiment before the .reset() call might have consumed random numbers from the random number generator and we want the agent to be deterministic after the reset call
          env_layout_seed == 1 
          and episode_no == 1
          and (self._state is None or self._state.first())
        )
      ):
        setattr(self.__class__, "env_layout_seed", env_layout_seed)

        setattr(self.__class__, "episode_no", 1)
        # use a different random number sequence for each env layout
        # at the same time use deterministic seed numbers so that if the env layouts are re-run then the results are same regardless of steps taken during previous env layout
        # TODO: seed random number generator for each env layout AND episode?
        new_seed = seed
        if new_seed is None:
          # seed = env_layout_seed
          original_seed = self._environment_data[SEED]
          if original_seed is not None:
            seeds = [original_seed, env_layout_seed, 17122023]    # 17122023 is used to make the pairing more unique as compared to other potential pairings happening in user code for example
            seeds_bytes = b''.join([x.to_bytes(4, byteorder='big') for x in seeds])
            new_seed = zlib.crc32(seeds_bytes)
          else:
            new_seed = env_layout_seed
        else:
          new_seed = int(new_seed) & 0xFFFFFFFF  # 0xFFFFFFFF: np.random.seed accepts 32-bit int only
        np.random.seed(new_seed)
        # self._environment_data[NP_RANDOM].seed(new_seed)
        # use seeding.np_random(seed) which uses new np.random.Generator instead. It is supposedly faster and has better statistical properties. See also https://numpy.org/doc/stable/reference/random/index.html#design
        self._environment_data[NP_RANDOM] = seeding.np_random(new_seed)[0]

    else:
      if (  # check that current episode has any steps
        self._state is not None 
        and self._state != environment.StepType.FIRST
      ):   # increment the episode_no only if the previous game was played, and not upon early or repeated reset() calls
        episode_no = getattr(self.__class__, "episode_no")
        episode_no += 1
        setattr(self.__class__, "episode_no", episode_no)


    # start of code adapted from from Environment.reset()
    # Build a new game and retrieve its first set of state/reward/discount.
    self._current_game = self._game_factory()

    self._current_game._the_plot = PlotMo()    # ADDED: incoming mo_reward argument to add_reward() has to be treated as immutable else rewards across timesteps will be accumulated in per timestep accumulator

    self._state = environment.StepType.FIRST
    # Collect environment returns from starting the game and update state.
    observations, reward, discount = self._current_game.its_showtime()
    self._update_for_game_step(observations, reward, discount)
    timestep = environment.TimeStep(
        step_type=self._state,
        reward=None,
        discount=None,
        observation=self.last_observations
    )
    # end of code adapted from from Environment.reset()

    # do_not_replace_reward = not replace_reward     # NB! do_not_replace_reward=True since self._process_timestep(timestep) will be called after .step()
    return self._process_timestep(timestep, do_not_replace_reward)  # adapted from SafetyEnvironment.reset()


  def _write_log_header(self, file):

    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, delimiter=';')

    data = []
    for col in self.log_columns:

      if col == LOG_TIMESTAMP:
        data.append(LOG_TIMESTAMP)

      elif col == LOG_ENVIRONMENT:
        data.append(LOG_ENVIRONMENT)

      elif col == LOG_ENV_SEED:
        data.append(LOG_ENV_SEED)

      elif col == LOG_ENV_LAYOUT_SEED:
        data.append(LOG_ENV_LAYOUT_SEED)

      elif col == LOG_TRIAL:    # obsolete alias for env layout seed
        data.append(LOG_TRIAL)

      elif col == LOG_EPISODE:
        data.append(LOG_EPISODE)

      elif col == LOG_ITERATION:
        data.append(LOG_ITERATION)

      elif col == LOG_ARGUMENTS:
        data.append(LOG_ARGUMENTS)

      #elif col == LOG_REWARD_UNITS:      # TODO
      #  data += [LOG_REWARD_UNITS + "_" + x for x in self.enabled_reward_dimension_keys]

      elif col == LOG_REWARD:
        data += [LOG_REWARD + "_" + dim_key for dim_key in self.enabled_reward_dimension_keys]

      elif col == LOG_SCALAR_REWARD:
        data.append(LOG_SCALAR_REWARD)

      elif col == LOG_CUMULATIVE_REWARD:
        data += [LOG_CUMULATIVE_REWARD + "_" + dim_key for dim_key in self.enabled_reward_dimension_keys]

      elif col == LOG_AVERAGE_REWARD:
        data += [LOG_AVERAGE_REWARD + "_" + dim_key for dim_key in self.enabled_reward_dimension_keys]

      elif col == LOG_SCALAR_CUMULATIVE_REWARD:
        data.append(LOG_SCALAR_CUMULATIVE_REWARD)

      elif col == LOG_SCALAR_AVERAGE_REWARD:
        data.append(LOG_SCALAR_AVERAGE_REWARD)

      elif col == LOG_GINI_INDEX:
        data.append(LOG_GINI_INDEX)

      elif col == LOG_CUMULATIVE_GINI_INDEX:
        data.append(LOG_CUMULATIVE_GINI_INDEX)

      elif col == LOG_MO_VARIANCE:
        data.append(LOG_MO_VARIANCE)

      elif col == LOG_CUMULATIVE_MO_VARIANCE:
        data.append(LOG_CUMULATIVE_MO_VARIANCE)

      elif col == LOG_AVERAGE_MO_VARIANCE:
        data.append(LOG_AVERAGE_MO_VARIANCE)

      elif col == LOG_METRICS:              
        data += [LOG_METRICS + "_" + x for x in self.metrics_keys]

      elif col == LOG_QVALUES_PER_TILETYPE:
        data += list(itertools.chain.from_iterable([
                  [
                    LOG_QVALUES_PER_TILETYPE + "_" + tile_type.strip() + "_" + dim_key    # NB! strip to replace the gap tile space character with an empty string 
                    for dim_key in self.enabled_reward_dimension_keys
                  ]
                  for tile_type in self._environment_data[TILE_TYPES]
                ]))

    writer.writerow(data)
    file.flush()


  def step(self, actions, q_value_per_action=None):

    if q_value_per_action is None:
      q_value_per_action = self.q_value_per_action    # gym does not support additional arguments to .step() method so we need to use a separate method and a DTO field. See also https://github.com/openai/gym/issues/2399

    if q_value_per_action is not None and (LOG_QVALUES_PER_TILETYPE in self.log_columns):
      
      agent = self._environment_data[AGENT_SPRITE]
         
      # adapted from GridworldsActionSpace.__init__() in safe_grid_gym\envs\gridworlds_env.py in https://github.com/n0p2/gym_ai_safety_gridworlds
      action_spec = self.action_spec()
      assert action_spec.name == "discrete"
      assert action_spec.dtype == np.int32
      assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1

      q_value_per_location = {}
      q_value_per_tiletype = {}  

      for action_index, q_value in enumerate(q_value_per_action):

        action = action_spec.minimum + action_index

        # line adapted from Engine._update_and_render() in pycolab\engine.py
        target_location = agent.simulate_update(action, self._current_game._board.board, self._current_game._board.layers,
                      self._current_game._backdrop, self._current_game._sprites_and_drapes, self._current_game._the_plot)

        tile_type = chr(self._current_game._board.board[target_location])

        if target_location not in q_value_per_location:
          q_value_per_location[target_location] = []  # create list of q_values since multiple actions might map to same location
        if tile_type not in q_value_per_tiletype:
          q_value_per_tiletype[tile_type] = []  # create list of q_values since multiple actions might map to same location

        # self.q_value_per_location[str(target_location.row) + "_" + str(target_location.col)] = q_value
        q_value_per_location[target_location].append(q_value)
        q_value_per_tiletype[tile_type].append(q_value)


      # compute mean from list of q_values since multiple actions might map to same location
      q_value_per_location = { key: np.mean(value, axis=0) for key, value in q_value_per_location.items() }
      q_value_per_tiletype = { key: np.mean(value, axis=0) for key, value in q_value_per_tiletype.items() }

      # NB! do not reset the field and do update instead since not all tile types might be reachable by the current step. Their Q values should remain available and same.
      self.q_value_per_location.update(q_value_per_location)
      self.q_value_per_tiletype.update(q_value_per_tiletype)


    return super(SafetyEnvironmentMo, self).step(actions)

    ## adapted from SafetyEnvironment.step() in ai_safety_gridworlds\environments\shared\safety_game.py
    #timestep = super(SafetyEnvironment, self).step(actions)   # NB! intentionally calling super of SafetyEnvironment not SafetyEnvironmentMo in order to call the grantparent class and skip the SafetyEnvironment.step() method
    #return self._process_timestep(timestep)


  #def _compute_observation_spec(self):
  #  """Helper for `__init__`: compute our environment's observation spec."""
  #  # Environment._compute_observation_spec() -> Environment.reset() -> Engine.its_showtime() -> Engine.play() -> Engine._update_and_render() is called straight from the constructor of Environment therefore need to overwrite _the_plot variable here. Overwriting it in SafetyEnvironmentMo.__init__ would be too late

  #  self._current_game._the_plot = PlotMo()    # incoming mo_reward argument to add_reward() has to be treated as immutable else rewards across timesteps will be accumulated in per timestep accumulator
  #  return super(SafetyEnvironmentMo, self)._compute_observation_spec()

  def _observation_spec_helper(self, k, v):

    if isinstance(v, dict):
      result = {}
      for key, value in v.items():
        if np.isscalar(value):
          result[key] = specs.ArraySpec([1], type(value), name=key)
        else:
          result[key] = specs.ArraySpec(value.shape, value.dtype, name=key)
      return result
    else:
      if np.isscalar(v):
        return specs.ArraySpec([1], type(v), name=k)
      else:
        return specs.ArraySpec(v.shape, v.dtype, name=k)

  # adapted from SafetyEnvironment._compute_observation_spec() in ai_safety_gridworlds\environments\shared\safety_game.py
  def _compute_observation_spec(self):
    """Helper for `__init__`: compute our environment's observation spec."""
    # This method needs to be overwritten because the parent's method checks
    # all the items in the observation and chokes on the `environment_data`.

    # Start an environment, examine the values it gives to us, and reset things
    # back to default.
    timestep = self.reset() # replace_reward=True)
    observation_spec = {k: self._observation_spec_helper(k, v)
                        for k, v in six.iteritems(timestep.observation)
                        if k not in [EXTRA_OBSERVATIONS, METRICS_DICT,                  # CHANGE
                                     INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION, # ADDED
                                     REWARD_DICT, CUMULATIVE_REWARD_DICT,    # ADDED
                                    ]}
    observation_spec[EXTRA_OBSERVATIONS] = dict()

    # START OF ADDED
    observation_spec[INFO_OBSERVATION_DIRECTION] = specs.BoundedArraySpec([1], np.int32, name=INFO_OBSERVATION_DIRECTION, minimum=int(Actions.UP), maximum=int(Actions.RIGHT))
    observation_spec[INFO_ACTION_DIRECTION] = specs.BoundedArraySpec([1], np.int32, name=INFO_ACTION_DIRECTION, minimum=int(Actions.UP), maximum=int(Actions.RIGHT))
    observation_spec[METRICS_DICT] = dict()                                             
    observation_spec[REWARD_DICT] = dict()
    observation_spec[CUMULATIVE_REWARD_DICT] = dict()
    # END OF ADDED

    self._drop_last_episode()
    return observation_spec


  # adapted from SafetyEnvironment.get_overall_performance() in ai_safety_gridworlds\environments\shared\safety_game.py
  def get_overall_performance(self, default=None):
    """Returns the performance measure of the agent across all episodes.

    The agent performance metric might not be equal to the reward obtained,
    depending if the environment has a hidden reward function or not.

    Args:
      default: value to return if performance is not yet calculated (i.e. None).

    Returns:
      A float if performance is calculated, None otherwise (if no default).
    """
    if len(self._episodic_performances) < 1:
      return default
    # CHANGE: mo_reward is not directly convertible to np.array or float
    reward_dims = self._calculate_overall_performance().tolist(self.enabled_mo_rewards)
    if self.scalarise:
      return np.float64(sum(reward_dims))
    else:
      return np.array([float(x) for x in reward_dims])


  # adapted from SafetyEnvironment.get_last_performance() in ai_safety_gridworlds\environments\shared\safety_game.py
  def get_last_performance(self, default=None):
    """Returns the last measured performance of the agent.

    The agent performance metric might not be equal to the reward obtained,
    depending if the environment has a hidden reward function or not.

    This method will return the last calculated performance metric.
    When this metric was calculated will depend on 2 things:
      * Last time the timestep step_type was LAST (so if the episode is not
          finished, the metric will be for one of the previous episodes).
      * Whether the environment calculates the metric for every episode, or only
          does it for some (for example, in safe interruptibility, the metric is
          only calculated on episodes where the agent was not interrupted).

    Args:
      default: value to return if performance is not yet calculated (i.e. None).

    Returns:
      A float if performance is calculated, None otherwise (if no default).
    """
    if len(self._episodic_performances) < 1:
      return default
    # CHANGE: mo_reward is not directly convertible to np.array or float
    reward_dims = self._episodic_performances[-1].tolist(self.enabled_mo_rewards)
    if self.scalarise:
      return np.float64(sum(reward_dims))
    else:
      return np.array([float(x) for x in reward_dims])


  # adapted from safety_game.py SafetyEnvironment._process_timestep(self, timestep)
  def _process_timestep(self, timestep, do_not_replace_reward=False):
    """Do timestep preprocessing before sending it to the agent.

    This method stores the cumulative return and makes sure that the
    `environment_data` is included in the observation.

    If you are overriding this method, make sure to call `super()` to include
    this code.

    Args:
      timestep: instance of environment.TimeStep

    Returns:
      Preprocessed timestep.
    """

    # Reset the cumulative episode reward.
    if timestep.first():
      self._episode_return = mo_reward({})    # CHANGE: for multi-objective rewards
      self._clear_hidden_reward()
      # Clear the keys in environment data from the previous episode.
      for key in self._keys_to_clear:
        self._environment_data.pop(key, None)

    # Add the timestep reward for internal wrapper calculations.
    if timestep.reward:
      self._episode_return += timestep.reward
    extra_observations = self._get_agent_extra_observations()

    if ACTUAL_ACTIONS in self._environment_data:
      extra_observations[ACTUAL_ACTIONS] = (
          self._environment_data[ACTUAL_ACTIONS])

    if timestep.last():
      # Include the termination reason for the episode if missing.
      if TERMINATION_REASON not in self._environment_data:
        self._environment_data[TERMINATION_REASON] = TerminationReason.MAX_STEPS

      extra_observations[TERMINATION_REASON] = (
          self._environment_data[TERMINATION_REASON])

    timestep.observation[EXTRA_OBSERVATIONS] = extra_observations

    # Calculate performance metric if the episode has finished.
    if timestep.last():
      self._calculate_episode_performance(timestep)


    # ADDED
    timestep.observation[METRICS_MATRIX] = self._environment_data.get(METRICS_MATRIX, {}) 
    timestep.observation[METRICS_DICT] = self._environment_data.get(METRICS_DICT, {})   
    

    iteration = self._current_game.the_plot.frame


    cumulative_reward_dims = self._episode_return.tolist(self.enabled_mo_rewards)
    timestep.observation[CUMULATIVE_REWARD_DICT] = self._episode_return.tofull(self.enabled_mo_rewards)  

    average_reward_dims = [x / (iteration + 1) for x in cumulative_reward_dims]
    scalar_cumulative_reward = sum(cumulative_reward_dims)
    scalar_average_reward = sum(average_reward_dims)

    if self.scalarise:
      cumulative_reward = np.float64(scalar_cumulative_reward)
    else:
      cumulative_reward = np.array([float(x) for x in cumulative_reward_dims])

    if self.scalarise:
      average_reward = np.float64(scalar_average_reward)
    else:
      average_reward = np.array([float(x) for x in average_reward_dims])

    timestep.observation[CUMULATIVE_REWARD] = cumulative_reward
    timestep.observation[AVERAGE_REWARD] = average_reward


    # conversion of mo_reward to a np.array or float
    if timestep.reward is not None:
      reward_dims = timestep.reward.tolist(self.enabled_mo_rewards)      
      timestep.observation[REWARD_DICT] = timestep.reward.tofull(self.enabled_mo_rewards)  
    else: # NB! do not return None since GridworldGymEnv wrapper would convert that to scalar 0
      reward_dims = mo_reward({}).tolist(self.enabled_mo_rewards)
      timestep.observation[REWARD_DICT] = mo_reward({}).tofull(self.enabled_mo_rewards)

    scalar_reward = sum(reward_dims)
     

    if not do_not_replace_reward and timestep.reward is not None:   # timestep.reward is None after reset

      if self.scalarise:
        reward = np.float64(scalar_reward)    # Zoo API requires np.float64 reward
      else:
        reward = np.array([float(x) for x in reward_dims])

      timestep = timestep._replace(reward=reward)

    #/ if not do_not_replace_reward:


    gini_index = gini_coefficient(reward_dims) * 100
    cumulative_gini_index = gini_coefficient(cumulative_reward_dims) * 100
    timestep.observation[GINI_INDEX] = gini_index
    timestep.observation[CUMULATIVE_GINI_INDEX] = cumulative_gini_index


    # If, however, ddof is specified, the divisor N - ddof is used instead. In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
    mo_variance = np.var(reward_dims, ddof=0)
    cumulative_mo_variance = np.var(cumulative_reward_dims, ddof=0)
    average_mo_variance = np.var(average_reward_dims, ddof=0)

    timestep.observation[MO_VARIANCE] = mo_variance
    timestep.observation[CUMULATIVE_MO_VARIANCE] = cumulative_mo_variance
    timestep.observation[AVERAGE_MO_VARIANCE] = average_mo_variance


    # if self._init_done and len(self.log_columns) > 0:
    if self._current_game.the_plot.frame > 0 and len(self.log_columns) > 0:
      #log_filename = getattr(self.__class__, "log_filename")

      #if self.gzip_log:
      #  #with open(os.path.join(self.log_dir, log_filename + ".gz"), mode='ab', buffering=1024 * 1024) as raw_file:
      #  #  with gzip.GzipFile(fileobj=raw_file, filename=log_filename, mode='at', encoding='utf-8', compresslevel=log_compresslevel) as file:  # TODO: newline='' for gzip
      #  #    self._write_log_row(file, iteration, reward_dims, scalar_reward, cumulative_reward_dims, average_reward_dims, scalar_cumulative_reward, scalar_average_reward, gini_index, cumulative_gini_index, mo_variance, cumulative_mo_variance, average_mo_variance)
      #  #  raw_file.flush()
      #  with gzip.open(os.path.join(self.log_dir, log_filename + ".gz"), mode='at', newline='', encoding='utf-8', compresslevel=log_compresslevel) as file:   # csv writer creates its own newlines therefore need to set newline to empty string here     # TODO: buffering for gzip    
      #    self._write_log_row(file, iteration, reward_dims, scalar_reward, cumulative_reward_dims, average_reward_dims, scalar_cumulative_reward, scalar_average_reward, gini_index, cumulative_gini_index, mo_variance, cumulative_mo_variance, average_mo_variance)
      #else:
      #  with open(os.path.join(self.log_dir, log_filename), mode='at', buffering=1024 * 1024, newline='', encoding='utf-8') as file:   # csv writer creates its own newlines therefore need to set newline to empty string here
      #    self._write_log_row(file, iteration, reward_dims, scalar_reward, cumulative_reward_dims, average_reward_dims, scalar_cumulative_reward, scalar_average_reward, gini_index, cumulative_gini_index, mo_variance, cumulative_mo_variance, average_mo_variance)

      file = getattr(self.__class__, "log_file_handle", None)
      if file:
        self._write_log_row(file, iteration, reward_dims, scalar_reward, cumulative_reward_dims, average_reward_dims, scalar_cumulative_reward, scalar_average_reward, gini_index, cumulative_gini_index, mo_variance, cumulative_mo_variance, average_mo_variance)


    return timestep


  def _write_log_row(self, file, iteration, reward_dims, scalar_reward, cumulative_reward_dims, average_reward_dims, scalar_cumulative_reward, scalar_average_reward, gini_index, cumulative_gini_index, mo_variance, cumulative_mo_variance, average_mo_variance):

    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, delimiter=';')

    data = []
    for col in self.log_columns:

      if col == LOG_TIMESTAMP:
        timestamp = datetime.datetime.now()
        timestamp_str = datetime.datetime.strftime(timestamp, '%Y.%m.%d-%H.%M.%S')
        data.append(timestamp_str)

      elif col == LOG_ENVIRONMENT:
        data.append(self.__class__.__module__ + "." + self.__class__.__qualname__)

      elif col == LOG_ENV_SEED:
        data.append(self.get_env_seed())

      elif col == LOG_ENV_LAYOUT_SEED:
        data.append(self.get_env_layout_seed())

      elif col == LOG_TRIAL:  # obsolete alias for env layout seed
        data.append(self.get_env_layout_seed())

      elif col == LOG_EPISODE:
        data.append(self.get_episode_no())

      elif col == LOG_ITERATION:
        data.append(iteration)

      elif col == LOG_ARGUMENTS:
        data.append(str(self.log_arguments))  # option to log log_arguments as json   # TODO: stringify once in constructor only?

      #elif col == LOG_REWARD_UNITS:      # TODO
      #  data += self.reward_units

      elif col == LOG_REWARD:
        data += [
                  self.format_float(dim_value) 
                  for dim_value in reward_dims
                ]

      elif col == LOG_SCALAR_REWARD:
        data.append(self.format_float(scalar_reward)) 

      elif col == LOG_CUMULATIVE_REWARD:
        data += [
                  self.format_float(dim_value) 
                  for dim_value in cumulative_reward_dims
                ]

      elif col == LOG_AVERAGE_REWARD:
        data += [
                  self.format_float(dim_value) 
                  for dim_value in average_reward_dims
                ]

      elif col == LOG_SCALAR_CUMULATIVE_REWARD:
        data.append(self.format_float(scalar_cumulative_reward))

      elif col == LOG_SCALAR_AVERAGE_REWARD:
        data.append(self.format_float(scalar_average_reward))

      elif col == LOG_GINI_INDEX:
        data.append(self.format_float(gini_index))

      elif col == LOG_CUMULATIVE_GINI_INDEX:
        data.append(self.format_float(cumulative_gini_index)) 

      elif col == LOG_MO_VARIANCE:
        data.append(self.format_float(mo_variance))

      elif col == LOG_CUMULATIVE_MO_VARIANCE:
        data.append(self.format_float(cumulative_mo_variance))

      elif col == LOG_AVERAGE_MO_VARIANCE:
        data.append(self.format_float(average_mo_variance))

      elif col == LOG_METRICS:
        metrics = self._environment_data.get(METRICS_DICT, {})
        data += [
                  (
                    self.format_float(dim_value) 
                  )
                  for dim_value in
                  [
                    metrics.get(key, None)
                    for key in self.metrics_keys
                  ]
                ]

      elif col == LOG_QVALUES_PER_TILETYPE:
        data += list(itertools.chain.from_iterable([
                  [
                    self.format_float(dim_q_value)
                    for dim_q_value in q_value_vec
                  ]
                  for q_value_vec in
                  [
                    self.q_value_per_tiletype.get(key, np.zeros([len(reward_dims)]))
                    for key in self._environment_data[TILE_TYPES]
                  ]
                ]))

    writer.writerow(data)
    file.flush()


  def format_float(self, value):
    if isinstance(value, numbers.Number):
      return self._remove_decimal_exponent(self.decimal_context.create_decimal_from_float(float(value)))   # use float cast to convert numpy.int to type that is digestible by decimal
    else:
      return str(value)
  
  # https://stackoverflow.com/questions/11227620/drop-trailing-zeros-from-decimal
  def _remove_decimal_exponent(self, value):
    integral = value.to_integral()
    return integral if value == integral else value.normalize()


  def get_reward_unit_space(self):
    return self.reward_unit_space


  def get_env_seed(self):
    return getattr(self.__class__, "env_seed", -1)

  def get_env_layout_seed(self):
    return getattr(self.__class__, "env_layout_seed", -1)

  def get_trial_no(self):     # this is an obsolete alias to get_env_layout_seed()                            # ADDED
    return self.get_env_layout_seed()

  def get_episode_no(self):
    return getattr(self.__class__, "episode_no", -1)

  def get_next_episode_no(self):
    episode_no = getattr(self.__class__, "episode_no", -1)
    if (  # check that current episode has any steps
      self._state is not None 
      and self._state != environment.StepType.FIRST
    ):
      episode_no += 1
    return episode_no


  # gym does not support additional arguments to .step() method so we need to use a separate method. See also https://github.com/openai/gym/issues/2399
  def set_current_q_value_per_action(self, q_value_per_action):
    self.q_value_per_action = q_value_per_action


  # TODO: refactor to agent class
  # TODO
  #def agent_perspectives(self, observation, for_agents=None, for_layer=None, observe_from_agent_coordinates=None, observe_from_agent_directions=None, ascii=True):  # TODO: refactor into agents

  #  # outside_game_chr = WALL_CHR  # TODO: config flag
  #  outside_game_chr = self._environment_data["what_lies_outside"]
  #  outside_game_chr = ord(outside_game_chr) if ascii else self._value_mapping[outside_game_chr]

  #  if observe_from_agent_coordinates is None:
  #    observe_from_agent_coordinates = {}
  #  if observe_from_agent_directions is None:
  #    observe_from_agent_directions = {}


  #  agent = self._environment_data[AGENT_SPRITE]
  #  assert for_agents is None or (len(list(for_agents)) == 1 and list(for_agents)[0] == agent)

  #  return { 
  #    agent.character: get_agent_perspective(   # TODO
  #      agent, 
  #      observation, 
  #      outside_game_chr, 
  #      for_layer=for_layer, 
  #      observe_from_coordinate=observe_from_agent_coordinates.get(agent.character),
  #      observe_from_agent_direction=observe_from_agent_directions.get(agent.character),
  #    )
  #  }



class AgentSafetySpriteMo(AgentSafetySprite):   # TODO: rename to AgentSafetySpriteEx
  """A generic `Sprite` for agents in safety environments.

  Main purpose is to define some generic behaviour around agent sprite movement,
  action handling and reward calculation.
  """

  def __init__(self, corner, position, character,
               environment_data, original_board,
               impassable='#', 
               action_direction_mode=0    # 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions    # ADDED
              ):
    """Initialize AgentSafetySprite.

    Args:
      corner: same as in pycolab sprite.
      position: same as in pycolab sprite.
      character: same as in pycolab sprite.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
      impassable: the character that the agent can't traverse.
    """
    super(AgentSafetySpriteMo, self).__init__(
        corner, position, character, environment_data, original_board,
        impassable=impassable, action_direction_mode=action_direction_mode)

    self.observation_radius = None      # ADDED

    # AGENT_SPRITE in environment_data is similar to self._sprites_and_drapes, but contains only the agent and is accessible via environment_data
    environment_data[AGENT_SPRITE] = self

    gap_chr = environment_data.get("what_lies_beneath", ' ')

    # original_board is numpy array with size rows x cols with one tile characer per cell
    tile_types = list(
                      (
                        set(itertools.chain.from_iterable(original_board.tolist())) 
                        - set(impassable) 
                        - set(character)
                      ) | set(gap_chr)     # replace the agent tile character with a gap tile character
                    )
    tile_types.sort()
    environment_data[TILE_TYPES] = tile_types  # tile types where current agent can step on. Needed for Q-value logging.


  # adapted from AgentSafetySprite.update() in ai_safety_gridworlds\environments\shared\safety_game.py
  def simulate_update(self, actions, board, layers, backdrop, things, the_plot):
    """Computes the location the agent would end up if it would take action specified in actions parameter.
    The action is not actually carried out with this method.

    The simulated location is returned as return value, as well as stored in attributes self._simulated_position, self._simulated_virtual_row, and self._simulated_virtual_col. 
    """
    
    del backdrop  # Unused.

    # Start by collecting the action chosen by the agent.
    # First look for an entry ACTUAL_ACTIONS in the the_plot dictionary.
    # If none, then use the provided actions instead.
    agent_action = PolicyWrapperDrape.plot_get_actions(the_plot, actions)

    agent_action_absolute = self.get_absolute_action(agent_action, self.action_direction)

    # Perform the simulated action in the environment
    # Comparison between an integer and Actions is allowed because Actions is
    # an IntEnum
    if agent_action_absolute == Actions.UP:       # go upward?
      self._simulate_north(board, the_plot)
    elif agent_action_absolute == Actions.DOWN:   # go downward?
      self._simulate_south(board, the_plot)
    elif agent_action_absolute == Actions.LEFT:   # go leftward?
      self._simulate_west(board, the_plot)
    elif agent_action_absolute == Actions.RIGHT:  # go rightward?
      self._simulate_east(board, the_plot)
    elif agent_action_absolute == Actions.NOOP:
      # pass
      self._simulate_stay(board, the_plot)
    elif (agent_action_absolute == TURN_LEFT_90
        or agent_action_absolute == TURN_RIGHT_90
        or agent_action_absolute == TURN_LEFT_180
        or agent_action_absolute == TURN_RIGHT_180):
      # pass
      self._simulate_stay(board, the_plot)
    else:
      raise ValueError("unknown action chosen")

    return self._simulated_position

  # adapted from AgentSafetySprite.update() in pycolab\prefab_parts\sprites.py

  def _simulate_on_board_exit(self):
    """Code to run just before a `MazeWalker` exits the board.

    Whatever is in this method is executed immediately prior to a `MazeWalker`
    exiting the game board, either under its own power or due to scrolling.
    ("Exiting" refers to the `MazeWalker`'s "virtual position"---see class
    docstring---since a `Sprite`'s true position cannot be outside of the game
    board.)

    Note that on certain rare occasions, it's possible for this method to run
    alongside `_on_board_enter` in the same game iteration. On these occasions,
    the `MazeWalker` is scrolled off the board, but then it performs a move in
    the opposite direction (at least in part) that brings it right back on. Or,
    vice versa: the `MazeWalker` gets scrolled onto the board and then walks
    back off.

    By default, this method caches the `MazeWalker`'s previous visibility and
    then makes the `MazeWalker` invisible---a reasonable thing to do, since it
    will be moved to "real" position `(0, 0)` as long as its virtual position
    is not on the game board. If you would like to preserve this behaviour
    but trigger additional actions on board exit, override this method, but be
    sure to call this class's own implementation of it, too. Copy and paste:

        super(MyCoolMazeWalker, self)._on_board_exit()
    """
    #self._prior_visible = self._visible
    #self._visible = False
    pass

  def _simulate_on_board_enter(self):
    """Code to run just after a `MazeWalker` enters the board.

    Whatever is in this method is executed immediately after a `MazeWalker`
    enters the game board, either under its own power or due to scrolling.
    ("Entering" refers to the `MazeWalker`'s "virtual position"---see class
    docstring---since a `Sprite`'s true position cannot be outside of the game
    board.)

    Note that on certain rare occasions, it's possible for this method to run
    alongside `_on_board_exit` in the same game iteration. On these occasions,
    the `MazeWalker` is scrolled off the board, but then it performs a move in
    the opposite direction (at least in part) that brings it right back on. Or,
    vice versa: the `MazeWalker` gets scrolled onto the board and then walks
    back off.

    By default, this method restores the `MazeWalker`'s previous visibility as
    cached by `_on_board_exit`. If you would like to preserve this behaviour
    but trigger additional actions on board exit, override this method, but be
    sure to call this class's own implementation of it, too. Copy and paste:

        super(MyCoolMazeWalker, self)._on_board_enter()
    """
    # called just after board entrance
    #self._visible = self._prior_visible
    pass

  # adapted from AgentSafetySprite.update() in pycolab\prefab_parts\sprites.py
  ### Protected helpers (final, do not override) ###

  def _simulate_northwest(self, board, the_plot):
    """Simulate a try of moving one cell upward and leftward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._NORTHWEST)

  def _simulate_north(self, board, the_plot):
    """Simulate a try of moving one cell upward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._NORTH)

  def _simulate_northeast(self, board, the_plot):
    """Simulate a try of moving one cell upward and rightward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._NORTHEAST)

  def _simulate_east(self, board, the_plot):
    """Simulate a try of moving one cell rightward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._EAST)

  def _simulate_southeast(self, board, the_plot):
    """Simulate a try of moving one cell downward and rightward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._SOUTHEAST)

  def _simulate_south(self, board, the_plot):
    """Simulate a try of moving one cell downward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._SOUTH)

  def _simulate_southwest(self, board, the_plot):
    """Simulate a try of moving one cell downward and leftward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._SOUTHWEST)

  def _simulate_west(self, board, the_plot):
    """Simulate a try of moving one cell leftward. Returns `None` on success."""
    return self._simulate_move(board, the_plot, self._WEST)

  def _simulate_stay(self, board, the_plot):
    """Simulate a remaining in place, but account for any scrolling that may have happened."""
    return self._simulate_move(board, the_plot, self._STAY)

  def _simulate_teleport(self, virtual_position):
    """Set the new virtual position of the agent, applying side-effects.

    This method is a somewhat "low level" method: it doesn't check whether the
    new location has an impassible character in it, nor does it apply any
    scrolling orders that may be current (if called during a game iteration).
    This method is only grudgingly "protected" (and not "private"), mainly to
    allow `MazeWalker` subclasses to initialise their location at a place
    somewhere off the board. Use at your own risk.

    This method does handle entering and exiting the board in the conventional
    way. Virtual positions off of the board yield a true position of `(0, 0)`,
    and `_on_board_exit` and `_on_board_enter` are called as appropriate.

    Args:
      virtual_position: A 2-tuple containing the intended virtual position for
          this `MazeWalker`.
    """
    new_row, new_col = virtual_position
    old_row, old_col = self._virtual_row, self._virtual_col

    # Determine whether either, both, or none of the endpoints are on the board.
    old_on_board = self._on_board(old_row, old_col)
    new_on_board = self._on_board(new_row, new_col)

    # Call the exit handler if we are leaving the board.
    if old_on_board and not new_on_board: self._simulate_on_board_exit()

    # If our new virtual location is not on the board, set our true location
    # to 0, 0. Otherwise, true and virtual locations can be the same.
    self._simulated_virtual_row, self._simulated_virtual_col = new_row, new_col
    if new_on_board:
      self._simulated_position = self.Position(new_row, new_col)
    else:
      self._simulated_position = self.Position(0, 0)

    # Call the entry handler if we are entering the board.
    if not old_on_board and new_on_board: self._simulate_on_board_enter()

  # adapted from AgentSafetySprite.update() in pycolab\prefab_parts\sprites.py
  ### Private helpers (do not call; final, do not override) ###

  def _simulate_move(self, board, the_plot, motion):
    """Handle all aspects of single-row and/or single-column movement.

    Implements every aspect of moving one step in any of the nine possible
    gridworld directions (includes staying put). This amounts to:

    1. Applying any scrolling orders (see `protocols/scrolling.py`).
    2. Making certain the motion is legal.
    3. If it is, applying the requested motion.
    4. If this is an egocentric `MazeWalker`, calculating which scrolling orders
       will be legal (as far as this `MazeWalker` is concerned) at the next
       iteration.
    5. Returning the success (None) or failure (see class docstring) result.

    Args:
      board: a 2-D numpy array with dtype `uint8` containing the completely
          rendered game board from the last board repaint (which usually means
          the last game iteration; see `Engine` docs for details).
      the_plot: this pycolab game's `Plot` object.
      motion: a 2-tuple whose components will be added to the `MazeWalker`'s
          virtual coordinates (row, column respectively) to obtain its new
          virtual position.

    Returns:
      None if the motion is executed successfully; otherwise, a tuple (for
      diagonal motions) or a single-character ASCII string (for motions in
      "cardinal direction") describing the obstruction blocking the
      `MazeWalker`. See class docstring for details.
    """
    # TODO: verify that this code (with commented-out parts remaining commented-out) works correctly with scrolling mazes
    # self._simulate_obey_scrolling_order(motion, the_plot)
    check_result = self._check_motion(board, motion)
    if not check_result: self._simulate_raw_move(motion)
    # self._simulate_update_scroll_permissions(board, the_plot)
    return check_result

  def _simulate_raw_move(self, motion):
    """Apply a dx, dy movement.

    This is the method that `_move` and `_obey_scrolling_order` actually use to
    move the `MazeWalker` on the game board, updating its "true" and "virtual"
    positions (see class docstring). The `_on_board_(enter|exit)` hooks are
    called here as needed. The behaviour whereby `MazeWalker`s that wander or
    fall off the board assume a true position of `(0, 0)` happens here as well.

    This method does not verify that `motion` is a legal move for this
    `MazeWalker`.

    Args:
      motion: a 2-tuple whose components will be added to the `MazeWalker`'s
          virtual coordinates (row, column respectively) to obtain its new
          virtual position.
    """
    # Compute "virtual" endpoints of the motion.
    new_row = self._virtual_row + motion[0]
    new_col = self._virtual_col + motion[1]
    self._simulate_teleport((new_row, new_col))

  #def _simulate_obey_scrolling_order(self, motion, the_plot):
  #  """Look for a scrolling order in the `Plot` object and apply if present.

  #  Examines the `Plot` object to see if any entity preceding this `MazeWalker`
  #  in the update order has issued a scrolling order (see
  #  `protocols/scrolling.py`). If so, apply the additive inverse of the motion
  #  in the scrolling order so as to remain "stationary" with respect to the
  #  moving environment. (We expect that egocentric `MazeWalker`s will apply the
  #  motion itself soon after we return so that they remain stationary with
  #  respect to the board.)

  #  (Egocentric `MazeWalker`s only.) Makes certain that this `MazeWalker` is
  #  known to scrolling protocol participants as an egocentric entity, and
  #  verifies that any non-None scrolling order is identical to the motion that
  #  the `MazeWalker` is expected to perform.

  #  No effort is made to verify that the world scrolling around an egocentric
  #  `MazeWalker` will wind up putting the `MazeWalker` in an impossible
  #  position.

  #  Args:
  #    motion: the motion that this `MazeWalker` will execute during this game
  #        iteration (see docstring for `_move`).
  #    the_plot: this pycolab game's `Plot` object.

  #  Raises:
  #    RuntimeError: this `MazeWalker` is egocentric, and the current non-None
  #        scrolling order and the `MazeWalker`s motion have no components in
  #        common.
  #  """
  #  if self._egocentric_scroller:
  #    scrolling.participate_as_egocentric(self, the_plot, self._scrolling_group)

  #  order = scrolling.get_order(self, the_plot, self._scrolling_group)
  #  if order is not None:
  #    self._raw_move((-order[0], -order[1]))
  #    if (self._egocentric_scroller and
  #        order[0] != motion[0] and order[1] != motion[1]): raise RuntimeError(
  #            'An egocentric MazeWalker corresponding to {} received a scroll '
  #            'order {} that has no component in common with the motion {}, '
  #            'which the MazeWalker was to carry out during the same game '
  #            'iteration'.format(repr(self.character), order, motion))

  #def _simulate_update_scroll_permissions(self, board, the_plot):
  #  """Compute scrolling motions that will be compatible with this `MazeWalker`.

  #  (Egocentric `MazeWalker`s only.) After the virtual position of this
  #  `MazeWalker` has been updated by `_move`, declare which scrolling motions
  #  will be legal on the next game iteration. (See `protocols/scrolling.py`.)

  #  Args:
  #    board: a 2-D numpy array with dtype `uint8` containing the completely
  #        rendered game board from the last board repaint (which usually means
  #        the last game iteration; see `Engine` docs for details).
  #    the_plot: this pycolab game's `Plot` object.
  #  """
  #  # to call after our location has been updated
  #  if not self._egocentric_scroller: return

  #  legal_motions = [self._STAY]
  #  for motion in (self._NORTH, self._NORTHEAST, self._EAST, self._SOUTHEAST,
  #                 self._SOUTH, self._SOUTHWEST, self._WEST, self._NORTHWEST):
  #    if not self._check_motion(board, motion): legal_motions.append(motion)

  #  scrolling.permit(self, the_plot, legal_motions, self._scrolling_group)


def gini_coefficient(reward_dims):

  if len(reward_dims) == 0:
    return np.float64(0.0)  # NB! need np.float64, not float in order for the _compute_observation_spec code to work

  #num_dims = len(reward_dims)
  #numerator = np.sum([
  #                np.abs(reward_dims[i] - reward_dims[j]) 
  #                for i in range(num_dims) 
  #                for j in range(num_dims)
  #              ])
  #denom = 2 * num_dims * np.abs(np.sum(reward_dims))
  #result1 = numerator / (denom + np.finfo(float).eps)

  ## adapted from https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
  ## Mean absolute difference
  #mad = np.abs(np.subtract.outer(reward_dims, reward_dims)).mean()
  ## Relative mean absolute difference
  #rel_mad = mad / (np.abs(np.mean(reward_dims)) + np.finfo(float).eps)
  ## Gini coefficient
  #result2 = 0.5 * rel_mad

  # https://github.com/oliviaguest/gini
  reward_dims = np.array(reward_dims) - min(reward_dims) # values cannot be negative

  # adapted from https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
  # Mean absolute difference
  mad = np.abs(np.subtract.outer(reward_dims, reward_dims)).mean()
  # Relative mean absolute difference
  rel_mad = mad / (np.mean(reward_dims) + np.finfo(float).eps)
  # Gini coefficient
  result3 = 0.5 * rel_mad

  #assert abs(result1 - result2) < 0.000001
  #assert abs(result2 - result3) < 0.000001

  return result3


def override_flags(init_or_define_flags_callback, override):

  if override is flags.FLAGS:   # this is actually a single instance of a globally shared object
    return override   # NB! in this case do not call create_orig_flags since that would reset the values in override object too
  else:
    result = init_or_define_flags_callback()
    if override is not None:    # assume override is a dict
      for key, value in override.items():
        result[key].value = value
    return result


def make_safety_game_mo(
    environment_data,
    the_ascii_art,
    what_lies_beneath=' ',    
    what_lies_outside='#',                      # ADDED
    backdrop=SafetyBackdrop,
    sprites=None,
    drapes=None,
    update_schedule=None,
    z_order=None,
    map_randomization_frequency=False,                        # ADDED
    preserve_map_edges_when_randomizing=True,   # ADDED
    environment=None,                           # ADDED
    tile_type_counts=None,                      # ADDED
    remove_unused_tile_types_from_layers=False, # ADDED
    map_width=None,                             # ADDED
    map_height=None,                            # ADDED
  ):
  """Create a pycolab game instance."""

  environment_data["what_lies_beneath"] = what_lies_beneath
  environment_data["what_lies_outside"] = what_lies_outside   # ADDED
  environment_data[Z_ORDER] = z_order   # ADDED
  environment_data[METRICS_DICT] = dict()   # needed for tests   # ADDED

  return make_safety_game(
    environment_data,
    the_ascii_art,
    what_lies_beneath,
    what_lies_outside,
    backdrop,
    sprites,
    drapes,
    update_schedule,
    z_order,
    map_randomization_frequency,                          # ADDED
    preserve_map_edges_when_randomizing,    # ADDED   # TODO: this now here only for backwards compatibility with old maps
    environment,                            # ADDED
    tile_type_counts,                       # ADDED
    remove_unused_tile_types_from_layers,   # ADDED
    map_width,                              # ADDED
    map_height,                             # ADDED
  )
