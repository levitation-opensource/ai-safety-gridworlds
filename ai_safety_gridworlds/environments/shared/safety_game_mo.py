# Copyright 2022 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 The AI Safety Gridworlds Authors. All Rights Reserved.
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
import os

# Dependency imports
from ai_safety_gridworlds.environments.shared.rl import array_spec as specs
from ai_safety_gridworlds.environments.shared.rl import environment
from ai_safety_gridworlds.environments.shared.mo_reward import mo_reward
from ai_safety_gridworlds.environments.shared.plot_mo import PlotMo
from ai_safety_gridworlds.environments.shared.safety_game import SafetyEnvironment, ACTUAL_ACTIONS, TERMINATION_REASON, EXTRA_OBSERVATIONS
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason


import numpy as np

import six


METRICS_DICT = 'metrics_dict'
METRICS_MATRIX = 'metrics_matrix'
CUMULATIVE_REWARD = 'cumulative_reward'


# timestamp, environment_name, episode_no, iteration_no, environment_flags, reward_unit_sizes, rewards, cumulative_rewards, metrics
LOG_TIMESTAMP = 'timestamp'
LOG_ENVIRONMENT = 'env'
LOG_TRIAL = 'trial'
LOG_EPISODE = 'episode'
LOG_ITERATION = 'iteration'
LOG_ARGUMENTS = 'arguments'
LOG_REWARD_UNITS = 'reward_unit'      # TODO
LOG_REWARD = 'reward'
LOG_SCALAR_REWARD = 'scalar_reward'
LOG_CUMULATIVE_REWARD = 'cumulative_reward'
LOG_SCALAR_CUMULATIVE_REWARD = 'scalar_cumulative_reward'
LOG_METRICS = 'metric'


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
  "trial_no",
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


class SafetyEnvironmentMo(SafetyEnvironment):
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
               environment_data={},
               #repainter=None,
               #max_iterations=100,
               FLAGS=None,
               scalarise=False,
               log_columns=[],
               log_dir="logs",
               log_filename_comment="",
               log_arguments=None,
               log_arguments_to_separate_file=True,
               trial_no=1,
               episode_no=None,
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
        environment_name, trial_no, episode_no, iteration_no, 
        environment_arguments, reward_unit_sizes, reward, scalar_reward, 
        cumulative_reward, scalar_cumulative_reward, metrics)
      log_dir: directory to save log files to.
      log_arguments: dictionary of environment arguments to log if LOG_ARGUMENTS 
        is set in log_columns or if log_arguments_to_separate_file is True. If
        log_arguments is None then all arguments are logged except the ones 
        listed in log_arguments_to_skip.
      log_arguments_to_separate_file: whether to log environment arguments to a 
        separate file.
      trial_no: trial number.
      episode_no: episode number. Use when you need to reset episode_no counter
        manually for some reason (for example, when changing flags).
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
    self.reward_unit_space[0] = np.array([float(x) for x in self.reward_unit_space[0]]) # min
    self.reward_unit_space[1] = np.array([float(x) for x in self.reward_unit_space[1]]) # max

    self.scalarise = scalarise


    if environment_data is None:
      self._environment_data = {}
    else:
      self._environment_data = environment_data

    self._environment_data[METRICS_DICT] = dict()
    self._environment_data[METRICS_MATRIX] = np.empty([0, 2], np.object)
    self._environment_data[CUMULATIVE_REWARD] = np.array(mo_reward({}).tolist(self.enabled_mo_rewards))



    # self._init_done = False   # needed in order to skip logging during _compute_observation_spec() call

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



    prev_trial_no = getattr(self.__class__, "trial_no", -1)
    setattr(self.__class__, "trial_no", trial_no)

    if (   # detect when a new experiment is started
      prev_experiment_no != next_experiment_no
      or prev_log_filename_comment != log_filename_comment 
      or prev_log_arguments != self.log_arguments
      or prev_flags != self.flags
      or prev_enabled_reward_dimension_keys != self.enabled_reward_dimension_keys
      or prev_metrics_keys != self.metrics_keys
    ):
      prev_trial_no = -1    # this causes a new log file to be created


    if prev_trial_no != trial_no: # if new trial is started then reset the episode_no counter
      setattr(self.__class__, "episode_no", 1)  # use static attribute so that the value survives re-construction of the environment
      # use a different random number sequence for each trial
      # at the same time use deterministic seed numbers so that if the trials are re-run then the results are same
      np.random.seed(int(trial_no) & 0xFFFFFFFF)  # 0xFFFFFFFF: np.random.seed accepts 32-bit int only
      # np.random.seed(int(time.time() * 10000000) & 0xFFFFFFFF)  # 0xFFFFFFFF: np.random.seed accepts 32-bit int only
    
    if episode_no is not None:
      setattr(self.__class__, "episode_no", episode_no)  # use static attribute so that the value survives re-construction of the environment



    self.log_dir = log_dir
    self.log_filename_comment = log_filename_comment
    self.log_columns = log_columns

    if len(self.log_columns) > 0:

      if self.log_dir and not os.path.exists(self.log_dir):
        os.makedirs(self.log_dir)

      # TODO: option to include log_arguments in filename

      if prev_trial_no == -1:  # save all episodes and all trials to same file

        classname = self.__class__.__name__
        timestamp = datetime.datetime.now()
        timestamp_str = datetime.datetime.strftime(timestamp, '%Y.%m.%d-%H.%M.%S')

        # NB! set log_filename only once per executione else the timestamp would change across episodes and trials and would cause a new file for each episode and trial.
        log_filename = classname + ("-" if self.log_filename_comment else "") + self.log_filename_comment + "-" + timestamp_str + ".csv"
        setattr(self.__class__, "log_filename", log_filename)
        arguments_filename = classname + ("-" if self.log_filename_comment else "") + self.log_filename_comment + "-arguments-" + timestamp_str + ".txt" 


        if log_arguments_to_separate_file:
          with open(os.path.join(self.log_dir, arguments_filename), 'w', 1024 * 1024) as file:
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


        with open(os.path.join(self.log_dir, log_filename), 'a', 1024 * 1024, newline='') as file:   # csv writer creates its own newlines therefore need to set newline to empty string here
          writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')

          data = []
          for col in self.log_columns:

            if col == LOG_TIMESTAMP:
              data.append(LOG_TIMESTAMP)

            elif col == LOG_ENVIRONMENT:
              data.append(LOG_ENVIRONMENT)

            elif col == LOG_TRIAL:
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
              data += [LOG_REWARD + "_" + x for x in self.enabled_reward_dimension_keys]

            elif col == LOG_SCALAR_REWARD:
              data.append(LOG_SCALAR_REWARD)

            elif col == LOG_CUMULATIVE_REWARD:
              data += [LOG_CUMULATIVE_REWARD + "_" + x for x in self.enabled_reward_dimension_keys]

            elif col == LOG_SCALAR_CUMULATIVE_REWARD:
              data.append(LOG_SCALAR_CUMULATIVE_REWARD)

            elif col == LOG_METRICS:              
              data += [LOG_METRICS + "_" + x for x in self.metrics_keys]

          writer.writerow(data)
          file.flush()
        
    else:
      setattr(self.__class__, "log_filename", None)


  # adapted from SafetyEnvironment.reset() in ai_safety_gridworlds\environments\shared\safety_game.py and from Environment.reset() in ai_safety_gridworlds\environments\shared\rl\pycolab_interface.py
  def reset(self, trial_no=None, start_new_experiment=False):
    """Start a new episode. 
    Increment the episode counter if the previous game was played.
    
    trial_no: trial number. If not specified then previous trial_no is reused.
    """
    # Environment._compute_observation_spec() -> Environment.reset() -> Engine.its_showtime() -> Engine.play() -> Engine._update_and_render() is called straight from the constructor of Environment therefore need to overwrite _the_plot variable here. Overwriting it in SafetyEnvironmentMo.__init__ would be too late
    
    if start_new_experiment:  # instruct the environment to start a new log file NEXT time the environment is constructed
      prev_experiment_no = getattr(self.__class__, "prev_experiment_no", 0)
      setattr(self.__class__, "next_experiment_no", prev_experiment_no + 1)


    # note: no elif here. env.reset(start_new_experiment=True) should still execute rest of the .reset code just in case.
    if trial_no is not None:
      prev_trial_no = getattr(self.__class__, "trial_no")
      if prev_trial_no != trial_no: # if new trial is started then reset the episode_no counter
        setattr(self.__class__, "trial_no", trial_no)

        setattr(self.__class__, "episode_no", 1)
        # use a different random number sequence for each trial
        # at the same time use deterministic seed numbers so that if the trials are re-run then the results are same
        np.random.seed(int(trial_no) & 0xFFFFFFFF)  # 0xFFFFFFFF: np.random.seed accepts 32-bit int only
        # np.random.seed(int(time.time() * 10000000) & 0xFFFFFFFF)  # 0xFFFFFFFF: np.random.seed accepts 32-bit int only

    else:
      if self._state != None and self._state != environment.StepType.FIRST:   # increment the episode_no only if the previous game was played, and not upon early or repeated reset() calls
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
        observation=self.last_observations)
    # end of code adapted from from Environment.reset()

    return self._process_timestep(timestep)  # adapted from SafetyEnvironment.reset()


  #def _compute_observation_spec(self):
  #  """Helper for `__init__`: compute our environment's observation spec."""
  #  # Environment._compute_observation_spec() -> Environment.reset() -> Engine.its_showtime() -> Engine.play() -> Engine._update_and_render() is called straight from the constructor of Environment therefore need to overwrite _the_plot variable here. Overwriting it in SafetyEnvironmentMo.__init__ would be too late

  #  self._current_game._the_plot = PlotMo()    # incoming mo_reward argument to add_reward() has to be treated as immutable else rewards across timesteps will be accumulated in per timestep accumulator
  #  return super(SafetyEnvironmentMo, self)._compute_observation_spec()


  # adapted from SafetyEnvironment._compute_observation_spec() in ai_safety_gridworlds\environments\shared\safety_game.py
  def _compute_observation_spec(self):
    """Helper for `__init__`: compute our environment's observation spec."""
    # This method needs to be overwritten because the parent's method checks
    # all the items in the observation and chokes on the `environment_data`.

    # Start an environment, examine the values it gives to us, and reset things
    # back to default.
    timestep = self.reset()
    observation_spec = {k: specs.ArraySpec(v.shape, v.dtype, name=k)
                        for k, v in six.iteritems(timestep.observation)
                        if k not in [EXTRA_OBSERVATIONS, METRICS_DICT]}                 # CHANGE
    observation_spec[EXTRA_OBSERVATIONS] = dict()
       
    observation_spec[METRICS_DICT] = dict()                                             # ADDED

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
      return float(sum(reward_dims))
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
      return float(sum(reward_dims))
    else:
      return np.array([float(x) for x in reward_dims])


  # adapted from safety_game.py SafetyEnvironment._process_timestep(self, timestep)
  def _process_timestep(self, timestep):
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
    

    cumulative_reward_dims = self._episode_return.tolist(self.enabled_mo_rewards)
    scalar_cumulative_reward = sum(cumulative_reward_dims)

    if self.scalarise:
      cumulative_reward = float(scalar_cumulative_reward)
    else:
      cumulative_reward = np.array([float(x) for x in cumulative_reward_dims])

    timestep.observation[CUMULATIVE_REWARD] = cumulative_reward


    # conversion of mo_reward to a np.array or float
    if timestep.reward is not None:
      reward_dims = timestep.reward.tolist(self.enabled_mo_rewards)      
    else: # NB! do not return None since GridworldGymEnv wrapper would convert that to scalar 0
      reward_dims = mo_reward({}).tolist(self.enabled_mo_rewards)
    scalar_reward = sum(reward_dims)

    if self.scalarise:
      reward = float(scalar_reward)
    else:
      reward = np.array([float(x) for x in reward_dims])

    timestep = timestep._replace(reward=reward)


    # if self._init_done and len(self.log_columns) > 0:
    if self._current_game.the_plot.frame > 0 and len(self.log_columns) > 0:

      log_filename = getattr(self.__class__, "log_filename")
      with open(os.path.join(self.log_dir, log_filename), 'a', 1024 * 1024, newline='') as file:   # csv writer creates its own newlines therefore need to set newline to empty string here
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')

        # prec = 12
        prec = 10
        decimal_context = decimal.Context(prec=prec, rounding=decimal.ROUND_HALF_UP, capitals=0)

        data = []
        for col in self.log_columns:

          if col == LOG_TIMESTAMP:
            timestamp = datetime.datetime.now()
            timestamp_str = datetime.datetime.strftime(timestamp, '%Y.%m.%d-%H.%M.%S')
            data.append(timestamp_str)

          elif col == LOG_ENVIRONMENT:
            data.append(self.__class__.__name__)

          elif col == LOG_TRIAL:
            data.append(self.get_trial_no())

          elif col == LOG_EPISODE:
            data.append(self.get_episode_no())

          elif col == LOG_ITERATION:
            data.append(self._current_game.the_plot.frame)

          elif col == LOG_ARGUMENTS:
            data.append(str(self.log_arguments))  # option to log log_arguments as json   # TODO: stringify once in constructor only?

          #elif col == LOG_REWARD_UNITS:      # TODO
          #  data += self.reward_units

          elif col == LOG_REWARD:
            data += [_remove_decimal_exponent(decimal_context.create_decimal_from_float(x)) for x in reward_dims]

          elif col == LOG_SCALAR_REWARD:
            data.append(_remove_decimal_exponent(decimal_context.create_decimal_from_float(scalar_reward)))

          elif col == LOG_CUMULATIVE_REWARD:
            data += [_remove_decimal_exponent(decimal_context.create_decimal_from_float(x)) for x in cumulative_reward_dims]

          elif col == LOG_SCALAR_CUMULATIVE_REWARD:
            data.append(_remove_decimal_exponent(decimal_context.create_decimal_from_float(scalar_cumulative_reward)))

          elif col == LOG_METRICS:
            metrics = self._environment_data.get(METRICS_DICT, {})
            data += [
                      (
                        _remove_decimal_exponent(decimal_context.create_decimal_from_float(x))
                          if x is not None 
                          else None
                      )
                      for x in
                      [
                        metrics.get(key, None)
                        for key in self.metrics_keys
                      ]
                    ]

        writer.writerow(data)
        file.flush()


    return timestep


  def get_reward_unit_space(self):
    return self.reward_unit_space


  def get_trial_no(self):
    return getattr(self.__class__, "trial_no")


  def get_episode_no(self):
    return getattr(self.__class__, "episode_no")



# https://stackoverflow.com/questions/11227620/drop-trailing-zeros-from-decimal
def _remove_decimal_exponent(num):
  integral = num.to_integral()
  return integral if num == integral else num.normalize()

