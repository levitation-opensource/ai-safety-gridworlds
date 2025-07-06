# Copyright 2023-2024 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
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

import abc
import sys

from collections import OrderedDict

# Dependency imports
from ai_safety_gridworlds.environments.shared import observation_distiller_ex   # CHANGED
from ai_safety_gridworlds.environments.shared.ma_reward import ma_reward        # ADDED
from ai_safety_gridworlds.environments.shared.rl import array_spec as specs
from ai_safety_gridworlds.environments.shared.rl import pycolab_interface_ma
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason
# from ai_safety_gridworlds.environments.shared.safety_game_moma import AGENT_SPRITE  # cannot import due to circular dependency
from ai_safety_gridworlds.environments.shared.rl.pycolab_interface_ma import INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION  # cannot import due to circular dependency

import enum
import numpy as np

from pycolab import ascii_art
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

import six
from six.moves import map
from six.moves import range


AGENT_SPRITE = 'agent_sprite'   # TODO: use safety_game_moma.AGENT_SPRITE instead


# NB! right now multi-objective and multi-agent environments need to use same action map values since the environment is not known yet when make_human_curses_ui_with_noop_keys is called for human play
from ai_safety_gridworlds.environments.shared.safety_game_mo_base import Directions, Actions

#class Directions(enum.IntEnum):
#  """Enum for observation and action directions of all the players.

#  Warning: Do not rely on these numbers staying as they are, they might change
#  in future iterations of the library. Always refer to all the action using
#  their respective enum names.
#  """
#  # currently the numbers should be in range 0-3 in order for the agent.observation_radius field to work
#  LEFT = 0
#  RIGHT = 1
#  UP = 2
#  DOWN = 3

#class Actions(enum.IntEnum):
#  """Enum for actions all the players can take.

#  Warning: Do not rely on these numbers staying as they are, they might change
#  in future iterations of the library. Always refer to all the action using
#  their respective enum names.
#  """
#  NOOP = 0    # CHANGED
#  LEFT = 1    # CHANGED
#  RIGHT = 2    # CHANGED
#  UP = 3    # CHANGED
#  DOWN = 4    # CHANGED
#  TURN_LEFT_90 = 5    # ADDED
#  TURN_RIGHT_90 = 6    # ADDED
#  TURN_LEFT_180 = 7    # ADDED
#  TURN_RIGHT_180 = 8    # ADDED
#  # Human only. Needs to be the last action in order for the action space sampling to work properly.
#  QUIT = 9    # CHANGED


# Colours common in all environments.
GAME_BG_COLOURS = {' ': (858, 858, 858),  # Environment floor.
                   '#': (599, 599, 599),  # Environment walls.
                   'A': (0, 706, 999),    # Player character.
                   '1': (0, 706, 999),    # Player character.
                   '2': (0, 706, 999),    # Player character.
                   '3': (0, 706, 999),    # Player character.
                   '4': (0, 706, 999),    # Player character.
                   '5': (0, 706, 999),    # Player character.
                   '6': (0, 706, 999),    # Player character.
                   '7': (0, 706, 999),    # Player character.
                   '8': (0, 706, 999),    # Player character.
                   '9': (0, 706, 999),    # Player character.
                   '0': (0, 706, 999),    # Player character.
                   'G': (0, 823, 196)}    # Goal.
GAME_FG_COLOURS = {' ': (858, 858, 858),
                   '#': (599, 599, 599),
                   'A': (0, 0, 0),
                   '1': (0, 0, 0),
                   '2': (0, 0, 0),
                   '3': (0, 0, 0),
                   '4': (0, 0, 0),
                   '5': (0, 0, 0),
                   '6': (0, 0, 0),
                   '7': (0, 0, 0),
                   '8': (0, 0, 0),
                   '9': (0, 0, 0),
                   '0': (0, 0, 0),
                   'G': (0, 0, 0)}

# If not specified otherwise, these are the actions a game will use.
DEFAULT_ACTION_SET = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]

# Some constants to use with the environment_data dictionary to avoid
ENV_DATA = 'environment_data'
ACTUAL_ACTIONS = 'actual_actions'
CURSES = 'curses'
TERMINATION_REASON = 'termination_reason'
HIDDEN_REWARD = 'hidden_reward'
ASCII_ART = 'ascii_art'   # ADDED
NP_RANDOM = 'np_random'   # ADDED
SEED = 'seed'   # ADDED

# Constants for the observations dictionary to the agent.
EXTRA_OBSERVATIONS = 'extra_observations'


class SafetyEnvironmentMa(pycolab_interface_ma.EnvironmentMa):
  """Base class for safety gridworld environments.

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

  def __init__(self,
               game_factory,
               game_bg_colours,
               game_fg_colours,
               actions=None,
               value_mapping=None,
               environment_data=None,
               repainter=None,
               max_iterations=100,
               observe_gaps_only_where_other_layers_are_blank=False,    # ADDED
               observable_attribute_categories=[],      # ADDED   
               observable_attribute_value_mapping:dict[str, dict[str, float]]={},      # ADDED 
               **kwargs     # just to avoid runtime errors when environments are passed flags via kwargs
              ):
    """Initialize a Python v2 environment for a pycolab game factory.

    Args:
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
    if environment_data is None:
      self._environment_data = {}
    else:
      self._environment_data = environment_data
    # Used to store agent performance per episode. Note that agent performance
    # metric might not be equal to the reward obtained.
    self._episodic_performances = []
    # Total environment reward for the current episode.
    self._episode_return = ma_reward({})      # CHANGED
    # Keys to clear from environment_data at start of each episode.
    self._keys_to_clear = [TERMINATION_REASON, ACTUAL_ACTIONS]
    #self._observable_attribute_categories = observable_attribute_categories    # ADDED
    #self._observable_attribute_value_mapping = observable_attribute_value_mapping    # ADDED

    if actions is None:
      actions = {
        'step': (min(DEFAULT_ACTION_SET).value, max(DEFAULT_ACTION_SET).value),
        'action_direction': (min(DEFAULT_ACTION_SET).value, max(DEFAULT_ACTION_SET).value), # TODO: direction set should not be based on action set
        'observation_direction': (min(DEFAULT_ACTION_SET).value, max(DEFAULT_ACTION_SET).value),  # TODO: direction set should not be based on action set
      }

    if value_mapping is None:
      value_mapping = {chr(i): i for i in range(256)}
    self._value_mapping = value_mapping

    array_converter = observation_distiller_ex.ObservationToArrayWithRGBEx(   # CHANGED
        value_mapping=value_mapping,
        colour_mapping=game_bg_colours,
        environment_data=environment_data,    # ADDED
        observable_attribute_categories=observable_attribute_categories,    # ADDED
        observable_attribute_value_mapping=observable_attribute_value_mapping,    # ADDED
        observe_gaps_only_where_other_layers_are_blank=observe_gaps_only_where_other_layers_are_blank,  # ADDED
      )

    super(SafetyEnvironmentMa, self).__init__(
        game_factory=game_factory,
        discrete_actions=actions,
        default_reward=0,
        observation_distiller=pycolab_interface_ma.Distiller(
            repainter=repainter,
            array_converter=array_converter),
        max_iterations=max_iterations,
        **kwargs)    # ADDED

  def set_observable_attribute_categories(self, observable_attribute_categories=[], observable_attribute_value_mapping:dict[str, dict[str, float]]={}):   # ADDED   
    #self.observable_attribute_categories = observable_attribute_categories
    #self.observable_attribute_value_mapping = observable_attribute_value_mapping
    self._observation_distiller._array_converter.set_observable_attribute_categories(observable_attribute_categories, observable_attribute_value_mapping)

  @property
  def environment_data(self):
    return self._environment_data

  @property
  def current_game(self):
    return self._current_game

  @property
  def episode_return(self):
    return self._episode_return

  def _compute_observation_spec(self, observation=None):    # CHANGED
    """Helper for `__init__`: compute our environment's observation spec."""
    # This method needs to be overwritten because the parent's method checks
    # all the items in the observation and chokes on the `environment_data`.

    # Start an environment, examine the values it gives to us, and reset things
    # back to default.

    if observation is None:   # ADDED
      timestep = self.reset() # replace_reward=True
      observation2 = timestep.observation
    else:
      observation2 = observation   # ADDED

    observation_spec = {k: specs.ArraySpec(v.shape, v.dtype, name=k)
                        for k, v in six.iteritems(observation2)   # CHANGED
                        if k != EXTRA_OBSERVATIONS}

    if observation is None:     # ADDED
      observation_spec[EXTRA_OBSERVATIONS] = dict()
      self._drop_last_episode()

    return observation_spec, observation2   # CHANGED

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
    return float(self._calculate_overall_performance())

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
    return float(self._episodic_performances[-1])

  def _calculate_overall_performance(self):
    """Calculates the agent performance across all the episodes.

    By default, the method will return the average across all episodes.
    You should override this method if you want to implement some other way of
    calculating the overall performance.

    Returns:
      A float value summarizing the performance of the agent.
    """
    return sum(self._episodic_performances) / len(self._episodic_performances)

  def _calculate_episode_performance(self, timestep):
    """Calculate performance metric for the agent for the current episode.

    Default performance metric is the average episode reward. You should
    override this method and implement your own if it differs from the default.

    Args:
      timestep: instance of environment_ma.TimeStep
    """
    self._episodic_performances.append(self._episode_return)

  def _get_hidden_reward(self, default_reward=0):
    """Extract the hidden reward from the plot of the current episode."""
    return self.current_game.the_plot.get(HIDDEN_REWARD, default_reward)

  def _clear_hidden_reward(self):
    """Delete hidden reward from the plot."""
    self.current_game.the_plot.pop(HIDDEN_REWARD, None)

  def _process_timestep(self, timestep):
    """Do timestep preprocessing before sending it to the agent.

    This method stores the cumulative return and makes sure that the
    `environment_data` is included in the observation.

    If you are overriding this method, make sure to call `super()` to include
    this code.

    Args:
      timestep: instance of environment_ma.TimeStep

    Returns:
      Preprocessed timestep.
    """
    # Reset the cumulative episode reward.
    # if timestep.first():
    if all( timestep.step_type[agent].first() for agent in self.environment_data[AGENT_SPRITE].keys() ):  # TODO: refactor into separate method    # CHANGED

      self._episode_return = 0
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

    # if timestep.last():
    if all( timestep.step_type[agent].dead()    # ADDED
            or timestep.step_type[agent].last() 
           for agent in self.environment_data[AGENT_SPRITE].keys() ):  # TODO: refactor into separate method    # CHANGED

      # Include the termination reason for the episode if missing.
      if TERMINATION_REASON not in self._environment_data:
        self._environment_data[TERMINATION_REASON] = { agent: TerminationReason.MAX_STEPS for agent in self._environment_data[AGENT_SPRITE].keys() }    # CHANGED

      extra_observations[TERMINATION_REASON] = { agent: (    # CHANGED
          self._environment_data[TERMINATION_REASON]) 
          for agent in self._environment_data[AGENT_SPRITE].keys() }    # ADDED

    timestep.observation[EXTRA_OBSERVATIONS] = extra_observations

    # Calculate performance metric if the episode has finished.
    # if timestep.last():
    if all( timestep.step_type[agent].dead()    # ADDED
            or timestep.step_type[agent].last() 
           for agent in self.environment_data[AGENT_SPRITE].keys() ):  # TODO: refactor into separate method    # CHANGED
      self._calculate_episode_performance(timestep)
    return timestep

  def _get_agent_extra_observations(self):
    """Overwrite this method to give additional information to the agent. The returned dictionary will be available under timestep.observation['extra_observations']"""
    return {}

  def reset(self):
    timestep = super(SafetyEnvironmentMa, self).reset()
    return self._process_timestep(timestep)

  def step(self, agents_actions):

    # if the actions are specified as numerics then interpret them as specifying step modality only
    agents_actions = OrderedDict({ agent: 
                        actions 
                        if isinstance(actions, dict)
                        else { "step": actions }
                       for agent, actions in agents_actions.items() })

    timestep = super(SafetyEnvironmentMa, self).step(agents_actions)
    return self._process_timestep(timestep)


class SafetyBackdrop(plab_things.Backdrop):
  """The backdrop for the game.

  Clear some values in the_plot.
  """

  def update(self, actions, board, layers, things, the_plot):
    super(SafetyBackdrop, self).update(actions, board, layers, things, the_plot)
    PolicyWrapperDrape.plot_clear_actions(the_plot)


class SafetySprite(prefab_sprites.MazeWalker):
  """A generic `Sprite` for objects that move in safety environments.

  Sprites in safety environments that can move, but do not represent the agent,
  should inherit from this class. Sprites that represent the agent should
  inherit from AgentSafetySprite class.

  This `Sprite` has logic tying actions to `MazeWalker` motion action helper
  methods, which keep the sprites from walking on top of obstacles.

  Its main purpose is to wrap the MazeWalker and get hold of the
  environment_data and original_board variables.
  """

  def __init__(self, corner, position, character,
               environment_data, original_board,
               impassable='#'):
    """Initialize SafetySprite.

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
    super(SafetySprite, self).__init__(
        corner, position, character, impassable=impassable,
        confined_to_board=True,   # multi-agent sprites are always confined to the board in order to avoid problems in computing observed object coordinates as well as confusion with agent perspectives  # TODO: fix that limitation    # ADDED
    )
    self._environment_data = environment_data
    self._original_board = original_board

  @abc.abstractmethod
  def update(self, actions, board, layers, backdrop, things, the_plot):
    """See pycolab Sprite class documentation."""
    pass


class AgentSafetySprite(SafetySprite):
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
    super(AgentSafetySprite, self).__init__(
        corner, position, character, environment_data, original_board,
        impassable=impassable)
    self._environment_data = environment_data
    self._original_board = original_board
    self.action_direction_mode = action_direction_mode      # ADDED
    self.action_direction = Directions.UP       # ADDED
    self.observable_attributes = {}       # ADDED


  # TODO: test this logic
  def get_absolute_action(self, agent_action, current_direction):  # ADDED

    agent_action = agent_action["step"]

    if self.action_direction_mode == 0:      # 0 - fixed
      absolute_action = agent_action

    elif self.action_direction_mode == 1 or self.action_direction_mode == 2:    # 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions

      if agent_action == Actions.UP:  # go forwards
        absolute_action = {
                              Directions.UP: Actions.UP,
                              Directions.DOWN: Actions.DOWN,
                              Directions.LEFT: Actions.LEFT,
                              Directions.RIGHT: Actions.RIGHT,
                            }[current_direction]

      elif agent_action == Actions.DOWN:  # go backwards
        absolute_action = {
                              Directions.UP: Actions.DOWN,
                              Directions.DOWN: Actions.UP,
                              Directions.LEFT: Actions.RIGHT,
                              Directions.RIGHT: Actions.LEFT,
                            }[current_direction]

      elif agent_action == Actions.LEFT:  # go left
        absolute_action = {
                              Directions.UP: Actions.LEFT,
                              Directions.DOWN: Actions.RIGHT,
                              Directions.LEFT: Actions.DOWN,
                              Directions.RIGHT: Actions.UP,
                            }[current_direction]

      elif agent_action == Actions.RIGHT: # go right
        absolute_action = {
                              Directions.UP: Actions.RIGHT,
                              Directions.DOWN: Actions.LEFT,
                              Directions.LEFT: Actions.UP,
                              Directions.RIGHT: Actions.DOWN,
                            }[current_direction]

      else:
        absolute_action = agent_action

    else:
      raise ValueError

    return absolute_action


  # TODO: test this logic
  def get_new_action_or_observation_direction(self, agent_action, current_direction):  # ADDED

    if self.action_direction_mode == 0:      # 0 - fixed
      absolute_direction = current_direction

    elif self.action_direction_mode == 1:    # 1 - relative, depending on last move

      if agent_action == Actions.UP:  # go forwards
        absolute_direction = {
                              Directions.UP: Directions.UP,
                              Directions.DOWN: Directions.DOWN,
                              Directions.LEFT: Directions.LEFT,
                              Directions.RIGHT: Directions.RIGHT,
                            }[current_direction]

      elif agent_action == Actions.DOWN:  # go backwards
        absolute_direction = {
                              Directions.UP: Directions.DOWN,
                              Directions.DOWN: Directions.UP,
                              Directions.LEFT: Directions.RIGHT,
                              Directions.RIGHT: Directions.LEFT,
                            }[current_direction]

      elif agent_action == Actions.LEFT:  # go left
        absolute_direction = {
                              Directions.UP: Directions.LEFT,
                              Directions.DOWN: Directions.RIGHT,
                              Directions.LEFT: Directions.DOWN,
                              Directions.RIGHT: Directions.UP,
                            }[current_direction]

      elif agent_action == Actions.RIGHT: # go right
        absolute_direction = {
                              Directions.UP: Directions.RIGHT,
                              Directions.DOWN: Directions.LEFT,
                              Directions.LEFT: Directions.UP,
                              Directions.RIGHT: Directions.DOWN,
                            }[current_direction]

      else:
        absolute_direction = agent_action

    elif self.action_direction_mode == 2:    # 2 - relative, controlled by separate turning actions

      if agent_action == Actions.TURN_LEFT_90:
        absolute_direction = {
                              Directions.UP: Directions.LEFT,
                              Directions.DOWN: Directions.RIGHT,
                              Directions.LEFT: Directions.DOWN,
                              Directions.RIGHT: Directions.UP,
                            }[current_direction]

      elif agent_action == Actions.TURN_RIGHT_90:
        absolute_direction = {
                              Directions.UP: Directions.RIGHT,
                              Directions.DOWN: Directions.LEFT,
                              Directions.LEFT: Directions.UP,
                              Directions.RIGHT: Directions.DOWN,
                            }[current_direction]

      elif agent_action == Actions.TURN_LEFT_180 or agent_action == Actions.TURN_RIGHT_180:
        absolute_direction = {
                              Directions.UP: Directions.DOWN,
                              Directions.DOWN: Directions.UP,
                              Directions.LEFT: Directions.RIGHT,
                              Directions.RIGHT: Directions.LEFT,
                            }[current_direction]

      else:
        absolute_direction = {
                              Actions.UP: Directions.UP,
                              Actions.DOWN: Directions.DOWN,
                              Actions.LEFT: Directions.LEFT,
                              Actions.RIGHT: Directions.RIGHT,
                            }[agent_action]

    else:
      raise ValueError

    return absolute_direction


  # TODO: change to a static method?
  # TODO: test this logic
  def map_action_to_observation_direction(self, proposed_actions, current_direction, action_direction_mode = 2, observation_direction_mode = 2):

    if "observation_direction" in proposed_actions:      # TODO
      proposed_actions = proposed_actions["observation_direction"]
    else:
      proposed_actions = proposed_actions["step"]


    if proposed_actions == Actions.NOOP or proposed_actions == Actions.QUIT:
      direction = current_direction

    elif observation_direction_mode == 0:   # fixed
      direction = current_direction

    elif observation_direction_mode == 1:   # relative, depending on last move
      assert(proposed_actions in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT])
      direction = self.get_new_action_or_observation_direction(proposed_actions, current_direction)

    elif observation_direction_mode == 2:   # relative, controlled by separate turning actions

      if action_direction_mode == 0:        # fixed (fixed mapping between keys/actions and relative direction)
        raise NotImplmentedError()  # TODO

      elif (action_direction_mode == 1      # relative, depending on last move
          or action_direction_mode == 2):   # relative, controlled by separate turning actions
        
        if proposed_actions == Actions.TURN_LEFT_90:
          direction = {
                                Directions.UP: Directions.LEFT,
                                Directions.DOWN: Directions.RIGHT,
                                Directions.LEFT: Directions.DOWN,
                                Directions.RIGHT: Directions.UP,
                              }[current_direction]

        elif proposed_actions == Actions.TURN_RIGHT_90:
          direction = {
                                Directions.UP: Directions.RIGHT,
                                Directions.DOWN: Directions.LEFT,
                                Directions.LEFT: Directions.UP,
                                Directions.RIGHT: Directions.DOWN,
                              }[current_direction]

        elif proposed_actions == Actions.TURN_LEFT_180 or proposed_actions == Actions.TURN_RIGHT_180:
          direction = {
                                Directions.UP: Directions.DOWN,
                                Directions.DOWN: Directions.UP,
                                Directions.LEFT: Directions.RIGHT,
                                Directions.RIGHT: Directions.LEFT,
                              }[current_direction]

        else:
          direction = current_direction

      else:
        raise ValueError("action_direction_mode")

    else: #/ elif observation_direction_mode == 2:
      raise ValueError("observation_direction_mode")

    return direction


  # TODO: change to a static method?
  def map_action_to_action_direction(self, proposed_actions, current_direction, action_direction_mode = 2):

    if "action_direction" in proposed_actions:
      proposed_actions = proposed_actions["action_direction"]
    else:
      proposed_actions = proposed_actions["step"]


    if proposed_actions == Actions.NOOP:
      direction = current_direction

    elif action_direction_mode == 0:   # fixed
      direction = current_direction

    elif action_direction_mode == 1:   # relative, depending on last move
      if proposed_actions == Actions.NOOP:
        direction = current_direction
      else:
        assert(proposed_actions in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT])
        direction = self.get_new_action_or_observation_direction(proposed_actions, current_direction)

    elif action_direction_mode == 2:   # relative, controlled by separate turning actions
        
      if proposed_actions == Actions.TURN_LEFT_90:
        direction = {
                              Directions.UP: Directions.LEFT,
                              Directions.DOWN: Directions.RIGHT,
                              Directions.LEFT: Directions.DOWN,
                              Directions.RIGHT: Directions.UP,
                            }[current_direction]

      elif proposed_actions == Actions.TURN_RIGHT_90:
        direction = {
                              Directions.UP: Directions.RIGHT,
                              Directions.DOWN: Directions.LEFT,
                              Directions.LEFT: Directions.UP,
                              Directions.RIGHT: Directions.DOWN,
                            }[current_direction]

      elif proposed_actions == Actions.TURN_LEFT_180 or proposed_actions == Actions.TURN_RIGHT_180:
        direction = {
                              Directions.UP: Directions.DOWN,
                              Directions.DOWN: Directions.UP,
                              Directions.LEFT: Directions.RIGHT,
                              Directions.RIGHT: Directions.LEFT,
                            }[current_direction]

      else:
        direction = current_direction

    else: #/ elif action_direction_mode == 2:
      raise ValueError("action_direction_mode")

    return direction


  def update(self, agents_actions, board, layers, backdrop, things, the_plot):
    del backdrop  # Unused.

    actions = agents_actions.get(self.character) if agents_actions is not None else None

    if actions is None or actions["step"] is None:
      return

    if actions["step"] == Actions.QUIT:
      self._environment_data[TERMINATION_REASON] = { agent: TerminationReason.QUIT for agent in self._environment_data[AGENT_SPRITE].keys() }     # CHANGED
      the_plot.terminate_episode()
      return

    # Start by collecting the action chosen by the agent.
    # First look for an entry ACTUAL_ACTIONS in the the_plot dictionary.
    # If none, then use the provided actions instead.
    agent_action = PolicyWrapperDrape.plot_get_actions(the_plot, self.character, actions) # MODIFIED

    # Remember the actual action so as to notify the agent so that it can
    # update on the action that was actually taken.
    if ACTUAL_ACTIONS not in self._environment_data:    # ADDED
      self._environment_data[ACTUAL_ACTIONS] = {}    # ADDED
    self._environment_data[ACTUAL_ACTIONS][self.character] = agent_action # MODIFIED

    agent_action_absolute = self.get_absolute_action(agent_action, self.action_direction)   # ADDED

    # Perform the actual action in the environment
    # Comparison between an integer and Actions is allowed because Actions is
    # an IntEnum
    if agent_action_absolute == Actions.UP:       # go upward?
      self._north(board, the_plot)
    elif agent_action_absolute == Actions.DOWN:   # go downward?
      self._south(board, the_plot)
    elif agent_action_absolute == Actions.LEFT:   # go leftward?
      self._west(board, the_plot)
    elif agent_action_absolute == Actions.RIGHT:  # go rightward?
      self._east(board, the_plot)

    self.action_direction = self.map_action_to_action_direction(agent_action, self.action_direction, self.action_direction_mode) 

    self.update_reward(actions, agent_action, layers, things, the_plot)

  def update_reward(self, proposed_actions, actual_actions,
                    layers, things, the_plot):
    """Updates the reward after the actions have been processed.

    Children should most likely define this method.

    Args:
      proposed_actions: actions that were proposed by the agent.
      actual_actions: action that is actually carried out in the environment.
        The two are likely to be the same unless a PolicyWrapperDrape changes
        the proposed actions.
      layers: as elsewhere.
      things: as elsewhere.
      the_plot: as elsewhere.
    """
    pass


class EnvironmentDataSprite(plab_things.Sprite):
  """A generic `Sprite` class for safety environments.

  All stationary Sprites in the safety environments should derive from this
  class.

  Its only purpose is to get hold of the environment_data dictionary and
  original_board variables.
  """

  def __init__(self, corner, position, character,
               environment_data, original_board):
    """Initialize environment data sprite.

    Args:
      corner: same as in pycolab sprite.
      position: same as in pycolab sprite.
      character: same as in pycolab sprite.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
    """
    super(EnvironmentDataSprite, self).__init__(corner, position, character)
    self._original_board = original_board
    self._environment_data = environment_data

  def update(self, actions, board, layers, backdrop, things, the_plot):
    """See parent class documentation."""
    pass


class EnvironmentDataDrape(plab_things.Drape):
  """A generic `Drape` class for safety environments.

  All Drapes in the safety environments should derive from this class.

  Its only purpose is to get hold of the environment_data and
  original_board variables.
  """

  def __init__(self, curtain, character,
               environment_data, original_board):
    """Initialize environment data drape.

    Args:
      curtain: same as in pycolab drape.
      character: same as in pycolab drape.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
    """
    super(EnvironmentDataDrape, self).__init__(curtain, character)
    self._original_board = original_board
    self._environment_data = environment_data

  def update(self, actions, board, layers, backdrop, things, the_plot):
    """See parent class documentation."""
    pass


class PolicyWrapperDrape(six.with_metaclass(abc.ABCMeta, EnvironmentDataDrape)):
  """A `Drape` parent class for policy wrappers.

  Policy wrappers change the entry ACTUAL_ACTIONS in the the_plot
  dictionary.
  Calls the child method `get_actual_action` with the current action
  (which may already have been modified by another sprite)
  and update the current value in the dictionary.
  This value may be used by the agent sprite in place of the agent's action.
  """

  ACTIONS_KEY = ACTUAL_ACTIONS

  def __init__(self, curtain, character,
               environment_data, original_board, agent_character):
    """Initialize policy wrapper drape.

    Args:
      curtain: same as in pycolab drape.
      character: same as in pycolab drape.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
      agent_character: the ascii character for the agent.
    """
    super(PolicyWrapperDrape, self).__init__(
        curtain, character, environment_data, original_board)
    self._agent_character = agent_character

  def update(self, actions, board, layers, backdrop, things, the_plot):
    agent_action = self.plot_get_actions(the_plot, self._agent_character, actions)  # MODIFIED

    if self._agent_character is not None:
      pos = things[self._agent_character].position
      # If the drape applies globally to all tiles instead of a specific tile,
      # redefine this function without the if statement on the following line.
      # (See example in 'whisky_gold.py.)
      if self.curtain[pos]:
        the_plot[self.ACTIONS_KEY][self._agent_character] = self.get_actual_actions(
            agent_action, things, the_plot)

  @abc.abstractmethod
  def get_actual_actions(self, actions, things, the_plot):
    """Takes the actions and returns new actions.

    A child `PolicyWrapperDrape` must implement this method.
    The PolicyWrapperDrapes are chained and can all change these actions.
    The actual actions returned by one drape are the actions input to the next
    one.

    See contrarian.py for a usage example.

    Args:
      actions: either the actions output by the agent (if no drape have modified
        them), or the actions modified by a drape (policy wrapper).
      things: Sprites, Drapes, etc.
      the_plot: the Plot, as elsewhere.
    """
    pass

  @classmethod
  def plot_get_actions(cls, the_plot, agent_character, actions):  # MODIFIED
    return the_plot.get(cls.ACTIONS_KEY, {}).get(agent_character, actions)  # MODIFIED

  @classmethod
  def plot_set_actions(cls, the_plot, agent_character, actions):  # MODIFIED
    the_plot[cls.ACTIONS_KEY][agent_character] = actions  # MODIFIED

  @classmethod
  def plot_clear_actions(cls, the_plot):
    if cls.ACTIONS_KEY in the_plot:
      del the_plot[cls.ACTIONS_KEY]


# Helper function used in various files.
def timestep_termination_reason(timestep, default=None):
  return timestep.observation[EXTRA_OBSERVATIONS].get(
      TERMINATION_REASON, default)


def add_hidden_reward(the_plot, reward, default=0):
  """Adds a hidden reward, analogous to pycolab add_reward.

  Args:
     the_plot: the game Plot object.
     reward: numeric value of the hidden reward.
     default: value with which to initialize the hidden reward variable.
  """
  the_plot[HIDDEN_REWARD] = the_plot.get(HIDDEN_REWARD, default) + reward


def terminate_episode(the_plot, environment_data, 
                      agent,        # ADDED
                      reason=TerminationReason.TERMINATED, discount=0.0):
  """Tells the pycolab game engine to terminate the current agent.

  This terminates agent, not episode. Episode terminates only when all agents are terminated

  Args:
    the_plot: the game Plot object.
    environment_data: dict used to pass around data in a single episode.
    reason: termination reason for the episode.
    discount: discount for the last observation.
  """
  if environment_data.get(TERMINATION_REASON) is None:    # ADDED
    environment_data[TERMINATION_REASON] = {}    # ADDED
  environment_data[TERMINATION_REASON][agent.character] = reason    # CHANGED

  if all( environment_data[TERMINATION_REASON].get(agent_char) is not None     # ADDED
         for agent_char in environment_data[AGENT_SPRITE].keys() ):            # ADDED
    the_plot.terminate_episode(discount=discount)


def get_player_positions(environment_data):

  agent_sprites = environment_data[AGENT_SPRITE]

  result = {}
  for agent in agent_sprites.values():
    result[agent_key] = agent.position

  return result


def get_players(environment_data):     # ADDED
  return environment_data[AGENT_SPRITE].values()

def is_last_step_of_round(environment_data):     # ADDED
  """Round is a sequence of steps where all alive agents step once. NOOP action counts as a step. None action does not count as a step."""

  maximum_step_count = -1
  minimum_step_count = sys.maxsize

  for key, agent in environment_data[AGENT_SPRITE].items():

    if agent.is_terminated():   # ignore terminated agents
      continue

    minimum_step_count = min(minimum_step_count, agent.step_count)
    maximum_step_count = max(maximum_step_count, agent.step_count)

    if minimum_step_count != maximum_step_count:  # some alive agent has not stepped yet during this round
      return False

  #/ for key, agent in environment_data[AGENT_SPRITE].items():

  return maximum_step_count > 0   # when no agents have stepped then consider the round as incomplete



# map_randomizations_per_environment = {}
randomized_maps_per_environment = {}

def make_safety_game(
    environment_data,
    the_ascii_art,
    what_lies_beneath,
    what_lies_outside,
    backdrop=SafetyBackdrop,
    sprites=None,
    drapes=None,
    update_schedule=None,
    z_order=None,
    map_randomization_frequency=False,                        # ADDED   # TODO: configuration to specify whether to randomize the map once per experiment, once per trial, or once per episode
    preserve_map_edges_when_randomizing=True,   # ADDED
    environment=None,                           # ADDED
    tile_type_counts=None,                      # ADDED
    remove_unused_tile_types_from_layers=False, # ADDED
    map_width=None,                             # ADDED
    map_height=None,                            # ADDED
  ):
  """Create a pycolab game instance."""


  # Keep a still copy of the initial board as a numpy array
  original_board = np.array(list(map(list, the_ascii_art[:])))


  # START OF ADDED

  np_random = environment_data[NP_RANDOM]
  seed = environment_data[SEED]


  if not tile_type_counts or not (map_randomization_frequency >= 1):
    enable_randomize = False

  elif environment is None:
    enable_randomize = True

  else:
    environment_class = environment.__class__.__module__ + "." + environment.__class__.__qualname__
    # last_randomization_done_for_environment = map_randomizations_per_environment.get(environment_class)
    env_layout_seed = environment.get_env_layout_seed()
    episode_no = environment.get_episode_no()

    # need to consider tile_type_counts in randomization_key since tests create different environment configurations in one go
    tile_type_counts_key = list(tile_type_counts.items())
    tile_type_counts_key.sort()

    ascii_art_key = "\n".join(the_ascii_art)

    # NB! if np_random argument is provided then that is not considered here for randomization_key calculation
    if map_randomization_frequency == 1:    # 1 - once per experiment run
      randomization_key = environment_class + "|" + str(seed) + "|" + str(tile_type_counts_key) + "|" + ascii_art_key + "|" + str(map_width) + "|" + str(map_height)
    elif map_randomization_frequency == 2:  # 2 - once per env seed update (there is a sequence of training episodes separated by env.reset call, but using a same model instance)
      randomization_key = environment_class + "|" + str(seed) + "|" + str(env_layout_seed) + "|" + str(tile_type_counts_key) + "|" + ascii_art_key + "|" + str(map_width) + "|" + str(map_height)
    elif map_randomization_frequency == 3:  # 3 - once per training episode
      randomization_key = environment_class + "|" + str(seed) + "|" + str(env_layout_seed) + "|" + str(episode_no) + "|" + str(tile_type_counts_key) + "|" + ascii_art_key + "|" + str(map_width) + "|" + str(map_height)
    else:
      raise ValueError("map_randomization_frequency")

    # enable_randomize = (last_randomization_done_for_environment != randomization_key)
    enable_randomize = randomization_key not in randomized_maps_per_environment

    if not enable_randomize:  # obtain earlier randomized map
      (the_ascii_art, original_board) = randomized_maps_per_environment[randomization_key]
    else:
      qqq = True    # for debugging


  resize_board = False
  if (
      (map_height is not None or map_width is not None) # at least one of the parameters is set
      and (map_height != original_board.shape[0] or map_width != original_board.shape[1]) # at least one of the parameters is different from original board
    ):

    assert tile_type_counts                   # NB! allow map resizing only when tile type counts is provided   # TODO: map resizing could work without tile type counts being explicitly provided as well, but then the tile type counts needs to be calculated by the code here
    assert map_randomization_frequency > 0    # NB! allow map resizing only when randomization is enabled

    if enable_randomize:  # if map was already generated during previous iteration and is reused now, then do not resize the board again
      resize_board = True
    
      if map_height is None:
        map_height = original_board.shape[0]

      if map_width is None:
        map_width = original_board.shape[1]

  #/ detect map resize need


  if resize_board:

    if preserve_map_edges_when_randomizing:
      assert map_height > 2 and map_width > 2
      shape = (map_height - 2, map_width - 2)
    else:
      shape = (map_height, map_width)


    submap = np.zeros(shape[0] * shape[1], original_board.dtype)  # need to use vector form since np.random.shuffle shuffles only one dimension at a time
      
    # linearly fill tile types into the vector form submap, reshuffle later
    next_i = 0
    for tile_type, tile_max_count in tile_type_counts.items():
      assert next_i + tile_max_count <= submap.shape[0]
      submap[next_i : next_i + tile_max_count] = tile_type
      next_i += tile_max_count

    submap[next_i : ] = what_lies_beneath

    np_random.shuffle(submap)
    submap = submap.reshape(shape)


    if preserve_map_edges_when_randomizing:
      original_board_initial = original_board
      original_board = np.zeros((map_height, map_width), original_board.dtype)

      # TODO: resample map edges from original_board_initial
      original_board.fill(what_lies_outside)

      original_board[1:-1, 1:-1] = submap

    else:
      original_board = submap

    the_ascii_art = list(map(''.join, original_board.tolist()))

  else: #/ if resize_board:

    # TODO: if tile type counts are bigger than on existing map, then add new drapes in random locations. And if map randomisation is set to 0 then trigger an assertion
    if tile_type_counts and enable_randomize:

      for tile_type, tile_max_count in tile_type_counts.items():

        if tile_max_count == 0 or not remove_unused_tile_types_from_layers:
          assert tile_type in drapes or tile_type in sprites, "Tile type not defined in drapes nor sprites"
          assert tile_type in update_schedule, "Tile type not defined in update_schedule"
          assert tile_type in z_order, "Tile type not defined in z_order"

        # tile_type_locations = (original_board == tile_type).nonzero()
        # num_locations = len(tile_type_locations[0])
        tile_type_locations = np.argwhere(original_board == tile_type)
        num_locations = len(tile_type_locations)
        num_items_to_remove = max(0, num_locations - tile_max_count)


        if num_items_to_remove > 0:

          assert map_randomization_frequency >= 1   # if tile type counts are specified and are different from the counts on original map then randomization should happen at least once

          # replace - Whether the sample is with or without replacement. Default is True, meaning that a value of a can be selected multiple times.
          indexes_to_remove = np_random.choice(num_locations, size=num_items_to_remove, replace=False)
          # locations_to_remove = (tile_type_locations[0][indexes_to_remove], tile_type_locations[1][indexes_to_remove])
          locations_to_remove = tile_type_locations[indexes_to_remove]
          row_coordinates = locations_to_remove[:, 0]
          col_coordinates = locations_to_remove[:, 1]
          original_board[row_coordinates, col_coordinates] = what_lies_beneath  # numpy requires unzipped coordinates for accessing multiple cells by individual coordinates

        #/ if num_items_to_remove > 0:


        if tile_max_count == 0 and remove_unused_tile_types_from_layers:

          sprites.pop(tile_type, None)
          drapes.pop(tile_type, None)
          if tile_type in update_schedule:
            del update_schedule[update_schedule.index(tile_type)]          
          if tile_type in z_order:
            del z_order[z_order.index(tile_type)]

        #/ if tile_max_count == 0 and remove_unused_tile_types_from_layers:

      #/ for tile_type, tile_max_count in tile_type_counts.items():

      the_ascii_art = list(map(''.join, original_board.tolist()))

    #/ if tile_type_counts and enable_randomize:


    if map_randomization_frequency >= 1 and enable_randomize:

      if preserve_map_edges_when_randomizing:
        submap = original_board[1:-1, 1:-1]
      else:
        submap = original_board

      shape = submap.shape
      submap = submap.reshape(shape[0] * shape[1])  # need to convert to vector form since np.random.shuffle shuffles only one dimension at a time
      np_random.shuffle(submap)
      submap = submap.reshape(shape)

      if preserve_map_edges_when_randomizing:
        original_board[1:-1, 1:-1] = submap
      else:
        original_board = submap

      the_ascii_art = list(map(''.join, original_board.tolist()))

    #/ if map_randomization_frequency >= 1 and enable_randomize:

  #/ if resize_board:
  
  
  if enable_randomize and environment is not None:  # save randomized map
    # map_randomizations_per_environment[environment_class] = randomization_key
    randomized_maps_per_environment[randomization_key] = (the_ascii_art, original_board)
    

  if remove_unused_tile_types_from_layers:

    used_tile_types = set(np.unique(original_board))
    sprites = { key: value for key, value in sprites.items() if key in used_tile_types }
    drapes = { key: value for key, value in drapes.items() if key in used_tile_types }
    update_schedule = [ key for key in update_schedule if key in used_tile_types ]
    z_order = [ key for key in z_order if key in used_tile_types ]


  environment_data[ASCII_ART] = the_ascii_art     # for debugging

  # END OF ADDED


  return ascii_art.ascii_art_to_game(
      the_ascii_art,
      what_lies_beneath,
      sprites=None 
        if sprites is None
        else {k: ascii_art.Partial(args[0],
                                environment_data,
                                original_board,
                                *args[1:])
          for k, args in sprites.items()},
      drapes=None 
        if drapes is None
        else {k: ascii_art.Partial(args[0],
                                environment_data,
                                original_board,
                                *args[1:])
          for k, args in drapes.items()},
      backdrop=backdrop,
      update_schedule=update_schedule,
      z_order=z_order,
      occlusion_in_layers=False)    # similar behaviour can be controlled by occlusion_in_layers in Zoo wrapper    # ADDED
