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
"""Frontends for humans who want to play pycolab games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import datetime

from ai_safety_gridworlds.environments.shared.mo_reward import mo_reward
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_game_mo
from ai_safety_gridworlds.environments.shared import safety_ui

import numpy as np

from pycolab.protocols import logging as plab_logging

import six


# adapted from ai_safety_gridworlds\environments\shared\safety_ui.py
class SafetyCursesUiEx(safety_ui.SafetyCursesUi):
  """A terminal-based UI for pycolab games.

  This is deriving from pycolab's `human_ui.CursesUi` class and shares a
  lot of its code. The main purpose of having a separate class is that we want
  to use the `play()` method on an instance of `SafetyEnvironment` and not just
  a pycolab game `Engine`. This way we can store information across
  episodes, conveniently call `get_overall_performance()` after the human has
  finished playing. It is also ensuring that human and agent interact with the
  environment in the same way (e.g. if `SafetyEnvironment` gets derived).
  """

  # adapted from SafetyCursesUi._init_curses_and_play(self, screen) in ai_safety_gridworlds\environments\shared\safety_ui.py
  def _init_curses_and_play(self, screen):
    """Set up an already-running curses; do interaction loop.

    This method is intended to be passed as an argument to `curses.wrapper`,
    so its only argument is the main, full-screen curses window.

    Args:
      screen: the main, full-screen curses window.

    Raises:
      ValueError: if any key in the `keys_to_actions` dict supplied to the
          constructor has already been reserved for use by `CursesUi`.
    """
    # This needs to be overwritten to use `self._env.step()` instead of
    # `self._game.play()`.

    # See whether the user is using any reserved keys. This check ought to be in
    # the constructor, but it can't run until curses is actually initialised, so
    # it's here instead.
    for key, action in six.iteritems(self._keycodes_to_actions):
      if key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
        raise ValueError(
            'the keys_to_actions argument to the CursesUi constructor binds '
            'action {} to the {} key, which is reserved for CursesUi. Please '
            'choose a different key for this action.'.format(
                repr(action), repr(curses.keyname(key))))

    # If the terminal supports colour, program the colours into curses as
    # "colour pairs". Update our dict mapping characters to colour pairs.
    self._init_colour()
    curses.curs_set(0)  # We don't need to see the cursor.
    if self._delay is None:
      screen.timeout(-1)  # Blocking reads
    else:
      screen.timeout(self._delay)  # Nonblocking (if 0) or timing-out reads

    # Create the curses window for the log display
    rows, cols = screen.getmaxyx()
    console = curses.newwin(rows // 2, cols, rows - (rows // 2), 0)

    # By default, the log display window is hidden
    paint_console = False

    # Kick off the game---get first observation, repaint it if desired,
    # initialise our total return, and display the first frame.
    self._env.reset()
    self._game = self._env.current_game
    # Use undistilled observations.
    observation = self._game._board  # pylint: disable=protected-access
    if self._repainter: observation = self._repainter(observation)
    self._display(screen, [observation], self._env.episode_return,
                  elapsed=datetime.timedelta())

    # Oh boy, play the game!
    while not self._env._game_over:  # pylint: disable=protected-access
      # Wait (or not, depending) for user input, and convert it to an action.
      # Unrecognised keycodes cause the game display to repaint (updating the
      # elapsed time clock and potentially showing/hiding/updating the log
      # message display) but don't trigger a call to the game engine's play()
      # method. Note that the timeout "keycode" -1 is treated the same as any
      # other keycode here.
      keycode = screen.getch()

      update_time_counter_only = False
      if keycode == curses.KEY_PPAGE:    # Page Up? Show the game console.
        paint_console = True
      elif keycode == curses.KEY_NPAGE:  # Page Down? Hide the game console.
        paint_console = False
      elif keycode in self._keycodes_to_actions:
        # Convert the keycode to a game action and send that to the engine.
        # Receive a new observation, reward, pcontinue; update total return.
        action = self._keycodes_to_actions[keycode]
        self._env.step(action)
        # Use undistilled observations.
        observation = self._game._board  # pylint: disable=protected-access
        if self._repainter: observation = self._repainter(observation)
      else:
        update_time_counter_only = True   # optimisation and flicker reduction: if keycode is -1 and delay does not trigger no-op (-1 not in self._keycodes_to_actions) then just update the time counter and not the whole screen

      # Update the game display, regardless of whether we've called the game's
      # play() method.
      elapsed = datetime.datetime.now() - self._start_time
      self._display(screen, [observation], self._env.episode_return, elapsed, update_time_counter_only=update_time_counter_only)

      # Update game console message buffer with new messages from the game.
      self._update_game_console(
          plab_logging.consume(self._game.the_plot), console, paint_console)

      # Show the screen to the user.
      curses.doupdate()


  # adapted from CursesUi._display(self, screen, observations, score, elapsed) in pycolab\human_ui.py
  def _display(self, screen, observations, score, elapsed, update_time_counter_only=False):

    if update_time_counter_only:  # optimisation and flicker reduction: if keycode is -1 and delay does not trigger no-op (-1 not in self._keycodes_to_actions) then just update the time counter and not the whole screen
      # Display the game clock and the current score.
      screen.addstr(0, 2, safety_ui._format_timedelta(elapsed), curses.color_pair(0))
      return


    super(SafetyCursesUiEx, self)._display(screen, observations, score, elapsed)

    start_row = 2
    start_col = 20
    padding = 2


    # compute max width of first column so that all content in second column can be aligned
    max_first_col_width = 0

    metrics = self._env._environment_data.get("metrics_matrix")
    if metrics is not None:
      metrics_cell_widths = [padding + max(len(str(cell)) for cell in col) for col in metrics.T]
      max_first_col_width = max(max_first_col_width, metrics_cell_widths[0])

    if (isinstance(self._env.episode_return, mo_reward) 
        and len(self._env.enabled_mo_rewards) > 0):  # avoid errors in case the reward dimensions are not defined
      reward_key_col_width = padding + max(len(str(key)) for key in self._env.enabled_reward_dimension_keys) # key may be None therefore need str(key)
      max_first_col_width = max(max_first_col_width, reward_key_col_width)
    else:
      max_first_col_width = max(max_first_col_width, padding + len("Episode return:"))


    if isinstance(self._env, safety_game_mo.SafetyEnvironmentMo):

      max_first_col_width = max(max_first_col_width, padding + len("Episode no:"))

      screen.addstr(start_row,     start_col, "Trial no:  ", curses.color_pair(0)) 
      screen.addstr(start_row + 1, start_col, "Episode no:", curses.color_pair(0)) 
      screen.addstr(start_row,     start_col + max_first_col_width, str(self._env.get_trial_no()), curses.color_pair(0)) 
      screen.addstr(start_row + 1, start_col + max_first_col_width, str(self._env.get_episode_no()), curses.color_pair(0)) 
      start_row += 3


    # print metrics
    if metrics is not None:

      screen.addstr(start_row, start_col, "Metrics:", curses.color_pair(0)) 
      for row_index, row in enumerate(metrics):
        for col_index, cell in enumerate(row):
          if col_index == 0:
            col_offset = 0
          elif col_index == 1:
            col_offset = max_first_col_width
          else:
            col_offset = metrics_cell_widths[col_index - 1]

          screen.addstr(start_row + 1 + row_index, start_col + col_offset, str(cell), curses.color_pair(0)) 

      start_row += len(metrics) + 2


    # print reward dimensions too
    if (isinstance(self._env.episode_return, mo_reward) 
        and len(self._env.enabled_mo_rewards) > 0):  # avoid errors in case the reward dimensions are not defined

      last_reward = self._env._last_reward.tofull(self._env.enabled_mo_rewards)
      episode_return = self._env.episode_return.tofull(self._env.enabled_mo_rewards)

      screen.addstr(start_row, start_col, "Last reward:", curses.color_pair(0)) 
      for row_index, (key, value) in enumerate(last_reward.items()):
        screen.addstr(start_row + 1 + row_index, start_col, key, curses.color_pair(0)) 
        screen.addstr(start_row + 1 + row_index, start_col + max_first_col_width, str(value), curses.color_pair(0)) 
      start_row += len(last_reward) + 2

      screen.addstr(start_row, start_col, "Episode return:", curses.color_pair(0)) 
      for row_index, (key, value) in enumerate(episode_return.items()):
        screen.addstr(start_row + 1 + row_index, start_col, key, curses.color_pair(0)) 
        screen.addstr(start_row + 1 + row_index, start_col + max_first_col_width, str(value), curses.color_pair(0)) 
      start_row += len(episode_return) + 2

    else:

      screen.addstr(start_row,     start_col, "Last reward:   ", curses.color_pair(0)) 
      screen.addstr(start_row + 1, start_col, "Episode return:", curses.color_pair(0)) 
      screen.addstr(start_row,     start_col + max_first_col_width, str(self._env._last_reward), curses.color_pair(0)) 
      screen.addstr(start_row + 1, start_col + max_first_col_width, str(self._env.episode_return), curses.color_pair(0)) 


# adapted from ai_safety_gridworlds\environments\shared\safety_ui.py
def make_human_curses_ui_with_noop_keys(game_bg_colours, game_fg_colours, noop_keys, delay=100):
  """Instantiate a Python Curses UI for the terminal game.

  Args:
    game_bg_colours: dict of game element background colours.
    game_fg_colours: dict of game element foreground colours.
    noop_keys: enables NOOP actions on keyboard using space bar and middle button on keypad.
    delay: in ms, how long does curses wait before emitting a noop action if
      such an action exists. If it doesn't it just waits, so this delay has no
      effect. Our situation is the latter case, as we don't have a noop.

  Returns:
    A curses UI game object.
  """

  keys_to_actions={curses.KEY_UP:       safety_game.Actions.UP,
                    curses.KEY_DOWN:    safety_game.Actions.DOWN,
                    curses.KEY_LEFT:    safety_game.Actions.LEFT,
                    curses.KEY_RIGHT:   safety_game.Actions.RIGHT,
                    'q':                safety_game.Actions.QUIT,
                    'Q':                safety_game.Actions.QUIT}
  if noop_keys:
     keys_to_actions.update({
        # middle button on keypad
        curses.KEY_B2: safety_game.Actions.NOOP,  # KEY_B2: Center of keypad - https://docs.python.org/3/library/curses.html
        # space bar
        ' ': safety_game.Actions.NOOP,
        # -1: Actions.NOOP,           # curses delay timeout "keycode" is -1
      })

  return SafetyCursesUiEx(  
      keys_to_actions=keys_to_actions,
      delay=delay,
      repainter=None,   # TODO
      colour_fg=game_fg_colours,
      colour_bg=game_bg_colours)


def map_contains(tile_char, map):
  """Returns True if some tile in the map contains given character"""

  assert len(tile_char) == 1
  return any(tile_char in row for row in map)


def save_metric(self, metrics_matrix_row_indexes, key, value):
  """Saves a metric both to metrics_matrix and metrics_dict"""

  # TODO: support for saving vectors into columns of metrix matrix
  if key in metrics_matrix_row_indexes:   # NB! if the metric is not activated then silently ignore calls to save it
    metric_row_index = metrics_matrix_row_indexes[key]
    self.environment_data[safety_game_mo.METRICS_MATRIX][metric_row_index, 1] = value
    self.environment_data[safety_game_mo.METRICS_DICT][key] = value

