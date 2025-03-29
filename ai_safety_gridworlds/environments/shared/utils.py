# Copyright 2022 - 2025 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
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

import sys
import traceback
from absl import flags
from ast import literal_eval

from ai_safety_gridworlds.environments.shared import safety_game_ma
from ai_safety_gridworlds.environments.shared import safety_game_moma
from ai_safety_gridworlds.environments.shared.safety_game_moma import METRICS_ROW_INDEXES
from ai_safety_gridworlds.environments.shared.safety_game_moma import LOG_TIMESTAMP, LOG_ENVIRONMENT, LOG_TRIAL, LOG_EPISODE, LOG_ITERATION, LOG_ARGUMENTS, LOG_REWARD_UNITS, LOG_REWARD, LOG_SCALAR_REWARD, LOG_CUMULATIVE_REWARD, LOG_AVERAGE_REWARD, LOG_SCALAR_CUMULATIVE_REWARD, LOG_SCALAR_AVERAGE_REWARD, LOG_GINI_INDEX, LOG_CUMULATIVE_GINI_INDEX, LOG_MO_VARIANCE, LOG_CUMULATIVE_MO_VARIANCE, LOG_AVERAGE_MO_VARIANCE, LOG_METRICS, LOG_QVALUES_PER_TILETYPE

from ai_safety_gridworlds.environments.shared.mo_reward import mo_reward

from ai_safety_gridworlds.environments.shared import safety_ui_ex
from ai_safety_gridworlds.environments.shared.safety_ui_ex import save_metric


def define_standard_flags(global_vars):

  # cannot use a module-global variable here since during testing, the environment may be created once, then another environment is created, which erases the flags, and then again current environment is creater later again
  if hasattr(flags.FLAGS, __name__ + "_flags_defined"):     # this function will be called multiple times via the experiments in the factory
    return flags.FLAGS
  flags.DEFINE_bool(__name__ + "_flags_defined", True, "")
  
  # reset flags state in case tests are being run, else exception occurs below while defining the flags
  # https://github.com/abseil/abseil-py/issues/36
  for name in list(flags.FLAGS):
    delattr(flags.FLAGS, name)
  flags.DEFINE_bool('eval', False, 'Print results to stderr for piping to file, otherwise print safety performance to user.') # recover flag defined in safety_ui.py


  # TODO: refactor standard flags to a shared method

  flags.DEFINE_integer('level',
                        global_vars["DEFAULT_LEVEL"],
                        'Which coloured resource sharing level to play.')

  flags.DEFINE_integer('max_iterations', global_vars["DEFAULT_MAX_ITERATIONS"], 'Max iterations.')

  flags.DEFINE_boolean('noops', global_vars["DEFAULT_NOOPS"], 
                        'Whether to include NOOP as a possible agent action.')
  flags.DEFINE_boolean('randomize_agent_actions_order', global_vars["DEFAULT_RANDOMIZE_AGENT_ACTIONS_ORDER"], 
                        'Whether to randomize the order the agent actions are carried out in order to resolve any tile collisions and resource availability collisions randomly.')

  flags.DEFINE_integer('map_randomization_frequency', global_vars["DEFAULT_MAP_RANDOMIZATION_FREQUENCY"],
                        'Whether and when to randomize the map. 0 - off, 1 - once per experiment run, 2 - once per trial (a trial is a sequence of training episodes separated by env.reset call, but using a same model instance), 3 - once per training episode.')
  
  flags.DEFINE_string('observation_radius', str(global_vars["DEFAULT_OBSERVATION_RADIUS"]), 
                       'How many tiles away from the agent can the agent see? -1 means the agent perspective is same as global perspective and the observation does not move when the agent moves. 0 means the agent can see only the tile underneath itself. None means the agent can see the whole board while still having agent-centric perspective; the observation size is 2*board_size-1.')
  flags.DEFINE_integer('observation_direction_mode', global_vars["DEFAULT_OBSERVATION_DIRECTION_MODE"], 
                       'Observation direction mode (0-2): 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions.')
  flags.DEFINE_integer('action_direction_mode', global_vars["DEFAULT_ACTION_DIRECTION_MODE"], 
                       'Action direction mode (0-2): 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions.')

  flags.DEFINE_boolean('remove_unused_tile_types_from_layers', global_vars["DEFAULT_REMOVE_UNUSED_TILE_TYPES_FROM_LAYERS"],
                       'Whether to remove tile types not present on initial map from observation layers.')

  flags.DEFINE_boolean('enable_logging', global_vars["DEFAULT_ENABLE_LOGGING"], 'Enable logging.')

  # if default map width and height are None, then original ascii art dimensions are used
  flags.DEFINE_integer('map_width', global_vars["DEFAULT_MAP_WIDTH"], 'Map width')
  flags.DEFINE_integer('map_height', global_vars["DEFAULT_MAP_HEIGHT"], 'Map height')

  flags.DEFINE_integer('amount_agents', global_vars["DEFAULT_AMOUNT_AGENTS"], 'Amount of agents.')

#/ def define_standard_flags(flags):


def parse_flags():

  FLAGS = flags.FLAGS

  # need to explicitly tell the flags library to parse argv before you can access FLAGS.xxx
  if __name__ == '__main__':
    FLAGS(sys.argv)
  else:
    FLAGS([""])


  # convert observation radius flag from string format to list/numeric format
  FLAGS.observation_radius = literal_eval(FLAGS.observation_radius) if FLAGS.observation_radius else None


  return FLAGS

#/ def parse_flags(flags):


def publish_metrics(drape_or_sprite, metrics_dict):

  metrics_row_indexes = drape_or_sprite.environment_data[METRICS_ROW_INDEXES]  

  for key, value in metrics_dict.items():
    save_metric(drape_or_sprite, metrics_row_indexes, key, value)

#/ def save_metrics(metrics_dict):


class DummyAgentDrape(safety_game_ma.EnvironmentDataDrape):
  """A `Drape` corresponding to agents not present on map.

  This class helps to keep the missing agent layers active in the observation.
  """
  pass


def run_human_playable_demo(env_class, global_vars):

  try:
    FLAGS = global_vars["define_flags"]()

    log_columns = [
      # LOG_TIMESTAMP,
      # LOG_ENVIRONMENT,
      LOG_TRIAL,       
      LOG_EPISODE,        
      LOG_ITERATION,
      # LOG_ARGUMENTS,     
      # LOG_REWARD_UNITS,     # TODO: use .get_reward_unit_space() method
      LOG_REWARD,
      LOG_SCALAR_REWARD,
      LOG_CUMULATIVE_REWARD,
      LOG_AVERAGE_REWARD,
      LOG_SCALAR_CUMULATIVE_REWARD, 
      LOG_SCALAR_AVERAGE_REWARD, 
      LOG_GINI_INDEX, 
      LOG_CUMULATIVE_GINI_INDEX,
      LOG_MO_VARIANCE, 
      LOG_CUMULATIVE_MO_VARIANCE,
      LOG_AVERAGE_MO_VARIANCE,
      LOG_METRICS,
      LOG_QVALUES_PER_TILETYPE,
    ]

    if not FLAGS.enable_logging:
      log_columns = None

    env = env_class(
      scalarise=False,
      log_columns=log_columns,
      log_arguments_to_separate_file=True,
      log_filename_comment="some_configuration_or_comment=1234",
      FLAGS=FLAGS,
      level=FLAGS.level, 
      max_iterations=FLAGS.max_iterations, 
      noops=FLAGS.noops,
    )


    enable_turning_keys = FLAGS.observation_direction_mode == 2 or FLAGS.action_direction_mode == 2


    while True:
      for trial_no in range(0, 2):
        # env.reset(options={"trial_no": trial_no + 1})  # NB! provide only trial_no. episode_no is updated automatically
        for episode_no in range(0, 2): 
          env.reset()   # it would also be ok to reset() at the end of the loop, it will not mess up the episode counter
          ui = safety_ui_ex.make_human_curses_ui_with_noop_keys(
            global_vars["GAME_BG_COLOURS"],
            global_vars["GAME_FG_COLOURS"],
            noop_keys=FLAGS.noops, 
            turning_keys=enable_turning_keys,
            custom_actions=global_vars["CUSTOM_ACTIONS"], 
          )
          ui.play(env)
        # TODO: randomize the map once per trial, not once per episode
        env.reset(options={"trial_no": env.get_trial_no()  + 1})  # NB! provide only trial_no. episode_no is updated automatically
        
  except Exception as ex:
    print(ex)
    print(traceback.format_exc())

#/ def run_human_playable_demo(env_class, global_vars):


def define_flags_and_update_from_kwargs(FLAGS, kwargs, define_flags):

  if FLAGS is None:
    FLAGS = define_flags()

  # TODO! refactor this code to a shared helper method
  arguments = kwargs    # override flags only when the keyword arguments are explicitly provided. Do not override flags with default keyword argument values
  for key, value in arguments.items():
    if key in ["FLAGS", "__class__", "kwargs", "self"]:
      continue
    if key in FLAGS:
      if isinstance(FLAGS[key].value, mo_reward):
        FLAGS[key].value = mo_reward.parse(value)
      else:
        FLAGS[key].value = value
    elif key.upper() in FLAGS:    # detect cases when flag has uppercase name
      if isinstance(FLAGS[key.upper()].value, mo_reward):
        FLAGS[key.upper()].value = mo_reward.parse(value)
      else:
        FLAGS[key.upper()].value = value

  return FLAGS

#/ def define_flags_and_update_from_kwargs():


def init_value_mapping(FLAGS, global_vars):
  """
  Returns a dictionary translating each cell type on the map to a float value.

  For example:
  {
    '#': 0.0
    ' ': 1.0
    'B': 2.0
    'G': 3.0
    'R': 4.0
    '0': 5.0
    '1': 6.0
  }
  """

  maps = global_vars["GAME_ART"]

  WALL_CHR = global_vars["WALL_CHR"]
  GAP_CHR = global_vars["GAP_CHR"]
  AGENT_CHRS = global_vars["AGENT_CHRS"]

  special_chars = [WALL_CHR, GAP_CHR] + AGENT_CHRS

  all_chars = set()
  for level_map in maps:
    for line in level_map:
      for char in line:
        if char not in special_chars:
          all_chars.add(char)

  # lets make the mapping deterministic
  all_chars = list(all_chars)
  all_chars.sort()

  # lets put wall and gap chr always at first positions
  all_chars = [WALL_CHR, GAP_CHR] + all_chars

  value_mapping = { char: float(index) for index, char in enumerate(all_chars) }

  # value_mapping = {}
  # for var_name, value in global_vars.items():
  #   if var_name.endswith("_CHR") and value in all_chars:
  #     value_mapping[var_name] = float(len(value_mapping))

  # value_mapping = { # TODO! create shared helper method for automatically building this value mapping from a list of characters
  #   WALL_CHR: 0.0,
  #   GAP_CHR: 1.0,
  #   RED_RESOURCE_CHR: 2.0, 
  #   BLUE_RESOURCE_CHR: 3.0, 
  #   GREEN_RESOURCE_CHR: 4.0,
  # }

  # lets put agent chars always at last positions
  value_mapping.update({
    AGENT_CHRS[agent_index]: float(len(value_mapping) + agent_index) 
    for agent_index 
    in range(0, FLAGS.amount_agents)
  })

  return value_mapping

#/ def init_value_mapping(FLAGS, GAME_ART, AGENT_CHRS):


def make_game_moma(global_vars, environment, environment_data, agent_sprite_class, drape_classes_dict,
                   preserve_map_edges_when_randomizing=True):

  FLAGS = flags.FLAGS
  # environment_data = local_vars["environment_data"]
  GAME_ART = global_vars["GAME_ART"]
  AGENT_CHRS = global_vars["AGENT_CHRS"]
  GAP_CHR = global_vars["GAP_CHR"]


  level = FLAGS.level
  amount_agents = FLAGS.amount_agents


  environment_data[METRICS_ROW_INDEXES] = dict()


  map = GAME_ART[level]


  sprites = {
              AGENT_CHRS[agent_index]: [agent_sprite_class, FLAGS] 
              for agent_index in range(0, amount_agents)
            }

  drapes = {
              char: [drape_class_data[0], FLAGS]
              for char, drape_class_data in drape_classes_dict.items()
           }

  for agent_character in AGENT_CHRS[amount_agents:]:  # populate unused agent layers
    drapes[agent_character] = [DummyAgentDrape]


  z_order = [char for char in drape_classes_dict.keys()]
  z_order += [AGENT_CHRS[agent_index] for agent_index in range(0, len(AGENT_CHRS))]

  # AGENT_CHR needs to be first else self.curtain[player.position]: does not work properly in drapes  # TODO: why?
  update_schedule = [AGENT_CHRS[agent_index] for agent_index in range(0, len(AGENT_CHRS))]
  update_schedule += [char for char in drape_classes_dict.keys()]


  tile_type_counts = {
              char: drape_class_data[1]
              for char, drape_class_data in drape_classes_dict.items()
            }

  # removing extra agents from the map
  # TODO: implement a way to optionally randomize the agent locations as well and move agent amount setting / extra agent disablement code to the make_safety_game method
  for agent_character in AGENT_CHRS[:amount_agents]:
    tile_type_counts[agent_character] = 1
  for agent_character in AGENT_CHRS[amount_agents:]:
    tile_type_counts[agent_character] = 0


  result = safety_game_moma.make_safety_game_mo(
      environment_data,
      map,
      what_lies_beneath=GAP_CHR,
      sprites=sprites,
      drapes=drapes,
      z_order=z_order,
      update_schedule=update_schedule,
      map_randomization_frequency=FLAGS.map_randomization_frequency,
      preserve_map_edges_when_randomizing=preserve_map_edges_when_randomizing,     
      environment=environment,
      tile_type_counts=tile_type_counts,
      remove_unused_tile_types_from_layers=FLAGS.remove_unused_tile_types_from_layers,
      map_width=FLAGS.map_width, 
      map_height=FLAGS.map_height,  
  )

  return result

#/ def make_game_moma(global_vars, environment, environment_data, agent_sprite_class, drape_classes_dict):


