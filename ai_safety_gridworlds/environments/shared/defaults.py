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

from ai_safety_gridworlds.environments.shared import safety_game_ma
from ai_safety_gridworlds.environments.shared.mo_reward import mo_reward


DEFAULT_LEVEL = 0     # level 0 is an empty template
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_NOOPS = True                      # Whether to include NOOP as a possible agent action.
DEFAULT_RANDOMIZE_AGENT_ACTIONS_ORDER = True    # Whether to randomize the order the agent actions are carried out in order to resolve any tile collisions and resource availability collisions randomly.
DEFAULT_MAP_RANDOMIZATION_FREQUENCY = 3                 # Whether to randomize the map.   # 0 - off, 1 - once per experiment run, 2 - once per trial (a trial is a sequence of training episodes separated by env.reset call, but using a same model instance), 3 - once per training episode
DEFAULT_OBSERVATION_RADIUS = [5] * 4            # How many tiles away from the agent can the agent see? -1 means the agent perspective is same as global perspective and the observation does not move when the agent moves. 0 means the agent can see only the tile underneath itself. None means the agent can see the whole board while still having agent-centric perspective; the observation size is 2*board_size-1.
DEFAULT_OBSERVATION_DIRECTION_MODE = 0    # 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions
DEFAULT_ACTION_DIRECTION_MODE = 0         # 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions
DEFAULT_REMOVE_UNUSED_TILE_TYPES_FROM_LAYERS = False    # Whether to remove tile types not present on initial map from observation layers.
DEFAULT_ENABLE_LOGGING = False

# if default map width and height are None, then original ascii art dimensions are used
DEFAULT_MAP_WIDTH = None
DEFAULT_MAP_HEIGHT = None


AGENT_CHR1 = '0'  # 'A'
AGENT_CHR2 = '1'
AGENT_CHR3 = '2'
AGENT_CHR4 = '3'
AGENT_CHR5 = '4'
AGENT_CHR6 = '5'
AGENT_CHR7 = '6'
AGENT_CHR8 = '7'
AGENT_CHR9 = '8'
AGENT_CHR10 = '9'

AGENT_CHRS = [ 
  AGENT_CHR1,
  AGENT_CHR2,
  #AGENT_CHR3,
  #AGENT_CHR4,
  #AGENT_CHR5,
  #AGENT_CHR6,
  #AGENT_CHR7,
  #AGENT_CHR8,
  #AGENT_CHR9,
  #AGENT_CHR10,
]


WALL_CHR = '#'
GAP_CHR = ' '


MOVEMENT_SCORE = mo_reward({"MOVEMENT": -1})    # TODO: tune

GAP_SCORE = mo_reward({})        


DEFAULT_AMOUNT_AGENTS = 1


CUSTOM_ACTIONS = {}


GAME_BG_COLOURS = {}
GAME_BG_COLOURS.update(safety_game_ma.GAME_BG_COLOURS)
GAME_BG_COLOURS.update({
    GAP_CHR: (0, 0, 0),
    WALL_CHR: (500, 500, 500),
})

GAME_FG_COLOURS = {}
GAME_FG_COLOURS.update(safety_game_ma.GAME_FG_COLOURS)
GAME_FG_COLOURS.update({
    GAP_CHR: (0, 0, 0), 
    WALL_CHR: (500, 500, 500),    # give wall chars foreground colour so that the walls look solid, not as # chars on gray background
})
