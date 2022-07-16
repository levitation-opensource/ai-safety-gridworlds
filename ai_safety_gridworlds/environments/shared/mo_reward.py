# Copyright 2022 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
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

from ast import literal_eval
import itertools

# Dependency imports
import numpy as np


class mo_reward(object):

  def __init__(self, reward_dimensions_dict, immutable=True):

    self._reward_dimensions_dict = reward_dimensions_dict
    self._immutable = immutable


  def copy(self):

    dict_clone = dict(self._reward_dimensions_dict) # clone
    return mo_reward(dict_clone, immutable=False)


  def __eq__(self, other):
    return self._reward_dimensions_dict == other._reward_dimensions_dict


  @staticmethod
  def parse(string):

    if string == "":
      return mo_reward({})
    else:
      object = literal_eval(string)
      # object = json.loads(string.replace("'", '"')) # mo_reward input python dictionary string is similar to json
      return mo_reward(object)


  @staticmethod
  def get_enabled_reward_dimension_keys(enabled_mo_rewards):
    """Returns keys of all dimensions defined in enabled_mo_rewards.

    Args:
      enabled_mo_rewards: a list of mo_reward objects.
    """

    if enabled_mo_rewards is None:

      return [None]

    else: # if enabled_mo_rewards is not None:

      # each reward may contain more than one enabled dimension
      keys_per_reward = [{ 
                            key for key, unit_value 
                            in reward._reward_dimensions_dict.items() if unit_value != 0 
                          }
                          for reward in enabled_mo_rewards]

      # select distinct keys:
      # enabled_reward_dimension_keys = list(set.union(*keys_per_reward))  # this does not preserve the order of the keys
      enabled_reward_dimension_keys = list(dict.fromkeys(itertools.chain.from_iterable(keys_per_reward)).keys())  # this preserves the order of the keys
      enabled_reward_dimension_keys.sort()  # for some reason the sorting order was still changing

      return enabled_reward_dimension_keys


  @staticmethod
  def get_enabled_reward_unit_space(enabled_mo_rewards):  
    """Returns tuple of min unit reward vector and max unit reward vector.

    Args:
      enabled_mo_rewards: a list of mo_reward objects.
    """

    if enabled_mo_rewards is None:

      return None

    else: # if enabled_mo_rewards is not None:

      enabled_reward_dimension_keys = mo_reward.get_enabled_reward_dimension_keys(enabled_mo_rewards)

      min_unit_value_per_key = {
                                  key: min([
                                    reward._reward_dimensions_dict.get(key, 0)
                                    for reward in enabled_mo_rewards
                                  ])
                                  for key in enabled_reward_dimension_keys
                               }
 
      max_unit_value_per_key = {
                                  key: max([
                                    reward._reward_dimensions_dict.get(key, 0)
                                    for reward in enabled_mo_rewards
                                  ])
                                  for key in enabled_reward_dimension_keys
                               }

      return [list(min_unit_value_per_key.values()), list(max_unit_value_per_key.values())]


  def tolist(self, enabled_mo_rewards):
    """Converts the mo_reward value to a list of all dimension values including dimensions with zero values."""

    if enabled_mo_rewards is None:

      reward_values = self._reward_dimensions_dict.values()
      return sum(reward_values)

    else: # if enabled_mo_rewards is not None:

      enabled_reward_dimension_keys = mo_reward.get_enabled_reward_dimension_keys(enabled_mo_rewards)

      for key, value in self._reward_dimensions_dict.items():
        if value != 0 and key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward with nonzero value" % key)

      result = [0] * len(enabled_reward_dimension_keys)
      for dimension_index, enabled_reward_dimension_key in enumerate(enabled_reward_dimension_keys):
        result[dimension_index] = self._reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def tofull(self, enabled_mo_rewards):
    """Converts the mo_reward value to dictionary containing keys of all dimensions including dimensions with zero values."""

    if enabled_mo_rewards is None:

      reward_values = self._reward_dimensions_dict.values()
      return {None: sum(reward_values)}

    else: # if enabled_mo_rewards is not None:

      enabled_reward_dimension_keys = mo_reward.get_enabled_reward_dimension_keys(enabled_mo_rewards)

      for key, value in self._reward_dimensions_dict.items():
        if value != 0 and key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward with nonzero value" % key)

      result = {}
      for enabled_reward_dimension_key in enabled_reward_dimension_keys:
        result[enabled_reward_dimension_key] = self._reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def __str__(self, enabled_mo_rewards=None): # tostring

    if enabled_mo_rewards is not None:
      enabled_reward_dimension_keys = mo_reward.get_enabled_reward_dimension_keys(enabled_mo_rewards)
      dict_with_enabled_keys = { key: self._reward_dimensions_dict.get(key, 0) for key in enabled_reward_dimension_keys }
      return str(dict_with_enabled_keys)
    else:
      return str({ key: value for key, value in self._reward_dimensions_dict.items() if value != 0 })


  def __repr__(self, enabled_mo_rewards=None): # tostring

    if enabled_mo_rewards is not None:
      enabled_reward_dimension_keys = mo_reward.get_enabled_reward_dimension_keys(enabled_mo_rewards)
      dict_with_enabled_keys = { key: self._reward_dimensions_dict.get(key, 0) for key in enabled_reward_dimension_keys }
      return "<" + repr(dict_with_enabled_keys) + ">"
    else:
      return "<" + repr({ key: value for key, value in self._reward_dimensions_dict.items() if value != 0 }) + ">"


  def __add__(self, other):

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) + other_value
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__add__, expecting a scalar or mo_reward")


  def __iadd__(self, other):  # in-place add

    if self._immutable:
      return self.__add__(other)

    if np.isscalar(other):
      for key, value in self._reward_dimensions_dict.items():
        self._reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        self._reward_dimensions_dict[other_key] = self._reward_dimensions_dict.get(other_key, 0) + other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__iadd__, expecting a scalar or mo_reward")

    return self


  def __radd__(self, other):  # reflected add (the order of operands is exchanged)

    return self + other


  def __sub__(self, other):

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) - other_value
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__sub__, expecting a scalar or mo_reward")


  def __isub__(self, other):  # in-place sub

    if self._immutable:
      return self.__sub__(other)

    if np.isscalar(other):
      for key, value in self._reward_dimensions_dict.items():
        self._reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        self._reward_dimensions_dict[other_key] = self._reward_dimensions_dict.get(other_key, 0) - other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__isub__, expecting a scalar or mo_reward")

    return self


  def __rsub__(self, other):  # reflected sub (the order of operands is exchanged)

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: other - value for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = other_value - result_dict.get(other_key, 0)
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__rsub__, expecting a scalar or mo_reward")


  def __neg__(self):  # unary -

    return mo_reward({ key: -value for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __mul__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__mul__, expecting a scalar")

    return mo_reward({ key: value * other for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __imul__(self, other):  # in-place mul

    if self._immutable:
      return self.__mul__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__imul__, expecting a scalar")

    for key, value in self._reward_dimensions_dict.items():
      self._reward_dimensions_dict[key] = value * other

    return self


  def __rmul__(self, other):  # reflected mul (the order of operands is exchanged)

    return self * other


  def __truediv__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__truediv__, expecting a scalar")

    return mo_reward({ key: value / other for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __itruediv__(self, other):  # in-place div

    if self._immutable:
      return self.__truediv__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__itruediv__, expecting a scalar")

    for key, value in self._reward_dimensions_dict.items():
      self._reward_dimensions_dict[key] = value / other

    return self


  def __rtruediv__(self, other):  # reflected div (the order of operands is exchanged)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__rtruediv__, expecting a scalar")

    return mo_reward({ key: other / value for key, value in self._reward_dimensions_dict.items() }, immutable=False)


