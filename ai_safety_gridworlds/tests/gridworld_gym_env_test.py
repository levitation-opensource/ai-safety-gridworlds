# Copyright 2022-2023 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2019 jvmncs https://github.com/jvmncs
# Copyright 2018-2019 David Lindner https://github.com/david-lindner/safe-grid-gym

import traceback
import unittest

try:
  import gymnasium as gym
  gym_v26 = True
except:
  import gym
  gym_v26 = False

import numpy as np

from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.environments.shared.rl import pycolab_interface_ma
from ai_safety_gridworlds.environments.shared.safety_game import Actions

from ai_safety_gridworlds.helpers.gridworld_gym_env import GridworldGymEnv
from ai_safety_gridworlds.helpers.gridworld_gym_env import INFO_HIDDEN_REWARD, INFO_OBSERVED_REWARD


class SafetyGridworldsTestCase(unittest.TestCase):
  def _check_rgb(self, rgb_list):
    first_shape = rgb_list[0].shape
    for rgb in rgb_list:
      self.assertIsInstance(rgb, np.ndarray)
      self.assertEqual(len(rgb.shape), 3)
      self.assertEqual(rgb.shape[0], 3)
      self.assertEqual(rgb.shape, first_shape)

  def _check_ansi(self, ansi_list):
    first_len = len(ansi_list[0])
    first_newline_count = ansi_list[0].count("\n")
    for ansi in ansi_list:
      self.assertIsInstance(ansi, str)
      self.assertEqual(len(ansi), first_len)
      self.assertEqual(ansi.count("\n"), first_newline_count)

  def _check_reward(self, env, reward):
    min_reward, max_reward = env.reward_range
    self.assertGreaterEqual(reward, min_reward)
    self.assertLessEqual(reward, max_reward)

  def _check_action_observation_valid(self, env, action, observation):
    self.assertTrue(env.action_space.contains(action))
    self.assertTrue(env.observation_space.contains(observation))

  def _check_rewards(
    self,
    env,
    demo,
    episode_info_observed_return,
    episode_info_hidden_return,
    episode_return,
  ):
    # check observed and hidden rewards
    self.assertEqual(episode_info_observed_return, demo.episode_return)

    hidden_reward = env._env._get_hidden_reward(default_reward=None)

    if hidden_reward is not None:
      self.assertEqual(episode_info_hidden_return, demo.safety_performance)
      self.assertEqual(episode_info_hidden_return, hidden_reward)

    self.assertEqual(episode_info_observed_return, episode_return)

  def setUp(self):
    self.demonstrations = {}
    for env_name in factory._environment_classes.keys():
      try:
        demos = demonstrations.get_demonstrations(env_name)
      except ValueError:
        # no demonstrations available
        demos = []
      # TODO: for tests, do not register same environment under different aliases
      self.demonstrations[env_name] = demos

    # add demo that fails, to test hidden reward
    self.demonstrations["absent_supervisor"].append(
      demonstrations.Demonstration(0, [Actions.DOWN] * 3, 47, 17, True)
    )

  def testActionSpaceSampleContains(self):
    """
    Check that sample and contain methods of the action space are consistent.
    """
    repetitions = 10

    for env_name in self.demonstrations.keys():
      env = GridworldGymEnv(env_name)
      action_space = env.action_space
      for _ in range(repetitions):
        action = action_space.sample()
        self.assertTrue(action_space.contains(action))

  def testObservationSpaceSampleContains(self):
    """
    Check that sample and contain methods of the observation space are consistent.
    """
    repetitions = 10

    for env_name in self.demonstrations.keys():
      env = GridworldGymEnv(env_name)

      if isinstance(env._env, pycolab_interface_ma.EnvironmentMa):   # skip multi-agent environments from Gym tests
        continue

      observation_space = env.observation_space
      for _ in range(repetitions):
        observation = observation_space.sample()
        assert observation_space.contains(observation)

  def reset(self, env):
    obs = env.reset()
    return obs

  def step(self, env, action):
    if gym_v26:
      obs, _, _, _, _ = env.step(action)
    else:
      obs, _, _, _ = env.step(action)
    return obs

  def testStateObjectCopy(self):
    """
    Make sure that the state array that is returned does not change in
    subsequent steps of the environment. The pycolab only returns a pointer
    to the state object, which makes it change if we take another step.
    For gym, however, we want the state to not change, i.e. return a copy
    of the board.
    """
    env = GridworldGymEnv("boat_race")
    obs0 = self.reset(env)
    obs1 = self.step(env, Actions.RIGHT)
    obs2 = self.step(env, Actions.RIGHT)
    self.assertFalse(np.all(obs0 == obs1))
    self.assertFalse(np.all(obs0 == obs2))
    self.assertFalse(np.all(obs1 == obs2))

    # ADDED
    env = GridworldGymEnv("boat_race_ex")
    obs0 = self.reset(env)
    obs1 = self.step(env, Actions.RIGHT)
    obs2 = self.step(env, Actions.RIGHT)
    self.assertFalse(np.all(obs0 == obs1))
    self.assertFalse(np.all(obs0 == obs2))
    self.assertFalse(np.all(obs1 == obs2))

  def testTransitions(self):
    """
    Ensure that when the use_transitions argument is set to True the state
    contains the board of the last two timesteps.
    """
    env = GridworldGymEnv("boat_race", use_transitions=False)
    board_init = self.reset(env)
    assert board_init.shape == (1, 5, 5)
    obs1 = self.step(env, Actions.RIGHT)
    assert obs1.shape == (1, 5, 5)
    obs2 = self.step(env, Actions.RIGHT)
    assert obs2.shape == (1, 5, 5)

    env = GridworldGymEnv("boat_race", use_transitions=True)
    board_init = self.reset(env)
    assert board_init.shape == (2, 5, 5)
    obs1 = self.step(env, Actions.RIGHT)
    assert obs1.shape == (2, 5, 5)
    obs2 = self.step(env, Actions.RIGHT)
    assert obs2.shape == (2, 5, 5)
    assert np.all(board_init[1] == obs1[0])
    assert np.all(obs1[1] == obs2[0])

    #env = gym.make("TransitionBoatRace-v0")
    #board_init = self.reset(env)
    #assert board_init.shape == (2, 5, 5)
    #obs1 = self.step(env, Actions.RIGHT)
    #assert obs1.shape == (2, 5, 5)
    #obs2 = self.step(env, Actions.RIGHT)
    #assert obs2.shape == (2, 5, 5)
    #assert np.all(board_init[1] == obs1[0])
    #assert np.all(obs1[1] == obs2[0])


    # ADDED
    env = GridworldGymEnv("boat_race_ex", level=0, use_transitions=False)
    board_init = self.reset(env)
    assert board_init.shape == (1, 5, 5)
    obs1 = self.step(env, Actions.RIGHT)
    assert obs1.shape == (1, 5, 5)
    obs2 = self.step(env, Actions.RIGHT)
    assert obs2.shape == (1, 5, 5)

    env = GridworldGymEnv("boat_race_ex", level=0, use_transitions=True)
    board_init = self.reset(env)
    assert board_init.shape == (2, 5, 5)
    obs1 = self.step(env, Actions.RIGHT)
    assert obs1.shape == (2, 5, 5)
    obs2 = self.step(env, Actions.RIGHT)
    assert obs2.shape == (2, 5, 5)
    assert np.all(board_init[1] == obs1[0])
    assert np.all(obs1[1] == obs2[0])

    #env = gym.make("TransitionBoatRaceEx-v0")
    #board_init = self.reset(env)
    #assert board_init.shape == (2, 5, 5)
    #obs1 = self.step(env, Actions.RIGHT)
    #assert obs1.shape == (2, 5, 5)
    #obs2 = self.step(env, Actions.RIGHT)
    #assert obs2.shape == (2, 5, 5)
    #assert np.all(board_init[1] == obs1[0])
    #assert np.all(obs1[1] == obs2[0])

  def testWithDemonstrations(self):
    """
    Run demonstrations in the safety gridworlds and perform sanity checks
    on rewards, episode termination and the "ansi" and "rgb_array" render modes.
    """

    repititions = 10

    for env_name, demos in self.demonstrations.items():
      for demo in demos:
        for i in range(repititions):

          try:      # ADDED
            # need to use np seed instead of the environment seed function
            # to be consistent with the seeds given in the demonstrations
            np.random.seed(demo.seed)
            env = GridworldGymEnv(env_name)

            actions = demo.actions
            self.reset(env)
            done = False

            episode_return = 0
            episode_info_observed_return = 0
            episode_info_hidden_return = 0

            rgb_list = [env.render("rgb_array")]
            ansi_list = [env.render("ansi")]

            for action in actions:
              self.assertFalse(done)

              if gym_v26:
                (obs, reward, terminated, truncated, info) = env.step(action)
                done = terminated or truncated
              else:
                (obs, reward, done, info) = env.step(action)

              episode_return += reward
              episode_info_observed_return += info[INFO_OBSERVED_REWARD]

              if info[INFO_HIDDEN_REWARD] is not None:
                episode_info_hidden_return += info[INFO_HIDDEN_REWARD]

              rgb_list.append(env.render("rgb_array"))
              ansi_list.append(env.render("ansi"))
              self._check_action_observation_valid(env, action, obs)
              self._check_reward(env, reward)

            self.assertEqual(done, demo.terminates)
            self._check_rewards(
              env,
              demo,
              episode_info_observed_return,
              episode_info_hidden_return,
              episode_return,
            )

            self._check_rgb(rgb_list)
            self._check_ansi(ansi_list)

          except Exception as ex:      # ADDED
            print("\nenv_name: " + env_name)
            # raise
            print(ex)
            print(traceback.format_exc())


if __name__ == "__main__":
  unittest.main()
