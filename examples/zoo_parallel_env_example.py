# Copyright 2022-2023 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 n0p2 https://github.com/n0p2/gym_ai_safety_gridworlds
"""
Example demonstrating the usage of the Zoo Parallel interface.

This was adapted from https://github.com/n0p2/gym_ai_safety_gridworlds
"""

import argparse

from pettingzoo import ParallelEnv

try:
  import gymnasium as gym
  gym_v26 = True
except:
  import gym
  gym_v26 = False

import logging

from ai_safety_gridworlds.demonstrations import demonstrations

# from safe_grid_gym_orig.envs import GridworldEnv
from ai_safety_gridworlds.helpers.gridworld_zoo_parallel_env import GridworldZooParallelEnv


logger = logging.getLogger(__name__)
hdlr = logging.FileHandler("env_example.log")
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def gym_env(args):
    env = mk_env(args)
    env.reset()
    actions = get_actions(args, env)

    rr = []
    episode_return = 0
    for (i, action) in enumerate(actions):
        
        agents_actions = {
            agent: action for agent in env.possible_agents
        }  

        if gym_v26:
            (_, reward, terminated, truncated, info) = env.step(agents_actions)
            done = terminated or truncated
        else:
            (_, reward, done, info) = env.step(agents_actions)

        episode_return += sum(reward)
        env.render(mode="human")
        if done:
            s = "episode {}, returns: {}".format(len(rr), episode_return)
            logger.info(s)
            rr.append(episode_return)
            episode_return = 0
            env.reset()


def mk_env(args):
    if args.gym_make:
        id_ = "ai_safety_gridworlds-" + args.env_name + "-v0"
        try:
            return gym.make(id_, disable_env_checker=True)    # gym version >= 24
        except TypeError:
            return gym.make(id_)    # gym version < 24
    else:
        return GridworldZooParallelEnv(env_name=args.env_name, render_animation_delay=args.pause)


def get_actions(args, env):
    if args.rand_act:
        return [env.action_space.sample() for _ in range(args.steps)]
    else:
        demo = demonstrations.get_demonstrations(args.env_name)[0]
        return demo.actions


# --------
# main io
# --------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env_name",
        default="distributional_shift",
        help="e.g. distributional_shift|side_effects_sokoban",
    )
    parser.add_argument("-r", "--rand_act", action="store_true")
    parser.add_argument("-g", "--gym_make", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gym_env(args)
