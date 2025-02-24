from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools

    use_setuptools()
    import setuptools

if sys.version_info[0] == 2:
    enum = ["enum34"]
elif sys.version_info[0] == 3:
    enum = []

if os.name == "nt":
    windows_curses = ["windows-curses"]
else:
    windows_curses = []

setuptools.setup(
    name="ai-safety-gridworlds",
    version="4.0",
    description="Extended, multi-agent and multi-objective environments based on DeepMind's "
        "AI Safety Gridworlds. This is a suite of reinforcement learning "
        "environments illustrating various safety properties of intelligent agents.",
    long_description=(
        "Extended, multi-agent and multi-objective environments based on DeepMind's "
        "AI Safety Gridworlds. "
        "This is a suite of reinforcement learning environments illustrating "
        "various safety properties of intelligent agents. "
        "It is made compatible with OpenAI's Gym and Gymnasium "
        "and Farama Foundation PettingZoo."
    ),
    url="https://github.com/levitation-opensource/ai-safety-gridworlds/",
    author="Roland Pihlakas, forked from jvmncs, David Lindner, n0p2, and from DeepMind Technologies",
    author_email="roland@simplify.ee",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: Console :: Curses",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.10",
        "Topic :: Games/Entertainment :: Arcade",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Testing",
    ],
    keywords=(
        "ai "
        "artificial intelligence "
        "ascii art "
        "game engine "
        "gridworld "
        "gym "
        "gymnasium "
        "mamorl "
        "multi-objective "
        "multi-agent "
        "pettingzoo "
        "reinforcement learning "
        "retro retrogaming "
        "rl "
    ),
    install_requires=[
      "absl-py", 
      "gym",    # If the user wants, they can manually install Gymnasium instead and then we will automatically use this newer package instead. Gym is needed as a fallback. Installation of Gymnasium is not forced in order to not override existing Gym installation, since Gymnasium would have priority during execution.
      "matplotlib",
      "numpy", 
      "pettingzoo",
      "pillow",
      # "pycolab", 
    ] + enum + windows_curses,
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
    test_suite="ai_safety_gridworlds.tests",
    tests_require=[],
    package_data={"ai_safety_gridworlds.helpers": ["*.ttf"]},
)
