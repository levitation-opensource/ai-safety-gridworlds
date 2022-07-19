# ai\_safety\_gridworlds changelog

## Version 2.8.3 - Tuesday 19. July 2022

* Added average_mo_variance column to CSV file which computes variance over multi-objective reward dimensions of the average reward over all iterations of the episode until current iteration.

## Version 2.8.2 - Sunday 17. July 2022

* Re-seed the random number generator during .reset() call for the first episode of the first trial, even if the random number generator was already seeded during the environment construction. This is because setting up the experiment before the .reset() call might have consumed random numbers from the random number generator and we want the agent to be deterministic after the reset call.
* Sort the reward dimensions alphabetically in the CSV log file.
* Fixed a bug which caused a new log file to be created for each newly constructed environment object even if the experiment stays same. Same experiment should log all trials to one log file regardless of whether the environment is reset or re-constructed.
* Added capability to override only select few flags from the agent side without having to call init_experiment_flags() from the correct experiment file in multiobjective-ai-safety-gridworlds in order to provide all flags during override.

## Version 2.8.1 - Thursday 14. July 2022

* Added variance between reward dimensions (not over time), variance between cumulative reward dimensions, gini index of reward dimensions, and gini index of cumulative reward dimensions to CSV logging and to agent observation. The gini index is a modified version - it is computed by substracting the minimum value, so the negative reward dimensions can also be handled.

## Version 2.8 - Wednesday 13. July 2022

* Implemented Q value logging. If the agent provides a matrix of Q values per action using .set_current_q_value_per_action() method before a call to .step() then, considering the agent's current location, the environment maps the Q values per action to Q values per tile type (according to the character on environment map) where that action would have ended up and adds this data to the CSV log file.

## Version 2.7.1 - Tuesday 12. July 2022

* Implemented tile visit count metrics for island navigation.
* Added capability to save non-numeric metric values to CSV log files.
* Added handling for calls to update metric values for metrics that are not activated.

## Version 2.7 - Wednesday 6. July 2022

* Implemented automatic registration of environments and experiments instead of manually declaring them in factory.py

## Version 2.6.3 - Wednesday 6. July 2022

* Log reward_unit_space in the environment parameters file in the reward_dimensions section.

## Version 2.6.2 - Tuesday 5. July 2022

* Added .get_reward_unit_space() method.

## Version 2.6.1 - Monday 4. July 2022

* Add code to automatically generate a new log file if the environment parameters change. This is helpful when multiple experiments are run in sequence from a Python batch file.
* Add .reset(start_new_experiment=True) which forces a new log file to be created next time the environment is created - even if the parameters do not change. This is helpful when multiple experiments are run in sequence from a Python batch file.

## Version 2.6 - Friday 1. July 2022

* Added register_with_gym() method to factory.py. This creates registrations for all environments in factory in such a way that they are gym compatible, using a GridworldGymEnv wrapper class included under helpers.
* Round off values caused by floating point drift before writing to CSV file so that the file will be smaller.

## Version 2.5.3 - Tuesday 28. June 2022

* Added gap tile reward to island navigation ex environment. Added Rolf 2020 experiments.

## Version 2.5.2 - Saturday 25. June 2022

* Create experiment reward and argument log files by default if not specified otherwise in the arguments.

## Version 2.5.1 - Friday 24. June 2022

* Added use_satiation_proportional_reward flag. The default value of this flag is False, which changes the previous behaviour which corresponded to use_satiation_proportional_reward=True flag value.

## Version 2.5 - Friday 17. June 2022

* Added support for succinctly configuring multiple experiments (configuration variations) based on a same base environment file. These "experiment environments" are child classes based on the main "template" environment classes. The experiment environments define variations on the flag values available in the main environment. The currently available experiment environments are described here https://docs.google.com/document/d/1AV566H0c-k7krBietrGdn-kYefSSH99oIH74DMWHYj0/edit#

## Version 2.4.2 - Wednesday 15. June 2022

* Added support for inserting additional comments to log filenames. Note that if there is a need to specify arbitrary arguments inside the arguments file then that was already possible before. The arguments file will save any arguments provided to the environment's constructor, except some blacklisted ones. It is allowed to provide argument names that the environment does not recognise as well.

## Version 2.4.1 - Saturday 11. June 2022

* Concatenate trial and episode logs into same CSV file. Move arguments to a separate TXT file. episode_no is incremented when reset() is called or when a new environment is constructed. trial_no is updated when reset() is called with a trial_no argument or when new environment is constructed with a trial_no argument. Automatically re-seeds the random number generator with a new seed for each new trial_no. The seeds being used are deterministic, which means that across executions the seed sequence will be same. Added get_trial_no and get_episode_no methods to environment. Save reward dimension names and metrics keys to environment arguments information file. Print trial number and episode number on screen. Improve visual alignment of reward values column and metric values column on screen.

## Version 2.4 - Friday 10. June 2022

* Added support for configurable logging of timestamp, environment_name, trial_no, episode_no, iteration_no, arguments, reward_unit_sizes, reward, scalar_reward, cumulative_reward, scalar_cumulative_reward, metrics.

## Version 2.3.3 - Wednesday 08. June 2022

* The cumulative rewards are also returned, in timestep.observation, under key cumulative_reward.

## Version 2.3.2 - Thursday 26. May 2022

* Added "scalarise" argument to SafetyEnvironmentMo which makes the timestep.reward, get_overall_performance, and get_last_performance to return ordinary scalar value like non-multi-objective environments do. This option is disabled by default. The scalarisation is computed using linear summing of the reward dimensions.
* The OpenAI Gym compatible GridworldGymEnv wrapper and AgentViewer are now available under ai_safety_gridworlds.helpers namespace.

## Version 2.3.1 - Tuesday 24. May 2022

* The metrics are also returned in timestep.observation under keys metrics_dict and metrics_matrix.

## Version 2.3 - Monday 23. May 2022

* Various bugfixes and minor refactorings.
* boat_race_ex.py has been implemented. The latter has now iterations penalty and repetition penalty (penalty for visiting the same tile repeatedly). The map contains human tiles which should be avoided. These aspects can be turned on and off using flags.

## Version 2.2 - Saturday 21. May 2022

* The multi-objective rewards are represented in vector form.
* Do not rerender the entire screen if only time counter needs to be updated. This reduces screen flicker.

## Version 2.1 - Thursday 19. May 2022

* Compatibility with OpenAI Gym using code from https://github.com/david-lindner/safe-grid-gym and https://github.com/n0p2/
* The multi-objective rewards are compatible with https://github.com/LucasAlegre/mo-gym

## Version 2.0 - Saturday 14. May 2022

* Refactored code for more consistency across environments. 
* Added the following flags to more environments: level, max_iterations, noops. 
* Added safety_ui_ex.make_human_curses_ui_with_noop_keys() method which enables human player to perform no-ops using keyboard. The RL agent had this capability in some environments already in the original code.
* Added SafetyCursesUiEx class which enables printing various custom drape and sprite metrics on the screen. 
* Started extending the maps and implementing multi-objective rewards for various environments.
* island_navigation_ex.py has been implemented. The latter has now food and drink sources with satiation and deficit aspects in the agent, as well as sustainability aspect in the environment. Also, the environment has gold and silver sources. All these aspects can be turned on and off, as well as their parameters can be configured using flags.
* Additionally planned multi-objective environment extensions: boat_race_ex.py, conveyor_belt_ex.py, safe_interruptibility_ex.py

## Version 1.5 - Tuesday, 13. October 2020

* Corrections for the side_effects_sokoban wall penalty calculation.
* Added new variants for the conveyor_belt and side_effects_sokoban environments.

## Version 1.4 - Tuesday, 13. August 2019

* Added the rocks_diamonds environment.

## Version 1.3.1 - Friday, 12. July 2019

* Removed movement reward in conveyor belt environments.
* Added adjustment of the hidden reward for sushi_goal at the end of the episode to make the performance scale consistent with other environments.
* Added tests for the sushi_goal variant.

## Version 1.3 - Tuesday, 30. April 2019

* Added a new variant of the conveyor_belt environment - *sushi goal*.
* Added optional NOOPs in conveyor_belt and side_effects_sokoban environments.


## Version 1.2 - Wednesday, 22. August 2018

* Python3 support!
* Compatibility with the newest version of pycolab.

Please make sure to see the new installation instructions in [README.md](https://github.com/deepmind/ai-safety-gridworlds/blob/master/README.md) in order to update to the correct version of pycolab.

## Version 1.1 - Monday, 25. June 2018

* Added a new side effects environment - **conveyor_belt.py**, described in
  the accompanying paper: [Measuring and avoiding side effects using relative reachability](https://arxiv.org/abs/1806.01186).

