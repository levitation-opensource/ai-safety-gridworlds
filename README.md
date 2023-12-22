# Descriptions of the added environments

The currently available experiment environments are described here https://docs.google.com/document/d/1AV566H0c-k7krBietrGdn-kYefSSH99oIH74DMWHYj0/edit# The descriptions are also available in this repo as a markdown file: https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds/blob/master/ai_safety_gridworlds/experiments/experiment_descriptions.md

One more elaborate environment is described separately here: https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds/blob/master/MultiAgentSafetyHackathon.md


# Major updates in this fork

* Added support for succinctly configuring multiple experiments (configuration variations) based on a same base environment file. These "experiment environments" are child classes based on the main "template" environment classes. The experiment environments define variations on the flag values available in the main environment. 
* Started extending the maps and implementing multi-objective rewards for various environments.
* island_navigation_ex.py has been implemented. The latter has now food and drink sources with satiation and deficit aspects in the agent, as well as sustainability aspect in the environment. Also, the environment has gold and silver sources. All these aspects can be turned on and off, as well as their parameters can be configured using flags.
* boat_race_ex.py has been implemented. The latter has now iterations penalty and repetition penalty (penalty for visiting the same tile repeatedly). The map contains human tiles which should be avoided. These aspects can be turned on and off using flags.
* Additionally planned multi-objective environment extensions: conveyor_belt_ex.py, safe_interruptibility_ex.py
* The multi-objective rewards are represented in vector form. The multi-objective environment constructor provides an additional option to automatically scalarise the rewards in order to return non-multi-objective-environment compatible values. This option is disabled by default. The scalarisation is computed using linear summing of the reward dimensions.
* The multi-objective rewards are compatible with https://github.com/LucasAlegre/mo-gym
* Compatibility with OpenAI Gym using code from https://github.com/david-lindner/safe-grid-gym and https://github.com/n0p2/ . The related GridworldGymEnv wrapper is available under ai_safety_gridworlds.helpers namespace. register_with_gym() method in factory.py creates registrations for all environments in such a way that they are Gym compatible, using the GridworldGymEnv wrapper class. The code is updated to be compatible with both Gym (gym package) and Gymnasium (gymnasium package).
* Compatibility with Farama Foundation PettingZoo. The related GridworldZooParallelEnv and GridworldZooAecEnv wrappers are available under ai_safety_gridworlds.helpers namespace.
* Added SafetyCursesUiEx class which enables printing various custom drape and sprite metrics on the screen. The metrics are also returned in timestep.observation under keys metrics_dict and metrics_matrix.
* Added safety_ui_ex.make_human_curses_ui_with_noop_keys() method which enables human player to perform no-ops using keyboard. The RL agent had this capability in some environments already in the original code.
* Support for configurable logging of timestamp, environment_name, trial_no, episode_no, iteration_no, arguments, reward_unit_sizes, reward, scalar_reward, cumulative_reward, scalar_cumulative_reward, metrics. Trial and episode logs are concatenated into same CSV file. The CSV files can be optionally automatically gzipped at the same time as they are written to, so gzipping is not postponed until CSV is complete. Environment arguments are saved to a separate TXT file. episode_no is incremented when reset() is called or when a new environment is constructed. trial_no is updated when reset() is called with a trial_no argument or when new environment is constructed with a trial_no argument. Automatically re-seeds the random number generator with a new seed for each new trial_no. The seeds being used are deterministic, which means that across executions the seed sequence will be same.
* Implemented Q value logging. If the agent provides a matrix of Q values per action using .set_current_q_value_per_action() method before a call to .step() then, considering the agent's current location, the environment maps the Q values per action to Q values per tile type (according to the character on environment map) where that action would have ended up and adds this data to the CSV log file.
* Support for multi-agent environments. Each agent has its own reward and termination accounting.
* Support for agent-specific observations. The agent-specific observations are returned via Zoo API as well as shown in the human ui. The number of agents is controlled by the environment. It can be configured for example via environment flags.
* Support for relative observation and action direction per agent. The observation direction can be configured to be always same, changing depending on last movement direction, or controlled by special turning actions. The observation and action direction is retuned in the info fields of the step function.
* Implemented a strategy for automatically resolving physical collisions as well as any kinds of distributed resource conflicts in the Zoo parallel environments without extra code in the concrete environment's code.
* Functionality to randomize the map and to configure tile type amounts without having to specify a new map. Randomization can be configured to take place once per experiment, once per trial (a trial is a sequence of training episodes using a same model instance), or once per training episode.


# Other updates

* Implemented automatic registration of environments and experiments instead of manually declaring them in factory.py
* Added the following flags to more environments: level, max_iterations, noops. 
* Refactored code for more consistency across environments. 
* The cumulative rewards are also returned, in timestep.observation, under key cumulative_reward.
* Added variance between reward dimensions (not over time), variance between cumulative reward dimensions, gini index of reward dimensions, and gini index of cumulative reward dimensions to CSV logging and to agent observation. The gini index is a modified version - it is computed by substracting the minimum value, so the negative reward dimensions can also be handled. Also added average_mo_variance column to CSV file which computes variance over multi-objective reward dimensions of the average reward over all iterations of the episode until current iteration.


# Minor updates

* Do not rerender the entire screen if only time counter needs to be updated. This reduces screen flicker.


# Roadmap

* Functionality for multi-modal actions. Currently the following modalities are planned:  
   * Movement on board (handled by the framework).
   * Additional actions (handled by the agent implementation). Multiple concurrent additional actions can be provided for each step. Examples of additional actions: "Eat", "Take", "Put", etc.
   * Change of agent's action direction and/or observation direction (two separate dimensions, handled by the framework). In case action direction and action are provided at the same time, the action direction is applied first.
   * Agent's expressions. Expressions are observable attributes (handled by the framework, can be changed via action or by the agent implementation). Multiple concurrent expressions can be provided for each step. Examples of expressions / observable attributes are:
      * Facial expressions.
      * Signs.
      * Numeric message bubbles.
      * Textual message bubbles.
* Functionality for observable agent attributes. These are attributes which always move on the map together with an agent and can be observed by other agents. The actions described above can change some of these attributes. Other attributes are controlled by environment's code.
   * Agent's action direction.
   * Agent's observation direction.
   * Expressions and numeric attributes:
      * Facial expressions.
      * Signs.
      * Numeric message bubbles.
      * Textual message bubbles.
   * Public metrics. Some of the agent's metrics can be made observable by other agents. Metrics are numeric attributes which change according to the environment's rules and cannot be changed directly by the agent's brain. Metrics are not rewards, but rewards may be dependent on metrics. Some metrics are agent-internal and not observable by others, but some can be made publicly observable.



# Other related resources

* For other interesting Gridworlds environments contributions, take a look at https://github.com/side-grids/ai-safety-gridworlds/tree/master/ai_safety_gridworlds/environments
* DeepMind's original readme file can be found here: https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds/blob/master/Original%20Readme.md


# Papers

* A published research paper based on experiments using this repository: Smith, B.J., Klassert, R. & Pihlakas, R. "Using soft maximin for risk averse multi-objective decision-making". Autonomous Agents and Multi-Agent Systems 37, Article 11 (2023). https://link.springer.com/article/10.1007/s10458-022-09586-2


## Acknowledgements
I would like to thank Ben Smith, Robert Klassert, Joel Pyykkö, AI Safety Camp V, AIntelope, EA Funds, Emergent Ventures / Mercatus Center at George Mason University, and Foresight Institute.

