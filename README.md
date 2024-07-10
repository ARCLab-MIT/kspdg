# MIT-ARCLab participation in KSPDG challenge

See the information about the challenge in the [upstream repository](https://github.com/mit-ll/spacegym-kspdg/tree/main)

## Resources

See papers and software tools related to this project in the `resources` folder

### Contact

Feel free to reach out if you have questions, feedback, or inquiries!

- <img src="https://raw.githubusercontent.com/gauravghongde/social-icons/master/PNG/Color/Gmail.png" width="20" height="20"/> **Email**: [victor.rfernandez@upm.es](mailto:victor.rfernandez@upm.es)
- <img src="https://raw.githubusercontent.com/gauravghongde/social-icons/master/PNG/Color/LinkedIN.png" width="20" height="20"/> **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/vrodriguezf90)
- <img src="https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/x-social-media-round-icon.png" width="20" height="20"/> **X**: [@YourTwitterHandle](https://twitter.com/vrodriguezf90)


### Environments / Challenge Scenarios

KSPDG is collection of orbital challenge problems defined within different scenario groups (i.e. "evaluation environments" to use nomenclature from reinforcement learning literature). As of Sep 2023, the challenge problems thus far implemented in KSPDG can be grouped into the following types of scenarios

- Basic [Pursuit-Evasion](https://github.com/mit-ll/spacegym-kspdg/tree/main/src/kspdg/pe1)
- Multi-agent target guarding, referred to as a [Lady-Bandit-Guard problem](https://github.com/mit-ll/spacegym-kspdg/tree/main/src/kspdg/lbg1)
- 1-v-1 [Sun-blocking problem](https://github.com/mit-ll/spacegym-kspdg/tree/main/src/kspdg/sb1)

---

## Agents Overview

MIT-ArcLab has developed a diverse range of agents during this research, categorized mainly into two types: Few-Shot Prompting Agents and Fine-Tuning Agents. Each category represent a different strategy in addressing this challenge.

- **Few-Shot Prompting Agents**: These agents are designed to perform tasks with zero training data. They are adept at understanding and executing instructions with only a few examples, showcasing the efficiency and adaptability of LLM models.

- **Fine-Tuning Agents**: These agents will use the fine-tuned models. They tend to be more consistent and reliable, but harder to lead them to high quality results.

- **Training Agent**: This agent called 'jason_keyboard_input' will collect the data of your gameplay in real time while the player accomplishes the mission.

You can explore and learn more about these agents in our [agents folder](https://github.com/ARCLab-MIT/kspdg/tree/main/arclab_mit/agents).

Feel free to delve into the folder for detailed information on each agent, including their design, capabilities, and examples.

---

## Data overview

The data associated with our research is organized into three main sections, each serving a unique purpose in the development and evaluation of our agents. Find all the scripts and folders [here](https://github.com/ARCLab-MIT/kspdg/tree/Alejandro/arclab_mit/agents_data).

### Weights & Biases (wandb) Integration

- **Purpose**: This section focuses on integrating with Weights & Biases (wandb) for managing training files. It provides tools and documentation for setting up and utilizing wandb to track and visualize the training process of our agents.
- **Contents**: Includes setup guides, configuration files, and examples of wandb dashboards and reports.

### Human Gameplay Data

- **Purpose**: This section houses all the data generated from human gameplay. It is crucial for understanding human strategies and behaviors, which aids in developing more human-like agents.
- **Contents**: Features raw gameplay data, processed datasets, and analysis scripts to glean insights from human play patterns.

### Evaluation Results

- **Purpose**: Dedicated to storing and presenting the evaluation results of our agents. This section is essential for assessing the performance and effectiveness of different agent strategies.
- **Contents**: Contains comprehensive evaluation reports, performance metrics, and comparison charts between different agents and scenarios.

Explore each section for detailed information, including methodologies, data formats, and access instructions.


## Citation
Direct link to arXiv preprint and BibTex citation.

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2404.00413)

```
@article{rodriguez2024language,
title={Language Models are Spacecraft Operators},
author={Victor Rodriguez-Fernandez, Alejandro Carrasco, Jason Cheng, Eli Scharf, Peng Mun Siew, Richard Linares},
journal={arXiv preprint arXiv:2404.00413},
year={2024}
}
```


---

## Installation

The installation process includes several components:

- [Kerbal Space Program](https://www.kerbalspaceprogram.com/)
- [Making History Expansion](https://store.privatedivision.com/game/kerbal-space-program-making-history-expansion-official-pc)
- [Mission Save Files](https://github.com/mit-ll/spacegym-kspdg/tree/master/ksp_saves/missions)
- [kRPC Server](https://krpc.github.io/krpc/getting-started.html#the-server-plugin)
- [PhysicsRangeExtender](https://github.com/jrodrigv/PhysicsRangeExtender)
- [`kspdg` python package](https://github.com/mit-ll/spacegym-kspdg/tree/master/src/kspdg)
- [Luna Multiplayer](http://lunamultiplayer.com/) (optional/future work)

> :warning: **Note**
> These instructions have been written and verified on a macOS. They have been partially tested on Ubuntu 18.04 as well. Other operating systems should be similar with deviations on file and directory names

### Install KSP & Making History Expansion

1. Purchase and Download Kerbal Space Program and Making History expansion: https://store.privatedivision.com/game/buy-kerbal-space-program-ksp
   - Make sure to purchase _Direct Download / DRM Free Private Division_ as the platform. Make sure you are buying KSP1 and not the recently released KSP2; none of this will work on KSP2!
   - Download the most recent version of KSP "On Final Approach" Portable (.zip). As of this writing and testing, the most recent version was v1.12.5
   - Download the most recent version of Making History expansion pack. As of this writing the most recent version was v1.12.1
2. Unzip `ksp-osx-1.12.5.zip` to desired location; for simplicity, all instructions assume the unzipped KSP folder is placed on the Desktop
3. Attempt to open the KSP game executable/app (e.g. `KSP.app` on Mac)

> :warning: **Troubleshooting**
>
> - On a Mac, you might run into an error where you can't open the KSP app because the developer can't be verified.
> - To change these preferences on your Mac, choose Apple menu > System Preferences, click Security & Privacy
> - Under the General tab there should be a notification saying something like "KSP.app was blocked ..." if you've attempted to open the KSP app. Click the "Open Anyway" button next to the notification

> :warning: **Troubleshooting**
>
> - On a Mac, after enabling KSP to be opened in Security and Privacy, you may encounter a bug where [the game loading screen stalls indefinitely](ttps://forum.kerbalspaceprogram.com/index.php?/topic/151986-just-purchased-and-stuck-on-loading-screen-mac-os/)
> - The workaround is to move the `KSP.app` icon onto the desktop and then back into the `KSP_osx` directory. For some reason bash commands didn't seem to work to fix this bug. Had to manually open Finder, drag the KSP.app icon onto the Desktop, and then drag it back into the KSP_osx/ directory

4. Unzip `KSP-Making_History_Expansion-en-us-mac-1.12.1.zip`
5. Follow the instructions in the `Instructions-xxx-xx-xx.txt` file located in the unzipped Making History Expansion directory.

> Instructions:
>
> 1. Copy the two other files located in this folder (.command and .zip) to the folder where the KSP app is located
> 2. Once you have copied the files, double click the .command file
>    Thats it! Enjoy the Making History Expansion of Kerbal Space Program!

6. Test installation by opening KSP (e.g. KSP.app on Mac). When main screen has loaded, select `Start Game` and you should see options for `Play Missions` and `Mission Builder` to confirm that the Making History Expansion was successfully installed

### Install KSPDG Mission Save Files

For each differential game environment there are associated mission files created using the "Making History" expansion pack that serves to populate the KSP game engine with the necessary spacecraft in the appropriate orbits. There is also a number of mission save files for in-game software development testing purposes.

The save files are located in this repo under `ksp_files/saves/missions` and `ksp_files/Missions`; both sets are necessary to populate the differential game environments. These mission save files must be downloaded and manaully installed into the correct directory of the KSP game installation.

Copy the contents of `ksp_files/saves/missions/` and `ksp_files/Missions` directory into your local installation of KSP. For example, on a Mac with this repo and KSP's install directories on the desktop this would look like:

```bash
git clone git@github.com:mit-ll/spacegym-kspdg.git
cd spacegym-kspdg
cp -r ksp_files/saves/missions/. ~/Desktop/KSP_osx/saves/missions
cp -r ksp_files/Missions/. ~/Desktop/KSP_osx/Missions
```

### Install kRPC Server

kRPC is what allows external scripts and processes (such as python programs) to send commands to and control the KSP game engine

1. Download latest version of kRPC from the [GitHub link on the kRPC Getting Started Page](https://krpc.github.io/krpc/getting-started.html#installation). As of this writing, you should download `krpc-0.5.2.zip`. **NOTE:** make sure to download a full version, not just the python package; v0.5.2 is a full version but v0.5.3 is just the python package
2. Unzip `krpc-0.5.2/` folder to `~/Desktop/krpc-0.5.2/`
3. Create a new directory in KSP's `GameData` directory and move all of the krpc contents there

```bash
mkdir ~/Desktop/KSP_osx/GameData/kRPC
mv ~/Desktop/krpc-0.5.2/* ~/Desktop/KSP_osx/GameData/kRPC/
```

### Install PhysicsRangeExtender

By default in KSP, high-fidelity physical simulation of spacecraft is only performed for spacecraft very near to the active spacecraft (e.g. only around [2km](https://steamcommunity.com/app/220200/discussions/0/3044985412465032716/)). [PhysicsRangeExtender](https://github.com/jrodrigv/PhysicsRangeExtender) allows for better simulation (e.g. thusting maneuvers) of more distant spacecraft.

1. Clone PhysicsRangeExtender (assumed to be cloned to Desktop in these instructions, but you can put it wherever you like since you will be copying things from the clone to `GameData`)
2. Copy necessary subfolder from PhysicsRangeExtender to your KSP install's `GameData` folder

```bash
# clone PhysicsRange Extender locally
cd ~/Desktop
git clone git@github.com:jrodrigv/PhysicsRangeExtender.git

# copy the necessary game data for the mod into your KSP install
mkdir ~/Desktop/KSP_osx/GameData/PhysicsRangeExtender
cp -r ~/Desktop/PhysicsRangeExtender/PhysicsRangeExtender/Distribution/GameData/PhysicsRangeExtender/* ~/Desktop/KSP_osx/GameData/PhysicsRangeExtender/
```

### Install `kspdg`

If you have not yet done so, clone this repository locally on your machine

```bash
git clone git@github.com:mit-ll/spacegym-kspdg.git
```

To install this package, run:

```bash
cd spacegym-kspdg
pip install -e .
```

For development of this package, we recommend using the conda environment defined in `environment.yml`. To create and activate this environment, run:

```bash
cd spacegym-kspdg
conda env create -f environment.yml
conda activate kspdg
```

> :warning: **Troubleshooting**
>
> - Note that the `kspdg` library depends upon [astropy](https://www.astropy.org/), which in turn depends upon [pyerfa](https://github.com/liberfa/pyerfa)
> - Note also that additionaly, our project includes extra dependencies beyond `kspdg`. One of them is [tiktoken](https://github.com/openai/tiktoken), which necessitates the [Rust compiler](https://www.rust-lang.org/tools/install).
> - **FOR MAC USERS with M1 chipsets:** as of this writing, [pyerfa has not fully supported M1's arm64 architecture](https://github.com/liberfa/pyerfa/issues/83)
> - This can lead to errors running `kspdg` such as
>
> ```
> (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e'))
> ```
>
> - The workaround for Mac users with M1 chipsets is described [here](https://github.com/liberfa/pyerfa/issues/83#issuecomment-1255333177). For Python 3.9, the workaround entails cloning pyerfa locally, checking out a specific version, and installing in the conda environment
>
> ```bash
> # get pyerfa source code and switch to specific release of pyerfa
> git clone --recursive https://github.com/liberfa/pyerfa/
> cd pyerfa
> git fetch origin
> git checkout v2.0.0.1
>
> # install specific version of pyerfa in conda environment
> conda activate kspdg
> pip install .
> ```
### Install Luna Multiplayer (LMP)

_Future Work_

### Verify Installation

**NOTE:** Because the KSPDG environments are inexorably linked to the KSP game engine, many of the library's unit/integration test can only be run when a particular game mission file has been loaded and running. This means that verifying installation and testing during code development is a bit more involved than just a single `pytest` call

**Serverless Tests:** Quick test to run without KSP game engine running nor kRPC server connection

```bash
cd spacegym-kspdg
conda activate kspdg
pytest tests/serverless_tests/
```

**KSP In-Game Tests:** These tests require the KSP game engine to be running, the test-specific mission to be loaded, and a connection to the kRPC server

1. Start KSP game application.
2. Select `Start Game` > `Play Missions` > `Community Created` > `pe1_i3` > `Continue`
3. In kRPC dialog box click `Add server`. Select `Show advanced settings` and select `Auto-accept new clients`. Then select `Start Server`
4. In a bash terminal:

```bash
cd spacegym-kspdg
conda activate kspdg
pytest tests/ksp_ingame_tests/test_pe1_e1_i3.py

# for additional tests, load a different mission in KSP:
# ESC > Quit to Main Menu > Exit to Main Menu > Play Missions > `lbg1_i2` > Continue
pytest tests/ksp_ingame_tests/test_lbg1_lg0_i2.py

# ESC > Quit to Main Menu > Exit to Main Menu > Play Missions > `sb1_i5` > Continue
pytest tests/ksp_ingame_tests/test_sb1_e1_i5.py
```

5. You should see the KSP game reset and focus on a vehicle that then performs several orientation and propulsive maneuvers. The pytest command should then indicate the number of passed tests.

> :warning: **Troubleshooting**
> If you are using a Mac with an arm64 architecture (e.g. M1 chipset) and recieve an error like `(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e'))`, please refer to instructions in the [kspdg library installation section](#install-kspdg) about installing `pyerfa` from source.

---

## Example: Few-Shot Agent Example

[TO BE DONE]

## Example: Fine-Tuned Agent Example

There is a specific folder for the fine-tuned agents that you can find in [here](https://github.com/ARCLab-MIT/kspdg/tree/Alejandro/arclab_mit/agents/fine-tuned_agent).

All these agents depend on GPT models as well as some parameters, these parameters are:

| Parameter | Description |
|-----------|-------------|
| `USE_RELATIVE_COORDINATES` | This parameter will be used to calculate relative velocity and position. |
| `USE_SHORT_ARGUMENT_NAMES` | This parameter will be used to save or communicate the observations data using abbreviated names for time, fuel usage, etc. |
| `OPENAI_API_KEY` | This parameter is mandatory for establishing a connection with OpenAI API. |
| `SCENARIO` | This will determine the scenario you will be running the agent in. |
| `MODEL` | Given the user's API key there can be multiple models with their unique ID; this parameter represents this ID. |
| `IGNORE_TIME` | If you don't want to keep the time in the logs and prompts. |
| `SLIDING_WINDOW_SIZE` | The size of the sliding window that will be used by the model. |
| `SLIDING_WINDOW_STRIDE` | The stride of the sliding window (skips) for each call. |
| `SYSTEM_PROMPTS` | All of the different system prompts for each strategy or scenario. |
| `CHAIN_OF_THOUGHT` | If you want the model to reason with Chain of Thought, this would be added in the system prompt. |


Once all the parameters are set, the agent runs as any other agent via kRPC. Select the desired mission in the game and run the script having the kRPC server open.

```bash
conda activate kspdg
cd arclab_mit/agents/fine-tuned_agent
python3 fine_tuning_agent.py
```

---

## Example: Agent-Environment Evaluator

This example walks through how to evaluate agents for scoring purpose. Due to the GUI-based interface of KSP that requires manual interaction, there is not a straight-forward process for hosting a centralized evaluation server upon which participants can submit agents. Instead there is a decentralized process where agents are evaluated locally on particapants' own computers. Participants will then upload their evaluation results to a centralized scoreboard where they will be authenticated and ranked against other participants.

> :warning: **Honor System**
> While we have taken some steps to prevent falsification of results, there is still potential for cheating.
> Therefore this competition works largely on the **honor system**.
>
> _If you think you are doing something inappropriate or unfair, you probably are._
>
> **We reserve the right to disqualify teams for unsporting behavior**

The agent evaluation process uses a compiled python script located in `scripts/evaluate.cpython-39.pyc` with input arguments in the ordering: `<agent-config-file> <environment-module> <environment-class>`.

See [`evaluation/configs/example_eval_cfg.yaml``](evaluation/configs/example_eval_cfg.yaml) for an example of the config file

Here is a basic example for running an agent-environment evaluation. As with other examples, you begin by:

1. Start KSP game application.
2. Select `Start Game` > `Play Missions` > `Community Created` > `pe1_i3` > `Continue`
3. In kRPC dialog box click `Add server`. Select `Show advanced settings` and select `Auto-accept new clients`. Then select `Start Server`

```bash
conda activate kspdg # while it is not strictly necessary to use conda environments, it is encouraged for development and debugging purpose
cd evaluation # working directory is important due to relative path in cfg.yaml
python evaluate.cpython-312.pyc configs/example_eval_cfg.yaml   # assuming your conda env has python 3.12
                                                                # ohterwise call evaluate.cpython-39.pyc for python 3.9
```

This should output to a file in the `results/` subdirectory with a name like `kspdg_results_20231018_125336.txt`. That file has JSON-formatted results that look like

```
{
    "agent_env_results": {
        "is_episode_done": true,
        "closest_approach": 235.2028250841451,
        "closest_approach_time": 200.80000000002892,
        "closest_approach_speed": 77.87944143686991,
        "closest_approach_pursuer_fuel_usage": 651.595703125,
        "pursuer_fuel_usage": 782.56884765625,
        "evader_fuel_usage": 0.0,
        "weighted_score": 983.3304428262093,
        "expected_deltav_at_final_time": 47.97165399572631
    },
    "user_id": "Team Baselines",
    "user_key": "b1bb536a-fe95-4dea-8564-4c8305ac963a",
    "kspdg_version": "0.0.23",
    "agent_name": "Naive-Ned",
    "scenario_environment": "PE1_E1_I3_V1"
}
1313515906
```

This results file will then be sent to the authentication and scoreboard server for official ranking in the KSPDG Challenge. 

_INSTRUCTIONS FOR UPLOADING RESULTS TO COMPETITION SCOREBOARD HAVE BEEN EMAILED TO PARTICIPANTS_

---

## Cautionary Notes

Here are some things you should NOT do as they will break/invalidate the proper functioning of the KSPDG environments

- Do not manually switch focus between different spacecraft in the KSP game engine while evaluating an agent. This can cause silent errors with the scipted agent's policies
- Do not save-over the mission save files. If you do so inadvertantly, you will need to re-download the orignal from this repo and copy these back into your local KSP game directory

---

## References

Throughout the documentation and code comments we refer to aerospace literature such as "Vallado Chp 3" for brevity. However this assumes anyone reading this code knows that "Vallado" is short hand for David Vallado's "Fundamentals of Astrodynamics and Applications", which is an unfair assumption to make. Here we list some of our short-hand references

- Vallado, David A. Fundamentals of astrodynamics and applications. Vol. 12. Springer Science & Business Media, 2001.
  - short hands: "Vallado"
  - There are multiple editions with slightly different section layouts. We try to specify which edition when referencing specific figures/sections but mostly pulling from 3rd or 4th edition
- Bate, R. R., Mueller, D. D., & White, J. E. (1971). Fundamentals of astrodynamics. Fundamentals of astrodynamics.
  - short hands: "BMW", "Bate, Mueller, White"

---
