# Dynamic Trajectory Planning of a Robotic Arm using Deep Reinforcement Learning

## Description
This project focuses on developing a dynamic trajectory planning system for a robotic arm using Deep Reinforcement Learning (DRL). The objective is to enable the robotic arm to plan and execute trajectories in real-time to achieve dynamic object grasping.

## Table of Contents
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Setting Up the Environment](#setting-up-the-environment)
- [Usage](#usage)
  - [Running the Project](#running-the-project)
  - [Important Notes](#important-notes)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Models and Training](#models-and-training)
- [Experiments and Results](#experiments-and-results)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation

### Prerequisites
- Python 3.7
- ROS Noetic
- TensorFlow 1.15.0
- PyTorch 1.13.1
- Other dependencies listed in `requirements.txt`

### Setting up the environment
conda create -n myrosenv python=3.7
conda activate myrosenv
pip install -r requirements.txt

### Cloning the Repository
```sh
$ mkdir catkin_ws/src
$ cd catkin_ws
$ catkin_make
$ cd src
$ git clone https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning.git
$ cd ..
$ caktin_make
```

### Setting up the environment
# Example for setting up a virtual environment
conda create -n drl_robotic_arm python=3.8
conda activate drl_robotic_arm
pip install -r requirements.txt

### Implementation

python ppo_controller_gpu.py

### Contact
For questions or collaboration, please contact devgoti1683@gmail.com
