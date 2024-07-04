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

### Training Video


### File Structure

https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning/assets/82582574/7c72c9ec-55e6-4e6b-bf67-a21a27a9c69c


```
project-root/
│
├── ppo_controller_gpu.py    # Main script for PPO training on GPU
├── requirements.txt         # List of project dependencies
├── README.md                # Project documentation
│
├── models/                  # Trained model checkpoints
│   ├── model_latest.pth
│   └── model_best.pth
│
├── data/                    # Dataset and configuration files
│   ├── robot_config.yaml
│   └── environment_params.json
│
├── scripts/
│   ├── preprocess_data.py   # Data preprocessing script
│   └── evaluate_model.py    # Model evaluation script
│
├── utils/
│   ├── env_wrapper.py       # Environment wrapper for RL
│   └── custom_layers.py     # Custom neural network layers
│
└── results/                 # Output directory for logs and results
├── training_logs/
└── evaluation_results/
```

### Implementation

python ppo_controller_gpu.py

### Contact
For questions or collaboration, please contact devgoti1683@gmail.com
