# Dynamic Trajectory Planning of a Robotic Arm using Deep Reinforcement Learning

## Introduction
This project focuses on developing a dynamic trajectory planning system for a robotic arm using Deep Reinforcement Learning (DRL). The objective is to enable the robotic arm to plan and execute trajectories in real-time to achieve dynamic object grasping.


## Table of Contents
- [Getting Started](getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Installation](#installation)
- [File Structure](#file-structure)
- [User Guide](#user-guide)
  - [Running the Project](#running-the-project)
  - [Training Video](training-video)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Models and Training](#models-and-training)
- [Experiments and Results](#experiments-and-results)
- [Demo](#demo)

- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Getting Started

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

### Installation

Create a catkin workspace
```sh
$ mkdir catkin_ws/src
$ cd catkin_ws
$ catkin_make
```

To automatically source this workspace every time a new shell is launched, run these commands
```
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
Clone the repository in the `src` folder in the catkin workspace
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning.git
```

Navigate back to workspace folder and build the packages
```
$ cd ~/catkin_ws
$ caktin_make
```


### File Structure

```
catkin_ws/src/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning-master
│
├── requirements.txt         # List of project dependencies
├── README.md                # Project documentation
├── kinova-ros/                  # Trained model checkpoints
│   ├── kinova_{all_files_created_by_k
│   ├── kinova_scripts/
|       ├──launch/
|          ├──position_control.launch  #Launches position control of kinova arm
|          ├──velocity_control.launch  #Launches velocity control of kinova arm
|       ├──src/
|          ├──runs/                    # contains all the training result files which was done
|          ├──ppo_controller_gpu.py    # main training file
|       ├──CMakeLists.txt
|       ├──package.xml

```

## User Guide

### Running the Project
```
$ cd ~/catkin_ws/src/kinova-ros/kinova_scripts/src/
$ python ppo_controller_gpu.py
````

This is main python file to launch gazebo environment of kinova arm , launch all necessary controllers, establish ros communication, start gym environment, setup PPO algorithm and then it will start training

### Training Video

https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning/assets/82582574/7c72c9ec-55e6-4e6b-bf67-a21a27a9c69c

### Acknowledgement
1. Developers of [kinova-ros](https://github.com/Kinovarobotics/kinova-ros)
   
### Contact
For questions or collaboration, please contact devgoti1683@gmail.com
