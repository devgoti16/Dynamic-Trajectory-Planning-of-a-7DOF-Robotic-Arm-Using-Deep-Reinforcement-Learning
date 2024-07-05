# Dynamic Trajectory Planning of a Robotic Arm using Deep Reinforcement Learning

## Introduction
This project focuses on developing a dynamic trajectory planning system for a KINOVA JACO2 7DOF robotic arm using Deep Reinforcement Learning (DRL). The objective is to enable the robotic arm to plan and execute trajectories in real-time to achieve dynamic object grasping.


## Table of Contents
- [Getting Started](getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Installation](#installation)
- [File Structure](#file-structure)
- [User Guide](#user-guide)
  - [Running the Project](#running-the-project)
  - [Algorithm Overview](#algorithm-ovreview)
  - [Training Video](training-video)
- [Acknowledgments](#acknowledgments)
- [Contribution](#contribution)
- [License](#license)
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
├── requirements.txt                                # List of project dependencies
├── README.md                                       # Project documentation
├── kinova-ros/                                     # Trained model checkpoints
│   ├── kinova_{all_folders_of_kinova_developers}
│   ├── kinova_scripts/                             #custom folder for files related to this project
|       ├──launch/
|          ├──position_control.launch               #Launches position control of kinova arm
|          ├──velocity_control.launch               #Launches velocity control of kinova arm
|       ├──src/
|          ├──runs/                                 # contains all the training result files which was done
|          ├──ppo_controller_gpu.py                 # main training file
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

### Algorithm Overview


1. Initialize PPO Agent
2. Initialize Gazebo Environment
3. Initialize Actor Network
4. Initialize Critic Network
5. Initialize Hyperparameters (learning rates, gamma, epsilon, lambda, etc.)
6. For each episode:
   1. Reset environment
   2. While not done and within max timesteps:
      1. Select action using current policy
      2. Take action, observe next state, reward, and done flag
      3. Store transition (state, action, reward, next_state, done)
   3. Compute advantages and returns
   4. For each epoch:
      1. For each mini-batch:
         1. Compute new action probabilities
         2. Compute probability ratios
         3. Compute surrogate objectives
         4. Compute actor (policy) loss
         5. Compute critic (value) loss
         6. Compute total loss
         7. Perform backpropagation
         8. Update network parameters
   5. Evaluate agent performance periodically
   6. Save best model if performance improved
7. Final evaluation

   
### Training Video

https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning/assets/82582574/7c72c9ec-55e6-4e6b-bf67-a21a27a9c69c

### Acknowledgements
1. Developers of [kinova-ros](https://github.com/Kinovarobotics/kinova-ros)

### Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/Newfeature`)
3. Commit your changes (`git commit -m 'Add some Newfeature'`)
4. Push to the branch (`git push origin feature/Newfeature`)
5. Open a Pull Request


### License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
   
### Contact
For questions or collaboration, please contact devgoti1683@gmail.com
