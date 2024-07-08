# Dynamic Trajectory Planning of a 7DOF Robotic Arm using Deep Reinforcement Learning

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
  - [Features of the code](#features-of-the-code)
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

Note : When building ROS packages, it's important to install dependencies in the ROS environment and the development environment (such as a Conda environment). Here's how you can manage dependencies for building ROS packages:

ROS Dependencies:Install ROS dependencies using ROS package manager (apt on Ubuntu) or from source if required by the package. Use ROS commands like rosdep to install system dependencies needed for building and running ROS packages.
For example:
```
sudo apt install ros-noetic-<package-name>
rosdep install --from-paths <path-to-your-ros-package> --ignore-src -r -y
```

Python Dependencies:
If your ROS package includes Python scripts or nodes, manage Python dependencies within a Conda environment. This ensures that Python dependencies are isolated and do not conflict with system-wide Python installations.
Create a Conda environment specifically for your ROS project and install Python dependencies using Conda or pip within this environment

Install Tenrflow, Tensorboard, PyTorch, Cuda in  conda environment (which is explained in next section)

### Setting up the environment

```
conda create -n myrosenv python=3.7
conda activate myrosenv
pip install -r requirements.txt
```
`requiremet.txt` files contains all the python dependencies that is needed in conda environment

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

When you run this code, it will initiate a comprehensive training process for a reinforcement learning agent designed to control a 7-DOF (Degree of Freedom) Jaco2 robotic arm in a simulated environment. The process begins by setting up the necessary infrastructure. It initializes a ROS node and launches the Gazebo simulator with a Jaco2 arm model. The code then establishes ROS publishers, subscribers, and services required for robot control, creating a bridge between the learning algorithm and the simulated robot. Next, the script sets up the reinforcement learning components. It creates a custom Gym environment tailored for the Jaco2 arm, which will serve as the interface for the agent to interact with the simulated robot. The code then initializes an Actor-Critic network architecture for the PPO (Proximal Policy Optimization) algorithm, which will be used to train the agent. This includes setting up the neural networks for both the actor (policy) and critic (value function) components of the algorithm. The core of the script is the training loop, which runs for a specified number of episodes. Each episode begins by resetting the environment to a starting state. The agent then interacts with the environment for a maximum number of timesteps per episode. During each timestep, the agent selects actions based on its current policy, executes these actions in the simulated environment, and collects the resulting rewards and observations. This trajectory data is stored for later use in updating the agent's policy.

After each episode, the script updates the agent using the PPO algorithm. This process involves using the collected trajectory data to improve the agent's policy and value function estimates. The code implements multiple epochs of training on mini-batches of the collected data, which is a key feature of the PPO algorithm. Throughout the training process, the script logs various metrics to track the agent's progress. It records training metrics such as episode rewards, episode lengths, and loss values for both the actor and critic networks. These metrics are logged using TensorBoard, allowing for easy visualization of the training progress. Additionally, the code periodically evaluates the agent's performance by running a separate set of evaluation episodes. The results of these evaluations are also logged. To ensure that progress is not lost in case of unexpected interruptions, the script implements a checkpointing system. It saves checkpoints of the model every 10 episodes, allowing training to be resumed from these points if necessary. The code also keeps track of the best-performing model based on evaluation results and saves this model separately. After the specified number of training episodes is complete, the script performs a final, more comprehensive evaluation of the agent's performance. This involves running the agent through a larger number of episodes to get a more statistically significant measure of its capabilities. Finally, the code saves the trained model and closes the TensorBoard writer. Throughout the entire process, it prints various information to the console, including episode rewards, evaluation results, and any errors encountered. All of this information, along with detailed metrics, is also logged to a text file for later analysis.

It's important to note that this script is designed to run in a ROS environment with Gazebo, so proper setup of these systems is crucial for the code to function correctly. The script also uses PyTorch for implementing the neural networks and reinforcement learning algorithm, so this library needs to be installed and configured properly.


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

### Features of the code

1. Integration of ROS and Gazebo with Reinforcement Learning: The code seamlessly integrates Robot Operating System (ROS) and Gazebo simulator with a reinforcement learning framework. This allows for training a RL agent on a highly realistic robotic simulation, bridging the gap between machine learning and robotics.

2. Dynamic Goal Updating: The environment features a dynamic goal-setting mechanism. The goal position is slightly perturbed in each step, adding an extra layer of complexity and realism to the task.

3. Comprehensive Reward Function: The reward calculation is highly detailed, considering factors such as distance to goal, progress towards the goal, action smoothness, energy consumption, joint limit violations, and velocity limit violations. This multi-faceted approach encourages the agent to learn a more nuanced and realistic control policy.

4. Robust Error Handling and Checkpointing: The script includes mechanisms to handle potential errors during the learning process and implements a regular checkpointing system. This allows for recovery from unexpected interruptions and provides a way to resume training from specific points.

5. Extensive Logging and Visualization: The code uses both TensorBoard and custom logging to track a wide variety of metrics throughout the training process. This includes detailed per-episode logs and periodic evaluation results.

6. Gradient Clipping: To prevent exploding gradients, the code implements gradient clipping for both the actor and critic networks during the learning process.

7. Entropy Regularization: The PPO implementation includes an entropy term in the loss function, which can help encourage exploration and prevent premature convergence to suboptimal policies.
   
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
