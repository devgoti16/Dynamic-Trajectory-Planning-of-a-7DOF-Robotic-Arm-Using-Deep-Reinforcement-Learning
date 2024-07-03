# Dynamic Trajectory Planning of a 7DOF Robotic Arm Using Deep Reinforcement Learning

## Description
This project implements dynamic trajectory planning for a 7 Degree of Freedom (DOF) robotic arm using Deep Reinforcement Learning techniques. It aims to optimize the arm's movement path in real-time, considering various environmental constraints and task objectives.

## Supported Version
Ubuntu 20.04 and ROS Noetic

## Features
- Dynamic trajectory planning for a 7DOF robotic arm
- Deep Reinforcement Learning implementation
- Real-time optimization of movement paths
- [Add more features specific to your project]

## Installation Guide

This guide will help you set up the project environment using Conda and install all necessary dependencies.

### Prerequisites
- Ensure you have Anaconda or Miniconda installed on your system. If not, download and install from [Anaconda's official website](https://www.anaconda.com/products/distribution).

### Steps to create the environment and install dependencies

1. Open your terminal (or Anaconda Prompt on Windows).

2. Create a new Conda environment named "myrosenv":
conda create --name myrosenv python=3.7

3. Activate the newly created environment:
   conda activate myrosenv
4. Install pip inside the Conda environment (if not already installed):
   conda install pip

5. Navigate to the directory containing your `requirements.txt` file.

6. Install the required packages using pip:
   pip install -r requirements.txt

7. Verify the installation by listing the installed packages:
   pip list
### Using the environment

- To activate the environment in the future, use:
conda activate myrosenv
- To deactivate the environment when you're done, use:
conda deactivate

Note: If you encounter any issues during installation, ensure that both Conda and pip are up to date:
conda update conda
pip install --upgrade pip

8.To set up this project locally, follow these steps:

a. Clone the repository:
git clone https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning.git

b. Navigate to the project directory:
cd Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning



## Usage
[Provide instructions on how to use your project, including any command-line instructions, API usage, or example code]

## Dependencies
- tensorflow==1.15.0
- torch==1.13.1+cu116
- numpy==1.21.5
- keras==2.11.0
- gazebo_ros==2.9.2
- gazebo_plugins==2.9.2
- gym==0.26.2
- tensorboard==2.11.2



## License
[Specify the license under which this project is released]

## Contact
Dev Goti - devgoti1683@gmail.com

Project Link: [https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning](https://github.com/devgoti16/Dynamic-Trajectory-Planning-of-a-7DOF-Robotic-Arm-Using-Deep-Reinforcement-Learning)

## Acknowledgments
- KINOVA JACO2 Arm repo  : [kinova-ros](https://github.com/Kinovarobotics/kinova-ros)











