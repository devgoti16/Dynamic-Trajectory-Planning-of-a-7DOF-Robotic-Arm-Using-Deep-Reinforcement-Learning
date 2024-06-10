#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from geometry_msgs.msg import Pose

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))                     
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the ROS node
class DQNTrajectoryPlanner:
    def __init__(self):
        rospy.init_node('dqn_trajectory_planner')
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        self.joint_velocity_pub = rospy.Publisher('/joint_velocity_controller', Float64MultiArray, queue_size=10)

        self.state_size = 7 * 2 + 3  # 7 joint angles, 7 joint velocities, 3 end effector positions
        self.action_size = 7  # 7 joint velocities
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        self.current_state = None
        self.joint_states = None
        self.end_effector_position = np.zeros(3)
        self.goal_position = np.random.uniform(-1, 1, size=3)

        self.episode = 0
        self.max_episodes = 2000

    def joint_states_callback(self, msg):
        self.joint_states = msg
        self.current_state = self.get_state_from_joint_states(msg)

    def get_state_from_joint_states(self, msg):
        joint_angles = np.array(msg.position)
        joint_velocities = np.array(msg.velocity)
        state = np.concatenate([joint_angles, joint_velocities, self.end_effector_position])
        return state

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randn(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        actions = self.model(state)
        return actions.detach().numpy()[0]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, torch.FloatTensor([target]))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def reset(self):
        # Set random positions for joints
        joint_positions = np.random.uniform(-1.0, 1.0, 7)
        # Simulate setting these positions in the simulation
        # This should be replaced with actual code to reset the simulation
        # Set a random goal position
        self.goal_position = np.random.uniform(-1, 1, size=3)
        rospy.loginfo(f"Resetting environment: Goal position set to {self.goal_position}")
        self.current_state = None

    def step(self, action):
        joint_velocity_msg = Float64MultiArray()
        joint_velocity_msg.data = action
        self.joint_velocity_pub.publish(joint_velocity_msg)
        
        next_state = self.current_state  # Update with actual next state from simulation
        reward = self.compute_reward()
        done = reward > -0.01  # Define the condition for episode termination

        self.memory.append((self.current_state, action, reward, next_state, done))

        self.current_state = next_state
        return next_state, reward, done

    def compute_reward(self):
        # Reward function based on distance to goal
        distance = np.linalg.norm(self.end_effector_position - self.goal_position)
        reward = -distance  # Negative reward for distance
        return reward

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown() and self.episode < self.max_episodes:
            self.reset()
            done = False
            while not done:
                if self.current_state is not None:
                    action = self.get_action(self.current_state)
                    next_state, reward, done = self.step(action)
                    self.replay()
                    self.update_target_model()

            self.episode += 1
            rospy.loginfo(f'Episode: {self.episode}, Reward: {reward}, Epsilon: {self.epsilon}')

if __name__ == '__main__':
    try:
        planner = DQNTrajectoryPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
