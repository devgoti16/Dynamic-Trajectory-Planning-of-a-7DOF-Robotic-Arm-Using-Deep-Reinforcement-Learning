#!/home/dev/anaconda3/envs/myrosenv/bin/python

import rospy
import gym
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
from gym import spaces
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import time
from std_srvs.srv import Empty
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from numpy import inf
import subprocess
from os import path
#import math
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from  stable_baselines.common.vec_env import DummyVecEnv
from tqdm import tqdm
from torch.distributions import MultivariateNormal
from collections import namedtuple,deque
TIME_DELTA = 0.1

class Jaco2Env(gym.Env):
    def __init__(self):


        super(Jaco2Env, self).__init__()
        self.action_dim = 7
        self.obs_dim = 17

        # Define action and observation space
        # Assuming the arm has 7 joints, each with a velocity range [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float64)

        # Observation space, assuming joint angles and velocities and end effefctor coordinates as states
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)


        port = "11311"
        self.launfile = "velocity_control.launch"
        subprocess.Popen(["roscore","-p",port])
        print("roscore launched!")
        rospy.init_node('dqn_controller')
        subprocess.Popen(["roslaunch","-p", "11311","kinova_scripts","velocity_control.launch"])
        print("Gazebo Launched")
        self.states = np.zeros(14)
        self.joint_state_sub = rospy.Subscriber('j2s7s300/joint_states', JointState, self.joint_state_callback)
        self.joint_vel_pub1 = rospy.Publisher('/j2s7s300/joint_1_velocity_controller/command', Float64, queue_size=1)
        self.joint_vel_pub2 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command', Float64, queue_size=1)
        self.joint_vel_pub3 = rospy.Publisher('/j2s7s300/joint_3_velocity_controller/command', Float64, queue_size=1)     
        self.joint_vel_pub4 = rospy.Publisher('/j2s7s300/joint_4_velocity_controller/command', Float64, queue_size=1)     
        self.joint_vel_pub5 = rospy.Publisher('/j2s7s300/joint_5_velocity_controller/command', Float64, queue_size=1)     
        self.joint_vel_pub6 = rospy.Publisher('/j2s7s300/joint_6_velocity_controller/command', Float64, queue_size=1)     
        self.joint_vel_pub7 = rospy.Publisher('/j2s7s300/joint_7_velocity_controller/command', Float64, queue_size=1)             
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics",Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics",Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation",Empty)
        # rospy.spin()
        print("all service and topics called")
        self.d_parameters = [0.2755,0.2050,0.2050,0.2073,0.1038,0.1038,0.1600,0.0098] #d1,d2,d3,d4,d5,d6,d7,e2  
        self.states = None

    def joint_state_callback(self,msg):
        angles = msg.position[:7]
        velocities = msg.velocity[:7]
        self.states = angles + velocities
        print("callback details:",self.states)

    def update_goal_position(self):
        self.goal_position += np.random.uniform(low=-0.05, high=0.05, size=3)  # Random continuous motion
        # goal_msg = Pose()
        # goal_msg.position.x, goal_msg.position.y, goal_msg.position.z = self.goal_position
        # self.goal_pub.publish(goal_msg)
        # self.state[14:] = self.goal_position

    def calculate_reward(self, distance):
        distance_reward = -distance
        exp_distance_penalty = -np.exp(distance)
        reward = distance_reward + exp_distance_penalty
        return reward
    
    def forward_kinematics(self, joint_angles):
        # Placeholder for forward kinematics calculation
        # This function should return the current end-effector position
        # based on the provided joint angles.

        dh_parameters = [
            (np.radians(90), 0, -self.d_parameters[0], joint_angles[0]),
            (np.radians(90), 0, 0,  joint_angles[1]),
            (np.radians(90), 0, -(self.d_parameters[1]+self.d_parameters[2]),  joint_angles[2]),
            (np.radians(90), 0, -self.d_parameters[7],  joint_angles[3]),
            (np.radians(90), 0, -(self.d_parameters[3]+self.d_parameters[4]), joint_angles[4]),
            (np.radians(90), 0, 0,  joint_angles[5]),
            (np.radians(180), 0, -(self.d_parameters[5]+self.d_parameters[6]),  joint_angles[6])
        ] #alpha,d,a,thetea
        T_0_n = np.eye(4)
        # print(dh_parameters)
        #transformation = []
        for (alpha,d, a, theta) in dh_parameters:
            T_i = self.dh_transformation(alpha, d, a, theta)
            T_0_n = np.dot(T_0_n, T_i)
            # transformation.append(T_0_n)
        return T_0_n
    
    def dh_transformation(self,alpha, d, a, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
 
    def compute_position(self,state):
        # Ensure the current joint states are available
        if state is None:
            return np.zeros(len(7))
        joint_angles = np.array(state)
        T__0 = self.forward_kinematics(joint_angles)
        current_position = T__0[:3,3]
        return current_position
    
    
    def step(self, action): #perform an action and read a new state
        joint_vel_msg_1 = Float64()
        joint_vel_msg_2 = Float64()
        joint_vel_msg_3 = Float64()
        joint_vel_msg_4 = Float64()
        joint_vel_msg_5 = Float64()
        joint_vel_msg_6 = Float64()
        joint_vel_msg_7 = Float64()
        joint_vel_msg_1.data = action[0]
        joint_vel_msg_2.data = action[1] 
        joint_vel_msg_3.data = action[2] 
        joint_vel_msg_4.data = action[3] 
        joint_vel_msg_5.data = action[4] 
        joint_vel_msg_6.data = action[5] 
        joint_vel_msg_7.data = action[6]  
        self.joint_vel_pub1.publish(joint_vel_msg_1)
        self.joint_vel_pub2.publish(joint_vel_msg_2)
        self.joint_vel_pub3.publish(joint_vel_msg_3)
        self.joint_vel_pub4.publish(joint_vel_msg_4)
        self.joint_vel_pub5.publish(joint_vel_msg_5)
        self.joint_vel_pub6.publish(joint_vel_msg_6)
        self.joint_vel_pub7.publish(joint_vel_msg_7)
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")            
        # time.sleep(TIME_DELTA)
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        #self.update_goal_position() 
        print("current state taken from topic:", self.states) # Update the goal position continuously
        end_effector_position = self.compute_position(self.states[:7])
        next_state = np.concatenate((self.states, end_effector_position))
        distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)     
        reward = self.calculate_reward(distance_to_goal)
        done = distance_to_goal < 0.05  # Close enough to goal
        if done:
            reward += 50  # Large reward for reaching the goal
        print("reward : ",reward)
        print("next state: ",next_state)
        print("distance to goal:",distance_to_goal)
        return next_state, reward, done, {}

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_simulation")
        try :
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        rospy.sleep(1)  # Allow time to reset
        #self.update_goal_position()  # Publish initial goal position
        # Initialize goal position at a random position within the workspace
        self.goal_position = np.random.uniform(low=-3, high=3, size=3)
        print("goal position :" , self.goal_position)
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")   
        # time.sleep(TIME_DELTA)
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        current_state = self.states
        print("current state taken from topic:", current_state)
        end_effector_position = self.compute_position(self.states[:7])
        return np.concatenate((current_state, end_effector_position))
    


class ActorNetwork(nn.Module):
    def __init__(self,n_actions, state_dim,fc1_dims = 256, fc2_dims = 128, chkpt_dir = 'tmp/ppo'):
        super(ActorNetwork,self).__init__()
        self.fc1 = nn.Linear(state_dim,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.mean = nn.Linear(fc2_dims,n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions)) 


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)  #.expand_as(mean)
        return mean, std



class CriticNetwork(nn.Module):
    def __init__(self, state_dim,fc1_dims = 256, fc2_dims = 128, chkpt_dir = 'tmp/ppo'):
        super(CriticNetwork,self).__init__()
        self.fc1 = nn.Linear(state_dim,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.value  = nn.Linear(fc2_dims,1)


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value




class Agent:
    def __init__ (self, state_dim, action_dim, lr = 0.002, gamma = 0.99, eps_clip = 0.2,epsilon = 0.2,lmbda = 0.95, epoch = 10, batch_size = 64):
        self.actor_network = ActorNetwork(action_dim,state_dim)
        self.critic_network = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),lr = lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr = lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.epochs = epoch
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    # def learn(self):
    #     for _ in range(self.n_epochs):
    #         state_arr

    def select_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0)
        print("state :", state)
        mean, std = self.actor_network(state)
        print("mean : ", mean, " & state : ", std)
        cov_matrix = torch.diag(std**2) 
        print("covariance matrix :",cov_matrix)
        dist = MultivariateNormal(mean,covariance_matrix=cov_matrix)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)   #.sum(dim=1)
        return action.detach().numpy()[0],action_log_prob.detach()
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lmbda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def learn(self, trajectories):
        states, actions, log_probs, rewards, next_states, dones = zip(*trajectories)
        # print(states,actions,log_probs,rewards,next_states,dones)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.stack(log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        values = self.critic_network(states).squeeze()
        next_values = self.critic_network(next_states).squeeze()
        advantages = self.compute_advantages(rewards, values.detach().numpy(), next_values.detach().numpy(), dones)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + values

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_indices = slice(i, i + self.batch_size)
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                mean, std = self.actor_network(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                ratios = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (self.critic_network(batch_states).squeeze() - batch_returns).pow(2).mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()





if __name__ == '__main__':

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    env = Jaco2Env()
    print("observation space dimension :",env.observation_space.shape[0])
    print("action space dimension :",env.action_space.shape[0])
    agent = Agent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    time.sleep(15)
    actor_lr = 0.001
    critic_lr = 0.001
    batch_size = 64
    TIME_DELTA = 0.1


    # #env = Dumm
    # # env.reset()
    n_episodes = 200
    max_timesteps = 10

    # agent.load_models()
    best_score = 0
    print("Training begins")
    for i in range(n_episodes):
        print("Epsiode :", i)
        observation = env.reset()
        done = False
        score = 0

        trajectories = []
        print("Environment resetted")
        for t in range(max_timesteps):
            print("Step :",t)
            # print("observation :", observation)
            action,log_prob = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            print("action :",action)
            trajectories.append([observation, action,log_prob, reward, next_observation, done])
            observation = next_observation

            if done :
                
                break
            print("score :", score)
        print("Learning.....")
        agent.learn(trajectories)
        print("Learning finished for ", i , "episode")

        if(i%10):
            agent.save_models()
        print(f"Episode : {i}, score : {score}")
    