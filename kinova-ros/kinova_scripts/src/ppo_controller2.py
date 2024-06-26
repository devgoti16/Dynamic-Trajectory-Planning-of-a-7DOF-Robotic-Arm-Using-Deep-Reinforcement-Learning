#!/home/dev/anaconda3/envs/myrosenv/bin/python

import rospy
import gym
import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
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
import warnings
from torch.distributions import Categorical
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


    def joint_state_callback(self,msg):
        angles = msg.position[:7]
        velocities = msg.velocity[:7]
        self.states = angles + velocities

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
        self.update_goal_position()  # Update the goal position continuously
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
        end_effector_position = self.compute_position(self.states[:7])
        return np.concatenate((current_state, end_effector_position))
    


# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,  action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                    )

        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)


    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()


    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
def train():

    ################TRAINING PARAMETERS##################
    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################



    env = Jaco2Env()

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update()

            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % log_freq == 0:

                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)


                log_running_reward = 0
                log_running_episodes = 0

            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    env.close()


if __name__ == '__main__':

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    env = Jaco2Env()
    print("observation space dimension ::",env.observation_space.shape[0])
    print("action space dimension ::",env.action_space.shape[0])
    train()

    