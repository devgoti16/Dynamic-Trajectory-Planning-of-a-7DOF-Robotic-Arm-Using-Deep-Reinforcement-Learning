#!/home/dev/anaconda3/envs/myrosenv/bin/python

import rospy
import gym
import threading
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
import os
#import math
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from  stable_baselines.common.vec_env import DummyVecEnv
from tqdm import tqdm
from torch.distributions import MultivariateNormal
from collections import namedtuple,deque

torch.autograd.set_detect_anomaly(True)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action,log_probs, reward, next_state, done):
        self.buffer.append((state, action, log_probs, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    

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
        self.states_lock = threading.Lock()
        self.callback_thread = threading.Thread(target = self.run_callback)
        self.callback_thread.daemon = True
        self.callback_thread.start()

    def run_callback(self):
        rospy.spin()

    def joint_state_callback(self,msg):
        angles = msg.position[:7]
        velocities = msg.velocity[:7]
        # self.states = angles + velocities
        with self.states_lock :
            self.states = angles + velocities
        #print("callback details:",self.states)

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
        #print("current state taken from topic:", self.states) # Update the goal position continuously
        end_effector_position = self.compute_position(self.states[:7])
        next_state = np.concatenate((self.states, end_effector_position))
        distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)     
        reward = self.calculate_reward(distance_to_goal)
        done = distance_to_goal < 0.05  # Close enough to goal
        if done:
            reward += 50  # Large reward for reaching the goal
        # print("reward : ",reward)
        # print("next state: ",next_state)
        # print("distance to goal:",distance_to_goal)
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
        #print("current state taken from topic:", current_state)
        end_effector_position = self.compute_position(self.states[:7])
        return np.concatenate((current_state, end_effector_position))
    


class ActorNetwork(nn.Module):
    def __init__(self,n_actions, state_dim,fc1_dims = 256, fc2_dims = 128, chkpt_dir = 'tmp/ppo'):
        super(ActorNetwork,self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dims).to(torch.float32)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims).to(torch.float32)
        self.mean = nn.Linear(fc2_dims, n_actions).to(torch.float32)
        self.log_std = nn.Parameter(torch.zeros(n_actions, dtype=torch.float32))


    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.log_std.exp() #.expand_as(mean)
        return mean, std



class CriticNetwork(nn.Module):
    def __init__(self, state_dim,fc1_dims = 256, fc2_dims = 128, chkpt_dir = 'tmp/ppo'):
        super(CriticNetwork,self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dims).to(torch.float32)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims).to(torch.float32)
        self.value = nn.Linear(fc2_dims, 1).to(torch.float32)

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
        self.replay_buffer = ReplayBuffer(10000)

    # def learn(self):
    #     for _ in range(self.n_epochs):
    #         state_arr
    def save_models(self, path='models'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor_network.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic_network.state_dict(), os.path.join(path, 'critic.pth'))

    def load_models(self, path='models'):
        self.actor_network.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.critic_network.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))


    def select_action(self,state):
        #print("state :", state)
        with torch.no_grad():
            state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
            mean, std = self.actor_network(state)
        #print("mean : ", mean, " & state : ", std)
        cov_matrix = torch.diag(std**2) 
        #print("covariance matrix :",cov_matrix)
        dist = MultivariateNormal(mean,covariance_matrix=cov_matrix)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)   #.sum(dim=1)
        return action.detach().numpy()[0],action_log_prob.detach()
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (~dones[step]) - values[step]
            gae = delta + self.gamma * self.lmbda * (~dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.epochs):
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, old_log_probs,rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32)
            old_log_probs = torch.tensor(old_log_probs,dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # Compute advantages
            with torch.no_grad():
                values = self.critic_network(states)
                next_values = self.critic_network(next_states)
                delta = rewards + self.gamma * next_values * (1 - dones) - values
                advantages = self.compute_gae(delta, dones)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute actor loss
            mean, std = self.actor_network(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute critic loss
            critic_loss = nn.MSELoss()(self.critic_network(states), rewards + self.gamma * next_values * (1 - dones))

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=0.5)
            self.critic_optimizer.step()

    def compute_gae(self, delta, dones):
        advantages = []
        gae = 0
        for δ, done in zip(reversed(delta), reversed(dones)):
            gae = δ + self.gamma * self.lmbda * gae * (1 - done)
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

def evaluate(agent, env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.select_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)



if __name__ == '__main__':

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    env = Jaco2Env()
    print("observation space dimension :",env.observation_space.shape[0])
    print("action space dimension :",env.action_space.shape[0])
    agent = Agent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    time.sleep(10)
    actor_lr = 0.001
    critic_lr = 0.001
    batch_size = 32
    TIME_DELTA = 0


    # #env = Dumm
    # # env.reset()
    n_episodes = 200
    max_timesteps = 200
    eval_interval = 10
    episode_rewards = []

    # agent.load_models()
    best_score = 0
    print("Training begins")
    for i in range(n_episodes):
        print("Epsiode :", i)
        observation = env.reset()
        done = False
        score = 0

        trajectories = []
        print("............Environment resetted..............")
        for t in range(max_timesteps):
            print(f"  Step: {t}")
            # print("observation :", observation)
            action,log_prob = agent.select_action(observation)
            print(f"  Action selected: {action}")
            next_observation, reward, done, info = env.step(action)
            print(f"  Reward: {reward}")
            score += reward
            # print("action :",action)
            agent.replay_buffer.push(observation, action,log_prob, reward, next_observation, done)
            observation = next_observation

            if done :
                print("Episodic task completed early")
                break
        print(f"Total for {i} episode is {score}")

        episode_rewards.append(score)
        print("Learning.....")
        # trajectories = torch.FloatTensor(np.array(trajectories))
        agent.learn()
        print("Learning finished for ", i , "episode")

        # print(f"Episode {i} finished. Reward: {episode_reward}")

        if (i + 1) % eval_interval == 0:
            mean_reward, std_reward = evaluate(agent, env)
            print(f"Evaluation after {i+1} episodes: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(episode_rewards)
    # Save the model after training
    agent.save_models()

    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('episode_rewards.png')
    plt.show()

    # Final evaluation
    mean_reward, std_reward = evaluate(agent, env, num_episodes=50)
    print(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


        # if(i%10):
        #     agent.save_models()
        # print(f"Episode : {i}, score : {score}")
    