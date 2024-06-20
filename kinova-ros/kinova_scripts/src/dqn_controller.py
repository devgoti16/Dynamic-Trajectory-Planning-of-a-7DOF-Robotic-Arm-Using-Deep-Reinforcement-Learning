#!/home/dev/anaconda3/envs/myrosenv/bin/python

import rospy
#import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
#from gym import spaces
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import time
from std_srvs.srv import Empty
#import torch
# import torch.nn as nn
# import torch.nn.functional as F
#from numpy import inf
import subprocess
from os import path
#import math

from jacogazebo import Jaco2Env



# def evaluate(network, epoch, eval_episodes = 20):
#     avg_reward = 0.0
#     col = 0
#     for _ in range(eval_episodes):

# class ReplayBuffer(object):
#     def __init__(self, buffer_size, random_seed = 123)
#         self.buffer_size = buffer_size
#         self.count = 0
#         self.buffer = deque(maxlen = buffer_size)
#         random.seed(random_seed)

#     def add(self,s,a,r,s2,done):
#         experience = (s,a,r,s2,done)
#         if self.count < self.buffer_size:

#             self.buffer.append(experience)
#             self.count += 1

#         else :
#             self.buffer.popleft()
#             self.buffer.append(experience)
        
#     def sample(self,batch_size):
#         batch = []
#         if self.count < batch_size:
#             batch = random.sample(self.buffer, self.count)
#         else :
#             batch = random.sample(self.buffer, batch_size)

        
        
#     def size(self):
#         return self.count
    
#     def clear(self):
#         self.buffer.clear()
#         self.count = 0



# class Jaco2Env():
#     def __init__(self):
#         port = "11311"
#         self.launfile = "velocity_control.launch"
#         subprocess.Popen(["roscore","-p",port])
#         print("roscore launched!")
#         rospy.init_node('dqn_controller')
#         subprocess.Popen(["roslaunch","-p", "11311","kinova_scripts","velocity_control.launch"])
#         print("Gazebo Launched")
#         self.states = np.zeros(14)
#         self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
#         self.joint_vel_pub1 = rospy.Publisher('/j2s7s300/joint_1_velocity_controller/command', Float64, queue_size=1)
#         self.joint_vel_pub2 = rospy.Publisher('/j2s7s300/joint_2_velocity_controller/command', Float64, queue_size=1)
#         self.joint_vel_pub3 = rospy.Publisher('/j2s7s300/joint_3_velocity_controller/command', Float64, queue_size=1)     
#         self.joint_vel_pub4 = rospy.Publisher('/j2s7s300/joint_4_velocity_controller/command', Float64, queue_size=1)     
#         self.joint_vel_pub5 = rospy.Publisher('/j2s7s300/joint_5_velocity_controller/command', Float64, queue_size=1)     
#         self.joint_vel_pub6 = rospy.Publisher('/j2s7s300/joint_6_velocity_controller/command', Float64, queue_size=1)     
#         self.joint_vel_pub7 = rospy.Publisher('/j2s7s300/joint_7_velocity_controller/command', Float64, queue_size=1)             
#         self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics",Empty)
#         self.pause = rospy.ServiceProxy("/gazebo/pause_physics",Empty)
#         self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation",Empty)
#         # rospy.spin()
#         print("all service and topics called")
#         self.d_parameters = [0.2755,0.2050,0.2050,0.2073,0.1038,0.1038,0.1600,0.0098] #d1,d2,d3,d4,d5,d6,d7,e2  


#     def joint_state_callback(self,msg):
#         angles = msg.position
#         velocities = msg.velocity
#         self.states = np.concatenate(((np.array(angles))[:7]), (np.array(velocities)[:7]))
#         rospy.loginfo("callback data : ",self.states)

#     def update_goal_position(self):
#         self.goal_position += np.random.uniform(low=-0.05, high=0.05, size=3)  # Random continuous motion
#         # goal_msg = Pose()
#         # goal_msg.position.x, goal_msg.position.y, goal_msg.position.z = self.goal_position
#         # self.goal_pub.publish(goal_msg)
#         # self.state[14:] = self.goal_position

#     def calculate_reward(self, distance):
#         distance_reward = -distance
#         exp_distance_penalty = -np.exp(distance)
#         reward = distance_reward + exp_distance_penalty
#         return reward
    
#     def forward_kinematics(self, joint_angles):
#         # Placeholder for forward kinematics calculation
#         # This function should return the current end-effector position
#         # based on the provided joint angles.

#         dh_parameters = [
#             (np.radians(90), 0, -self.d_parameters[0], joint_angles[0]),
#             (np.radians(90), 0, 0,  joint_angles[1]),
#             (np.radians(90), 0, -(self.d_parameters[1]+self.d_parameters[2]),  joint_angles[2]),
#             (np.radians(90), 0, -self.d_parameters[7],  joint_angles[3]),
#             (np.radians(90), 0, -(self.d_parameters[3]+self.d_parameters[4]), joint_angles[4]),
#             (np.radians(90), 0, 0,  joint_angles[5]),
#             (np.radians(180), 0, -(self.d_parameters[5]+self.d_parameters[6]),  joint_angles[6])
#         ] #alpha,d,a,thetea
#         T_0_n = np.eye(4)
#         # print(dh_parameters)
#         #transformation = []
#         for (alpha,d, a, theta) in dh_parameters:
#             print(alpha,d,a,theta)
#             T_i = self.dh_transformation(alpha, d, a, theta)
#             T_0_n = np.dot(T_0_n, T_i)
#             # transformation.append(T_0_n)
#         return T_0_n
    
#     def dh_transformation(self,alpha, d, a, theta):
#         return np.array([
#             [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
#             [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
#             [0, np.sin(alpha), np.cos(alpha), d],
#             [0, 0, 0, 1]
#         ])
 
#     def compute_position(self,state):
#         # Ensure the current joint states are available
#         if state is None:
#             return np.zeros(len(7))
#         joint_angles = np.array(state)
#         T__0 = self.forward_kinematics(joint_angles)
#         current_position = T__0[:3,3]
#         return current_position
    
    
#     def step(self, action): #perform an action and read a new state
#         joint_vel_msg_1 = Float64()
#         joint_vel_msg_2 = Float64()
#         joint_vel_msg_3 = Float64()
#         joint_vel_msg_4 = Float64()
#         joint_vel_msg_5 = Float64()
#         joint_vel_msg_6 = Float64()
#         joint_vel_msg_7 = Float64()
#         joint_vel_msg_1.data = action[0]
#         joint_vel_msg_2.data = action[1] 
#         joint_vel_msg_3.data = action[2] 
#         joint_vel_msg_4.data = action[3] 
#         joint_vel_msg_5.data = action[4] 
#         joint_vel_msg_6.data = action[5] 
#         joint_vel_msg_7.data = action[6]  
#         self.joint_vel_pub1.publish(joint_vel_msg_1)
#         self.joint_vel_pub2.publish(joint_vel_msg_2)
#         self.joint_vel_pub3.publish(joint_vel_msg_3)
#         self.joint_vel_pub4.publish(joint_vel_msg_4)
#         self.joint_vel_pub5.publish(joint_vel_msg_5)
#         self.joint_vel_pub6.publish(joint_vel_msg_6)
#         self.joint_vel_pub7.publish(joint_vel_msg_7)
#         rospy.wait_for_service("/gazebo/unpause_physics")
#         try:
#             self.unpause()
#         except (rospy.ServiceException) as e:
#             print("/gazebo/unpause_physics service call failed")            
#         # time.sleep(TIME_DELTA)
#         rate.sleep()
#         rospy.wait_for_service("/gazebo/pause_physics")
#         try:
#             self.pause()
#         except (rospy.ServiceException) as e:
#             print("/gazebo/pause_physics service call failed")
#         #self.update_goal_position()  # Update the goal position continuously
#         end_effector_position = self.compute_position(self.states[:7])
#         next_state = np.concatenate((self.states, end_effector_position))
#         distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)     
#         reward = self.calculate_reward(distance_to_goal)
#         done = distance_to_goal < 0.05  # Close enough to goal
#         if done:
#             reward += 50  # Large reward for reaching the goal
#         print("reward : ",reward)
#         print("next state: ",next_state)
#         return next_state, reward, done, {}

#     def reset(self):
#         rospy.wait_for_service("/gazebo/reset_simulation")
#         try :
#             self.reset_proxy()
#         except rospy.ServiceException as e:
#             print("/gazebo/reset_simulation service call failed")
#         rospy.sleep(1)  # Allow time to reset
#         #self.update_goal_position()  # Publish initial goal position
#         # Initialize goal position at a random position within the workspace
#         self.goal_position = np.random.uniform(low=-3, high=3, size=3)
#         print("goal position :" , self.goal_position)
#         rospy.wait_for_service("/gazebo/unpause_physics")
#         try:
#             self.unpause()
#         except (rospy.ServiceException) as e:
#             print("/gazebo/unpause_physics service call failed")   
#         # time.sleep(TIME_DELTA)
#         rate.sleep()
#         rospy.wait_for_service("/gazebo/pause_physics")
#         try:
#             self.pause()
#         except (rospy.ServiceException) as e:
#             print("/gazebo/pause_physics service call failed")
#         current_state = self.states
#         end_effector_position = self.compute_position(self.states[:7])
#         return np.concatenate((current_state, end_effector_position))


class DQNAgent:
    def __init__(self):
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=17, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(7, activation='tanh'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("random action taken")
            return np.random.uniform(-1,1,size=7)
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state= tf.expand_dims(state_tensor, axis=0)
        act_values = agent.model.predict(state)
        return act_values.detach().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action + 3] = target  # Adjust action index for the range shift
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
# episodes = 2000
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

if __name__ == '__main__':
        
    try:

        # while not rospy.is_shutdown():

        
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        episodes = 2000
        TIME_DELTA = 0.1
        batch_size = 32
        max_timesteps = 500
        scores = []
        buffer_size = 10000
        save_model = True
        load_model = False
        seed = 0
        state_size = 17
        action_size = 7
        agent = DQNAgent()
        env = Jaco2Env()
        time.sleep(15)
        rate = rospy.Rate(10)
        #tf.set_random_seed(seed)
        np.random.seed(seed)

        print("going for training")

        # replay_buffer = ReplayBuffer(buffer_size,seed)
        for e in range(episodes):
            done = False
            print("epsiode :", e)
            state = env.reset()
            print(" environment has been reseted")            
            for times in range(max_timesteps):
                print("step :",times)
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, 17])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    agent.update_target_model()
                    scores.append(times)
                    print(f"episode: {e}/{episodes}, score: {times}, e: {agent.epsilon:.2}")
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            print("maximum timesteps achieved")

        # plt.plot(scores)
        # plt.xlabel('Episode')
        # plt.ylabel('Score')
        # plt.show()
        # rospy.spin()

    except rospy.ROSInterruptException:
            rospy.loginfo("Node closed")
    # except Exception as e :
    #     rospy.logerr("An error occurred : %s", e)
    # finally :
    #     rospy.signal_shutdown("Node terminated due to error")