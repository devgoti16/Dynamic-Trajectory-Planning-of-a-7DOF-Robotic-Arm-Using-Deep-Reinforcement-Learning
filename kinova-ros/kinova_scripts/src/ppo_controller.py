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
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from  stable_baselines.common.vec_env import DummyVecEnv
from tqdm import tqdm
from torch.distributions import MultivariateNormal, Normal
from collections import namedtuple,deque
import matplotlib
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
import datetime 

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
        #self.states = angles + velocities
        with self.states_lock :
            self.states = angles + velocities
        #print("callback details:",self.states)

    def get_current_state(self):
        with self.states_lock:
            return self.states

    def update_goal_position(self):
        self.goal_position += np.random.uniform(low=-0.05, high=0.05, size=3)  # Random continuous motion
        # goal_msg = Pose()
        # goal_msg.position.x, goal_msg.position.y, goal_msg.position.z = self.goal_position
        # self.goal_pub.publish(goal_msg)
        # self.state[14:] = self.goal_position

    def calculate_reward(self,distance, current_position, action, previous_action=None,previous_distance=None):
        # Distance to goal
        distance_reward = -distance

        # Progress towards goal
        
        progress = previous_distance - distance
        if progress > 0:
            progress_reward = 3 * progress  # Reward for moving towards the goal
        else:
            progress_reward = -10

        # Action smoothness
        if previous_action is not None:
            action_smoothness = -0.1*np.sum(np.square(action - previous_action))
        else:
            action_smoothness = 0

        # Encourage exploration in early stages
        exploration_reward = 0

        # Penalize being close to joint limits
        # joint_limit_penalty = self.calculate_joint_limit_penalty()

        # Energy efficiency
        energy_penalty = -0.01 * np.sum(np.square(action))

        # Combine rewards
        reward = (
            3 * distance_reward +
            5 * progress_reward +
             action_smoothness +
            exploration_reward  +
            3 * energy_penalty
        )

        # Bonus for reaching the goal
        if distance < 0.5:
            reward += 100
        return reward
    
    def forward_kinematics(self, joint_angles):
        dh_parameters = [
            (np.radians(90), 0, -self.d_parameters[0], joint_angles[0]),
            (np.radians(90), 0, 0,  joint_angles[1]),
            (np.radians(90), 0, -(self.d_parameters[1]+self.d_parameters[2]),  joint_angles[2]),
            (np.radians(90), 0, -self.d_parameters[7],  joint_angles[3]),
            (np.radians(90), 0, -(self.d_parameters[3]+self.d_parameters[4]), joint_angles[4]),
            (np.radians(90), 0, 0,  joint_angles[5]),
            (np.radians(180), 0, -(self.d_parameters[5]+self.d_parameters[6]),  joint_angles[6])
        ] 
        T_0_n = np.eye(4)
        for (alpha,d, a, theta) in dh_parameters:
            T_i = self.dh_transformation(alpha, d, a, theta)
            T_0_n = np.dot(T_0_n, T_i)
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
        self.update_goal_position() 
        #print("current state taken from topic:", self.states) # Update the goal position continuously
        current_state = self.get_current_state()
        end_effector_position = self.compute_position(current_state[:7])
        next_state = np.concatenate((current_state, end_effector_position))
        distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)     
        reward = self.calculate_reward(distance_to_goal,current_state,action,self.last_action,self.last_distance)
        done = distance_to_goal < 0.05  # Close enough to goal
         # Large reward for reaching the goal
        # print("reward : ",reward)
        # print("next state: ",next_state)
        # print("distance to goal:",distance_to_goal)
        last_action = action
        last_distance = distance_to_goal
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
        current_state = self.get_current_state()
        self.last_action = 0
        self.last_distance = 0
        #print("current state taken from topic:", current_state)
        end_effector_position = self.compute_position(current_state[:7])
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
    def __init__ (self, state_dim, action_dim, lr = 3e-4, gamma = 0.99, eps_clip = 0.2,epsilon = 0.2,lmbda = 0.95, epoch = 30, batch_size = 32):
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
        #self.replay_buffer = ReplayBuffer(10000)
        self.actor_loss = 0
        self.critic_loss = 0
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)


    # def learn(self):
    #     for _ in range(self.n_epochs):
    #         state_arr
    def save_models(self, path='models'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor_network.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic_network.state_dict(), os.path.join(path, 'critic.pth'))
        print(f"Models saved to {path}")

    def load_models(self, path='models'):
        self.actor_network.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.critic_network.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        print(f"Models loaded from {path}")

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
        action = 2*(torch.tanh(action))
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

    def learn(self, trajectories):

        if not trajectories:
            print("Warning: Empty trajectories. Skipping learning step.")
            return
        
        states, actions, log_probs, rewards, next_states, dones = zip(*trajectories)
        # print(states,actions,log_probs,rewards,next_states,dones)
        # print(states.dtype())
        #print(f"Learning step - Trajectories: {len(trajectories)}, States shape: {np.shape(states)}")
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        

        with torch.no_grad():
            values = self.critic_network(states).squeeze()
            next_values = self.critic_network(next_states).squeeze()
            advantages = self.compute_advantages(rewards.detach().numpy(), values.detach().numpy(), next_values.detach().numpy(), dones)
            
            advantages = torch.tensor(advantages,dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                #print("mean : ",mean)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratios = torch.exp(new_log_probs - batch_log_probs)
                
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                #print("policy loss :",policy_loss)
                value_pred = self.critic_network(batch_states).squeeze()
                value_loss = F.mse_loss(value_pred , batch_returns)
                #print("value loss :",value_loss)
                # self.actor_optimizer.zero_grad()
                # policy_loss.backward(retain_graph=True)
                # self.actor_optimizer.step()

                # self.critic_optimizer.zero_grad()
                # value_loss.backward(retain_graph=True)
                # self.critic_optimizer.step()

                total_loss = policy_loss + 0.5 * value_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.actor_loss = policy_loss.item()
        self.critic_loss = value_loss.item()

def evaluate(agent, env, num_episodes=5):
    total_rewards = []
    episode_lengths = []
    for _ in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        done = False
        step = 0
        while not done and step < 200:
            action, _ = agent.select_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            step += 1
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
    return np.mean(total_rewards), np.std(total_rewards), np.mean(episode_lengths)


def evaluate_saved_model(model_path, env, num_episodes=50):
    evaluation_agent = Agent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    evaluation_agent.load_models(model_path)
    
    mean_reward, std_reward, mean_length = evaluate(evaluation_agent, env, num_episodes=num_episodes)
    print(f"Evaluation of model from {model_path}:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, Mean length: {mean_length:.2f}")
    
    return mean_reward, std_reward, mean_length



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
    n_episodes = 500
    max_timesteps = 500
    eval_interval = 100
    episode_rewards = []
    evaluate_interval = 100  # Set to None to disable intermediate evaluation
    save_best_model = True   # Set to False if you only want to save the final model

    best_eval_reward = float('-inf')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/jaco2_ppo_{current_time}'
    writer = SummaryWriter(log_dir)

    # agent.load_models()
    print("Training begins")
    try :
        for i in range(n_episodes):
            print("Epsiode :", i)
            observation = env.reset()
            # print(("test2"))
            done = False
            score = 0
            # print("test3")
            trajectories = []
            print("............Environment resetted..............")
            for t in range(max_timesteps):
                #print(f"  Step: {t}")
                # print("observation :", observation)
                action,log_prob = agent.select_action(observation)
                #print(f"  Action selected: {action}")
                next_observation, reward, done, info = env.step(action)
                #print(f"  Reward: {reward}")
                score += reward
                # print("action :",action)
                trajectories.append([observation, action,log_prob, reward, next_observation, done])
                observation = next_observation

                if done :
                    print("Episodic task completed early")
                    break
            print(f"Total for {i} episode is {score}")

            episode_rewards.append(score)
            writer.add_scalar('Training/Episode Reward', score, i)
            writer.add_scalar('Training/Episode Length', t+1, i)


            #print("Learning.....")
            # trajectories = torch.FloatTensor(np.array(trajectories))
            agent.learn(trajectories)
            #print("Learning finished for ", i , "episode")

            writer.add_scalar('Training/Actor Loss', agent.actor_loss, i)
            writer.add_scalar('Training/Critic Loss', agent.critic_loss, i)

            # print(f"Episode {i} finished. Reward: {episode_reward}")

            if (i + 1) % eval_interval == 0:
                mean_reward, std_reward, mean_length = evaluate(agent, env)
                writer.add_scalar('Evaluation/Mean Reward', mean_reward, i)
                writer.add_scalar('Evaluation/Std Reward', std_reward, i)
                writer.add_scalar('Evaluation/Mean Episode Length', mean_length, i)
                
                print(f"Evaluation after {i+1} episodes: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, Mean length: {mean_length:.2f}")
                
                if mean_reward > best_eval_reward:
                    best_eval_reward = mean_reward
                    agent.save_models(f'best_model_episode_{i+1}')
                    print(f"New best model saved at episode {i+1}")

        print(episode_rewards)
        #Save the model after training
        agent.save_models()

        #Plot episode rewards
        # plt.figure(figsize=(10, 5))
        # plt.plot(episode_rewards)
        # plt.title('Episode Rewards')
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.savefig('episode_rewards.png')
        # plt.show()

    
        # Final evaluation
        print("loading model")
        agent.load_models()
        print("model loaded")
        final_mean_reward, final_std_reward, final_mean_length = evaluate(agent, env, num_episodes=50)
        print(f"Final evaluation: Mean reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}, Mean length: {final_mean_length:.2f}")
        
        # # Evaluate the best model
        # best_model_path = f'best_model_episode_{best_episode}'  # Replace best_episode with the actual episode number
        # evaluate_saved_model(best_model_path, env)

        # # Evaluate the final model
        # evaluate_saved_model('final_model', env)

        writer.close()

            # if(i%10):
        #     agent.save_models()
        # print(f"Episode : {i}, score : {score}")
    
    except rospy.ROSInterruptException:
        pass

    finally : 
        rospy.signal_shutdown("Training complete")