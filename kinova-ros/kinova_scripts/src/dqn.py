import numpy as np
import itertools
import torch 
from torch import nn
from collections import deque
import random
import gym

gamma = 0.99
batch_size = 32
replay_bsize = 50000  # Max number of transitions to be stored
min_replay_size = 1000  # Number of samples sampled for gradient descent
epsilon_start = 1.0 
epsilon_end = 0.1
epsilon_decay = 10000
max_steps = 100000
episodes = 1000
target_update = 1000

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))
        self.model = nn.Sequential(
            nn.Linear(in_features,256),
            nn.Tanh(),
            nn.Linear(256, env.action_space.n)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        q_value = self(obs_t.unsqueeze(0))
        max_q_value = torch.argmax(q_value, dim=1)[0]
        action = max_q_value.detach().item()
        return action
    

env = gym.make("LunarLander-v2", render_mode="human")
replay_buffer = deque(maxlen=replay_bsize)
reward_buffer = deque([0.0], maxlen=1000)  # Reward received in episodes are stored here.
episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())  # Need to do this because both were initialized differently

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initial replay buffer
obs = env.reset()[0]  # Unpack observation from tuple
for _ in range(min_replay_size):
    action = env.action_space.sample()
    new_obs, rew, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    transition = (obs, action, rew, done, new_obs)  # SARS
    replay_buffer.append(transition)
    obs = new_obs 

    if done:
        obs = env.reset()[0]
print("entering training")

for step in itertools.count():
    epsilon = np.interp(step, [0, epsilon_decay], [epsilon_start, epsilon_end])

    rnd_sample = random.random()
    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)
    
    new_obs, rew, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    transition = (obs, action, rew, done, new_obs)  # SARS
    replay_buffer.append(transition)
    obs = new_obs
    episode_reward += rew 

    if done:
        obs = env.reset()[0]
        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    if len(replay_buffer) < batch_size:
        continue

    transitions = random.sample(replay_buffer, batch_size)
    obse = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obse = np.asarray([t[4] for t in transitions])

    obse_t = torch.as_tensor(obse, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obse_t = torch.as_tensor(new_obse, dtype=torch.float32)

    tgt_q_value = target_net(new_obse_t)
    max_tgt_q_value = tgt_q_value.max(dim=1, keepdim=True)[0]

    target = rews_t + gamma * (1 - dones_t) * max_tgt_q_value  # If it's terminal state, dones_t = 1

    q_values = online_net(obse_t)
    action_q_values = torch.gather(q_values, dim=1, index=actions_t)
    loss = nn.functional.smooth_l1_loss(action_q_values, target)

    # Gradient Descent Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % target_update == 0:
        target_net.load_state_dict(online_net.state_dict())

    if step % 1000 == 0:
        #print(replay_buffer)
        print(f"Step: {step} -----> Mean Reward: {np.mean(reward_buffer)}")
