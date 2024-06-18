import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from agent import Agent
from utils import plot_learning_curve, make_env, manage_memory
from gym import wrappers
import tensorflow as tf

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DeepQNetwork(keras.Model):
    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=(4, 4), activation='relu',
                            data_format='channels_first',
                            input_shape=input_dims)
        self.conv2 = Conv2D(64, 4, strides=(2, 2), activation='relu',
                            data_format='channels_first')
        self.conv3 = Conv2D(64, 3, strides=(1, 1), activation='relu',
                            data_format='channels_first')
        self.flat = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(n_actions, activation=None)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.fname = self.chkpt_dir + self.env_name + '_' + self.algo + '_'

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(input_dims, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DeepQNetwork(input_dims, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        self.q_eval.save(self.fname+'q_eval')
        self.q_next.save(self.fname+'q_next')
        print('... models saved successfully ...')

    def load_models(self):
        self.q_eval = keras.models.load_model(self.fname+'q_eval')
        self.q_next = keras.models.load_model(self.fname+'q_next')
        print('... models loaded successfully ...')

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                  self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation])
            actions = self.q_eval(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = tf.range(self.batch_size, dtype=tf.int32)
        print(actions.shape)
        exit()
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.q_eval(states), indices=action_indices)
            q_next = self.q_next(states_)

            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - dones.numpy())

            loss = keras.losses.MSE(q_pred, q_target)

        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)

        self.q_eval.optimizer.apply_gradients(zip(grads, params))

        self.learn_step_counter += 1

        self.decrement_epsilon()

if __name__ == '__main__':
    manage_memory()
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    record_agent = False
    n_games = 250
    agent = Agent(gamma=0.99, epsilon=1, lr=0.0001,
                  input_dims=(env.observation_space.shape),
                  n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                  batch_size=32, replace=1000, eps_dec=1e-5,
                  chkpt_dir='models/', algo='DQNAgent',
                  env_name='PongNoFrameskip-v4')
    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.eps_min

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
        + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a
    # mkdir video
    if record_agent:
        env = wrappers.Monitor(env, "video",
                               video_callable=lambda episode_id: True,
                               force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode {} score {:.1f} avg score {:.1f} '
              'best score {:.1f} epsilon {:.2f} steps {}'.
              format(i, score, avg_score, best_score, agent.epsilon,
                     n_steps))

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)