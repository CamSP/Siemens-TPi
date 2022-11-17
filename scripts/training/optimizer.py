# +
import numpy as np
import gym
import random
import tensorflow as tf
from env import factory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

model_path = "../../models/optimization/model.h5"
log_path = "../../models/optimization/logs"
PPO_path = "../../models/optimization/PPO_model"
DQN_path = "../../models/optimization/DQN_model"
log_training_path = "../../models/optimization/logs/DQN"
# -

env = factory(model_path)

env = Monitor(env, log_path)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=6)

# # Callbacks

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=2, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best = stop_callback,
                             eval_freq = 50,
                             best_model_save_path = DQN_path,
                             verbose = 1
                            )

net_arch = [dict(pi=[32, 64, 128, 32], vf=[32, 64, 128, 32])]
model = DQN("MlpPolicy", env, verbose=2, tensorboard_log=log_path, device="cuda")

model.learn(total_timesteps = 10000, callback = eval_callback)

evaluate_policy(model, env, n_eval_episodes=10, render=True)







from stable_baselines3.common.env_checker import check_env
from env import factory
env = factory()
check_env(env)

























action = env.action_space.sample()
action

99601.976562/905.514221

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print("Episode: {} Score: {}".format(episode, score))
env.close()

states = env.observation_space.shape
actions = env.action_space.n


def build_model(states, actions):
    model = Sequential()
    
    model.add(Flatten(input_shape = (1, states[0])))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    
    return model


model = build_model(states, actions)
model.summary()

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length = 1)
    dqn = DQNAgent(model = model, 
                   memory = memory, 
                   policy = policy, 
                   nb_actions = actions,
                   nb_steps_warmup = 100,
                   target_model_update = 1e-2
                )
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr = 1e-3), metrics = ["mae"])

dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=10, visualize=False)
plt.plot(scores.history["episode_reward"])
print(np.mean(scores.history["episode_reward"]))













import gym
from gym.spaces import Discrete, Box


class envShower(gym.Env):
    def __init__(self):
        # Posible Actions
        self.action_space = Discrete(3)
        # Production Array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start state
        self.state = 38 + random.randint(-3, 3)
        # Set time length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        
        # Deep model prediction
        # Realizar la predicción y evaluar el nuevo estado
        self.state += action - 1
        # Reduce time
        self.shower_length -= 1
        
        # Calculate reward (calcular promedio de eficiencia y reemplazarlo por -100)
        if self.state >= 37 and self.state <=39:
            reward = 1
        else:
            reward = -1
        
        # Check if time is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False
            
        # Apply producction/energy noise
        #self.state += random.randint(-1, 1)
        
        # Place holders for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info
    
    def reset(self):
        # Reset vars
        self.state = 38 + random.randint(-3, 3)
        
        self.shower_length = 60
        return self.state
        
    def render(self, mode="human", close=False):
        # Implement visualization
        pass


env = envShower()

states = env.observation_space.shape
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print("Episode: {} Score: {}".format(episode, score))


def build_model(states, actions):
    input_layer = Input(shape=states)
    x = Flatten(input_shape=(1,states))(input_layer)
    x = Dense(24, activation="relu")(x)
    x = Dense(24, activation="relu")(x)
    x = Dense(actions, activation="linear")(x)
    model = tf.keras.Model(inputs = input_layer, outputs = x)
    return model


model = build_model(states, actions)

model.summary()

model.output.shape


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length = 1)
    dqn = DQNAgent(model = model, 
                   memory = memory, 
                   policy = policy, 
                   nb_actions = env.action_space.n,
                   nb_steps_warmup = 100,
                   target_model_update = 1e-2
                )
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr = 1e-3), metrics = ["mae"])

dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
plt.plot(scores.history["episode_reward"])
print(np.mean(scores.history["episode_reward"]))








