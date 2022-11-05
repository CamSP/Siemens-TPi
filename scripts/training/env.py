import gym
from gym.spaces import Discrete, Box
import numpy as np
import random


class factory(gym.Env):
    def __init__(self):
        # Posible Actions
        self.action_space = Discrete(81)
        # Production Array
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]))
        
        ## Aplicar EDA
        # Set start Production
        self.production = 100000 + random.randint(-10000,10000)
        # Set start Energy
        self.energy = 900 + random.randint(-100,100)
        
        # Set start state
        self.state = self.production/self.energy
        # Set time length
        self.time = 1000
        
    def step(self, action):
        # Apply action
        
        # Deep model prediction
        # Realizar la predicción y evaluar el nuevo estado
        self.state += action - 1
        # Reduce time
        self.time -= 1
        
        # Calculate reward (calcular promedio de eficiencia y reemplazarlo por -100)
        reward = (self.production/self.energy)/112-1
        
        # Check if time is done
        if self.time <= 0:
            done = True
        else:
            done = False
            
        # Apply producction/energy noise
        self.state += random.randint(-1000, 1000)
        
        # Place holders for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info
    
    def reset(self):
        # Reset vars
        self.production = 93329
        self.energy = 881
        self.state = self.production/self.energy
        
        self.time = 100
        return self.state
        
    def render(self, mode="human", close=False):
        # Implement visualization
        pass
        


env2 = factory()


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

# # Test

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

# # DL

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

states = env.observation_space.shape
actions = env.action_space.n


def build_model(states, action):
    model = Sequential()
    model.add(Dense(24, activation="relu", input_shape=states))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


model = build_model(states, action)

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
                   nb_steps_warmup = 10,
                   target_model_update = 1e-2
                )
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr = 1e-3), metrics = ["mae"])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
plt.plot(scores.history["episode_reward"])
print(np.mean(scores.history["episode_reward"]))

import matplotlib.pyplot as plt



