import gym
from gym.spaces import Box, Discrete
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model


class factory(gym.Env):
    def __init__(self, model_path = "../../models/optimization/model.h5"):
        # Import model
        self.model = load_model(model_path)
        # Posible Actions
        self.action_space = Discrete(9)
        # Production Array
        self.observation_space = Box(low=np.array([60000, 0]), high=np.array([200000, 2000]))
        #
        # Valves
        self.POZ_PIT_1401B = random.randint(50,2000)
        self.POZ_PIT_1400A = random.randint(50,2000)
        self.POZ_PIT_1400B = random.randint(50,2000)        
        self.POZ_PIT_1501A = random.randint(20,2000)
        # Start Values
        self.energy, self.production = self.prediction(self.POZ_PIT_1401B, 
                                                  self.POZ_PIT_1400A, 
                                                  self.POZ_PIT_1400B, 
                                                  self.POZ_PIT_1501A)
        
        # Set start state
        self.state = np.array([self.production, self.energy])
        # Set time length
        self.time = 100
        
    def step(self, action):
        # Apply action
        if action == 1:
            self.POZ_PIT_1401B += 10
        if action == 2:
            self.POZ_PIT_1401B -= 10
        if action == 3:
            self.POZ_PIT_1400A += 10
        if action == 4:
            self.POZ_PIT_1400A -= 10
        if action == 5:
            self.POZ_PIT_1400B += 10
        if action == 6:
            self.POZ_PIT_1400B -= 10
        if action == 7:
            self.POZ_PIT_1501A += 10
        if action == 8:
            self.POZ_PIT_1501A -= 10
        
        # Deep model prediction
        self.energy, self.production = self.prediction(self.POZ_PIT_1401B, 
                                                  self.POZ_PIT_1400A, 
                                                  self.POZ_PIT_1400B, 
                                                  self.POZ_PIT_1501A)
        
        # Realizar la predicción y evaluar el nuevo estado
        self.state = np.array([self.production, self.energy])
        # Reduce time
        self.time -= 1
        
        # Calculate reward (calcular promedio de eficiencia y reemplazarlo por -100)
        reward = (self.production/self.energy)/112-1
        
        # Check if time is done
        if self.time <= 0:
            done = True
        else:
            done = False
        
        # Place holders for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info
    
    def reset(self):
         # Valves
        self.POZ_PIT_1401B = random.randint(50,2000)
        self.POZ_PIT_1400A = random.randint(50,2000)
        self.POZ_PIT_1400B = random.randint(50,2000)        
        self.POZ_PIT_1501A = random.randint(20,2000)
        # Start Values
        self.energy, self.production = self.prediction(self.POZ_PIT_1401B, 
                                                  self.POZ_PIT_1400A, 
                                                  self.POZ_PIT_1400B, 
                                                  self.POZ_PIT_1501A)
        
        # Set start state
        self.state = np.array([self.production, self.energy])
        # Set time length
        self.time = 100
        return self.state
    
    def prediction(self, POZ_PIT_1401B, POZ_PIT_1400A, POZ_PIT_1400B, POZ_PIT_1501A):
        array = np.array([[POZ_PIT_1401B, POZ_PIT_1400A, POZ_PIT_1400B, POZ_PIT_1501A]])
        tensor = tf.convert_to_tensor(array)    
        self.pred = self.model.predict(tensor, steps = 1, verbose=0)
        self.energy_pred = self.pred[0][0][0]
        self.production_pred = self.pred[1][0][0]
        return self.energy_pred, self.production_pred
        
    def render(self, mode="human", close=False):
        # Implement visualization
        print("1401B: %5f, 1400B: %5f, 1400B: %5f, 1501A: %5f, production: %8f, energy: %6f" % (
            self.POZ_PIT_1401B,
            self.POZ_PIT_1400A,
            self.POZ_PIT_1400B,
            self.POZ_PIT_1501A,
            self.state[0],
            self.state[1]
        ))
        pass
