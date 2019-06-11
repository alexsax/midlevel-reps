from gym import spaces
import gym
import torch
from evkit.sensors import SensorDict

class SensorEnvWrapper(gym.ObservationWrapper):
    ''' Wraps a typical gym environment so to work with our package
        obs = env.step(),
        obs = {sensor_name: env.step()}
        
        Parameters:
            name: what to name the sensor
    '''
    def __init__(self, env, name='obs'):
        super().__init__(env)
        self.name = name
        self.observation_space = spaces.Dict({self.name: self.observation_space})
        
    def observation(self, observation):
        return SensorDict({self.name: observation})
