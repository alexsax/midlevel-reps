from gym.spaces.box import Box
import gym
import torch

class ProcessObservationWrapper(gym.ObservationWrapper):
    ''' Wraps an environment so that instead of
            obs = env.step(),
            obs = transform(env.step())
        
        Args:
            transform: a function that transforms obs
            obs_shape: the final obs_shape is needed to set the observation space of the env
    '''
    def __init__(self, env, transform, obs_space):
        super().__init__(env)
        self.observation_space = obs_space
        self.transform = transform
        
    def observation(self, observation):
        return self.transform(observation)
