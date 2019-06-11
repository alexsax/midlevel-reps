import gym

class BaseEmbodiedEnv(gym.Env):
    ''' Abstract class for all embodied environments. '''
    
    is_embodied = True
    