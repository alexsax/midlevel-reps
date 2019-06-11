import gym
from scipy.misc import imsave
import random

def random_policy(env, save_name, n_episodes=5):
    obs_s = env.observation_space
    episodes = 0
    steps = 0
    obs = env._reset()
    while episodes < n_episodes:
        
        action = random.randrange(2)
        #print(action)
        obs, rew, done, info = env._step(action)
        if steps % 25 == 0:
            imsave("/home/bradleyemi/svl/images3/{}-{}.png".format(save_name, steps), obs["rgb_filled"])
        steps += 1
         
        if done:
            obs = env._reset()
            episodes += 1