import gym

__all__ = ['SkipWrapper']

def SkipWrapper(repeat_count):
    class SkipWrapper(gym.Wrapper):
        """
            Generic common frame skipping wrapper
            Will perform action for `x` additional steps
        """
        def __init__(self, env):
            super(SkipWrapper, self).__init__(env)
            self.repeat_count = repeat_count
            self.stepcount = 0

        def step(self, action):
            done = False
            total_reward = 0
            current_step = 0
            while current_step < (self.repeat_count + 1) and not done:
                self.stepcount += 1
                if (current_step < self.repeat_count):
                    _, reward, done, info = self.env.step_physics(action)
                else:
                    self.obs, reward, done, info = self.env.step(action)
                total_reward += reward
                current_step += 1
            if 'skip.stepcount' in info:
                raise gym.error.Error('Key "skip.stepcount" already in info. Make sure you are not stacking ' \
                                      'the SkipWrapper wrappers.')
            info['skip.stepcount'] = self.stepcount
            info['skip.repeat_count'] = self.repeat_count
            return self.obs, total_reward, done, info

        def reset(self):
            self.stepcount = 0
            self.obs = self.env.reset()
            return self.obs

    return SkipWrapper