import gym
from gym import spaces
from vizdoom import *
import numpy as np
import os
# from gym.envs.classic_control import rendering

CONFIGS = [['basic.cfg', 3],                # 0
           ['deadly_corridor.cfg', 7],      # 1
           ['defend_the_center.cfg', 3],    # 2
           ['defend_the_line.cfg', 3],      # 3
           ['health_gathering.cfg', 3],     # 4
           ['my_way_home.cfg', 5],          # 5
           ['predict_position.cfg', 3],     # 6
           ['take_cover.cfg', 2],           # 7
           ['deathmatch.cfg', 20],          # 8
           ['health_gathering_supreme.cfg', 3]]  # 9

FPS    = 50


class VizdoomEnv(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    
    def __init__(self, level):

        # init game
        self.game = DoomGame()
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        game_path = os.path.join(os.path.dirname(__file__), 'freedoom2.wad')
        assert os.path.isfile(game_path)
        self.game.set_doom_game_path(game_path)
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[level][0]))
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)

        args = []
        args.append('+sv_cheats 1')
        for arg in args:
            self.game.add_game_args(arg)
        
        self.game.init()
        self.state = None

        self.action_space = spaces.Discrete(CONFIGS[level][1])
        self.observation_space = spaces.Box(0, 255, (self.game.get_screen_height(),
                                                     self.game.get_screen_width(),
                                                     self.game.get_screen_channels()),
                                            dtype=np.uint8)
        self.viewer = None
        self.done = False

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act)
        self.state = self.game.get_state()
        self.done = self.game.is_episode_finished()
        self.obs = self._get_obs()

        info = {'dummy': 0}
        return self.obs, reward, self.done, info

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        self.obs = self._get_obs()
        return self._get_obs()

    def render(self, mode='human'):
        # img = np.zeros_like(self.game.get_state().screen_buffer)
        # img = self.game.get_state().screen_buffer
        # img = np.transpose(img, [1, 2, 0])
        # print(self.obs.shape)
        # print(self.obs.dtype)
        if self.viewer is not None:
            # self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.obs)
        return self.obs

    def seed(self, seed):
        self.game.set_seed(seed)

    def _get_obs(self):
        if not self.done:
            return np.transpose(self.state.screen_buffer, (1, 2, 0))
        else:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {(): 2,
                (ord('a'),): 0,
                (ord('d'),): 1,
                (ord('w'),): 3,
                (ord('s'),): 4,
                (ord('q'),): 5,
                (ord('e'),): 6}
        return keys

