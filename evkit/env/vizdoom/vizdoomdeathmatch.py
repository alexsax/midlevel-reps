from .vizdoomenv import VizdoomEnv


class VizdoomDeathmatch(VizdoomEnv):

    def __init__(self):
        super(VizdoomDeathmatch, self).__init__(8)
