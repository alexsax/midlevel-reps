from .vizdoomenv import VizdoomEnv


class VizdoomCorridor(VizdoomEnv):

    def __init__(self):
        super(VizdoomCorridor, self).__init__(1)
