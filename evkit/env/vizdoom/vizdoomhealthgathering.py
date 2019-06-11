from .vizdoomenv import VizdoomEnv


class VizdoomHealthGathering(VizdoomEnv):

    def __init__(self):
        super(VizdoomHealthGathering, self).__init__(4)
