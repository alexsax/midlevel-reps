from .vizdoomenv import VizdoomEnv


class VizdoomPredictPosition(VizdoomEnv):

    def __init__(self):
        super(VizdoomPredictPosition, self).__init__(6)
