

from ..utils.doom import DoomRoom
from evkit.utils.misc import Bunch

objects_dict = {k: i for i, k in enumerate([
             'blue_torch',
             'blue_alien',
             'red_torch',
             'short_red_column',
             'red_ball',
             'helmet',
             'big_torch',
             'blue_card',
             'blue_skull',
             'green_torch',
             'green_alien',
             'green_armor',
             'green_column',
             'short_green_column',
             'red_column',
             'red_skull',
             'red_card',
             'red_armor', 
             'unk1'
            ])}
OBJECTS = Bunch(objects_dict)


objects_blocking_dict = {k: objects_dict[k] for k in 
                    [
                     'blue_torch',
                     'blue_alien',
                     'red_torch',
                     'short_red_column',
                     'big_torch',
                     'green_torch',
                     'green_alien',
                     'green_column',
                     'short_green_column',
                     'red_column',
                    ]}
OBJECTS_BLOCKING = Bunch(objects_blocking_dict)

class SemanticGoalMapCfg(object):
    # Size of the map
    MAP_SIZE_X = 384
    MAP_SIZE_Y = 384

    # Map offsets in doom coordinates
    X_OFFSET = 0
    Y_OFFSET = 320

    DEST_WALL_MARGIN = 20

    objects = OBJECTS
    objects_blocking = OBJECTS_BLOCKING

    def __init__(self):
        self.room = DoomRoom(self.X_OFFSET, self.X_OFFSET + self.MAP_SIZE_X, self.DEST_WALL_MARGIN,
                             self.Y_OFFSET, self.Y_OFFSET + self.MAP_SIZE_Y, self.DEST_WALL_MARGIN,
                             0, 0, 0)
        self.valid_space = self.room.space
    
    def doom_coords(self, x, y):
        # (0, 1) -> doom coords
        doom_x = x * self.MAP_SIZE_X + self.X_OFFSET
        doom_y = y * self.MAP_SIZE_Y + self.Y_OFFSET
        return doom_x, doom_y

    def normalized_coords(self, doom_x, doom_y):
        # Doom coords -> (0, 1)
        x = (doom_x - self.X_OFFSET) / (self.MAP_SIZE_X)
        y = (doom_y - self.Y_OFFSET) / (self.MAP_SIZE_Y)
        return x, y
