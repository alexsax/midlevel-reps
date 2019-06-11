from vizdoom import *
import re
import cv2
import numpy as np
from gym import spaces

def get_doom_coordinates(x, y):
    return int(x) * 256 * 256, int(y) * 256 * 256


def get_world_coordinates(x):
    return x / (256 * 256)


def get_agent_location(game):
    x = get_world_coordinates(game.get_game_variable(GameVariable.USER3))
    y = get_world_coordinates(game.get_game_variable(GameVariable.USER4))
    return x, y

def get_agent_orientation(game):
    o = get_world_coordinates(game.get_game_variable(GameVariable.USER5))*360.0
    return o


def spawn_object(game, object_id, x, y, idx):
    x_pos, y_pos = get_doom_coordinates(x, y)
    # call spawn function twice because vizdoom objects are not spawned
    # sometimes if spawned only once for some unknown reason
    for _ in range(1):
        game.send_game_command("pukename spawn_object_by_id_and_location%i \
                                %i %i %i" % (idx, object_id, x_pos, y_pos))
        pause_game(game, 0.01)


def spawn_agent(game, x, y, orientation):
    x_pos, y_pos = get_doom_coordinates(x, y)
    game.send_game_command("pukename set_position %i %i %i" %
                           (x_pos, y_pos, orientation))
    pause_game(game, 0.01)


def pause_game(game, steps):
    for i in range(1):
        r = game.make_action([False, False, False])


def split_object(object_string):
    split_word = re.findall('[A-Z][^A-Z]*', object_string)
    split_word.reverse()
    return split_word


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points
    """
    return ((x1-y1)**2 + (x2-y2)**2)**0.5


def process_screen(screen, gray, height, width):
    """
    Resize the screen.
    """
    if gray:
        screen = screen.astype(np.float32).mean(axis=0)
        if screen.shape != (height, width):
            screen = cv2.resize(screen, (width, height), interpolation=cv2.INTER_AREA)
            screen_reshaped = screen
    else:
        if screen.shape[1] != height or screen.shape[2] != width:
            screen = cv2.resize(screen, (width, height), interpolation=cv2.INTER_AREA)
            if len(screen.shape) ==2:
                screen = np.expand_dims(screen, axis=2)
            screen_reshaped = screen.transpose(2,0,1)
    return screen_reshaped

def process_batch_images(obs, gray, height, width):
    obs_reshaped = np.zeros((obs.shape[0],obs.shape[1],height,width))
    for i in range(obs.shape[0]):
        obs_reshaped[i,:] = process_screen(obs[i,:].transpose(1,2,0),gray,height,width)
    return obs_reshaped


class DoomObject(object):
    def __init__(self, *args):
        self.name = ''.join(list(reversed(args)))
        self.type = args[0]

        if self.type == "Column":
            self.type = "pillar"
        elif self.type == "Skull":
            self.type = "skullkey"
        elif self.type == "Card":
            self.type = "keycard"
        elif self.type == "Torch":
            self.type = "torch"
        elif self.type == "Armor":
            self.type = "armor"

        try:
            # Bug in Vizdoom, BlueArmor is actually red.
            # I can see your expression ! ;-)
            if self.name == 'BlueArmor':
                self.color = 'Red'
            else:
                self.color = args[1]
        except IndexError:
            self.color = None

        try:
            self.relative_size = args[2]
        except IndexError:
            self.relative_size = None
        if self.type == "torch" and self.relative_size is None:
            self.relative_size = "Tall"

        try:
            self.absolute_size = args[3]
        except IndexError:
            self.absolute_size = None
            


class DoomRoom(object):
    def __init__(self, x_min, x_max, margin_x,
                       y_min, y_max, margin_y,
                       z_min, z_max, margin_z):
        mins = np.array([x_min + margin_x, y_min + margin_y, z_min + margin_z])
        maxes = np.array([x_max - margin_x, y_max - margin_y, z_max - margin_z])
        overlap = maxes < mins
        averages = (maxes + mins) / 2
        mins[overlap] = averages[overlap]
        maxes[overlap] = averages[overlap]
        self.space = spaces.Box(mins, maxes, dtype=np.float32)

    