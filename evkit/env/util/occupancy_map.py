import copy
import functools
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
import torchvision as vision
import warnings

from evkit.env.habitat.utils import FORWARD_VALUE, STOP_VALUE, LEFT_VALUE, RIGHT_VALUE

MAP_SIZE = 84
TURN_ANGLE = np.deg2rad(10)
# FORWARD_STEP_SIZE = 0.25  # unused since forward motion is calculated analytically


class OccupancyMap(object):
    # goal at origin, pointing in direction of init_heading
    def __init__(self, initial_pg=np.array([999, 0]), map_kwargs={}):
        r, self.init_heading = initial_pg
        self.pgs = [initial_pg]  # in polar coordinates
        self.history = [np.array([r, self.init_heading + np.pi, self.init_heading], dtype=np.float32)]
        # history stores a list of states with format (r, global angle, heading)

        self.history_size = map_kwargs['history_size']
        self.max_building_size = map_kwargs['map_building_size']
        self.cell_size = map_kwargs['map_building_size'] / MAP_SIZE
        self.pos_to_map_fixed = functools.partial(pos_to_map, cell_size = self.cell_size)
        self.max_pool = map_kwargs['map_max_pool']

        if self.max_pool:
            warnings.warn("You want to max pool the map. Be careful! This is moved to a transform for speedup. For old checkpoints, this code no longer rotates the map")

        self.device = 'cuda' if map_kwargs['use_cuda'] else 'cpu'

    def add_pointgoal(self, pg):
        self.pgs.append(pg)

    def compute_eps_ball_ratio(self, eps):
        # finds ratio of points inside eps ball
        num_points = len(self.history)
        hist = np.array(self.history)[:,:2]  # throw away heading
        hist = convert_polar_to_xy(hist)
        dists = cdist(hist, hist[-1].unsqueeze(0))
        num_close = np.count_nonzero(dists < eps)
        r = 1.0 * num_close / num_points
        return r

    @property
    def last_state(self):
        return self.history[-1]

    @property
    def last_pointgoal(self):
        # numpy array [r, theta] in local polar coordinates
        return self.pgs[-1]

    def step(self, action):
        # Run forward model to get agent new state
        # we need the resulting observation/pointgoal of the action prior to calling step(action)
        # e.g.
        # action = agent.step(obs)
        # obs = env.step(action)
        # omap.add_pointgoal(obs)
        # omap.step(action)
        assert len(self.pgs) >= 2, 'reset provides an obs, first action provides an obs, list must be >= 2 in length'
        if action == FORWARD_VALUE:
            # assumption: forward movement can move us along wall or get us stuck but does NOT change our heading
            second_last_pointgoal = self.pgs[-2]
            dr, dphi = self.last_pointgoal - second_last_pointgoal
            step = np.array([dr, dphi, 0])
        elif action == RIGHT_VALUE:
            step = np.array([0, 0, -TURN_ANGLE])
        elif action == LEFT_VALUE:
            step = np.array([0, 0,  TURN_ANGLE])
        elif action == STOP_VALUE:
            step = np.array([0, 0, 0])
        else:
            assert False, 'What new fancy action did those people add :)'

        new_loc = self.last_state + step
        self.history.append(new_loc)
        if self.history_size is not None and len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

    def get_current_global_pos(self):
        return self.history[-1]

    def construct_occupancy_map(self) -> np.ndarray:
        # does not need second_last_pointgoal
        if len(self.history) == 0:
            return np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

                
        cur_agent_pos_polar, cur_agent_heading = self.last_state[:2], self.last_state[2]
        cur_agent_pos_xy = convert_polar_to_xy(cur_agent_pos_polar)

        global_coords_polar = copy.deepcopy(np.array(self.history))[:,:2]  # throw away heading
        global_coords_xy = convert_polar_to_xy(global_coords_polar)


        # translate then rotate by negative angle only because we rotate everything by PI before return
        # rotation subtracts initial heading so that the initial agent always points 'north'
        agent_coords = global_coords_xy - cur_agent_pos_xy
        agent_coords = rotate(agent_coords, -1 * (cur_agent_heading - self.init_heading))
#         agent_coords = rotate(agent_coords, -1 * (np.pi + cur_agent_heading - self.init_heading))


        # calculate goal coordinates (independent of forward model)
        last_pointgoal_rotated = self.last_pointgoal #+ np.array([0, np.pi])
        goal = convert_polar_to_xy(last_pointgoal_rotated)
        goal_coords = np.array([goal])

        # quantize
        visitation_cells = pos_to_map(agent_coords + self.max_building_size / 2, cell_size=self.cell_size)
        goal_cells = pos_to_map(goal_coords + self.max_building_size / 2, cell_size=self.cell_size)


        # plot (make ambient pixels 128 so that they are 0 when pass into nn)
        omap = torch.full((3, MAP_SIZE, MAP_SIZE), fill_value=128, dtype=torch.uint8, device=None, requires_grad=False) # Avoid multiplies, stack, and copying to torch
        omap[0][visitation_cells[:, 0], visitation_cells[:, 1]] = 255 # Agent visitation
        omap[1][goal_cells[:, 0], goal_cells[:, 1]] = 255 # Goal
        # omap[2][visitation_cells[-1][0], visitation_cells[-1][1]] = 255 # Agent itself

        # omap = np.rot90(omap, k=2, axes=(0,1))
        # WARNING: with code checkpoints, we need the map to be rotated
        # omap = torch.rot90(omap, k=2, dims=(1,2)) # Face north (this could be computed using a different world2agent transform)
        
        if self.max_pool:
            omap = F.max_pool2d(omap.float(), kernel_size=3, stride=1, padding=1).byte()

        omap = omap.permute(1, 2, 0).cpu().numpy()
        assert omap.dtype == np.uint8, f'Omap needs to be uint8, currently {omap.dtype}'
        return omap

def convert_polar_to_xy(polar_coords:np.ndarray):
    r, t = polar_coords.T
    return np.array([r * np.cos(t), r * np.sin(t)]).T


def pos_to_map(pos_coord: np.ndarray, cell_size: float):
    # quantize coordinates
    map_coord = (pos_coord / cell_size).astype(np.int32)
#     if np.any(map_coord >= MAP_SIZE):
#         warnings.warn('going off the map!')
    return np.clip(map_coord, a_min=0, a_max=MAP_SIZE - 1)

def rotate(coords_to_rotate, theta):
    # coords are in xy
    # counter clockwise. Input, output both N x 2
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    return rot_matrix.dot(coords_to_rotate.T).T


def visualize_map(mapp: np.ndarray):
    plt.imshow(mapp)
    plt.show()


