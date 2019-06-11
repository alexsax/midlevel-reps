from gibson.envs.husky_env import HuskyNavigateEnv
import numpy as np
import scipy.ndimage
from scipy.misc import imresize
import pickle
import math
from gibson.envs.map_renderer import ExplorationMapRenderer
import os
from gibson.data.datasets import get_model_path

class HuskyExplorationEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0, start_locations_file=None, fixed_endpoints=False):
        HuskyNavigateEnv.__init__(self, config, gpu_count)
        self.fixed_endpoints = fixed_endpoints
        self.cell_size = self.config["cell_size"]
        self.map_x_range = self.config["map_x_range"]
        self.map_y_range = self.config["map_y_range"]
        self.default_z = self.config["initial_pos"][2]

        self.x_vals = np.arange(self.map_x_range[0], self.map_x_range[1], self.cell_size)
        self.y_vals = np.arange(self.map_y_range[0], self.map_y_range[1], self.cell_size)
        self.occupancy_map = np.zeros((self.x_vals.shape[0], self.y_vals.shape[0]))
        
        self.start_locations_file = start_locations_file
        self.use_valid_locations = False
        if self.start_locations_file is not None:
            self.valid_locations = np.loadtxt(self.start_locations_file, delimiter=',')
            self.n_points = self.valid_locations.shape[0]
            self.use_valid_locations = True

    def get_quadrant(self, angle):
        if angle > np.pi / 4 and angle <= 3 * np.pi / 4:
            return (0,1)
        elif angle > -np.pi / 4 and angle <= np.pi / 4:
            return (1,0)
        elif angle > -3 * np.pi / 4 and angle <= -np.pi / 4:
            return (0,-1)
        else:
            return (-1,0)

    def _rewards(self, action=None, debugmode=False):
        position = self.robot.get_position()
        x, y = position[0:2]
        orientation = self.robot.get_rpy()[2]
        quadrant = self.get_quadrant(orientation)
        x_idx = int((x - self.map_x_range[0]) / self.cell_size)
        y_idx = int((y - self.map_y_range[0]) / self.cell_size)
        new_block = 0.
        if (x_idx + quadrant[0] >= self.x_vals.shape[0]) or (x_idx + quadrant[0] < 0) or \
           (y_idx + quadrant[1] >= self.y_vals.shape[0]) or (y_idx + quadrant[1] < 0):
           return [0.]
        if self.occupancy_map[x_idx + quadrant[0], y_idx + quadrant[1]] == 0:
            self.occupancy_map[x_idx + quadrant[0], y_idx + quadrant[1]] = 1.
            new_block = 1.

        return [new_block]
    
    def _step(self, action):
        self.obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        yaw = self.robot.get_rpy()[2]
        self.obs["map"] = self.render_map(rotate=True, rotate_angle=yaw, translate=True)
        return self.obs, rew, done, info

    def _reset(self):
        if self.start_locations_file is not None and (not self.fixed_endpoints):
            new_start_location = self.valid_locations[np.random.randint(self.n_points), :]
            self.config["initial_pos"] = [new_start_location[0], new_start_location[1], self.default_z]
        obs = HuskyNavigateEnv._reset(self)
        self.occupancy_map = np.zeros((self.x_vals.shape[0], self.y_vals.shape[0]))
        yaw = self.robot.get_rpy()[2]
        obs["map"] = self.render_map(rotate=True, rotate_angle=yaw, translate=True)
        
        return obs

    def _termination(self, debugmode=False):
        # some checks to make sure the husky hasn't flipped or fallen off
        z = self.robot.get_position()[2]
        z_initial = self.config["initial_pos"][2]
        roll, pitch = self.robot.get_rpy()[0:2]
        if (abs(roll) > 1.22):
            print("Agent roll too high")
        if (abs(pitch) > 1.22):
            print("Agent pitch too high")
        if (abs(z - z_initial) > 1.0):
            print("Agent fell off")
        done = abs(z - z_initial) > 1.0 or self.nframe >= self.config["episode_length"] # or abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        return done


    def render_map(self, rotate=False, rotate_angle=0.0, translate=False):
        x = self.occupancy_map * 255.
        x = x.astype(np.uint8)
        if translate:
            position = self.robot.get_position()
            x_idx = int((position[0] - self.map_x_range[0]) / self.cell_size) - (self.x_vals.shape[0] // 2)
            y_idx = int((position[1] - self.map_y_range[0]) / self.cell_size) - (self.y_vals.shape[0] // 2)
            x = scipy.ndimage.shift(x, np.array([-x_idx, -y_idx]))

        x = imresize(x, (self.config["resolution"], self.config["resolution"]))
        if rotate:
            x = scipy.ndimage.rotate(x, 180.0 - math.degrees(rotate_angle), reshape=False)
        return x[:,:,np.newaxis]

    def render_map_rgb(self):
        yaw = self.robot.get_rpy()[2]
        x = self.render_map(rotate=True, rotate_angle=yaw, translate=True)
        return np.repeat(x, 3, axis=2)


class HuskyVisualExplorationEnv(HuskyExplorationEnv):
    def __init__(self, config, gpu_count=0, start_locations_file=None, fixed_endpoints=False, render=True):
        HuskyExplorationEnv.__init__(self, config, gpu_count, start_locations_file, fixed_endpoints)
        self.min_depth = 0.0
        self.max_depth = 1.5
        self.fov = self.config["fov"]
        self.screen_dim = self.config["resolution"]
        self.render = render
        if render:
            mesh_file = os.path.join(get_model_path(self.config["model_id"]), "mesh_z_up.obj")
            self.map_renderer = ExplorationMapRenderer(mesh_file, self.default_z, 0.1, self.cell_size, render_resolution = self.screen_dim)

    def _step(self, action):
        orig_found = np.sum(self.occupancy_map)
        self.obs, _, done, info = HuskyExplorationEnv._step(self, action)
        self._update_occupancy_map(self.obs['depth'])
        rew = (np.sum(self.occupancy_map) - orig_found) * 0.1
        self.obs["map"] = self.render_map()
        if self.render:
            self.map_renderer.update_agent(*self.robot.get_position()[:2])
            self.obs["map_render"] = self.map_renderer.render()
        return self.obs, rew, done, info

    def _update_occupancy_map(self, depth_image):
        clipped_depth_image = np.clip(depth_image, self.min_depth, self.max_depth)
        xyz = self._reproject_depth_image(clipped_depth_image.squeeze())
        xx, yy = self.rotate_origin_only(xyz[self.screen_dim//2:, self.screen_dim//2, :], math.radians(90) - self.robot.get_rpy()[2])
        xx += self.robot.get_position()[0]
        yy += self.robot.get_position()[1]
        for x, y in zip(xx, yy):
            self.insert_occupancy_map(x, y)
            if self.render:
                self.map_renderer.update_grid(x, y)

    def insert_occupancy_map(self, x, y):
        idx_x = int((x - self.map_x_range[0]) / self.cell_size)
        idx_y = int((y - self.map_y_range[0]) / self.cell_size)
        idx_x = np.clip(idx_x, 0, self.x_vals.shape[0] - 1)
        idx_y = np.clip(idx_y, 0, self.y_vals.shape[0] - 1)
        if idx_y < 0 or idx_y < 0:
            raise ValueError("Trying to set occupancy in grid cell ({}, {})".format(idx_x, idx_y))
        self.occupancy_map[idx_x, idx_y] = 1
        return idx_x, idx_y


    def _reproject_depth_image(self, depth, unit_scale=1.0):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.
        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        y = depth * unit_scale
        x = y * ((c - self.screen_dim // 2) / self.fov / self.screen_dim // 2)
        z = y * ((r - self.screen_dim // 2) / self.fov / self.screen_dim // 2)
        return np.dstack((x, y, z))

    def rotate_origin_only(self, xy, radians):
        x, y = xy[:,:2].T
        xx = x * math.cos(radians) + y * math.sin(radians)
        yy = -x * math.sin(radians) + y * math.cos(radians)
        return xx, yy

    def _reset(self):
        obs = HuskyExplorationEnv._reset(self)
        if self.render:
            self.map_renderer.clear_nonstatic_layers()
            self.map_renderer.update_spawn(*self.config["initial_pos"][0:2], radius=0.125)
            self.map_renderer.update_agent(*self.config["initial_pos"][0:2])
            obs["map_render"] = self.map_renderer.render()
        return obs

    def render_map_rgb(self):
        if self.render:
            return self.map_renderer.render()
        else:
            return np.zeros((self.screen_dim, self.screen_dim, 3))



