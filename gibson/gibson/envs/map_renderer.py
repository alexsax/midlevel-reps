import numpy as np
from gibson import assets
from gibson.utils import meshcut
import matplotlib.pyplot as plt
from scipy.misc import imresize
import math

class OccupancyMap(object):
    
    def __init__(self, xmin, xmax, ymin, ymax, voxel_length):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.voxel_length = voxel_length      
        self.size_x = math.ceil((xmax - xmin) / voxel_length)
        self.size_y = math.ceil((ymax - ymin) / voxel_length)
        self.bitmap = np.full((self.size_x, self.size_y), False, dtype=np.bool_)
        
    def update(self, x, y, val=True, thickness=0):
        idx_x, idx_y = self._get_voxel_coords(x, y)
        self.bitmap[idx_x, idx_y] = val
        for t in range(thickness):
            for xc in range(idx_x - t, idx_x + t):
                for yc in range(idx_y - t, idx_y + t):
                    self.bitmap[xc, yc] = val 
        return self.bitmap
    
    def get(self, x, y):
        idx_x, idx_y = self._get_voxel_coords(x, y)
        return self.bitmap[idx_x, idx_y]

    def clear(self):
        self.bitmap = np.full((self.size_x, self.size_y), False, dtype=np.bool_)

    def _get_voxel_coords(self, x, y):
        idx_x = int((x - self.xmin) / self.voxel_length)
        idx_y = int((y - self.ymin) / self.voxel_length)
        idx_x = np.clip(idx_x, 0, self.size_x - 1)
        idx_y = np.clip(idx_y, 0, self.size_y - 1)
        if idx_y < 0 or idx_y < 0:
            raise ValueError("Trying to set occupancy in grid cell ({}, {})".format(idx_x, idx_y))
        return idx_x, idx_y

def load_obj(fn):
    verts = []
    faces = []
    with open(fn) as f:
        for line in f:
            if line[:2] == 'v ':
                verts.append(list(map(float, line.strip().split()[1:4])))
            if line[:2] == 'f ':
                face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
                faces.append(face)
    verts = np.array(verts)
    faces = np.array(faces) - 1
    return verts, faces

class NavigationMapRenderer(object):
    def __init__(self, mesh_file, mesh_z, voxel_size, line_resolution=50, render_resolution=128):
        self.mesh_file = mesh_file
        self.mesh_z = mesh_z
        self.voxel_size = voxel_size
        self.line_resolution = line_resolution
        self.render_resolution = render_resolution
        self.map_color = np.array([0,0,0])
        self.agent_color = np.array([0,0,255])
        self.target_color = np.array([0,255,0])
        self.spawn_color = np.array([255,0,0])
        self.last_pose = None
        self.create_map_layer(self.mesh_file, self.mesh_z)
        self.create_target_layer()
        self.create_spawn_layer()
        self.create_agent_layer()
        self.render_image = np.ones((self.map_layer.bitmap.shape[0], self.map_layer.bitmap.shape[1], 3), dtype=np.uint8) * 255
        self.draw_map()

    def create_map_layer(self, mesh_file, mesh_z, map_margin=0.5):
        print("Creating map layer for mesh file {}...".format(mesh_file))
        verts, faces = load_obj(mesh_file)
        cross_section = meshcut.cross_section(verts, faces, plane_orig=(0,0,mesh_z), plane_normal=(0,0,1))
        
        self.x_min = np.inf
        self.y_min = np.inf
        self.x_max = -np.inf
        self.y_max = -np.inf
        for item in cross_section:
            x = item[:,0]
            y = item[:,1]
            self.x_min = min(x.min(), self.x_min)
            self.x_max = max(x.max(), self.x_max)
            self.y_min = min(y.min(), self.y_min)
            self.y_max = max(y.max(), self.y_max)

        delta_x = self.x_max - self.x_min
        delta_y = self.y_max - self.y_min
        if delta_x > delta_y:
            midpoint = (self.y_max + self.y_min) / 2
            self.y_max = midpoint + (delta_x / 2)
            self.y_min = midpoint - (delta_x / 2)
        else:
            midpoint = (self.x_max + self.x_min) / 2
            self.x_max = midpoint + (delta_y / 2)
            self.x_min = midpoint - (delta_y / 2)
        
        self.x_min -= map_margin
        self.y_min -= map_margin
        self.x_max += map_margin
        self.y_max += map_margin

        self.map_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)
        
        for item in cross_section:
            xy = item[:,0:2]
            for i in range(xy.shape[0] - 1):
                xl = xy[i,0]
                xh = xy[i+1,0]
                yl = xy[i,1]
                yh = xy[i+1,1]
                self.draw_line(self.map_layer, xl, yl, xh, yh, self.line_resolution)
        print("Done.")
        
    def draw_map(self):
        self.render_image[self.map_layer.bitmap] = self.map_color

    def create_target_layer(self):
        self.target_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)

    def create_spawn_layer(self):
        self.spawn_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)
    
    def create_agent_layer(self):
        self.agent_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)

    def draw_box(self, layer, x_min, x_max, y_min, y_max):
        x_idx_min, y_idx_min = layer._get_voxel_coords(x_min, y_min)
        x_idx_max, y_idx_max = layer._get_voxel_coords(x_max, y_max)
        for x in range(x_idx_min, x_idx_max + 1):
            for y in range(y_idx_min, y_idx_max + 1):
                layer.bitmap[x, y] = True

    def draw_line(self, layer, x0, y0, x1, y1, resolution, thickness=0):
        xs = np.linspace(x0, x1, num=resolution)
        ys = np.linspace(y0, y1, num=resolution)
        for x, y in zip(xs, ys):
            layer.update(x, y, thickness=thickness)

    def update_agent(self, agent_x, agent_y):
        if self.last_pose is not None:
            x_old, y_old = self.last_pose
            self.draw_line(self.agent_layer, x_old, y_old, agent_x, agent_y)
            self.last_pose = [agent_x, agent_y]
        self.agent_layer.update(agent_x, agent_y, thickness=2)


    def update_target(self, target_x, target_y, radius=0.25):
        self.draw_box(self.target_layer,
                      target_x - radius,
                      target_x + radius,
                      target_y - radius,
                      target_y + radius)

    def update_spawn(self, spawn_x, spawn_y, radius=0.25):
        self.draw_box(self.spawn_layer,
                      spawn_x - radius,
                      spawn_x + radius,
                      spawn_y - radius,
                      spawn_y + radius)

    def clear_nonstatic_layers(self):
        self.last_pose = None
        self.clear_agent_layer()
        self.clear_target_layer()
        self.clear_spawn_layer()
        self.render_image = np.ones((self.map_layer.bitmap.shape[0], self.map_layer.bitmap.shape[1], 3), dtype=np.uint8) * 255
        self.draw_map()

    def clear_agent_layer(self):
        self.agent_layer.clear()

    def clear_target_layer(self):
        self.target_layer.clear()

    def clear_spawn_layer(self):
        self.spawn_layer.clear()

    def render(self):
        self.render_image[self.agent_layer.bitmap] = self.agent_color
        self.render_image[self.target_layer.bitmap] = self.target_color
        self.render_image[self.spawn_layer.bitmap] = self.spawn_color
        return imresize(self.render_image, (self.render_resolution, self.render_resolution, 3))

class ExplorationMapRenderer(NavigationMapRenderer):
    def __init__(self, mesh_file, mesh_z, voxel_size, grid_voxel_size, line_resolution=50, render_resolution=128):
        self.mesh_file = mesh_file
        self.mesh_z = mesh_z
        self.voxel_size = voxel_size
        self.grid_voxel_size = grid_voxel_size
        self.line_resolution = line_resolution
        self.render_resolution = render_resolution
        self.map_color = np.array([0,0,0])
        self.agent_color = np.array([0,0,255])
        self.grid_color = np.array([160,214,255])
        self.spawn_color = np.array([255,0,0])
        self.last_pose = None
        self.create_map_layer(self.mesh_file, self.mesh_z)
        self.create_spawn_layer()
        self.create_agent_layer()
        self.create_grid_layer()
        self.render_image = np.ones((self.map_layer.bitmap.shape[0], self.map_layer.bitmap.shape[1], 3), dtype=np.uint8) * 255
        self.draw_map()

    def create_grid_layer(self):
        self.grid_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.grid_voxel_size)

    def clear_grid_layer(self):
        self.grid_layer.clear()

    def update_grid(self, x, y):
        self.grid_layer.update(x, y)

    def clear_nonstatic_layers(self):
        self.last_pose = None
        self.clear_agent_layer()
        self.clear_grid_layer()
        self.clear_spawn_layer()
        self.render_image = np.ones((self.map_layer.bitmap.shape[0], self.map_layer.bitmap.shape[1], 3), dtype=np.uint8) * 255
        self.draw_map()

    def render(self):
        self.grid_render_image = np.zeros((self.grid_layer.bitmap.shape[0], self.grid_layer.bitmap.shape[1]), dtype=np.uint8)
        self.grid_render_image[self.grid_layer.bitmap] = 1
        self.grid_render_image = imresize(self.grid_render_image, (self.map_layer.bitmap.shape[0], self.map_layer.bitmap.shape[1]))

        self.grid_render_bitmap = self.grid_render_image == 1

        self.render_image[self.grid_render_bitmap] = self.grid_color
        self.render_image[self.agent_layer.bitmap] = self.agent_color
        self.render_image[self.spawn_layer.bitmap] = self.spawn_color
        self.draw_map()

        return imresize(self.render_image, (self.render_resolution, self.render_resolution, 3))
