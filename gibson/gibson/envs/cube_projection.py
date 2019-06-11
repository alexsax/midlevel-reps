from skimage.draw import polygon
import math
import numpy as np
import glob
import itertools
import transforms3d

import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)



class Cube(object):
    def __init__(self, origin=(0.0, 0.0, 0.0), scale=1.0, rotation_mat=None):
        STANDARD_CUBE_VERTS = []
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    STANDARD_CUBE_VERTS.append([x, y, z])
        STANDARD_CUBE_VERTS = np.array(STANDARD_CUBE_VERTS, dtype=np.float32)
        STANDARD_CUBE_VERTS -= 0.5

        # Find the faces of the cube
        CUBE_FACES = list(itertools.combinations(STANDARD_CUBE_VERTS, 4))
        def is_face(verts):
            eps = 1e-8
            edge_lengths = [np.linalg.norm(v0 - v1) for v0, v1 in itertools.combinations(verts, 2)] # We reqire 2 to have length sqrt(2) and 4 to be length 1
            return len([e for e in edge_lengths if np.isclose(e, np.sqrt(2.0))]) == 2 and len([e for e in edge_lengths if np.isclose(e, 1.0)]) == 4

        def clockwise(f):
            for p in itertools.permutations(f, 4):
                v1, v2, v3, v4 = p
                if np.isclose(np.linalg.norm(v1 - v2), 1.0) and \
                   np.isclose(np.linalg.norm(v2 - v3), 1.0) and \
                   np.isclose(np.linalg.norm(v3 - v4), 1.0) and \
                   np.isclose(np.linalg.norm(v4 - v1), 1.0):
                    return p
            raise ValueError

        CUBE_FACES = [clockwise(f) for f in CUBE_FACES if is_face(f)]

        # Map these faces to vertex indices
        def index_of_vert(query, verts):
            for i, v in enumerate(verts):
                if np.isclose(np.linalg.norm(v - query), 0):
                    return i
            raise KeyError

        self.cube_face_idxs = [[index_of_vert(q, STANDARD_CUBE_VERTS) for q in face] for face in CUBE_FACES]
        self.verts = np.copy(STANDARD_CUBE_VERTS)
        self.verts *= scale
        self.rotation_mat = rotation_mat
        if rotation_mat is not None:
            self.verts = self.rotation_mat.dot(self.verts.T).T
        self.verts += origin
        self.homogeneous_verts = np.ones((8, 4))
        self.homogeneous_verts[:, :3] = self.verts


def generate_projection_matrix(x_world, y_world, z_world, yaw, pitch, roll, fov_x, fov_y, size_x, size_y):
   # Camera Extrinsics
   R_camera = transforms3d.euler.euler2mat(roll, pitch, yaw)
   t_camera = np.array([x_world, y_world, z_world])
   RT = np.eye(4)
   RT[:3, :3] = R_camera.T
   RT[:3, 3] = -R_camera.T.dot(t_camera)
   rotation = np.array([[0,0,-1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]]) # So that z-dimension points out
   RT = np.dot(rotation, RT)

   # Camera Intrinsics
   f_x = size_x / math.tan(fov_x / 2)
   f_y = size_y /  math.tan(fov_y / 2)
   K = np.array([
       [f_x, 0.0, size_x],
       [0.0, f_y, size_y],
       [0.0, 0.0, 1.0]  
   ])
   world_to_image_mat = K.dot(RT[:3])
   return world_to_image_mat

def draw_cube(cube, world_to_image_mat, im_size_x, im_size_y, fast_depth=True, debug=False):
   depth_image = world_to_image_mat.dot(cube.homogeneous_verts.T).T
   depth_image[:,:2] /= depth_image[:,2][:, np.newaxis]
   im = np.full((im_size_x, im_size_y), np.inf)
   xx, yy, depth_zz = depth_image.T
   xx_in_range = np.logical_and(xx >= 0, xx < im_size_x)
   yy_in_range = np.logical_and(yy >= 0, yy < im_size_y)
   valid_coords = np.logical_and(xx_in_range, yy_in_range)
   valid_coords = np.logical_and(valid_coords, depth_zz > 0)
   for i, idxs in enumerate(cube.cube_face_idxs):
       if fast_depth:
           depth_to_fill = np.abs(min(depth_zz[idxs])) # Just use the max depth of this face. Not accurate, but probably sufficient
       else:
           raise NotImplementedError("We'd need to interpolate between the vertices")
       if np.any(valid_coords[idxs]):
           im[polygon(xx[idxs], yy[idxs], shape=im.shape)] = depth_to_fill
   return im, xx, yy

def get_cube_depth_and_faces(cube, world_to_image_mat, im_size_x, im_size_y, fast_depth=True, debug=False):
    depth_image = world_to_image_mat.dot(cube.homogeneous_verts.T).T
    depth_image[:,:2] /= depth_image[:,2][:, np.newaxis]
    xx, yy, depth_zz = depth_image.T
    xx_in_range = np.logical_and(xx >= 0, xx < im_size_x)
    yy_in_range = np.logical_and(yy >= 0, yy < im_size_y)
    valid_coords = np.logical_and(xx_in_range, yy_in_range)
    valid_coords = np.logical_and(valid_coords, depth_zz > 0)
    xx_faces = []
    yy_faces = []
    masks = []
    for i, idxs in enumerate(cube.cube_face_idxs):
        im = np.full((im_size_x, im_size_y), np.inf)
        if fast_depth:
            depth_to_fill = np.abs(max(depth_zz[idxs])) # Just use the max depth of this face. Not accurate, but probably sufficient
        else:
            raise NotImplementedError("We'd need to interpolate between the vertices")
        if np.any(valid_coords[idxs]):
            im[polygon(xx[idxs], yy[idxs], shape=im.shape)] = depth_to_fill
            xx_faces.append(xx[idxs])
            yy_faces.append(yy[idxs])
            masks.append(im)
    return masks, xx_faces, yy_faces

if __name__ == '__main__':
    x_world, y_world, z_world = -2.0, -0.9, 0.0
    yaw, pitch, roll = 0.0, 0.0, 0.0

    # Camera Intrinsics
    SIZE_X = 128 // 2
    SIZE_Y = 128 // 2
    FOV_X = math.radians(90)
    FOV_Y = math.radians(90)
    f_x = SIZE_X / math.tan(FOV_X / 2)
    f_y = SIZE_Y /  math.tan(FOV_Y / 2)

    cube = Cube()
    world_to_image_mat = generate_projection_matrix(x_world, y_world, z_world, yaw, pitch, roll, FOV_X, FOV_Y, SIZE_X, SIZE_Y)
    plt.imshow(draw_cube(cube, world_to_image_mat, SIZE_X*2, SIZE_Y*2))
    plt.show()
