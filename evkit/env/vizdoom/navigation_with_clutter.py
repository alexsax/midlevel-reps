import gym
from gym import spaces
import numpy as np
from random import choice

from .utils.doom import DoomRoom
from .utils import commands, poisson_disc

from .scenarios.semantic_goal_map_cfg import SemanticGoalMapCfg

import os
from .vizdoomgoalenv import VizdoomPointGoalEnv, \
        DISTANCE_TOLERANCE_DEFAULT, \
        PointGoal, DoomAgent

AGENT_SIZE = 40
OBJECT_SIZE = 40
RADIUS_AROUND_INIT_SPAWN = 32
# N_CLUTTER = 15
MAX_N_TRIES_GENERATING_CLUTTER = 100

class VizdoomPointGoalWithClutterEnv(VizdoomPointGoalEnv):
    
    def __init__(self,
                 # n_goal_objects=1,
                 n_clutter_objects=None,
                 randomize_clutter=True,
                 target_object='green_torch',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if n_clutter_objects is None or n_clutter_objects < 1:
            n_clutter_objects = 1
        self.n_clutter_objects = n_clutter_objects
        self.randomize_clutter = randomize_clutter
        self.target_object = target_object
        
    def reset(self,
              agent_location=None,
              goal_location=None,
              save_replay_file_path=""):
        self.episode_number += 1
        self.total_reward = 0
        self.last_reward = 0
        self.step_count = 0
        self.action_list = []
        
        # print("e: ", os.getpid())
        if self.randomize_maps is not None:
            low, high = self.randomize_maps
            map_no = 'map{0:02d}'.format(
                np.random.randint(high - low + 1) + low )
            self.map_no = map_no
            self.game.set_doom_map(map_no)
        
        if save_replay_file_path and ".lmp" not in save_replay_file_path:
            save_replay_file_path = save_replay_file_path + ".lmp"
        # self.game.new_episode(save_replay_file_path)
        self.game.new_episode()
        # Make clutter
        clutter_points = None
        n_tries = 0
        while clutter_points is None and n_tries < MAX_N_TRIES_GENERATING_CLUTTER:
            n_tries += 1
            clutter_points = poisson_disc.generate_points(AGENT_SIZE + OBJECT_SIZE,
                                                  self.map_cfg.MAP_SIZE_X - 2 * self.map_cfg.DEST_WALL_MARGIN,
                                                  self.map_cfg.MAP_SIZE_Y - 2 * self.map_cfg.DEST_WALL_MARGIN,
                                                  self.n_clutter_objects)
            clutter_points = [(point[0] + self.map_cfg.X_OFFSET + self.map_cfg.DEST_WALL_MARGIN,
                               point[1] + self.map_cfg.Y_OFFSET + self.map_cfg.DEST_WALL_MARGIN)
                              for point in clutter_points]

            # Set the goal.
            #   Something to consider:
            #       Setting the goal as one of the clutter points 
            #       This is so that the goal is not identifiable by a particularly dense cluster
            if goal_location is not None:
                goal_x, goal_y = goal_location
            else:
                agent_initial_location = self._get_agent_frame_of_reference()
                ax, ay = agent_initial_location['x'], agent_initial_location['y']
                ax, ay = self.map_cfg.doom_coords(ax, ay)
                try:
                    gx, gy = self._pick_clutter_point_for_goal(clutter_points, ax, ay)  
                    goal_x, goal_y = self.map_cfg.normalized_coords(gx, gy)
                except RuntimeError:
                    clutter_points = None
        assert n_tries < MAX_N_TRIES_GENERATING_CLUTTER, "Could not generate clutter. This could be a problem with fitting so many objects on the map, or it could be that the clutter was too close to the spawn point."
            
        self.goal = PointGoal(x=goal_x, y=goal_y)
        doom_x, doom_y = self.map_cfg.doom_coords(self.goal.x, self.goal.y)
        self.goals = [self.goal]

        # Send game commands at the end. Otherwise, multiple spawns will show up on the minimap
        self.game.send_game_command("pukename player_spawn")
        self._randomize_textures(self.randomize_textures)
        
        # Spawn a green torch AND a red skull so that the torch is visible on the minimap
        commands.spawn_object(self.game,
                              getattr(self.map_cfg.objects, self.target_object),
                              doom_x,
                              doom_y,
                              self.n_clutter_objects + 2)
        # Spawn clutter
        for i, point in enumerate(clutter_points):
            self._spawn_clutter_object(point[0], point[1], i)

        # Make agent next
        if agent_location is not None:
            agent_x, agent_y = self.map_cfg.doom_coords(*agent_location)
        else:
            while True:
                agent_x, agent_y = self.map_cfg.valid_space.sample()[:2]
                agent_x_normed, agent_y_normed = self.map_cfg.normalized_coords(agent_x, agent_y)
                if np.linalg.norm(self.goal.relative_loc(agent_x,
                                                         agent_y,
                                                         0.0), 
                                   ord=1) > DISTANCE_TOLERANCE_DEFAULT:
                    break
        commands.spawn_agent(self.game, agent_x, agent_y, orientation=choice([0,1,2,3]))

        # We need to manually record the agent at the desired location since the game will
        #   not update the agent location until the next tick
        self.agent = DoomAgent(x=agent_x, y=agent_y, theta=0.0)

        # Spawn this at the end (forget why)
        commands.spawn_object(self.game, self.map_cfg.objects.red_skull, doom_x, doom_y, self.n_clutter_objects + 3)

        self.state = self.game.get_state()
        self.obs = self._get_obs()
        return self.obs

    
    def _set_goal(self):
        pass
    
    def _pick_clutter_point_for_goal(self, clutter_points, spawn_x, spawn_y):
        '''
            Args:
                spawn_x, spawn_y: In doom coords
        
            returns:
                Chosen clutter point
                
            note: pops the chosen clutter point
        '''
        for i, (px, py) in enumerate(clutter_points):
            if np.linalg.norm([px - spawn_x, py - spawn_y], ord=np.inf) > RADIUS_AROUND_INIT_SPAWN:
                gx, gy = clutter_points.pop(i)                
                return gx, gy
        raise RuntimeError("All clutter points are too close to spawn and would not appear on the map. Therefore, cannot choose an approptiate goal from the these points, as it would not appear.")

    def _spawn_clutter_object(self, doom_x, doom_y, idx):
        obj = None
        if self.randomize_clutter:
            while True:
                obj = choice(self.map_cfg.objects_blocking.vals())
                if obj != getattr(self.map_cfg.objects, self.target_object):
                    break
        else:
            obj = self.map_cfg.objects.short_green_column
        commands.spawn_object(self.game, obj, doom_x, doom_y, idx)