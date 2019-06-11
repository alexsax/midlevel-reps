import gym
from gym import spaces
import numpy as np
from random import choice

from .utils.doom import DoomRoom
from .utils import commands, poisson_disc

from .scenarios.semantic_goal_map_cfg import SemanticGoalMapCfg

import os
from .vizdoomgoalenv import VizdoomPointGoalEnv, \
        PointGoal, DoomAgent, VIZDOOM_NOOP

DISTANCE_TOLERANCE_NORMALIZED_COORDS = 0.1
AGENT_SIZE = 40
OBJECT_SIZE = 40
RADIUS_AROUND_INIT_SPAWN = 32
# N_CLUTTER = 15
MAX_N_TRIES_GENERATING_CLUTTER = 100

class VizdoomNearestGoalWithClutterEnv(VizdoomPointGoalEnv):
    
    def __init__(self,
                 n_goal_objects=1,
                 n_clutter_objects=None,
                 randomize_clutter=True,
                 target_object='green_torch',
                 distance_to_goal_thresh=DISTANCE_TOLERANCE_NORMALIZED_COORDS,
                 *args, **kwargs):
        kwargs['distance_to_goal_thresh'] = distance_to_goal_thresh
        super().__init__(*args, **kwargs)
        if n_clutter_objects is None:
            n_clutter_objects = 1
        self.n_total_objects = n_goal_objects + n_clutter_objects
        self.n_goal_objects = n_goal_objects
        self.n_clutter_objects = n_clutter_objects
        self.randomize_clutter = randomize_clutter
        self.target_object = target_object
        self.n_spawned = 0
        
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
        location_proposals = None
        n_tries = 0
        while location_proposals is None and n_tries < MAX_N_TRIES_GENERATING_CLUTTER:
            n_tries += 1
            location_proposals = poisson_disc.generate_points(AGENT_SIZE + OBJECT_SIZE,
                                                  self.map_cfg.MAP_SIZE_X - 2 * self.map_cfg.DEST_WALL_MARGIN,
                                                  self.map_cfg.MAP_SIZE_Y - 2 * self.map_cfg.DEST_WALL_MARGIN,
                                                  self.n_total_objects)
            location_proposals = [(point[0] + self.map_cfg.X_OFFSET + self.map_cfg.DEST_WALL_MARGIN,
                               point[1] + self.map_cfg.Y_OFFSET + self.map_cfg.DEST_WALL_MARGIN)
                              for point in location_proposals]

            # Set the goal.
            #   Something to consider:
            #       Setting the goal as one of the clutter points 
            #       This is so that the goal is not identifiable by a particularly dense cluster
            if goal_location is not None:
                goal_x, goal_y = goal_location
            else:
                agent_initial_location = self._get_agent_frame_of_reference()
                ax, ay = agent_initial_location['x'], agent_initial_location['y']
                try:
                    goals_unnormalized, clutter = self._partition_into_goals_and_clutter(location_proposals, ax, ay, self.n_goal_objects)
                    self.goals = [PointGoal(x=goal_x, y=goal_y)
                                  for (goal_x, goal_y) in goals_unnormalized]
                    self.clutter = clutter
                except RuntimeError:
                    location_proposals = None
        assert n_tries < MAX_N_TRIES_GENERATING_CLUTTER, "Could not generate clutter. This could be a problem with fitting so many objects on the map, or it could be that the clutter was too close to the spawn point."
            
        

        # Send game commands at the end. Otherwise, multiple spawns will show up on the minimap
        self.game.send_game_command("pukename player_spawn")
        self._randomize_textures(self.randomize_textures)
        
        # # Spawn a green torch AND a red skull so that the torch is visible on the minimap
        # commands.spawn_object(self.game,
        #                       getattr(self.map_cfg.objects, self.target_object),
        #                       doom_x,
        #                       doom_y,
        #                       self.n_clutter_objects + 2)
        # Spawn goals
        for g in self.goals:
            self._spawn_goal_point(g.x, g.y)
        
        # Spawn clutter
        for i, point in enumerate(self.clutter):
            self._spawn_clutter_object(point[0], point[1], i)

        # Make agent next
        agent_x, agent_y = self._set_agent_location_away_from_goals(force_agent_location=agent_location)
        commands.spawn_agent(self.game, agent_x, agent_y, orientation=choice([0,1,2,3]))

        # Spawn this at the end (forget why)
        # commands.spawn_object(self.game, self.map_cfg.objects.red_skull, doom_x, doom_y, self.n_clutter_objects + 3)
        for g in self.goals:
            commands.spawn_object(self.game, self.map_cfg.objects.red_skull,
                                  g.x, g.y, self.n_spawned)
            self.n_spawned += 1   
            
        # Only the nearest one counts as a goal. The others are essentially decoys
        dist_to_clutter = [np.linalg.norm(
                                          g.relative_loc(self.agent.x,
                                                         self.agent.y,
                                                         0.0), 
                                          ord=1)
                           for g in self.goals]
        closest_idx = np.argmin(dist_to_clutter)
        self.goals = [self.goals[closest_idx]]
        
        _game_reward = self.game.make_action(VIZDOOM_NOOP, 1)
        self.state = self.game.get_state()
        self.obs = self._get_obs()
        # print(self.obs['color'].mean(), self.game.is_episode_finished(), 
        #       self.distance_to_a_goal(ord=1), self.distance_to_goal_thresh, 
        #       self.max_actions, self.step_count,
        #       self.done)
        # raise NotImplementedError()
        return self.obs

    
    def _set_goal(self):
        pass
    
    def _spawn_goal_point(self, x, y):
        # Spawn a green torch AND a red skull so that the torch is visible on the minimap
        commands.spawn_object(self.game,
                              getattr(self.map_cfg.objects, self.target_object),
                              x, y, self.n_spawned)
        self.n_spawned += 1
        commands.spawn_object(self.game, self.map_cfg.objects.red_skull,
                              x, y, self.n_spawned)
        self.n_spawned += 1

    def _spawn_clutter_object(self, doom_x, doom_y, idx=None):
        obj = None
        if self.randomize_clutter:
            while True:
                obj = choice(self.map_cfg.objects_blocking.vals())
                if obj != getattr(self.map_cfg.objects, self.target_object):
                    break
        else:
            obj = self.map_cfg.objects.short_green_column
        commands.spawn_object(self.game, obj, doom_x, doom_y, self.n_spawned)
        self.n_spawned += 1

    def _partition_into_goals_and_clutter(self, location_proposals, spawn_x, spawn_y, n_goals):
        '''
            Args:
                location_proposals: In doom coords
                spawn_x, spawn_y: In doom coords
        
            returns:
                Chosen clutter point
                
            note: pops the chosen clutter point
        '''
        goal_points = []
        clutter_points = []
        for i, (px, py) in enumerate(location_proposals):
            if len(goal_points) < n_goals and \
                    np.linalg.norm([px - spawn_x, py - spawn_y], ord=np.inf) > RADIUS_AROUND_INIT_SPAWN:
                goal_points.append((px, py))
            else:
                clutter_points.append((px, py))
        if len(goal_points) == n_goals:
            return goal_points, clutter_points
        raise RuntimeError("All clutter points are too close to spawn and would not appear on the map. Therefore, cannot choose an approptiate goal from the these points, as it would not appear.")

    def _set_agent_location_away_from_goals(self, force_agent_location=None):
        '''
            Args:
                spawn_x, spawn_y: In doom coords
        
            returns:
                Chosen clutter point
                
            note: pops the chosen clutter point
        '''
        # Make agent next
        if force_agent_location is not None:
            agent_x_normed, agent_y_normed = force_agent_location
            agent_x, agent_y = self.map_cfg.doom_coords(*force_agent_location)
        else:
            while True:
                agent_x, agent_y = self.map_cfg.valid_space.sample()[:2]
                dist_to_goal = [np.linalg.norm(g.relative_loc(agent_x,
                                                              agent_y,
                                                              0.0), 
                                               ord=np.inf)
                                 for g in self.goals]
                dist_to_clutter = [np.linalg.norm(g.relative_loc(agent_x,
                                                              agent_y,
                                                              0.0), 
                                               ord=np.inf)
                                 for g in self.goals]
                if min(dist_to_goal) > self.distance_to_goal_thresh and \
                   min(dist_to_clutter) > AGENT_SIZE:
                    break

        self.agent = DoomAgent(x=agent_x, y=agent_y, theta=0.0)
        return agent_x, agent_y

    @property
    def done(self):
        done = self.game.is_episode_finished() or \
                    self.distance_to_a_goal(ord=2) < self.distance_to_goal_thresh
        if self.max_actions is not None and self.step_count >= self.max_actions:
            done = True
        # print("is_done:", done)
        # print(self.game.is_episode_finished(), 
        #       self.distance_to_a_goal(ord=1), self.distance_to_goal_thresh, 
        #       self.max_actions, self.step_count,
        #       done)
        return done

    def _compute_reward(self, _game_reward, tol=1):
        if self.distance_to_a_goal(ord=2) < self.distance_to_goal_thresh:
            return 100
        else:
            return -1 #+ (1. / (np.linalg.norm(self.agent.loc - self.goal.loc , ord=1) + 1e-4)) / 1000.

