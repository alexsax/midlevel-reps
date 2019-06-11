from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv, SemanticRobotEnv
from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.core.physics.robot_locomotors import Husky
from gibson.data.datasets import get_model_path
from gibson import assets
import numpy as np
import csv
import os
import subprocess, signal

tracking_camera = {
    'yaw': 110,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

class HuskyRandomEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0):
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        CameraRobotEnv.__init__(self, self.config, gpu_count, 
                                scene_type="building",
                                tracking_camera=tracking_camera)
        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()
        self.model_id = self.config["model_id"]
        self.scenarios = self.get_scenarios(self.config["scenarios"])
        self.n_scenarios = len(self.scenarios)

    def get_scenarios(self, scenario_size):
        scenarios_path = os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), 'navigation_scenarios')
        scenario_file = os.path.join(scenarios_path, 'pointgoal_gibson_{}_v1.csv'.format(scenario_size))
        scenarios = []
        with open(scenario_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if row['sceneId'] == self.model_id:
                    scenarios.append(row)
        return scenarios

    def _reset(self):
        scenario_index = np.random.randint(self.n_scenarios)
        scenario = self.scenarios[scenario_index]
        self.config["initial_pos"] = [float(scenario['startX']),
                                      float(scenario['startY']),
                                      float(scenario['startZ']) + 0.5]
        self.config["target_pos"] = [float(scenario['goalX']),
                                     float(scenario['goalY']),
                                     float(scenario['goalZ'])]
        print("xi", self.config["initial_pos"])
        return super(HuskyRandomEnv, self)._reset()


class HuskyMultiSceneEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0):
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        CameraRobotEnv.__init__(self, self.config, gpu_count, 
                                scene_type="building",
                                tracking_camera=tracking_camera)
        self.robot_introduce(Husky(self.config, env=self))
        self.scenarios = self.get_scenarios(self.config["scenarios"])
        self.n_scenarios = len(self.scenarios)

    def get_scenarios(self, scenario_size):
        scenarios_path = os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), 'navigation_scenarios')
        scenario_file = os.path.join(scenarios_path, 'pointgoal_gibson_{}_v1.csv'.format(scenario_size))
        with open(scenario_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            return [row for row in reader]

    def kill_depth_render(self):
        process = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = process.communicate()
        for line in out.splitlines():
            if 'depth_render' in str(line):
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)
                print("Successfully killed depth_render")

    def _reset(self):
        # Randomly select a scenario
        scenario_index = np.random.randint(self.n_scenarios)
        scenario = self.scenarios[scenario_index]
        print("Selected scenario:", scenario)
        self.model_id = self.config["model_id"] = scenario['sceneId']
        self.model_path = get_model_path(self.model_id)
        self.config["initial_pos"] = [float(scenario['startX']),
                                      float(scenario['startY']),
                                      float(scenario['startZ'])]
        self.config["target_pos"] = [float(scenario['goalX']),
                                     float(scenario['goalY']),
                                     float(scenario['goalZ'])]

        self.config["target_orn"] = [0, 0, 0]
        self.config["initial_orn"] = [0, 0, float(scenario['startAngle'])]
        self.kill_depth_render()
        self.setup_rendering_camera()
        self.scene_introduce()
        return super(HuskyMultiSceneEnv, self)._reset()