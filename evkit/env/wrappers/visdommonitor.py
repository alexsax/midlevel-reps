from gym.wrappers.monitoring import stats_recorder, video_recorder
from gym.wrappers import Monitor
import gym
import json
import numpy as np
import os
try:
    from tnt.torchnet.logger import VisdomLogger
except ModuleNotFoundError:
    from torchnet.logger import VisdomLogger

class VisdomMonitor(Monitor):
    
    def __init__(self, env, directory,
                 video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None,
                 server="localhost", visdom_env='main', port=8097,
                 visdom_log_file=None):
        self.visdom_env = visdom_env
        self.server = server
        self.port = port
        self.video_logger = None
        self.visdom_log_file = visdom_log_file
        super(VisdomMonitor, self).__init__(env, directory,
                                            video_callable=video_callable, force=force,
                                            resume=resume, write_upon_reset=write_upon_reset,
                                            uid=uid, mode=mode)

    def step_physics(self, action):
        self._before_step(action)
        if hasattr(self.env, 'step_physics'):
            observation, reward, done, info = self.env.step_physics(action)
        else:
            observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._before_reset()
        if 'save_replay_file_path' not in kwargs:
            if self.enabled:
                kwargs['save_replay_file_path'] = self.episode_base_path
        try:
            observation = self.env.reset(**kwargs)
        except TypeError:
            kwargs.pop('save_replay_file_path')
            observation = self.env.reset(**kwargs)
        self._after_reset(observation)
        return observation

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        #
        # TODO: calculate a more correct 'episode_id' upon merge
        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=self.episode_base_path,
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )
        self._create_video_logger()
        self.video_recorder.capture_frame()
    
    def _create_video_logger(self):
        self.video_logger = VisdomLogger('video', env=self.visdom_env, server=self.server,
                                         port=self.port, log_to_filename=self.visdom_log_file,
                                         opts={'title': 'env{}.ep{:06}'.format(self.file_infix, self.episode_id)})

    def _close_video_recorder(self):
        self.video_recorder.close()
        if self.video_recorder.functional:
            path, metadata_path = self.video_recorder.path, self.video_recorder.metadata_path
            self.videos.append((path, metadata_path))
            self.video_logger.log(videofile=path)
    
    
    @property
    def episode_base_path(self):
        return os.path.join(self.directory,
                            '{}.video.{}.video{:06}'.format(
                                self.file_prefix,
                                self.file_infix,
                                self.episode_id)
                            )
        
#     def _after_step(self, observation, reward, done, info):
#         print("Enabled: {} | Done: {}".format(self.enabled, done))
#         if not self.enabled: return done

#         if done and self.env_semantics_autoreset:
#             # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
#             self.reset_video_recorder()
#             self.episode_id += 1
#             self._flush()

#         # Record stats
#         self.stats_recorder.after_step(observation, reward, done, info)
#         # Record video
#         print("Capturing frame")
#         self.video_recorder.capture_frame()

#         return done

class VideoRecorder(video_recorder.VideoRecorder):

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = video_recorder.ImageEncoder(self.path, frame.shape, self.frames_per_sec)
            self.metadata['encoder_version'] = self.encoder.version_info

        try:
            self.encoder.capture_frame(frame)
        except error.InvalidFrame as e:
            logger.warn('Tried to pass invalid video frame, marking as broken: %s', e)
            self.broken = True
        else:
            self.empty = False
