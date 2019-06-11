from gym.wrappers import Monitor
import gym


class VisdomMonitor(Monitor):
    
    def __init__(self, env, directory,
                 video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None,
                 server="localhost", env='main', port=8097):
        super(VisdomMonitor, self).__init__(env, directory,
                                            video_callable=video_callable, force=force,
                                            resume=resume, write_upon_reset=write_upon_reset,
                                            uid=uid, mode=mode)

    
    
    def _close_video_recorder(self):
        video_recorder