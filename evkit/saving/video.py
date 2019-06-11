import skvideo.io


class VideoLogger(object):
    ''' Logs a video to a file, frame-by-frame  
        
        All frames must be the same height.
        
        Example:
        >>> logger = VideoLogger("output.mp4")
        >>> for i in range(30):
        >>>     logger.log(color_transitions_(i, n_frames, width, height) )
        >>> del logger  #or, just let the logger go out of scope
    '''

    def __init__(self, save_path, fps=30):
        fps = str(fps)
        self.writer = skvideo.io.FFmpegWriter(save_path,
                                              inputdict={'-r': fps},
                                              outputdict={
                                                  '-vcodec': 'libx264',
                                                  '-r': fps,
                                              })
        self.f_open = False

    def log(self, frame):
        ''' Adds a frame to the file
            Parameters:
                frame: A WxHxC numpy array (uint8). All frames must be the same height
        '''
        self.writer.writeFrame(frame)
    
    def close(self):
        try:
            self.writer.close()
        except AttributeError:
            pass

    def __del__(self):
        self.close()


# def color_video_logger(logger_path, fps=30):
#     n_frames = 900
#     width, height = 640, 480
#     rate = '30'
#     writer = skvideo.io.FFmpegWriter("writer_test.mp4", inputdict={
#           '-r': rate,
#         },
#         outputdict={
#           '-vcodec': 'libx264',
#           '-r': rate,
#     })
#     for i in range(n_frames):
#             writer.writeFrame((fade_to_white(i, n_frames, width, height) * 255).astype(np.uint8))
#     writer.close()



######################################
# TESTING
######################################
import numpy as np
def color_transitions_(i, k, width, height):
    x = np.linspace(0, 1.0, width)
    y = np.linspace(0, 1.0, height)
    bg = np.array(np.meshgrid(x, y))
    bg = (1.0 - (i / k)) * bg + (i / k) * (1 - bg)
    r = np.ones_like(bg[0][np.newaxis, ...]) * i / k
    return np.uint8(np.rollaxis(np.concatenate([bg, r], axis=0), 0, 3) * 255)
