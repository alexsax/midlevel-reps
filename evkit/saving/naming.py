import os

def checkpoint_name(checkpoint_dir, epoch='latest'):
    return os.path.join(checkpoint_dir, 'ckpt-{}.dat'.format(epoch))
