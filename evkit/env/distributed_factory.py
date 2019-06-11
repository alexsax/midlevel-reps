from evkit.env.util.vec_env.subproc_vec_embodied_env import SubprocVecEmbodiedEnv

from evkit.utils.misc import Bunch

_distribution_schemes = Bunch(
    {'vectorize': 'VECTORIZE',
     'independent': 'INDEPENDENT'})

class DistributedEnv(object):
    
    distribution_schemes = _distribution_schemes

    @classmethod
    def new(cls, envs, gae_gamma=None, distribution_method=_distribution_schemes):
        if distribution_method == cls.distribution_schemes.vectorize:
            return cls.vectorized(envs, gae_gamma)
        elif distribution_method == cls.distribution_schemes.independent:
            return cls.independent(envs, gae_gamma)
        else:
            raise NotImplementedError

    def vectorized(envs, gae_gamma=None):
        ''' Vectorizes an interable of environments 
            Params:
                envs: an iterable of environments
                gae_gamma: if not none and there observation space is one-dimensional, then apply the gamma parameter from GAE
        '''
        envs = SubprocVecEmbodiedEnv(envs)

        # if len(envs) > 1:
        #     envs = SubprocVecEmbodiedEnv(envs)
        # else:
        #     envs = DummyVecEnv(envs) # TODO(sasha): Update this to work with sensordict
        
        if gae_gamma is not None:
            if hasattr(envs.observation_space, "spaces") \
                and len(envs.observation_space.spaces) == 1 \
                and len(list(envs.observation_space.spaces.values())[0].shape) == 1:
                    envs = VecNormalize(envs, gamma=gae_gamma)
            elif not hasattr(envs.observation_space, "spaces") and len(envs.observation_space.shape) == 1:
                envs = VecNormalize(envs, gamma=gae_gamma)
        
        return envs
    
    def independent(envs, gae_gamma=None):
        if gae_gamma is not None:
            raise NotImplementedError('gae_gamma not supported for "independent" distributed environments')
        
        envs = [e() for e in envs]
        return envs