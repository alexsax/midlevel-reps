''' A version of the Gym registry that allows more flexible use of kwargs'''

from gym import error, logger
from gym.envs.registration as registration #import register, EnvSpec, spec



def make(self, id, kwargs):
        logger.info('Making new env: %s', id)
        spec = registration.spec(id)
        env = spec.make()
        # We used to have people override _reset/_step rather than
        # reset/step. Set _gym_disable_underscore_compat = True on
        # your environment if you use these methods and don't want
        # compatibility code to be invoked.
        if hasattr(env, "_reset") and hasattr(env, "_step") and not getattr(env, "_gym_disable_underscore_compat", False):
            patch_deprecated_methods(env)
        if (env.spec.timestep_limit is not None) and not spec.tags.get('vnc'):
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env,
                            max_episode_steps=env.spec.max_episode_steps,
                            max_episode_seconds=env.spec.max_episode_seconds)
        return env