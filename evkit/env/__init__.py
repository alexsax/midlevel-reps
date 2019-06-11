from gym.envs.registration import register
from gym.error import Error as GymError
from .distributed_factory import DistributedEnv
from .envs import EnvFactory

'''
   ViZDoom Environments
'''
try: 

    register(
        id='VizdoomRoom-v0',
        entry_point='evkit.env.vizdoom:VizdoomPointGoalEnv'
    )
    
    register(
        id='VizdoomBasic-v0',
        entry_point='evkit.env.vizdoom:VizdoomBasic'
    )

    register(
        id='VizdoomCorridor-v0',
        entry_point='evkit.env.vizdoom:VizdoomCorridor'
    )

    register(
        id='VizdoomDefendCenter-v0',
        entry_point='evkit.env.vizdoom:VizdoomDefendCenter'
    )

    register(
        id='VizdoomDefendLine-v0',
        entry_point='evkit.env.vizdoom:VizdoomDefendLine'
    )

    register(
        id='VizdoomHealthGathering-v0',
        entry_point='evkit.env.vizdoom:VizdoomHealthGathering'
    )

    register(
        id='VizdoomMyWayHome-v0',
        entry_point='evkit.env.vizdoom:VizdoomMyWayHome'
    )

    register(
        id='VizdoomPredictPosition-v0',
        entry_point='evkit.env.vizdoom:VizdoomPredictPosition'
    )

    register(
        id='VizdoomTakeCover-v0',
        entry_point='evkit.env.vizdoom:VizdoomTakeCover'
    )

    register(
        id='VizdoomDeathmatch-v0',
        entry_point='evkit.env.vizdoom:VizdoomDeathmatch'
    )

    register(
        id='VizdoomHealthGatheringSupreme-v0',
        entry_point='evkit.env.vizdoom:VizdoomHealthGatheringSupreme'
    )
except GymError:
    pass
