import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .distributions import Categorical, DiagGaussian
from .model import CNNBase, MLPBase
from evkit.models.taskonomy_network import TaskonomyDecoder
from evkit.env.habitat.utils import DEFAULT_TAKEOVER_ACTIONS, NON_STOP_VALUES, TAKEOVER_ACTION_SEQUENCES, LEFT_VALUE, RIGHT_VALUE, FORWARD_VALUE
from evkit.models.architectures import FrameStacked

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError


    def act(self, inputs, states, masks, deterministic=False):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value

    def evaluate_actions(self, observations, internal_states, masks, action):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value, action_log_probs, dist_entropy, states


class Policy(BasePolicy):
    def __init__(self, obs_shape, action_space, recurrent_policy):
        super().__init__()
    
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            self.base = MLPBase(obs_shape[0])
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.internal_state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError


    def act(self, inputs, states, masks, deterministic=False):
        ''' TODO: find out what these do '''
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            # Select MAP/MLE estimate (depending on if we are feeling Bayesian)
            action = dist.mode()
        else:
            # Sample from trained posterior distribution
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        ''' TODO: find out what these do '''
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

class TakeoverPolicy(BasePolicy):
    def taking_over(self, obs):
        raise NotImplementedError

def apply(func, tensor):
    tList = [func(m) for m in torch.unbind(tensor, dim=0) ]
    res = torch.stack(tList, dim=0)
    return res

def convert_polar_to_xy(polar_coords:torch.Tensor):
    assert polar_coords.shape[0] == 2, 'Expect polar coordinates, but array length != 2'
    r, t = polar_coords
    return torch.Tensor([r * torch.cos(t), r * torch.sin(t)]).cuda()

class BackoutPolicy(TakeoverPolicy):
    # Take over policy and apply backout sequence when stuck
    # I imagine that in the future, we want more hardcoded action sequences to take over the policy
    # Then, I think creating a HardcodedTakeoverPolicy class will be useful and good programming practice
    # But right now, it is a bit hard to think about
    def __init__(self, patience, num_processes, unstuck_dist=0.5, takeover_action_sequences=TAKEOVER_ACTION_SEQUENCES, randomize_actions=False):
        super().__init__()
        self.patience = patience  # number of steps to wait for action
        self.unstuck_dist = unstuck_dist  # distance in both x and y coord that agent needs to move to not be stuck
        self.pos_history = torch.zeros((patience, num_processes, 2))
        # WARNING: we start with 0's everywhere which could be erroneous,
        # but since (0,0) is goal position
        # if our observation is close to (0,0), then the episode would be over
        # BUT may not be true for other tasks

        self.takeover_action_sequences = takeover_action_sequences
        self.randomize_actions = randomize_actions
        if self.randomize_actions:
            self.takeover_action_sequence = self.takeover_action_sequences[np.random.randint(0, len(self.takeover_action_sequences))]  # trouble with np.choice
        else:
            self.takeover_action_sequence = takeover_action_sequences[0]
        self.num_takeover_steps = len(self.takeover_action_sequence)

        self.t = 0
        self.num_processes = num_processes
        self.takeover_index = torch.zeros((num_processes,), dtype=torch.long)
        self.is_takeover = torch.zeros((num_processes,), dtype=torch.uint8)

    def _is_stuck(self, obs):
        global_pos = copy.deepcopy(obs['global_pos'][:,-3:-1])  # r, theta in world coordinates
        assert global_pos.shape[0] == self.num_processes, 'number of batch operations do not match'
        global_pos_xy = apply(convert_polar_to_xy, global_pos)  # x, y in world coordinates  TODO This can probably be faster
        self.pos_history[self.t % self.patience] = global_pos_xy
        if self.t % 1000 == 999 and self.randomize_actions:  # refresh random action sequence
            self.takeover_action_sequence = self.takeover_action_sequences[np.random.randint(0, len(self.takeover_action_sequences))]
            self.num_takeover_steps = len(self.takeover_action_sequence)
            self.is_takeover[self.takeover_index >= self.num_takeover_steps] = 0
            self.takeover_index[self.takeover_index >= self.num_takeover_steps] = 0

        self.t += 1
        p_max, _ = torch.max(self.pos_history, dim=0)
        p_min, _ = torch.min(self.pos_history, dim=0)
        p_box = p_max - p_min
        stuck_vector = torch.all(p_box < self.unstuck_dist, dim=1)
        return stuck_vector

    def act_with_mask(self, observations, model_states, masks, deterministic=False) -> (torch.Tensor, torch.Tensor):
        # check if need to take over any setups
        new_to_takeover = self._is_stuck(observations)
        self.is_takeover = self.is_takeover | new_to_takeover  # need this to be binary since we add it later

        # get actions to takeover
        actions = torch.index_select(self.takeover_action_sequence, 0, self.takeover_index)
        takeover_mask = copy.deepcopy(self.is_takeover)

        # increment action sequence indices
        self.takeover_index += self.is_takeover.long()
        self.is_takeover[self.takeover_index >= self.num_takeover_steps] = 0
        self.takeover_index[self.takeover_index >= self.num_takeover_steps] = 0
        return actions, takeover_mask

class TrainedBackoutPolicy(BackoutPolicy):
    def __init__(self, patience, num_processes, policy: BasePolicy, unstuck_dist=0.5, num_takeover_steps=8):
        super().__init__(patience, num_processes, unstuck_dist=unstuck_dist, randomize_actions=False)
        self.base = policy.base
        self.dist = policy.dist
        self.num_takeover_steps = num_takeover_steps

    def act_with_mask(self, observations, model_states, masks, deterministic=False) -> (torch.Tensor, torch.Tensor):
        # check if need to take over any setups
        new_to_takeover = self._is_stuck(observations)
        self.is_takeover = self.is_takeover | new_to_takeover  # need this to be binary since we add it later

        # get actions to takeover
        _, actor_features, _ = self.base(observations, model_states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            # Select MAP/MLE estimate (depending on if we are feeling Bayesian)
            actions = dist.mode()
        else:
            # Sample from trained posterior distribution
            actions = dist.sample()
        takeover_mask = copy.deepcopy(self.is_takeover)

        # increment action sequence indices
        self.takeover_index += self.is_takeover.long()
        self.is_takeover[self.takeover_index >= self.num_takeover_steps] = 0
        self.takeover_index[self.takeover_index >= self.num_takeover_steps] = 0
        return actions, takeover_mask



class ActionValidator():
    # makes sure you do not make stupid actions
    def __init__(self):
        self.action_history = []

    def reset(self):
        self.action_history = []

    def check_action(self, action: int) -> int:
        return action

class JerkAvoidanceValidator(ActionValidator):
    def __init__(self):
        super().__init__()

    def reset(self):  # TODO this is not properly done (its not called)
        super().reset()

    def check_action(self, action_cur: int) -> int:
        if len(self.action_history) < 2:
            self.action_history.append(action_cur)
            return action_cur

        action_second_last = self.action_history[-2]
        action_last = self.action_history[-1]
        action_final = action_cur

        if action_second_last == LEFT_VALUE and action_last == RIGHT_VALUE and action_cur == LEFT_VALUE:
            action_final = np.random.choice([FORWARD_VALUE, RIGHT_VALUE])

        if action_second_last == RIGHT_VALUE and action_last == LEFT_VALUE and action_cur == RIGHT_VALUE:
            action_final = np.random.choice([FORWARD_VALUE, LEFT_VALUE])

        self.action_history.append(action_final)
        return action_final


from evkit.models.taskonomy_network import SINGLE_IMAGE_TASKS, TASKS_TO_CHANNELS

class PolicyWithBase(BasePolicy):
    def __init__(self, base, action_space, decoder_path=None, num_stack=4, takeover:TakeoverPolicy = None, validator:ActionValidator = None):
        '''
            Args:
                base: A unit which of type ActorCriticModule
        '''
        super().__init__()
        self.base = base
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.internal_state_size = self.base.internal_state_size
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.takeover = takeover
        self.action_validator = validator

        if decoder_path is not None:
            task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
            decoder = TaskonomyDecoder(out_channels=TASKS_TO_CHANNELS[task], eval_only=True)
            checkpoint = torch.load(decoder_path)
            decoder.load_state_dict(checkpoint['state_dict'])
            self.decoder = FrameStacked(decoder, num_stack)
            self.decoder.cuda()
            self.decoder.eval()



    def forward(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError


    def act(self, observations, model_states, masks, deterministic=False):
        ''' TODO: find out what these do 
            
            inputs: Observations?
            states: Model state?
            masks: ???
            deterministic: Boolean, True if the policy is deterministic
        '''

        value, actor_features, states = self.base(observations, model_states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            # Select MAP/MLE estimate (depending on if we are feeling Bayesian)
            action = dist.mode()
        else:
            # Sample from trained posterior distribution
            action = dist.sample()

        self.probs = dist.probs
        action_log_probs = dist.log_probs(action)
        self.entropy = dist.entropy().mean()
        self.perplexity = torch.exp(self.entropy)

        # used to force agent to one action in training env (not in all envs!) useful for debugging
        # from habitat.sims.habitat_simulator import SimulatorActions
        # action[0][0] = SimulatorActions.FORWARD.value

        # apply takeover
        if self.takeover is not None:
            takeover_action, takeover_mask = self.takeover.act_with_mask(observations, model_states, masks, deterministic)
            takeover_action, takeover_mask = takeover_action.cuda(), takeover_mask.cuda()
            value.squeeze(1).masked_fill_(takeover_mask, 0).unsqueeze(1)
            action.squeeze(1).masked_scatter_(takeover_mask, takeover_action).unsqueeze(1)
            action_log_probs.squeeze(1).masked_fill_(takeover_mask, 0).unsqueeze(1)
            # states.masked_scatter_(takeover_mask, torch.zeros_like(states, device='cuda'))  TODO

        if self.action_validator is not None:
            action = self.action_validator.check_action(action)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action, cache={}):
        ''' TODO: find out what these do '''
        value, actor_features, states = self.base(inputs, states, masks, cache)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

    def compute_intrinsic_losses(self, intrinsic_losses, inputs, states, masks, action, cache):
        losses = {}
        for intrinsic_loss in intrinsic_losses:
            if intrinsic_loss == 'activation_l2':
                assert 'residual' in cache,  f'cache does not contain residual. it contains {cache.keys()}' # (8*4) x 16 x 16
                diff = self.l2(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'activation_l1':
                assert 'residual' in cache,  f'cache does not contain residual. it contains {cache.keys()}'
                diff = self.l1(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'perceptual_l1':  # only L1 since decoder
                assert 'residual' in cache,  f'cache does not contain residual. it contains {cache.keys()}'
                act_teacher = self.decoder(inputs['taskonomy'])
                act_student = self.decoder(cache['residual'])     # this uses a lot of memory... make sure that ppo_num_epoch=16
                diff = self.l1(act_teacher, act_student)
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'weight':
                pass
        return losses
