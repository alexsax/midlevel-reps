import torch
import torch.nn as nn

from evkit.models.architectures import atari_nature
from evkit.rl.utils import init, init_normc_
from evkit.sensors import SensorDict

class ActorCriticModule(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def internal_state_size(self):
        raise NotImplementedError("internal_state_size not implemented in abstract class LearnerModel")

    @property
    def output_size(self):
        raise NotImplementedError("output_size not implemented in abstract class LearnerModel")

    def forward(self, inputs, states, masks):
        ''' value, actor_features, states = self.base(inputs, states, masks)  '''
        raise NotImplementedError("forward not implemented in abstract class LearnerModel")

        
class NaivelyRecurrentACModule(ActorCriticModule):
    ''' consists of a perception unit, a recurrent unit. 
        The perception unit produces a state representation P of shape internal_state_shape 
        The recurrent unit learns a function f(P) to generate a new internal_state
        The action and value should both be linear combinations of the internal state
    '''
    def __init__(self, perception_unit, use_gru=False, internal_state_size=512):
        super(NaivelyRecurrentACModule, self).__init__()
        self._internal_state_size = internal_state_size
        
        if use_gru:
            self.gru = nn.GRUCell(input_size=internal_state_size, hidden_size=internal_state_size)
            # nn.init.orthogonal_(self.gru.weight_ih.data)
            # nn.init.orthogonal_(self.gru.weight_hh.data)
            # self.gru.bias_ih.data.fill_(0)
            # self.gru.bias_hh.data.fill_(0)
        
        self.perception_unit = perception_unit
    
        # Make the critic
        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(internal_state_size, 1))

        #self.train()

        
    @property
    def internal_state_size(self):
        return self._internal_state_size

    @property
    def output_size(self):
        return self.internal_state_size

    def forward(self, observations, internal_states, masks, cache={}):
        ''' 
            Returns:
                values: estimates of the values of new states
                dist_params: Something which paramaterizes the distribtion that gives action_log_probabilities
                internal_states: next states
        '''
        try:
            x = self.perception_unit(observations, cache)
        except:
            x = self.perception_unit(observations)  # cache is not implemented yet for this perception unit
        if hasattr(self, 'gru'):
            N = internal_states.size(0)  # N parallel envs
            # print(N, observations.size(0), internal_states.size())
            if observations.size(0) == N:
                x = internal_states = self.gru(x, internal_states * masks)
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                T = int(x.size(0) / N)
                x = x.view(T, N, x.size(1))  # unflatten
                masks = masks.view(T, N, 1)  # Same deal with masks
                
                outputs = []
                for i in range(T):
                    hx = self.gru(x[i], internal_states * masks[i])
                    internal_states = hx
                    outputs.append(hx)
                    
                # assert len(outputs) == T
                x = torch.stack(outputs, dim=0)  # a (T, N, -1) tensor
                x = x.view(T * N, -1)  # flatten
    
    
        return self.critic_linear(x), x, internal_states
