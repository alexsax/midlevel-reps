from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
import random
import time

from .segment_tree import SumSegmentTree, MinSegmentTree
from evkit.rl.storage.memory import ReplayMemory
from evkit.sensors import SensorDict

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

import torch.nn as nn

class RolloutSensorDictDQNReplayBuffer(object):
    def __init__(self, obs_shape, action_dim, memory_size=10000, use_priority=True):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.memory_size = memory_size

        self.observations = self.observations = SensorDict({
            k: torch.zeros(memory_size, 1, *ob_shape)
            for k, ob_shape in obs_shape.items()
        })

        self.rewards = torch.zeros(memory_size, 1)
        self.actions = torch.zeros(memory_size, 1, dtype=torch.long)
        self.masks = torch.zeros(memory_size, 1)
        self.use_priority = use_priority
        if self.use_priority:
            # Set prioritized replay buffer vars
            it_capacity = 1
            while it_capacity < self.memory_size:
                it_capacity *= 2
            self.alpha = 0.6
            self.beta = 0.4
            self.max_priority = 1.0
            self._it_sum = SumSegmentTree(it_capacity)
            self._it_min = MinSegmentTree(it_capacity)

        self.step = 0
        self.memory_occupied = 0

    def cuda(self):
        self.observations = self.observations.apply(lambda k, v: v.cuda())
        self.rewards = self.rewards.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, current_obs, action, reward, mask):
        next_step = (self.step + 1) % self.memory_size
        modules = [self.observations[k][next_step].copy_ for k in self.observations]
        inputs = tuple([current_obs[k].peek() for k in self.observations])
        nn.parallel.parallel_apply(modules, inputs)
        # for k in self.observations:
        # self.observations[k][self.step + 1].copy_(current_obs[k].peek())
        self.actions[self.step].copy_(action.squeeze())
        self.rewards[self.step].copy_(reward.squeeze())
        self.masks[next_step].copy_(mask.squeeze())
        self.step = (self.step + 1) % self.memory_size
        if self.memory_occupied < self.memory_size:
            self.memory_occupied += 1

        if self.use_priority:
            self._it_sum[next_step] = self.max_priority * self.alpha
            self._it_min[next_step] = self.max_priority * self.alpha

    def get_current_observation(self):
        return self.observations.at(self.step)

    def sample(self, num_transitions, beta=1.0):
        observations_sample = SensorDict(
            {k: torch.zeros(num_transitions, *ob_shape).cuda() for k, ob_shape in self.obs_shape.items()})
        next_observations_sample = SensorDict(
            {k: torch.zeros(num_transitions * ob_shape).cuda() for k, ob_shape in self.obs_shape.items()})
        rewards_sample = torch.zeros(num_transitions, 1).cuda()
        actions_sample = torch.zeros(num_transitions, 1).cuda()
        masks_sample = torch.ones(num_transitions, 1).cuda()
        weights_sample = torch.ones(num_transitions, 1).cuda()

        if self.use_priority:
            indices = self._sample_proportional(num_transitions)
        else:
            indices = random.sample(range(self.memory_occupied), num_transitions)

        next_indices = [(idx + 1) % self.memory_size for idx in indices]
        for k, sensor_ob in self.observations.items():
            observations_sample[k] = sensor_ob.view(-1, *sensor_ob.size()[2:])[indices]
            next_observations_sample[k] = sensor_ob.view(-1, *sensor_ob.size()[2:])[next_indices]
        actions_sample = self.actions.view(-1, 1)[indices]
        rewards_sample = self.rewards.view(-1, 1)[indices]
        masks_sample = self.masks.view(-1, 1)[next_indices]
        
        if self.use_priority:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * self.memory_occupied) ** (-beta)

            for i, idx in enumerate(indices):
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * self.memory_occupied) ** (-beta)
                weights_sample[i] = weight / max_weight

        return observations_sample, actions_sample, rewards_sample, masks_sample, next_observations_sample, weights_sample, indices

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.memory_occupied - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        for idx, priority in zip(idxes, priorities):
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            self._max_priority = max(self.max_priority, priority)


class RolloutSensorDictReplayBuffer(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size, actor_critic, use_gae, gamma, tau,
                 memory_size=10000):
        assert memory_size > num_steps, "RolloutSensorDictReplayBuffer must be larger than the number of steps per update"
        #assert num_processes == 1
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.state_size = state_size
        self.memory_size = memory_size
        self.obs_shape = obs_shape
        self.sensor_names = set(obs_shape.keys())
        self.observations = SensorDict({
            k: torch.zeros(memory_size, num_processes, *ob_shape)
            for k, ob_shape in obs_shape.items()
        })
        self.states = torch.zeros(memory_size, num_processes, state_size, requires_grad=False)
        self.rewards = torch.zeros(memory_size, num_processes, 1, requires_grad=False)
        self.value_preds = torch.zeros(memory_size, num_processes, 1, requires_grad=False)
        self.returns = torch.zeros(memory_size, num_processes, 1, requires_grad=False)
        self.action_log_probs = torch.zeros(memory_size, num_processes, 1, requires_grad=False)
        self.actions = torch.zeros(memory_size, num_processes, 1, requires_grad=False)
        self.masks = torch.ones(memory_size, num_processes, 1, requires_grad=False)

        self.actor_critic = actor_critic
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau

        self.num_steps = num_steps
        self.step = 0
        self.memory_occupied = 0

    def cuda(self):
        # self.observations = self.observations.apply(lambda k, v: v.cuda())
        # self.states = self.states.cuda()
        # self.rewards = self.rewards.cuda()
        # self.value_preds = self.value_preds.cuda()
        # self.returns = self.returns.cuda()
        # self.action_log_probs = self.action_log_probs.cuda()
        # self.actions = self.actions.cuda()
        # self.masks = self.masks.cuda()
        self.actor_critic = self.actor_critic.cuda()
    
    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        next_step = (self.step + 1) % self.memory_size
        modules = [self.observations[k][next_step].copy_ for k in self.observations]
        inputs = tuple([(current_obs[k].peek(),) for k in self.observations])
        nn.parallel.parallel_apply(modules, inputs)
        # for k in self.observations:
        # self.observations[k][self.step + 1].copy_(current_obs[k].peek())
#         self.states[next_step].copy_(state)
#         self.actions[self.step].copy_(action)
#         self.action_log_probs[self.step].copy_(action_log_prob)
#         self.value_preds[self.step].copy_(value_pred)
#         self.rewards[self.step].copy_(reward)
#         self.masks[next_step].copy_(mask)
        modules = [self.states[next_step].copy_,
                   self.actions[self.step].copy_,
                   self.action_log_probs[self.step].copy_,
                   self.value_preds[self.step].copy_,
                   self.rewards[self.step].copy_,
                   self.masks[next_step].copy_]
        inputs = [state, action, action_log_prob, value_pred, reward, mask]
        nn.parallel.parallel_apply(modules, inputs)
    
        self.step = (self.step + 1) % self.memory_size
        if self.memory_occupied < self.memory_size:
            self.memory_occupied += 1

    def get_current_observation(self):
        return self.observations.at(self.step).apply(lambda k, v: v.cuda())

    def get_current_state(self):
        return self.states[self.step].cuda()

    def get_current_mask(self):
        return self.masks[self.step].cuda()

    def after_update(self):
        pass

    def feed_forward_generator(self, advantages, num_mini_batch, on_policy=True):
        # Randomly sample a trajectory if off policy
        if on_policy or self.memory_occupied < self.memory_size:
            stop_idx = self.step
            start_idx = (self.step - self.num_steps) % self.memory_size
        else:
            # Make sure the start index is at least n+1 steps behind the current step
            start_idx = (self.step - np.random.randint(self.num_steps + 1, self.memory_size)) % self.memory_size
            stop_idx = (start_idx + self.num_steps) % self.memory_size

        # Create buffers for the current sample
        # todo: clean this cuda mess up
        observations_sample = SensorDict(
            {k: torch.zeros(self.num_steps + 1, self.num_processes, *ob_shape) for k, ob_shape in
             self.obs_shape.items()}).apply(lambda k, v: v.cuda())
        states_sample = torch.zeros(self.num_steps + 1, self.num_processes, self.state_size).cuda()
        rewards_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        values_sample = torch.zeros(self.num_steps + 1, self.num_processes, 1).cuda()
        returns_sample = torch.zeros(self.num_steps + 1, self.num_processes, 1).cuda()
        action_log_probs_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        actions_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        masks_sample = torch.ones(self.num_steps + 1, self.num_processes, 1).cuda()

        # Fill the buffers and get values
        idx = start_idx
        sample_idx = 0
        while idx != stop_idx:
            for k in self.observations:
                observations_sample[k][sample_idx] = self.observations[k][idx].cuda()
            states_sample[sample_idx] = self.states[idx].cuda()
            rewards_sample[sample_idx] = self.rewards[idx].cuda()
            action_log_probs_sample[sample_idx] = self.action_log_probs[idx].cuda()
            actions_sample[sample_idx] = self.actions[idx].cuda()
            masks_sample[sample_idx] = self.masks[idx].cuda()
            with torch.no_grad():
                values_sample[sample_idx] = self.actor_critic.get_value(
                    self.observations.at(idx).apply(lambda k, v: v.cuda()), self.states[idx].cuda(),
                    self.masks[idx].cuda())
            idx = (idx + 1) % self.memory_size
            sample_idx += 1

        # we need to compute returns and advantages on the fly, since we now have updated value function predictions
        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.observations.at(stop_idx).apply(lambda k, v: v.cuda()),
                                                     self.states[stop_idx].cuda(), self.masks[stop_idx].cuda())
        if self.use_gae:
            values_sample[-1] = next_value
            gae = 0
            for step in reversed(range(rewards_sample.size(0))):
                delta = rewards_sample[step] + self.gamma * values_sample[step + 1] * masks_sample[step + 1] - \
                        values_sample[step]
                gae = delta + self.gamma * self.tau * masks_sample[step + 1] * gae
                returns_sample[step] = gae + values_sample[step]
        else:
            returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                returns_sample[step] = returns_sample[step + 1] * self.gamma * masks_batch[step + 1] + rewards_sample[
                    step]

        mini_batch_size = self.num_steps // num_mini_batch
        observations_batch = {}
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps)), mini_batch_size, drop_last=False)
        advantages = returns_sample[:-1] - values_sample[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for indices in sampler:
            for k, sensor_ob in observations_sample.items():
                observations_batch[k] = sensor_ob[:-1].view(-1, *sensor_ob.size()[2:])[indices]
            states_batch = states_sample[:-1].view(-1, states_sample.size(-1))[indices]
            actions_batch = actions_sample.view(-1, actions_sample.size(-1))[indices]
            return_batch = returns_sample[:-1].view(-1, 1)[indices]
            masks_batch = masks_sample[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = action_log_probs_sample.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield observations_batch, states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class RolloutSensorDictStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
        '''
            num_steps:
            num_processes: number of parallel rollouts to store
            obs_shape: Dict from sensor_names -> sensor_obs_shape
            action_space:
            state_size: Internal state size
        '''
        self.sensor_names = set(obs_shape.keys())
        self.observations = SensorDict({
            k: torch.zeros(num_steps + 1, num_processes, *ob_shape)
            for k, ob_shape in obs_shape.items()
        })
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def cuda(self):
        self.observations = self.observations.apply(lambda k, v: v.cuda())
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        modules = [self.observations[k][self.step + 1].copy_ for k in self.observations]
        inputs = tuple([(current_obs[k].peek(),) for k in self.observations])
        nn.parallel.parallel_apply(modules, inputs)
        # for k in self.observations:
        # self.observations[k][self.step + 1].copy_(current_obs[k].peek())
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def get_current_observation(self):
        return self.observations.at(self.step)

    def get_current_state(self):
        return self.states[self.step]

    def get_current_mask(self):
        return self.masks[self.step]

    def after_update(self):
        for k in self.observations:
            self.observations[k][0].copy_(self.observations[k][-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"* number of steps ({num_steps}) = {num_processes * num_steps} "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        mini_batch_size = batch_size // num_mini_batch
        observations_batch = {}
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            for k, sensor_ob in self.observations.items():
                observations_batch[k] = sensor_ob[:-1].view(-1, *sensor_ob.size()[2:])[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield observations_batch, states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)
            states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for k, sensor_ob in self.observations.items():
                    observations_batch[k].append(sensor_ob[:-1, ind])
                states_batch.append(self.states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            for k, v in observations_batch.items():
                observations_batch[k] = torch.stack(observations_batch[k], 1)
            actions_batch = torch.stack(actions_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            states_batch = torch.stack(states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for k, sensor_ob in observations_batch.items():
                observations_batch[k] = _flatten_helper(T, N, sensor_ob)
            actions_batch = _flatten_helper(T, N, actions_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield SensorDict(observations_batch), \
                  states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class RolloutTensorStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"* number of steps ({num_steps}) = {num_processes * num_steps} "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1,
                                                             *self.observations.size()[2:])[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = []
            states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                observations_batch.append(self.observations[:-1, ind])
                states_batch.append(self.states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            observations_batch = torch.stack(observations_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            states_batch = torch.stack(states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            observations_batch = _flatten_helper(T, N, observations_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield observations_batch, states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ