import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torch
import time
import numpy as np

class QLearner(nn.Module):
    def __init__(self, actor_network, target_network,
                 action_dim, batch_size, lr, eps, gamma,
                 copy_frequency,
                 start_schedule, schedule_timesteps,
                 initial_p, final_p):
        super(QLearner, self).__init__()
        self.actor_network = actor_network
        self.target_network = target_network
        self.learning_schedule = LearningSchedule(start_schedule, schedule_timesteps, initial_p, final_p)
        self.beta_schedule = LearningSchedule(start_schedule, schedule_timesteps, 0.4, 1.0)
        self.action_dim = action_dim
        self.copy_frequency = copy_frequency
        self.batch_size = batch_size
        self.gamma = gamma

        self.optimizer = optim.Adam(actor_network.parameters(),
                                    lr=lr,
                                    eps=eps)

        self.step = 0

    def cuda(self):
        self.actor_network = self.actor_network.cuda()
        self.target_network = self.target_network.cuda()

    def act(self, observation, greedy=False):
        self.step += 1
        if self.step % self.copy_frequency == 1:
            self.target_network.load_state_dict(self.actor_network.state_dict())
        if random.random() > self.learning_schedule.value(self.step) or greedy:
            with torch.no_grad():
                return self.actor_network(observation).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]])
        

    def update(self, rollouts):
        loss_epoch = 0
        observations, actions, rewards, masks, next_observations, weights, indices = rollouts.sample(self.batch_size,
                                                                                                     beta=self.beta_schedule.value(self.step))
        next_state_values = self.target_network(next_observations).detach().max(1)[0].unsqueeze(1)

        state_action_values = self.actor_network(observations).gather(1, actions)
        targets = rewards + self.gamma * masks * next_state_values
        if rollouts.use_priority:
            with torch.no_grad():
                td_errors = torch.abs(targets - state_action_values).detach() + 1e-6
            
            rollouts.update_priorities(indices, td_errors)
        loss = torch.sum(weights * (targets - state_action_values) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_epoch += loss.item()
        return loss_epoch

    def get_epsilon(self):
        return self.learning_schedule.value(self.step)


class LearningSchedule(object):
    def __init__(self, start_schedule, schedule_timesteps, initial_p=1.0, final_p=0.05):
        self.initial_p = initial_p
        self.final_p = final_p
        self.schedule_timesteps = schedule_timesteps
        self.start_schedule = start_schedule

    def value(self, t):
        fraction = min(max(0.0, float(t - self.start_schedule)) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)