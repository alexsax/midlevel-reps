import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class PPOReplay(object):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 on_policy_epoch,
                 off_policy_epoch,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 amsgrad=True,
                 weight_decay=0.0,
                 intrinsic_losses=None,  # list of loss key words
                 intrinsic_loss_coef=0.0
                 ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.on_policy_epoch = on_policy_epoch
        self.off_policy_epoch = off_policy_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.intrinsic_loss_coef=intrinsic_loss_coef  # TODO make this a list

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(),
                                    lr=lr,
                                    eps=eps,
                                    weight_decay=weight_decay,
                                    amsgrad=amsgrad)
        self.last_grad_norm = None
        self.intrinsic_losses = intrinsic_losses if intrinsic_losses is not None else []

    def update(self, rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        max_importance_weight_epoch = 0
        on_policy = [0] * self.on_policy_epoch
        off_policy = [1] * self.off_policy_epoch
        epochs = on_policy + off_policy
        random.shuffle(epochs)
        info = {}

        for e in epochs:
            if e == 0:
                data_generator = rollouts.feed_forward_generator(
                        None, self.num_mini_batch, on_policy=True)
            else:
                data_generator = rollouts.feed_forward_generator(
                        None, self.num_mini_batch, on_policy=False)

            for sample in data_generator:
                observations_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                cache = {} 
                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                    observations_batch, states_batch,
                    masks_batch, actions_batch, cache
                )

                intrinsic_loss_dict = self.actor_critic.compute_intrinsic_losses(
                    self.intrinsic_losses,
                    observations_batch, states_batch,
                    masks_batch, actions_batch, cache
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, return_batch)
                self.optimizer.zero_grad()

                total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                for loss_name, loss_val in intrinsic_loss_dict.items():
                    total_loss += loss_val * self.intrinsic_loss_coef
                total_loss.backward()
                self.last_grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                for loss in self.intrinsic_losses:
                    try:
                        info[loss] += intrinsic_loss_dict[loss].item()
                    except:
                        info[loss] = intrinsic_loss_dict[loss].item()
                max_importance_weight_epoch = max(torch.max(ratio).item(), max_importance_weight_epoch)

        num_updates = 2 * self.ppo_epoch * self.num_mini_batch  # twice since on_policy and off_policy
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        for loss in self.intrinsic_losses:
            info[loss] /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, max_importance_weight_epoch, info
