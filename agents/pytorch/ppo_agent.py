import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from agents.pytorch.utilities import get_device
from agents.general_agent import PolicyAgent


class PPO(PolicyAgent):
    def __init__(self, parameters: dict, actor: nn.Module, critic: nn.Module):
        device = get_device("auto")
        super(PPO, self).__init__(parameters=parameters, actor=actor.to(device), critic=critic.to(device))
        self.actor_old = copy.deepcopy(actor).to(device)
        self.critic_old = copy.deepcopy(critic).to(device)
        # Hyper-parameters
        self.gamma = self._config['gamma']
        self.gae_lambda = self._config['gae_lambda']
        self.actor_lr = self._config['actor_lr']
        self.critic_lr = self._config['critic_lr']
        self.epochs = self._config['epochs']
        self.ratio_clipping = self._config['ratio_clipping']
        # 표준편차의 최솟값과 최대값 설정
        self.std_bound = self._config['std_bound']
        self.state_dim = self._config['state_dim']
        # Optimizer
        opt_arg = [
            {'params': self.actor.parameters(), 'lr': self.actor_lr},
            {'params': self.critic.parameters(), 'lr': self.critic_lr}
        ]

        self.optimizer = getattr(torch.optim, parameters['optimizer'])(opt_arg)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        self.loss = getattr(nn, parameters['loss_function'])()

        self.device = device

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self._policy_old.act(state)

            self.buffer.States.append(state)
            self.buffer.Actions.append(action)
            self.buffer.LogProbs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self._policy_old.act(state)
            self.buffer.States.append(state)
            self.buffer.Actions.append(action)
            self.buffer.LogProbs.append(action_logprob)
            # 하나의 액션이 나오도록 *****
            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.Rewards), reversed(self.buffer.IsTerminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.States, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.Actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.LogProbs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self._policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self._policy_old.load_state_dict(self._policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self._policy.set_action_std(new_action_std)
            self._policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
            print("-------------------------------------------------------------------------------------------")
