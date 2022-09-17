import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from algorithms.learner import ActorCritic, ActorCriticModel, RolloutBuffer
from algorithms.learner import get_device


class PPO(ActorCriticModel):
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                 has_continuous_action_space=False, action_std_init=0.6):

        device = get_device("auto")
        policy = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                             action_std_init, device).to(device)

        policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                                 action_std_init, device).to(device)

        parameters = {'policy': policy, 'policy_old': policy_old}

        # 습관이지만 파이썬에서는 어떤 것의 super인지 명명하는 것이 중요(다중 상속에 용이하게 사용됨)
        super(PPO, self).__init__(parameters=parameters)

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.optimizer = torch.optim.Adam([
            {'params': self._policy.actor.parameters(), 'lr': lr_actor},
            {'params': self._policy.critic.parameters(), 'lr': lr_critic}
        ])

        self._policy_old.load_state_dict(self._policy.state_dict())

        self.MseLoss = nn.MSELoss()

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
