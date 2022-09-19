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
        with torch.no_grad():
            (spatial_x, non_spatial_x) = state
            spatial_x = torch.FloatTensor(spatial_x).to(self.device)
            non_spatial_x = torch.FloatTensor(non_spatial_x).to(self.device)
            actions, action_logprobs = self.actor.act(spatial=spatial_x, non_spatial=non_spatial_x)

        self.batch_state.append(state)
        self.batch_action(actions)
        self.batch_log_old_policy_pdf(action_logprobs)

        return [action.item() for action in actions]

    def update(self, next_state=None, done=None):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.batch_reward), reversed(self.batch_done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.batch_state, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.batch_action, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.batch_log_old_policy_pdf, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Evaluating old actions and values
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            spatial_features = old_states[0]
            non_spatial_features = old_states[1]
            input_states = self.actor.pre_forward(x1=spatial_features, x2=non_spatial_features)
            state_values = self.critic(input_states)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ratio_clipping, 1 + self.ratio_clipping) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # clear buffer
        self.batch_clear()

    def save(self, checkpoint_path: str):
        actor_path = checkpoint_path.replace(".pth", "actor.pth")
        critic_path = checkpoint_path.replace(".pth", "critic.pth")
        torch.save(self.actor_old.state_dict(), actor_path)
        torch.save(self.critic_old.state_dict(), critic_path)

    def load(self, checkpoint_path: str):
        if "actor" in checkpoint_path:
            actor_path = checkpoint_path
            critic_path = checkpoint_path.replace("actor.pth", "critic.pth")
        elif "critic" in checkpoint_path:
            critic_path = checkpoint_path
            actor_path = checkpoint_path.replace("critic.pth", "actor.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
        self.actor_old.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(critic_path, map_location=lambda storage, loc: storage))
        self.critic_old.load_state_dict(torch.load(critic_path, map_location=lambda storage, loc: storage))
