import copy
import numpy as np
import torch
import torch.nn as nn
from buffer.rollout_buffer import RolloutBuffer
from agents.general_agent import PolicyAgent
from torch.distributions import Categorical


class PPO(PolicyAgent):

    def __init__(self, parameters: dict, actor: nn.Module, critic: nn.Module):
        super(PPO, self).__init__(parameters=parameters, actor=actor, critic=critic)
        self.actor_old = copy.deepcopy(actor).to(self.device)
        self.critic_old = copy.deepcopy(critic).to(self.device)
        # Hyper-parameters
        self.gamma = self._config['gamma']
        self.gae_lambda = self._config['gae_lambda']
        self.actor_lr = self._config['actor_lr']
        self.critic_lr = self._config['critic_lr']
        self.epochs = self._config['epochs']
        self.ratio_clipping = self._config['ratio_clipping']
        # Optimizer
        opt_arg = [
            {'params': self.actor.parameters(), 'lr': self.actor_lr},
            {'params': self.critic.parameters(), 'lr': self.critic_lr}
        ]
        self.optimizer = getattr(torch.optim, parameters['optimizer'])(opt_arg)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
        self.loss = getattr(nn, parameters['loss_function'])()
        self.hidden_state = copy.deepcopy(self.actor.init_h_state)
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.to(self.device)

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            actions, action_logprobs, next_hidden = self.act(state=state, hidden=self.hidden_state)
        self.hidden_state = next_hidden

        return {"action": actions, "action_logprobs": action_logprobs, "next_hidden": next_hidden}

    def act(self, state, hidden=[]):
        rtn_action = []
        action_logprob = None
        outputs, hidden = self.actor_old(x=state, h=hidden)
        last = 0
        if len(self.actor_old.action_mask) > 0:
            outputs *= self.actor_old.action_mask

        for idx, output_dim in enumerate(self.actor_old.outputs_dim):
            dist = Categorical(outputs[:, last:last + output_dim])
            action = dist.sample()
            if action_logprob is None:
                action_logprob = dist.log_prob(action)
            else:
                action_logprob += dist.log_prob(action)
            rtn_action.append(action.detach())
            last += output_dim

        return torch.stack(rtn_action, dim=0).detach(), action_logprob.detach(), hidden

    def update(self, next_state=None, done=None):
        transitions = self.buffer.sample()
        # Monte Carlo estimate of returns
        # Agent 수 만큼 생성
        discounted_reward = np.zeros(transitions['reward'].shape[1])
        rewards = np.zeros(transitions['reward'].shape)
        batch_count = transitions['reward'].shape[0] - 1
        # b, n 구조로 계산
        for reward, is_terminal in zip(reversed(transitions['reward']), reversed(transitions['done'])):
            # batch iteration n about r(or d) shape
            for idx in range(reward.shape[0]):
                if is_terminal[idx]:
                    discounted_reward[idx] = 0
                discounted_reward[idx] = reward[idx] + (self.gamma * discounted_reward[idx])
                rewards[batch_count, idx] = discounted_reward[idx]
            batch_count -= 1

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = rewards.flatten()
        # convert list to tensor
        old_state = dict()
        if 'spatial_feature' in self.actor.networks:
            old_state.update({'matrix': torch.FloatTensor(transitions['matrix_state']).to(self.device)})
        if 'non_spatial_feature' in self.actor.networks:
            old_state.update({'vector': torch.FloatTensor(transitions['vector_state']).to(self.device)})

        old_hiddens = None
        if self.actor.recurrent:
            old_hiddens = transitions['next_hidden']
        # Optimize policy for K epochs
        dump = torch.zeros(len(rewards), 1)
        metrics = {'reward': dump, 'entropy': dump, 'state_value': dump, 'loss': dump}
        for _ in range(self.epochs):
            # Evaluating old actions and values
            logprobs, dist_entropy = self.evaluate(old_state, transitions['action'], hidden=old_hiddens)
            state_values, _ = self.critic(old_state)
            # match state_values tensor dimensions with rewards tensor
            state_values = state_values.flatten()
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - transitions['action_logprobs'].squeeze().detach())
            # ratios = ratios.clamp(min=3.0e-9, max=88)
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

            metrics = {'reward': rewards.detach().cpu(),
                       'entropy': dist_entropy.detach().cpu(),
                       'state_value': state_values.detach().cpu(),
                       'loss': loss.detach().cpu()}
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # insert metric
        self.insert_metrics(metrics)

    def evaluate(self, state, actions, hidden=None):
        outputs, _ = self.actor(x=state, h=hidden)
        last = 0

        action_logprobs = None
        dist_entropy = None
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            action = actions[:, idx, :].flatten()
            dist = Categorical(outputs[:, last:last + output_dim])

            if action_logprobs is None:
                action_logprobs = dist.log_prob(action)
                dist_entropy = dist.entropy()
            else:
                action_logprobs += dist.log_prob(action)
                dist_entropy += dist.entropy()
            last += output_dim

        return action_logprobs, dist_entropy

    def save(self, checkpoint_path: str):
        if ".pth" not in checkpoint_path:
            checkpoint_path = checkpoint_path + '.pth'
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
        else:
            raise "error wrong path"

        self.actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
        self.actor_old.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(critic_path, map_location=lambda storage, loc: storage))
        self.critic_old.load_state_dict(torch.load(critic_path, map_location=lambda storage, loc: storage))

    def set_mask(self, mask):
        if len(mask) > 0:
            self.actor_old.action_mask = mask
