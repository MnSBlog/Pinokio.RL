import copy

import torch
import torch.nn as nn
from torch.distributions import Normal, Multinomial, Categorical
import os
import numpy as np
from buffer.replay_buffer import ReplayBuffer
from agents.general_agent import PolicyAgent


class SAC(PolicyAgent):
    def __init__(self, parameters: dict, actor: nn.Module, critic: nn.Module):
        super(SAC, self).__init__(parameters=parameters, actor=actor, critic=critic)
        # actor/critic head 교체하기
        for seq in self.actor.networks.keys():
            if 'head' in seq:
                index = seq.replace('head', '')
                dump = self.actor.networks[seq]
                self.critic.networks[seq] = nn.Sequential(nn.Linear(dump[0].in_features,
                                                                    self.actor.outputs_dim[int(index)])).to(self.device)
        self.critic.outputs_dim = copy.deepcopy(self.actor.outputs_dim)
        self.optimizer = dict()
        opt_arg = [
            {'params': self.actor.parameters(), 'lr': self._config['actor_lr']}
        ]
        self.optimizer['actor'] = getattr(torch.optim, parameters['actor_optimizer'])(opt_arg)

        self.critic1 = self.critic
        self.target_critic1 = copy.deepcopy(self.critic)
        opt_arg = [
            {'params': self.critic1.parameters(), 'lr': self._config['critic_lr']},
            {'params': self.target_critic1.parameters(), 'lr': self._config['critic_lr']}
        ]
        self.optimizer['critic1'] = getattr(torch.optim, parameters['critic_optimizer'])(opt_arg)

        self.critic2 = copy.deepcopy(self.critic)
        self.target_critic2 = copy.deepcopy(self.critic)
        opt_arg = [
            {'params': self.critic2.parameters(), 'lr': self._config['critic_lr']},
            {'params': self.target_critic2.parameters(), 'lr': self._config['critic_lr']}
        ]
        self.optimizer['critic2'] = getattr(torch.optim, parameters['critic_optimizer'])(opt_arg)

        self.use_dynamic_alpha = self._config['use_dynamic_alpha']
        if self.use_dynamic_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            opt_arg = [
                {'params': [self.log_alpha], 'lr': self._config['alpha_lr']},
            ]
            self.optimizer['alpha'] = getattr(torch.optim, parameters['alpha_optimizer'])(opt_arg)
        else:
            self.log_alpha = torch.tensor(self._config['static_log_alpha']).to(self.device)
            self.optimizer['alpha'] = None
        self.alpha = self.log_alpha.exp()

        # if self.action_type == "continuous":
        #     self.target_entropy = -action_size
        # else:
        self.target_entropy = -np.log(1 / sum(self.actor.outputs_dim)) * 0.98

        self.gamma = self._config['gamma']
        self.tau = self._config['tau']
        self.buffer = ReplayBuffer()
        self.num_learn = 0
        self.target_update_stamp = 0
        self.time_t = 0

        self.hidden_state = copy.deepcopy(self.actor.init_h_state)
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.to(self.device)

        self.loss = getattr(nn, parameters['loss_function'])()

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
        outputs, hidden = self.actor(x=state, h=hidden)
        last = 0
        if len(self.actor.action_mask) > 0:
            outputs *= self.actor.action_mask

        for idx, output_dim in enumerate(self.actor.outputs_dim):
            dist = Categorical(outputs[:, last:last + output_dim])
            action = dist.sample()
            if action_logprob is None:
                action_logprob = dist.log_prob(action)
            else:
                action_logprob += dist.log_prob(action)
            rtn_action.append(action.detach())
            last += output_dim

        return torch.stack(rtn_action, dim=0).detach(), action_logprob.detach(), hidden

    @staticmethod
    def sample_action(self, mu, std):
        m = Normal(mu, std)
        z = m.rsample()
        action = torch.tanh(z)
        log_prob = m.log_prob(z)
        # Enforcing Action Bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def update(self, next_state=None, done=None):
        transitions = self.buffer.sample(self._config['batch_size'])

        state = dict()
        if 'spatial_feature' in self.actor.networks:
            state.update({'matrix': torch.FloatTensor(transitions['matrix_state']).detach().to(self.device)})
        if 'non_spatial_feature' in self.actor.networks:
            state.update({'vector': torch.FloatTensor(transitions['vector_state']).detach().to(self.device)})

        next_state = dict()
        if 'spatial_feature' in self.actor.networks:
            next_state.update({'matrix': torch.FloatTensor(transitions['next_matrix_state']).detach().to(self.device)})
        if 'non_spatial_feature':
            next_state.update({'vector': torch.FloatTensor(transitions['next_vector_state']).detach().to(self.device)})
        action = transitions["action"].detach().to(self.device)
        reward = transitions["reward"].detach().to(self.device)
        done = transitions["done"].detach().to(self.device)

        q1, _ = self.critic1(state)
        q1 = q1.gather(1, action.long()).squeeze()
        q2, _ = self.critic2(state)
        q2 = q2.gather(1, action.long()).squeeze()

        with torch.no_grad():
            next_pi, _ = self.actor(next_state)
            n_q1, _ = self.target_critic1(next_state)
            next_q1 = (next_pi * n_q1).sum(
                -1, keepdim=True
            )
            n_q2, _ = self.target_critic2(next_state)
            next_q2 = (next_pi * n_q2).sum(
                -1, keepdim=True
            )
            m = Categorical(next_pi)
            entropy = m.entropy()

        with torch.no_grad():
            min_next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * self.gamma * (
                    min_next_q.squeeze() + self.alpha * entropy
            )

        max_Q = torch.max(target_q).item()

        # Critic
        critic_loss1 = self.loss(q1, target_q)
        critic_loss2 = self.loss(q2, target_q)

        self.optimizer['critic1'].zero_grad(set_to_none=True)
        critic_loss1.backward()
        self.optimizer['critic1'].step()

        self.optimizer['critic2'].zero_grad(set_to_none=True)
        critic_loss2.backward()
        self.optimizer['critic2'].step()

        # Actor
        pi, _ = self.actor(state)
        c1, _ = self.critic1(state)
        c2, _ = self.critic2(state)
        q1 = (pi * c1).sum(-1, keepdim=True)
        q2 = (pi * c2).sum(-1, keepdim=True)
        m = Categorical(pi)
        entropy = m.entropy().unsqueeze(-1)

        min_q = torch.min(q1, q2)
        actor_loss = -((self.alpha.detach() * entropy) + min_q).mean()
        self.optimizer['actor'].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optimizer['actor'].step()

        # Alpha
        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach().mean()

        self.alpha = self.log_alpha.exp()

        if self.use_dynamic_alpha:
            self.optimizer['alpha'].zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.optimizer['alpha'].step()

        self.update_target_hard()

        result = {
            "critic_loss1": critic_loss1.item(),
            "critic_loss2": critic_loss2.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "max_Q": max_Q,
            "mean_Q": min_q.mean().item(),
            "alpha": self.alpha.item(),
            "entropy": entropy.mean().item(),
        }

    def update_target_soft(self):
        for t_p, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)
        for t_p, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)

    def update_target_hard(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def save(self, checkpoint_path: str):
        pass
        # print(f"...Save model to {path}...")
        # save_dict = {
        #     "actor": self.actor.state_dict(),
        #     "actor_optimizer": self.actor_optimizer.state_dict(),
        #     "critic1": self.critic1.state_dict(),
        #     "critic2": self.critic2.state_dict(),
        #     "critic_optimizer1": self.critic_optimizer1.state_dict(),
        #     "critic_optimizer2": self.critic_optimizer2.state_dict(),
        # }
        # if self.use_dynamic_alpha:
        #     save_dict["log_alpha"] = self.log_alpha
        #     save_dict["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        #
        # torch.save(save_dict, os.path.join(path, "ckpt"))

    def load(self, checkpoint_path: str):
        pass
        # print(f"...Load model from {path}...")
        # checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        # self.actor.load_state_dict(checkpoint["actor"])
        # self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        #
        # self.critic1.load_state_dict(checkpoint["critic1"])
        # self.critic1.load_state_dict(checkpoint["critic2"])
        # self.target_critic1.load_state_dict(self.critic1.state_dict())
        # self.target_critic2.load_state_dict(self.critic2.state_dict())
        # self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        # self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])
        #
        # if self.use_dynamic_alpha and "log_alpha" in checkpoint.keys():
        #     self.log_alpha = checkpoint["log_alpha"]
        #     self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    def sync_in(self, weights):
        self.actor.load_state_dict(weights)

    def sync_out(self, device="cpu"):
        weights = self.actor.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device)
        sync_item = {
            "weights": weights,
        }
        return sync_item
