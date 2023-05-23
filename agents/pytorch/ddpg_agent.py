import copy
from utils.calculator import multiple
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from agents.pytorch.utilities import get_device
from buffer.replay_buffer import ReplayBuffer
from agents.general_agent import PolicyAgent
from agents.pytorch.utilities import OuNoise


class DDPG(PolicyAgent):
    def __init__(self, parameters: dict, actor: nn.Module, critic: nn.Module):
        device = get_device("auto")
        super(DDPG, self).__init__(parameters=parameters, actor=actor.to(device), critic=critic.to(device))
        gamma = 0.99,
        buffer_size = 50000,
        batch_size = 128,
        start_train_step = 2000,
        tau = 1e-3,
        run_step = 1e6,
        lr_decay = True,
        # OU noise
        mu = 0,
        theta = 1e-3,
        sigma = 2e-3,

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # Optimizer
        self.optimizer = dict()
        self.loss = dict()
        # Actor Optimizer
        opt_arg = [
            {'params': self.actor.parameters(), 'lr': self._config['actor_lr']},
        ]
        self.optimizer['actor'] = getattr(torch.optim, self._config['actor_optimizer'])(opt_arg)
        self.loss['actor'] = getattr(nn, self._config['loss_function'])()
        # Critic Optimizer
        opt_arg = [
            {'params': self.critic.parameters(), 'lr': self._config['critic_lr']}
        ]
        self.optimizer['critic'] = getattr(torch.optim, self._config['critic_optimizer'])(opt_arg)
        self.loss['critic'] = getattr(nn, self._config['loss_function'])()

        action_size = multiple(self.actor.outputs_dim)
        self.OU = OuNoise(action_size, mu, theta, sigma)

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.num_learn = 0
        self.run_step = run_step
        self.lr_decay = lr_decay

    @torch.no_grad()
    def act(self, state, training=True):
        self.actor.train(training)
        mu = self.actor(self.as_tensor(state))
        mu = mu.cpu().numpy()
        action = mu + self.OU.sample().clip(-1.0, 1.0) if training else mu
        return {"action": action}

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        # Critic Update
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            next_q = self.target_critic(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * next_q
        q = self.critic(state, action)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        max_Q = torch.max(target_q, axis=0).values.cpu().numpy()[0]

        # Actor Update
        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.num_learn += 1

        result = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "max_Q": max_Q,
        }
        return result

    def update_target_soft(self):
        for t_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)
        for t_p, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)

        if self.memory.size >= self.batch_size and step >= self.start_train_step:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(
                    step, [self.actor_optimizer, self.critic_optimizer]
                )
        if self.num_learn > 0:
            self.update_target_soft()

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        save_dict = {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        torch.save(save_dict, os.path.join(path, "ckpt"))

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

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
