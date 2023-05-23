import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.calculator import multiple
from buffer.replay_buffer import ReplayBuffer
from agents.general_agent import PolicyAgent


class TD3(PolicyAgent):
    def __init__(self, parameters: dict, actor: nn.Module, critic: nn.Module):
        super(TD3, self).__init__(parameters=parameters, actor=actor, critic=critic)

        hidden_size = 256,
        actor = "deterministic_policy",
        critic = "continuous_q_network",
        head = "mlp",
        optim_config = {
            "actor": "adam",
            "critic": "adam",
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
        },
        gamma = 0.99,
        buffer_size = 50000,
        batch_size = 128,
        start_train_step = 1000,
        initial_random_step = 0,
        tau = 1e-3,
        update_delay = 2,
        action_noise_std = 0.1,
        target_noise_std = 0.2,
        target_noise_clip = 0.5,
        run_step = 1e6,
        lr_decay = True,
        device = None,

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic)
        self.target_critic2 = copy.deepcopy(self.critic)

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
        self.optimizer['critic1'] = getattr(torch.optim, self._config['critic_optimizer'])(opt_arg)
        self.optimizer['critic2'] = getattr(torch.optim, self._config['critic_optimizer'])(opt_arg)
        self.loss['critic'] = getattr(nn, self._config['loss_function'])()

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.initial_random_step = initial_random_step
        self.num_random_step = 0
        self.num_learn = 0
        self.run_step = run_step
        self.lr_decay = lr_decay

        self.action_size = multiple(self.actor.outputs_dim)
        self.update_delay = update_delay
        self.action_noise_std = action_noise_std
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.actor_loss = 0.0

    @torch.no_grad()
    def act(self, state, training=True):
        self.actor.train(training)
        if training and self.num_random_step < self.initial_random_step:
            action = np.random.uniform(-1.0, 1.0, (1, self.action_size))
            self.num_random_step += 1
        else:
            action = self.actor(self.as_tensor(state))
            action = action.cpu().numpy()
            if training:
                noise = np.random.normal(0, self.action_noise_std, self.action_size)
                action = (action + noise).clip(-1.0, 1.0)
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
            noise = (torch.randn_like(action) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_action = (self.target_actor(next_state) + noise).clamp(-1.0, 1.0)
            next_q1 = self.target_critic1(next_state, next_action)
            next_q2 = self.target_critic2(next_state, next_action)
            min_next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * self.gamma * min_next_q

        critic_loss1 = F.mse_loss(target_q, self.critic1(state, action))
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        critic_loss2 = F.mse_loss(target_q, self.critic2(state, action))
        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        max_Q = torch.max(target_q, axis=0).values.cpu().numpy()[0]

        # Delayed Actor Update
        if self.num_learn % self.update_delay == 0:
            action_pred = self.actor(state)
            actor_loss = -self.critic1(state, action_pred).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_loss = actor_loss.item()
            if self.num_learn > 0:
                self.update_target_soft()

        self.num_learn += 1

        self.result = {
            "critic_loss1": critic_loss1.item(),
            "critic_loss2": critic_loss2.item(),
            "actor_loss": self.actor_loss,
            "max_Q": max_Q,
        }

        return self.result

    def update_target_soft(self):
        for t_p, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)
        for t_p, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
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
                    step,
                    [
                        self.actor_optimizer,
                        self.critic_optimizer1,
                        self.critic_optimizer2,
                    ],
                )

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        save_dict = {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic_optimizer1": self.critic_optimizer1.state_dict(),
            "critic_optimizer2": self.critic_optimizer2.state_dict(),
        }
        torch.save(save_dict, os.path.join(path, "ckpt"))

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic1.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])

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
