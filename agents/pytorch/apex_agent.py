from collections import deque
import torch
import numpy as np
from buffer.per_buffer import PERBuffer
from agents.pytorch.dqn_agent import DQN


class ApeX(DQN):
    def __init__(self, parameters: dict, actor, **kwargs):
        super(ApeX, self).__init__(parameters=parameters, actor=actor)
        # ApeX
        self.num_transitions = 0

        # PER
        self.learn_period_stamp = 0
        self.beta_add = (1 - self._config['beta']) / self._config['actor_lr']
        self.buffer = PERBuffer(uniform_sample_prob=self._config['uniform_sample_prob'])

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            actions, q, next_hidden = self.act(state=state, hidden=self.hidden_state)

        return {"action": actions, "q": q, "next_hidden": next_hidden}

    def act(self, state, hidden=None):
        epsilon = self.epsilon
        rtn_action = []
        last = 0
        q, hidden = self.actor(x=state, h=hidden)
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            if np.random.random() < epsilon:
                batch_size = q.shape[0]
                action = torch.randint(0, output_dim, size=(batch_size, 1))
            else:
                action = torch.argmax(q[:, last:last + output_dim], -1, keepdim=True).cpu().numpy()
                last += output_dim
                q = np.take(q.cpu().numpy(), action)
            rtn_action.append(action)
        return torch.stack(rtn_action, dim=0).detach(), q, hidden

    def update(self, next_state=None, done=None):
        transitions, weights, indices, sampled_p, mean_p = self._buffer.sample(
            self._config['beta'], self._config['batch_size']
        )
        state = dict()
        if 'spatial_feature' in self.actor.networks:
            state.update({'matrix': torch.FloatTensor(transitions['matrix_state']).to(self.device)})
        if 'non_spatial_feature' in self.actor.networks:
            state.update({'vector': torch.FloatTensor(transitions['vector_state']).to(self.device)})

        next_state = dict()
        if 'spatial_feature' in self.actor.networks:
            next_state.update({'matrix': torch.FloatTensor(transitions['next_matrix_state']).to(self.device)})
        if 'non_spatial_feature':
            next_state.update({'vector': torch.FloatTensor(transitions['next_vector_state']).to(self.device)})

        action = transitions["action"]
        reward = transitions["reward"]
        reward = torch.FloatTensor(reward).to(self.device)
        done = transitions["done"]
        done = torch.FloatTensor(done).to(self.device)
        loss = None
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            eye = torch.eye(output_dim).to(self.device)
            one_hot_action = eye[action.view(-1).long()]
            q = (self.actor(state)[0] * one_hot_action).sum(1, keepdims=True)

            with torch.no_grad():
                max_Q = torch.max(q).item()
                next_q, _ = self.actor(next_state)
                max_a = torch.argmax(next_q, dim=1)
                max_one_hot_action = eye[max_a.long()]

                next_target_q = self.target_actor(next_state)
                target_q = (next_target_q * max_one_hot_action).sum(1, keepdims=True)

                for i in reversed(range(self._config['n_step'])):
                    target_q = reward[:, i] + (1 - done[:, i]) * self._config['gamma'] * target_q

            # Update sum tree
            td_error = abs(target_q - q)
            p_j = torch.pow(td_error, self._config['alpha'])
            for i, p in zip(indices, p_j):
                self.buffer.update_priority(p.item(), i)

            weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

            if loss is None:
                loss = (weights * (td_error ** 2)).mean()
            else:
                loss += (weights * (td_error ** 2)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._config['clip_grad_norm'])
        self.optimizer.step()

        metrics = {
            "loss": loss.item(),
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
            "num_transitions": self.num_transitions,
        }

        self._epsilon_decay()
        self.update_target()
