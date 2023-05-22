from collections import deque
import torch
import numpy as np
from buffer.per_buffer import PERBuffer
from agents.pytorch.dqn_agent import DQN


class ApeX(DQN):
    def __init__(self, parameters: dict, network):
        super(ApeX, self).__init__(parameters=parameters, network=network)
        # ApeX
        self.num_transitions = 0

        # PER
        self.learn_period_stamp = 0
        self.beta_add = (1 - self._config['beta']) / self._config['learning_rate']

        # MultiStep
        self._buffer = PERBuffer(self._config['buffer_size'], self._config['uniform_sample_prob'])
        self.tmp_buffer = deque(maxlen=self._config['n_step'] + 1)

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            actions, q, next_hidden = self.act(state=state, hidden=self.hidden_state)

        return actions

    def act(self, state, hidden=None):
        epsilon = self.epsilon

        q = self.network(state)
        if np.random.random() < epsilon:
            batch_size = (
                state[0].shape[0] if isinstance(state, list) else state.shape[0]
            )
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            action = torch.argmax(q, -1, keepdim=True).cpu().numpy()
        q = np.take(q.cpu().numpy(), action)
        return action, q, hidden

    def update(self, next_state=None, done=None):
        transitions, weights, indices, sampled_p, mean_p = self._buffer.sample(
            self._config['beta'], self._config['batch_size']
        )
        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.network(next_state)
            max_a = torch.argmax(next_q, axis=1)
            max_one_hot_action = eye[max_a.long()]

            next_target_q = self.target_network(next_state)
            target_q = (next_target_q * max_one_hot_action).sum(1, keepdims=True)

            for i in reversed(range(self._config['n_step'])):
                target_q = reward[:, i] + (1 - done[:, i]) * self._config['gamma'] * target_q

        # Update sum tree
        td_error = abs(target_q - q)
        p_j = torch.pow(td_error, self._config['alpha'])
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

        loss = (weights * (td_error**2)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._config['clip_grad_norm'])
        self.optimizer.step()

        metrics = {
            "loss": loss.item(),
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
            "num_transitions": self.num_transitions,
        }

        self.__epsilon_decay()

    def interact_callback(self, transition):
        _transition = {}
        self.tmp_buffer.append(transition)
        if len(self.tmp_buffer) == self.tmp_buffer.maxlen:
            _transition["state"] = self.tmp_buffer[0]["state"]
            _transition["action"] = self.tmp_buffer[0]["action"]
            _transition["next_state"] = self.tmp_buffer[-1]["state"]

            for key in self.tmp_buffer[0].keys():
                if key not in ["state", "action", "next_state"]:
                    _transition[key] = np.stack(
                        [t[key] for t in self.tmp_buffer][:-1], axis=1
                    )

            target_q = self.tmp_buffer[-1]["q"]
            for i in reversed(range(self.n_step)):
                target_q = (
                    self.tmp_buffer[i]["reward"]
                    + (1 - self.tmp_buffer[i]["done"]) * self.gamma * target_q
                )
            priority = abs(target_q - self.tmp_buffer[0]["q"])

            _transition["priority"] = priority
            del _transition["q"]

        return _transition
