import numpy as np
import copy
import torch
import torch.nn as nn
from easydict import EasyDict
from buffer.replay_buffer import ReplayBuffer
from agents.general_agent import GeneralAgent


class DQN(GeneralAgent):
    def __init__(self, parameters: dict, actor, **kwargs):
        super(DQN, self).__init__(parameters=parameters, actor=actor)
        # Hyperparameters
        gamma = self._config['gamma']  # 0.99,
        epsilon_init = self._config['epsilon_init']  # 1.0
        epsilon_min = self._config['epsilon_min']  # 0.1
        epsilon_eval = self._config['epsilon_eval']  # 0.0
        explore_ratio = self._config['epsilon_ratio']  # 0.1
        buffer_size = self._config['buffer_size']  # 50000
        batch_size = self._config['batch_size']  # 64
        start_train_step = self._config['start_step']  # 2000
        target_update_period = self._config['update_period']  # 500
        run_step = self._config['learning_rate']  # 1e6
        lr_decay = self._config['lr_decay']  # True
        self.epsilon = epsilon_init
        self.explore_step = run_step * explore_ratio
        self.epsilon_delta = (epsilon_init - epsilon_min) / self.explore_step

        self._parameter = EasyDict(self._config)
        # Network
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer
        opt_arg = [
            {'params': self.network.parameters(), 'lr': self._config['learning_rate']},
        ]
        self.optimizer = getattr(torch.optim, self._config['optimizer'])(opt_arg)  # adam
        self.loss = getattr(nn, self._config['loss_function'])()  # smooth_l1_loss
        self._buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        epsilon = self.epsilon

        if np.random.random() < epsilon:
            batch_size = (
                state[0].shape[0] if isinstance(state, list) else state.shape[0]
            )
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            action = (
                torch.argmax(self.network(self.as_tensor(state)), -1, keepdim=True)
                .cpu()
                .numpy()
            )
        return {"action": action}

    def update(self, next_state=None, done=None):
        transitions = self._buffer.sample(self.batch_size)
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        eye = torch.eye(self.action_size, device=self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        with torch.no_grad():
            max_q = torch.max(q).item()
            next_q = self.target_network(next_state)
            target_q = (
                    reward + (1 - done) * self._parameter.gamma * next_q.max(1, keepdims=True).values
            )

        loss = self.loss(q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        metrics = {'reward': max_q.detach().cpu(),
                   'entropy': self.epsilon,
                   'state_value': q.detach().cpu(),
                   'loss': loss.detach().cpu()}
        self.__epsilon_decay()

    def save(self, checkpoint_path: str):
        torch.save(self.network.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str):
        self.network.load_state_dict(torch.load(checkpoint_path))
        self.target_network.load_state_dict(torch.load(checkpoint_path))

    def __epsilon_decay(self):
        new_epsilon = self.epsilon - self.epsilon_delta
        self.epsilon = max(self._parameter.epsilon_min, new_epsilon)
