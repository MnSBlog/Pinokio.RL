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
        epsilon_init = self._config['epsilon_init']  # 1.0
        epsilon_min = self._config['epsilon_min']  # 0.1
        explore_ratio = self._config['explore_ratio']  # 0.1
        run_step = 100000
        self.epsilon = epsilon_init
        self.explore_step = run_step * explore_ratio
        self.epsilon_delta = (epsilon_init - epsilon_min) / self.explore_step

        self._parameter = EasyDict(self._config)
        # Network
        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Optimizer
        opt_arg = [
            {'params': self.actor.parameters(), 'lr': self._config['actor_lr']},
        ]
        self.optimizer = getattr(torch.optim, self._config['optimizer'])(opt_arg)  # adam
        self.loss = getattr(nn, self._config['loss_function'])()  # smooth_l1_loss
        self._buffer = ReplayBuffer()

        self.hidden_state = copy.deepcopy(self.actor.init_h_state)
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.to(self.device)

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
            next_q = self.target_actor(next_state)
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
        torch.save(self.actor.state_dict(), checkpoint_path + 'actor.pth')
        torch.save(self.target_actor.state_dict(), checkpoint_path + 'target_actor.pth')

    def load(self, checkpoint_path: str):
        self.actor.load_state_dict(torch.load(checkpoint_path + 'actor.pth'))
        self.target_actor.load_state_dict(torch.load(checkpoint_path + 'target_actor.pth'))

    def __epsilon_decay(self):
        new_epsilon = self.epsilon - self.epsilon_delta
        self.epsilon = max(self._parameter.epsilon_min, new_epsilon)

    def update_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
