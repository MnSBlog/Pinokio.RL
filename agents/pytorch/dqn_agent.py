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
        self.update_num = 0
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
        self.buffer = ReplayBuffer()

        self.hidden_state = copy.deepcopy(self.actor.init_h_state)
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.to(self.device)

    def select_action(self, state):
        state = self.convert_to_torch(state)
        epsilon = self.epsilon
        rtn_action = []
        last = 0
        q, _ = self.actor(state)
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            if np.random.random() < epsilon:
                batch_size = q.shape[0]
                action = torch.randint(0, output_dim, size=(batch_size, 1))
            else:
                action = torch.argmax(q[:, last:last + output_dim], -1, keepdim=True)
                last += output_dim
            rtn_action.append(action)
        return {"action": torch.stack(rtn_action, dim=0).detach()}

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

        q, _ = self.actor(state, True)
        next_q, _ = self.target_actor(next_state)
        loss = None
        last = 0
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            eye = torch.eye(output_dim, device=self.device)
            one_hot_action = eye[action.view(-1).long()]
            sub_q = (q[:, last:last + output_dim] * one_hot_action).sum(1, keepdims=True).squeeze()
            with torch.no_grad():
                max_q = torch.max(sub_q).item()
                sub_next = next_q[:, last:last + output_dim].max(1, keepdims=True).values.squeeze()
                target_q = (
                        reward + (1 - done) * self._config['gamma'] * sub_next
                )

            if loss is None:
                loss = self.loss(sub_q, target_q)
            else:
                loss += self.loss(sub_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # metrics = {'reward': max_q
        #            'entropy': self.epsilon,
        #            'state_value': q,
        #            'loss': loss.detach().cpu()}
        self._epsilon_decay()
        self.update_target()

    def save(self, checkpoint_path: str):
        torch.save(self.actor.state_dict(), checkpoint_path + 'actor.pth')
        torch.save(self.target_actor.state_dict(), checkpoint_path + 'target_actor.pth')

    def load(self, checkpoint_path: str):
        self.actor.load_state_dict(torch.load(checkpoint_path + 'actor.pth'))
        self.target_actor.load_state_dict(torch.load(checkpoint_path + 'target_actor.pth'))

    def _epsilon_decay(self):
        new_epsilon = self.epsilon - self.epsilon_delta
        self.epsilon = max(self._config['epsilon_min'], new_epsilon)

    def update_target(self):
        self.update_num += 1
        if self.update_num == self._config['target_update_period']:
            self.update_num = 0
            self.target_actor.load_state_dict(self.actor.state_dict())
            return True
        else:
            return False
