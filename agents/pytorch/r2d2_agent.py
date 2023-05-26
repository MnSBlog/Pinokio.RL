import copy
from collections import deque
from itertools import islice
import torch
import torch.nn.functional as F
import numpy as np
from agents.pytorch.apex_agent import ApeX


class R2D2(ApeX):
    """Recurrent Replay Distributed DQN (R2D2) agent.

    Args:
        seq_len (int): sequence length of RNN input.
        n_burn_in (int): burn-in period. (unit: step)
        zero_padding (bool): parameter that determine whether to use zero padding.
        eta (float): priority exponent.
    """

    def __init__(self, parameters: dict, actor, **kwargs):
        super(R2D2, self).__init__(parameters=parameters, actor=actor, **kwargs)
        assert 0 < self._config['n_burn_in'] < self._config['seq_len']
        # actor non-spatial 도입단 LSTM layer 추가
        old_layer = self.actor.networks['input_layer']
        input_size = old_layer[0].in_features + sum(self.actor.outputs_dim)
        old_layer = self.actor.networks['neck']
        self.hidden_size = old_layer[0].in_features
        self.actor.networks['input_layer'] = torch.nn.LSTM(input_size=input_size,
                                                           hidden_size=self.hidden_size, batch_first=True)

        # actor output value layer 추가
        value_index = 0
        in_feature = self.actor.networks['head0'][0].in_features
        for seq in self.actor.networks.keys():
            if 'head' in seq:
                index = seq.replace('head', '')
                if value_index < int(index):
                    value_index = index
        self.actor.networks['head' + str(value_index + 1)] = torch.nn.Linear(in_feature, 1)

        # actor network 설정 변경
        self.actor.recurrent = True
        self.actor.rnn_len = 1
        self.value_index = value_index + 1
        self.actor.to(self.device)
        self.target_actor = copy.deepcopy(self.actor)

        # R2D2
        self.seq_len = self._config['seq_len']
        self.n_burn_in = self._config['n_burn_in']
        self.zero_padding = self._config['zero_padding']
        self.eta = self._config['eta']

        self.prev_action = None
        self.store_period = self.seq_len // 2
        self.store_period_stamp = 0
        self.store_start = True
        self.n_step = self.actor.local_len

        hidden_in = (
            torch.zeros(1, 1, self.hidden_size).to(self.device),
            torch.zeros(1, 1, self.hidden_size).to(self.device),
        )
        self.hidden_state = hidden_in
        self.tmp_buffer = deque(maxlen=self.n_step + self.seq_len)

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            return self.act(state=state, hidden=self.hidden_state)

    def act(self, state, hidden=None):
        epsilon = self.epsilon
        rtn_action = []
        rtn_q = []
        last = 0
        state = self.__model_preforward(state)
        q, hidden_in, hidden_out, prev_action_onehot = self.__model_forward(state)
        hidden_h = hidden_in[0]
        hidden_c = hidden_in[1]
        prev_action_onehot = prev_action_onehot[:, -1]
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            if np.random.random() < epsilon:
                batch_size = q.shape[0]
                action = torch.randint(0, output_dim, size=(batch_size, 1)).to(self.device)
                q = torch.take(q[:, -1], action)
            else:
                sub_q = q[:, last:last + output_dim]
                action = torch.argmax(sub_q, -1)
                q = torch.take(sub_q[:, -1], action)
            rtn_action.append(action)
            rtn_q.append(q)
            self.hidden_state = hidden_out
            self.prev_action = action

        return {
            "action": torch.cat(rtn_action, dim=0).detach(),
            "state": state,
            "prev_action_onehot": prev_action_onehot,
            "q": torch.cat(rtn_q, dim=0).detach(),
            "hidden_h": hidden_h,
            "hidden_c": hidden_c,
        }

    def update(self, next_state=None, done=None):
        transitions, weights, indices, sampled_p, mean_p = self.buffer.sample(
            self._config['beta'], self._config['batch_size']
        )
        for key, value in transitions.items():
            if isinstance(transitions[key], np.ndarray):
                continue
            else:
                transitions[key] = transitions[key].squeeze(1)

        state = transitions["state"][:, : self.seq_len]
        prev_action_onehot = transitions["prev_action_onehot"][:, : self.seq_len]
        next_state = transitions["state"][:, self.n_step:]
        next_prev_action_onehot = transitions["prev_action_onehot"][:, self.n_step:]
        action = transitions["action"][:, : self.seq_len]
        reward = transitions["reward"]
        done = transitions["done"]
        hidden_h = transitions["hidden_h"].transpose(0, 1).contiguous()
        hidden_c = transitions["hidden_c"].transpose(0, 1).contiguous()
        next_hidden_h = transitions["next_hidden_h"].transpose(0, 1).contiguous()
        next_hidden_c = transitions["next_hidden_c"].transpose(0, 1).contiguous()
        hidden = (hidden_h, hidden_c)
        next_hidden = (next_hidden_h, next_hidden_c)
        loss = None
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            eye = torch.eye(output_dim).to(self.device)
            one_hot_action = eye[action.view(-1, self.seq_len).long()][:, self.n_burn_in:]
            print('================================================')
            print(state.shape)
            print('------------------------------------------------')
            print(prev_action_onehot.shape)
            print('================================================')
            q_pred = self.get_q(state, prev_action_onehot, hidden, self.actor)
            q = (q_pred * one_hot_action).sum(-1, keepdims=True)
            with torch.no_grad():
                max_Q = torch.max(q).item()
                next_q = self.get_q(
                    next_state, next_prev_action_onehot, next_hidden, self.actor
                )
                max_a = torch.argmax(next_q, dim=-1)
                max_one_hot_action = eye[max_a.long()]

                next_target_q = self.get_q(
                    next_state, next_prev_action_onehot, next_hidden, self.target_actor
                )
                target_q = (next_target_q * max_one_hot_action).sum(-1, keepdims=True)
                target_q = self.inv_val_rescale(target_q)

                for i in reversed(range(self.n_step)):
                    target_q = (
                            reward[:, i + self.n_burn_in: i + self.seq_len]
                            + (1 - done[:, i + self.n_burn_in: i + self.seq_len])
                            * self._config['gamma']
                            * target_q
                    )

                target_q = self.val_rescale(target_q)

            # Update sum tree
            td_error = abs(target_q - q)
            priority = self.eta * torch.max(td_error, dim=1).values + (
                    1 - self.eta
            ) * torch.mean(td_error, dim=1)
            p_j = torch.pow(priority, self._config['alpha'])
            for i, p in zip(indices, p_j):
                self.buffer.update_priority(p.item(), i)

            # Annealing beta
            self._config['beta'] = min(1.0, self._config['beta'] + self.beta_add)

            #         weights = torch.FloatTensor(weights[..., np.newaxis, np.newaxis]).to(self.device)
            #         loss = (weights * (td_error**2)).mean()

            weights = torch.FloatTensor(weights[..., np.newaxis]).to(self.device)
            if loss is None:
                loss = (weights * (td_error[:, -1] ** 2)).mean()
            else:
                loss += (weights * (td_error[:, -1] ** 2)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._config['clip_grad_norm'])
        self.optimizer.step()

        result = {
            "loss": loss.item(),
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
            "num_transitions": self.num_transitions,
        }

        return result

    def interact_callback(self, transition):
        _transition = {}
        self.tmp_buffer.append(transition)

        if (self.store_start or self.store_period_stamp == self.store_period) and (
                (self.zero_padding and len(self.tmp_buffer) >= self.n_step + 1)
                or (
                        not self.zero_padding and len(self.tmp_buffer) == self.tmp_buffer.maxlen
                )
        ):
            _transition["hidden_h"] = self.tmp_buffer[0]["hidden_h"]
            _transition["hidden_c"] = self.tmp_buffer[0]["hidden_c"]
            _transition["next_hidden_h"] = self.tmp_buffer[self.n_step]["hidden_h"]
            _transition["next_hidden_c"] = self.tmp_buffer[self.n_step]["hidden_c"]

            for key in self.tmp_buffer[0].keys():
                # if key not in ['action', 'hidden_h', 'hidden_c', 'next_state']:
                if key not in ["hidden_h", "hidden_c", "next_vector_state",
                               "next_matrix_state", "vector_state", "matrix_state"]:
                    if key in ["q", "prev_action_onehot"]:
                        _transition[key] = torch.stack(
                            [t[key] for t in self.tmp_buffer], dim=1
                        )
                    elif key == "state":
                        _transition[key] = torch.cat(
                            [t[key] for t in self.tmp_buffer], dim=1
                        )
                    else:
                        if len(self.tmp_buffer[0][key]) != 0:
                            _transition[key] = torch.stack(
                                [t[key] for t in self.tmp_buffer][:-1], dim=1
                            )

            # state sequence zero padding
            if self.zero_padding and len(self.tmp_buffer) < self.tmp_buffer.maxlen:
                lack_dims = self.tmp_buffer.maxlen - len(self.tmp_buffer)
                zero_state = torch.zeros((1, lack_dims, *transition["state"].shape[2:])).to(self.device)
                _transition["state"] = torch.concat((zero_state, _transition["state"]), dim=1)
                zero_prev_action_onehot = torch.zeros(
                    (1, lack_dims, *transition["prev_action_onehot"].shape[1:])
                ).to(self.device)
                _transition["prev_action_onehot"] = torch.concat(
                    (zero_prev_action_onehot, _transition["prev_action_onehot"]), dim=1
                )
                zero_action = torch.zeros((1, lack_dims, *transition["action"].shape[1:])).to(self.device)
                _transition["action"] = torch.concat(
                    (zero_action, _transition["action"]), dim=1
                )
                zero_reward = torch.zeros((1, lack_dims, *transition["reward"].shape[1:]))
                _transition["reward"] = torch.concat(
                    (zero_reward, _transition["reward"]), dim=1
                )
                zero_done = torch.zeros((1, lack_dims, *transition["done"].shape[1:]))
                _transition["done"] = torch.concat(
                    (zero_done, _transition["done"]), dim=1
                )
                zero_q = torch.zeros((1, lack_dims, *transition["q"].shape[1:])).to(self.device)
                _transition["q"] = torch.concat((zero_q, _transition["q"]), dim=1)

                if lack_dims > self.n_step:
                    _transition["next_hidden_h"] = self.tmp_buffer[0]["hidden_h"]
                    _transition["next_hidden_c"] = self.tmp_buffer[0]["hidden_c"]
                else:
                    _transition["next_hidden_h"] = self.tmp_buffer[
                        self.n_step - lack_dims
                        ]["hidden_h"]
                    _transition["next_hidden_c"] = self.tmp_buffer[
                        self.n_step - lack_dims
                        ]["hidden_c"]

            _transition["reward"] = _transition["reward"].to(self.device).unsqueeze(2)
            _transition["done"] = _transition["done"].to(self.device).unsqueeze(2)
            target_q = self.inv_val_rescale(
                _transition["q"][:, self.n_burn_in + self.n_step:]
            )
            for i in reversed(range(self.n_step)):
                target_q = (
                        _transition["reward"][:, i + self.n_burn_in: i + self.seq_len]
                        + (
                                1
                                - _transition["done"][:, i + self.n_burn_in: i + self.seq_len]
                        )
                        * self._config['gamma']
                        * target_q
                )

            target_q = self.val_rescale(target_q)
            td_error = abs(
                target_q - _transition["q"][:, self.n_burn_in: self.seq_len]
            )
            priority = self.eta * torch.max(td_error, dim=1)[0] + (1 - self.eta) * torch.mean(
                td_error, dim=1
            )
            _transition["priority"] = priority
            del _transition["q"]

            self.store_start = False
            self.store_period_stamp -= self.store_period

        if (
                len(self.tmp_buffer) > self.n_step
                and self.tmp_buffer[-self.n_step - 1]["done"]
        ):
            self.store_start = True
            self.tmp_buffer = deque(
                islice(self.tmp_buffer, len(self.tmp_buffer) - self.n_step, None),
                maxlen=self.tmp_buffer.maxlen,
            )

        self.store_period_stamp += 1
        if transition["done"]:
            self.hidden_state = None
            self.prev_action = None

        return _transition

    def get_q(self, state, prev_action_onehot, hidden_in, network):
        with torch.no_grad():
            burn_in_q, hidden_in, hidden_out, _ = self.__model_forward(
                state[:, : self.n_burn_in],
                prev_action_onehot[:, : self.n_burn_in],
                hidden_in,
                network
            )
        q, hidden_in, hidden_out, _ = self.__model_forward(
            state[:, self.n_burn_in :],
            prev_action_onehot[:, self.n_burn_in:],
            hidden_out,
            network
        )

        return q

    @staticmethod
    def val_rescale(val, eps=1e-3):
        return (val / (abs(val) + 1e-10)) * ((abs(val) + 1) ** (1 / 2) - 1) + (
                eps * val
        )

    @staticmethod
    def inv_val_rescale(val, eps=1e-3):
        # Reference: Proposition A.2 in paper "Observe and Look Further: Achieving Consistent Performance on Atari"
        return (val / (abs(val) + 1e-10)) * (
                (((1 + 4 * eps * (abs(val) + 1 + eps)) ** (1 / 2) - 1) / (2 * eps)) ** 2 - 1
        )

    def __model_preforward(self, state):
        x = self.actor.pre_forward(state)
        return x.unsqueeze(dim=1)

    def __model_forward(self, x, prev_action_onehot=None, hidden_state=None, network=None):
        if hidden_state is None:
            hidden_state = copy.deepcopy(self.hidden_state)
        if network is None:
            network = self.actor

        if self.prev_action is None:
            prev_action_onehot = torch.zeros(
                (x.shape[0], 1, sum(network.outputs_dim)), device=self.device
            )
        else:
            if prev_action_onehot is None:
                prev_action_onehot = F.one_hot(
                    torch.tensor(self.prev_action, dtype=torch.long, device=self.device),
                    sum(network.outputs_dim),
                )
        x = torch.cat([x, prev_action_onehot], dim=-1)
        if hidden_state is None:
            hidden_state = (
                torch.zeros(1, x.size(0), self.hidden_size).to(x.device),
                torch.zeros(1, x.size(0), self.hidden_size).to(x.device),
            )

        x, hidden_out = network.networks['input_layer'](x, hidden_state)
        if 'neck' in network.networks:
            x = network.networks['neck'](x)
        outputs = []
        dim = len(x.shape) - 1
        value = network.networks["head" + str(network.n_of_heads)](x)
        for index in range(network.n_of_heads):
            key = "head" + str(index)
            outputs.append(network.networks[key](x) + value)

        return torch.cat(outputs, dim=dim), hidden_state, hidden_out, prev_action_onehot
