import copy
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
from buffer.per_buffer import PERBuffer
from agents.pytorch.dqn_agent import DQN
from utils.calculator import multiple


class Rainbow(DQN):
    def __init__(self, parameters: dict, actor, **kwargs):
        super(Rainbow, self).__init__(parameters=parameters, actor=actor)

        # self.action_size = multiple(self.actor.outputs_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.gamma = self._config['gamma']
        self.random_flag = True

        # PER
        self.alpha = self._config['alpha']
        self.beta = self._config['beta']
        self.learn_period = self._config['learn_period']
        self.learn_period_stamp = 0
        self.uniform_sample_prob = self._config['uniform_sample_prob']
        self.beta_add = (1 - self.beta) / 100000

        # C51
        self.v_min = self._config['v_min']
        self.v_max = self._config['v_max']
        self.num_support = self._config['num_support']

        # MultiStep
        self.buffer = PERBuffer(uniform_sample_prob=self.uniform_sample_prob)

        # C51
        self.delta_z = (self.v_max - self.v_min) / (self.num_support - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.num_support, device=self.device).view(1, -1)

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            actions, next_hidden = self.act(state=state, hidden=self.hidden_state)
        self.hidden_state = next_hidden

        return {"action": actions, "next_hidden": next_hidden}

    def act(self, state, hidden=[]):
        rtn_action = []
        last = 0
        logits, hidden = self.actor(x=state, h=hidden)
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            if self.random_flag:
                output_dim = output_dim // self.num_support
                batch_size = logits.shape[0]
                action = torch.randint(0, output_dim, size=(batch_size, 1))
            else:
                _, q_action = self.logits2q(logits[:, last:last + output_dim], output_dim)
                action = torch.argmax(q_action, -1, keepdim=True).cpu().numpy()
                last += output_dim
            rtn_action.append(action)
        return torch.stack(rtn_action, dim=0).detach(), hidden

    def update(self, next_state=None, done=None):
        transitions, weights, indices, sampled_p, mean_p = self.buffer.sample(
            self.beta, self._config['batch_size']
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

        logit, _ = self.actor(state, True)
        last = 0
        KL = None
        max_Q = None
        max_logit = None
        min_logit = None
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            p_logit, q_action = self.logits2q(logit[:, last:last + output_dim], output_dim)
            last += output_dim
            action_eye = torch.eye(q_action.shape[1]).to(self.device)
            sub_action = action[:, idx, :]
            sub_action = sub_action.squeeze(1)
            action_onehot = action_eye[sub_action.long()]
            p_action = torch.squeeze(action_onehot @ p_logit, 1)

            target_dist = torch.zeros(
                self._config['batch_size'], self.num_support, device=self.device, requires_grad=False
            )
            with torch.no_grad():
                # Double
                _, next_q_action = self.logits2q(self.actor(next_state)[0], output_dim)
                target_p_logit, _ = self.logits2q(self.target_actor(next_state)[0], output_dim)

                target_action = torch.argmax(next_q_action, -1, keepdim=True)
                target_action_onehot = action_eye[target_action.long()]
                target_p_action = torch.squeeze(target_action_onehot @ target_p_logit, 1)

                Tz = self.z

                b = torch.clamp(Tz - self.v_min, 0, self.v_max - self.v_min) / self.delta_z
                l = torch.floor(b).long()
                u = torch.ceil(b).long()

                support_eye = torch.eye(self.num_support, device=self.device)
                l_support_onehot = support_eye[l]
                u_support_onehot = support_eye[u]

                l_support_binary = torch.unsqueeze(u - b, -1)
                u_support_binary = torch.unsqueeze(b - l, -1)
                target_p_action_binary = torch.unsqueeze(target_p_action, -1)

                lluu = (
                    l_support_onehot * l_support_binary
                    + u_support_onehot * u_support_binary
                )

                target_dist += done * torch.mean(
                    l_support_onehot * u_support_onehot + lluu, 1
                )
                target_dist += (1 - done) * torch.sum(
                    target_p_action_binary * lluu, 1
                )
                target_dist /= torch.clamp(
                    torch.sum(target_dist, 1, keepdim=True), min=1e-8
                )

            if max_Q is None:
                max_Q = torch.max(q_action).item()
            else:
                max_Q += torch.max(q_action).item()
            if max_logit is None:
                max_logit = torch.max(logit).item()
            else:
                max_logit *= torch.max(logit).item()
            if min_logit is None:
                min_logit = torch.min(logit).item()
            else:
                min_logit *= torch.min(logit).item()

            # PER
            if KL is None:
                KL = -(target_dist * torch.clamp(p_action, min=1e-8).log()).sum(-1)
            else:
                KL += -(target_dist * torch.clamp(p_action, min=1e-8).log()).sum(-1)

        p_j = torch.pow(KL, self.alpha)

        for i, p in zip(indices, p_j):
            self.buffer.update_priority(p.item(), i)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

        loss = (weights * KL).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.update_target()
        result = {
            "loss": loss.item(),
            "beta": self.beta,
            "max_Q": max_Q,
            "max_logit": max_logit,
            "min_logit": min_logit,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
        }

        return result

    def logits2q(self, logits, action_size):
        action_size = action_size // self.num_support
        _logits = logits.view(logits.shape[0], action_size, self.num_support)
        p_logit = torch.exp(F.log_softmax(_logits, dim=-1))

        z_action = self.z.expand(p_logit.shape[0], action_size, self.num_support)
        q_action = torch.sum(z_action * p_logit, dim=-1)

        return p_logit, q_action
