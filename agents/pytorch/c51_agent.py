import numpy as np
import torch
import torch.nn.functional as F
from agents.pytorch.dqn_agent import DQN


class C51(DQN):
    """C51 agent.
    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        v_min (float): minimum value of support. # 이거 우리 리워드레인지를 올바르게 스케일링하고 있는지 확인필요
        v_max (float): maximum value of support.
        num_support (int): number of support.
    """

    def __init__(self, parameters: dict, actor, **kwargs):
        super(C51, self).__init__(parameters=parameters, actor=actor)

        self.v_min = self._config['v_min']
        self.v_max = self._config['v_max']
        self.num_support = self._config['num_support']
        self.delta_z = (self.v_max - self.v_min) / (self.num_support - 1)
        self.z = torch.linspace(
            self.v_min, self.v_max, self.num_support, device=self.device
        ).view(1, -1)

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            actions = self.act(state=state, hidden=self.hidden_state)

        return actions

    def act(self, state, hidden=[]):
        rtn_action = []
        last = 0
        logits, self.hidden_state = self.actor(x=state, h=hidden)
        epsilon = self.epsilon
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            if np.random.random() < epsilon:
                output_dim = output_dim // self.num_support
                batch_size = logits.shape[0]
                action = torch.randint(0, output_dim, size=(batch_size, 1))
            else:
                _, q_action = self.logits2q(logits[:, last:last + output_dim], output_dim)
                action = torch.argmax(q_action, -1, keepdim=True)
                last += output_dim
            rtn_action.append(action)
        return {"action": torch.stack(rtn_action, dim=0).detach()}

    def update(self, next_state=None, done=None):
        transitions = self.buffer.sample(self._config['batch_size'])

        transitions["reward"] = transitions["reward"].unsqueeze(-1)
        transitions["done"] = transitions["done"].unsqueeze(-1)
        for key, transition in transitions.items():
            transitions[key] = transition.detach().to(self.device)

        state = dict()
        if 'spatial_feature' in self.actor.networks:
            state.update({'matrix': transitions['matrix_state']})
        if 'non_spatial_feature' in self.actor.networks:
            state.update({'vector': transitions['vector_state']})

        next_state = dict()
        if 'spatial_feature' in self.actor.networks:
            next_state.update({'matrix': transitions['matrix_state']})
        if 'non_spatial_feature' in self.actor.networks:
            next_state.update({'vector': transitions['vector_state']})

        action = transitions["action"]
        reward = transitions["reward"]
        done = transitions["done"]

        logit, _ = self.actor(state)
        last = 0
        loss = None
        max_Q = None
        max_logit = None
        min_logit = None
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            p_logit, q_action = self.logits2q(logit[:, last:last + output_dim], output_dim)
            last += output_dim
            action_eye = torch.eye(q_action.shape[1]).to(self.device)
            sub_action = action[:, idx, :]
            action_onehot = action_eye[sub_action.long()]
            p_action = torch.squeeze(action_onehot @ p_logit, 1)

            target_dist = torch.zeros(
                self._config['batch_size'], self.num_support, device=self.device, requires_grad=False
            )

            with torch.no_grad():
                target_p_logit, target_q_action = self.logits2q(self.target_actor(next_state)[0], output_dim)

                target_action = torch.argmax(target_q_action, -1, keepdim=True)
                target_action_onehot = action_eye[target_action.long()]
                target_p_action = torch.squeeze(target_action_onehot @ target_p_logit, 1)

                Tz = reward.expand(-1, self.num_support) + (1 - done) * self._config['gamma'] * self.z
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
                target_dist += (1 - done) * torch.sum(target_p_action_binary * lluu, 1)
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

            if loss is None:
                loss = -(target_dist * torch.clamp(p_action, min=1e-8).log()).sum(-1).mean()
            else:
                loss += -(target_dist * torch.clamp(p_action, min=1e-8).log()).sum(-1).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.update_target()
        self._epsilon_decay()

        result = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "max_Q": max_Q,
            "max_logit": max_logit,
            "min_logit": min_logit,
        }
        return result

    def logits2q(self, logits, action_size):
        action_size = action_size // self.num_support
        _logits = logits.view(logits.shape[0], action_size, self.num_support)
        _logits_max = torch.max(_logits, -1, keepdim=True).values
        p_logit = torch.exp(F.log_softmax(_logits - _logits_max, dim=-1))

        z_action = self.z.expand(p_logit.shape[0], action_size, self.num_support)
        q_action = torch.sum(z_action * p_logit, dim=-1)

        return p_logit, q_action
