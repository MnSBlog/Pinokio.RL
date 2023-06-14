import copy
import numpy as np
import torch
import torch.nn as nn
from buffer.rollout_buffer import RolloutBuffer
from agents.pytorch.utilities import get_device
from agents.general_agent import PolicyAgent
from torch.distributions import Categorical


class A2C(PolicyAgent):

    def __init__(self, parameters: dict, actor: nn.Module, critic: nn.Module):
        device = get_device("auto")
        super(A2C, self).__init__(parameters=parameters, actor=actor.to(device), critic=critic.to(device))
        # Hyper-parameters
        gamma = self._config['gamma']
        batch_size = self._config['epochs']
        actor_lr = self._config['actor_lr']
        critic_lr = self._config['critic_lr']

        # Buffer
        self.buffer = RolloutBuffer()
        # Optimizer
        self.optimizer = dict()
        self.loss = dict()
        # Actor Optimizer
        opt_arg = [
            {'params': self.actor.parameters(), 'lr': actor_lr},
        ]
        self.optimizer['actor'] = getattr(torch.optim, self._config['actor_optimizer'])(opt_arg)
        self.loss['actor'] = getattr(nn, self._config['loss_function'])()
        # Critic Optimizer
        opt_arg = [
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ]
        self.optimizer['critic'] = getattr(torch.optim, self._config['critic_optimizer'])(opt_arg)
        self.loss['critic'] = getattr(nn, self._config['loss_function'])()

        self.device = device
        self.hidden_state = copy.deepcopy(self.actor.init_h_state)
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.to(self.device)

        # Metric list
        self.metric_list = ['reward', 'state_value']

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            actions, action_logprobs, next_hidden = self.act(state=state, hidden=self.hidden_state)

        return {"action": actions, "action_logprobs": action_logprobs, "next_hidden": next_hidden}

    def act(self, state, hidden=None):
        rtn_action = []
        action_logprob = None
        outputs, hidden = self.actor(x=state, h=hidden)
        last = 0
        if len(self.actor.action_mask) > 0:
            outputs *= self.actor.action_mask

        for idx, output_dim in enumerate(self.actor.outputs_dim):
            dist = Categorical(outputs[:, last:last + output_dim])
            action = dist.sample()
            if action_logprob is None:
                action_logprob = dist.log_prob(action)
            else:
                action_logprob += dist.log_prob(action)
            rtn_action.append(action.detach())
            last += output_dim
        return torch.stack(rtn_action, dim=0).detach(), action_logprob.detach(), hidden

    def update(self, next_state=None, done=None):
        # unpack memory
        transitions = self.buffer.sample()
        # get td target
        next_v_values = self.critic(transitions['state'])
        td_targets = self.get_td_target(transitions['reward'], next_v_values.numpy(), transitions['done'])
        # critic update
        self.update_x(x='critic', state=transitions['state'], td_targets=td_targets)

        # 어드밴티지 계산
        v_values = self.critic(transitions['state'])
        next_v_values = self.critic(transitions['next_state'])
        advantages = transitions['reward'] + self._config['gamma'] * next_v_values - v_values

        metrics = {'reward': transitions['reward'].detach().cpu(),
                   'state_value': next_v_values.detach().cpu()}

        # actor update
        self.update_x(x='actor', state=transitions['state'],
                      actions=transitions['actions'],
                      advantage=advantages)
        # insert metric
        self.insert_metrics(metrics)

    def get_td_target(self, rew, next_v, done):
        y_i = np.zeros(next_v.shape)
        for i in range(next_v.shape[0]):
            if done[i]:
                y_i[i] = rew[i]
            else:
                y_i[i] = rew[i] + self._config['gamma'] * next_v[i]
        return y_i

    def update_x(self, **kwargs):
        # MSE error 사용
        if kwargs['x'] == 'critic':
            td_hat = self.critic(kwargs['state'])
            loss = self.loss['critic'](kwargs['td_targets'] - td_hat)
        else:
            _, action_logprobs, _ = self.act(state=kwargs['state'], hidden=self.hidden_state)
            # 손실함수
            loss_policy = action_logprobs * kwargs['advantage']
            loss = torch.sum(-loss_policy)

        self.optimizer[kwargs['x']].zero_grad()
        loss.mean().backward()
        self.optimizer[kwargs['x']].step()

    def save(self, checkpoint_path: str):
        if ".pth" not in checkpoint_path:
            checkpoint_path = checkpoint_path + '.pth'
        actor_path = checkpoint_path.replace(".pth", "actor.pth")
        critic_path = checkpoint_path.replace(".pth", "critic.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, checkpoint_path: str):
        if "actor" in checkpoint_path:
            actor_path = checkpoint_path
            critic_path = checkpoint_path.replace("actor.pth", "critic.pth")
        elif "critic" in checkpoint_path:
            critic_path = checkpoint_path
            actor_path = checkpoint_path.replace("critic.pth", "actor.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(critic_path, map_location=lambda storage, loc: storage))

    def set_mask(self, mask):
        if len(mask) > 0:
            self.actor.action_mask = mask
