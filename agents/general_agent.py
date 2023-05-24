import os
import torch
from buffer.base import DummyBuffer
from agents.tf.actorcritic import Actor, Critic
from agents.pytorch.utilities import get_device


class GeneralAgent:
    def __init__(self, parameters: dict, actor, **kwargs):
        self.device = get_device("auto")
        self._config = parameters
        self.actor = actor.to(self.device)
        self.buffer = DummyBuffer()
        self.exconfig = dict()
        self.metric_list = ['reward', 'entropy', 'state_value', 'loss']
        self.statistics = ['max', 'min', 'std', 'mean']
        self.metric = self.make_metrics()

    def select_action(self, state):
        raise NotImplementedError

    def update(self, next_state=None, done=None):
        raise NotImplementedError

    def save(self, checkpoint_path: str):
        raise NotImplementedError

    def load(self, checkpoint_path: str):
        raise NotImplementedError

    def set_mask(self, mask):
        raise NotImplementedError

    def evaluate(self, state, actions, hidden=None):
        raise NotImplementedError

    def insert_metrics(self, sub_metric):
        for key in self.metric_list:
            if key in sub_metric:
                for statistic in self.statistics:
                    value = getattr(torch, statistic)(sub_metric[key])
                    key_name = key + '_' + statistic
                    self.metric[key_name].append(value.item())

    def make_metrics(self):
        metric = dict()
        for title in self.metric_list:
            for statistic in self.statistics:
                key_name = title + '_' + statistic
                metric[key_name] = []
        return metric

    def convert_to_torch(self, state):
        spatial_x = state['matrix']
        non_spatial_x = state['vector']
        mask = state['action_mask']

        if torch.is_tensor(non_spatial_x) is False:
            non_spatial_x = torch.FloatTensor(non_spatial_x)
        non_spatial_x = non_spatial_x.to(self.device)

        if torch.is_tensor(spatial_x) is False:
            if len(spatial_x) > 0:
                spatial_x = torch.FloatTensor(spatial_x).to(self.device)
        else:
            spatial_x = spatial_x.to(self.device)

        if torch.is_tensor(mask) is False:
            if len(mask) > 0:
                mask = torch.FloatTensor(mask).to(self.device)
                mask = mask.unsqueeze(dim=0)
        else:
            mask = mask.to(self.device)

        state['matrix'] = spatial_x
        state['vector'] = non_spatial_x
        state['action_mask'] = mask

        return state


class PolicyAgent(GeneralAgent):
    def __init__(self, parameters: dict, actor, critic):
        super(PolicyAgent, self).__init__(parameters=parameters, actor=actor)
        self.critic = critic.to(self.device)

    def select_action(self, state):
        return super(PolicyAgent, self).select_action(state)

    def update(self, next_state=None, done=None):
        return super(PolicyAgent, self).update()

    def save(self, checkpoint_path: str):
        return super(PolicyAgent, self).save(checkpoint_path=checkpoint_path)

    def load(self, checkpoint_path: str):
        return super(PolicyAgent, self).load(checkpoint_path=checkpoint_path)

    def set_mask(self, mask):
        return super(PolicyAgent, self).set_mask(mask=mask)

    def evaluate(self, state, actions, hidden=None):
        return super(PolicyAgent, self).evaluate(state=state, actions=actions, hidden=hidden)
