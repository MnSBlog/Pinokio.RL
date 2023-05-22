import os
import torch
from buffer.base import BaseBuffer
from agents.tf.actorcritic import Actor, Critic


class GeneralAgent:
    def __init__(self, parameters: dict):
        self._config = parameters
        self._buffer = BaseBuffer()
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


class PolicyAgent(GeneralAgent):
    def __init__(self, parameters: dict, actor, critic):
        super(PolicyAgent, self).__init__(parameters=parameters)
        self.actor = actor
        self.critic = critic

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
