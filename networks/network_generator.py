import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.distributions import MultivariateNormal, Categorical


def make_sequential(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


class CustomTorchNetwork(nn.Module):
    def __init__(self, config):
        super(CustomTorchNetwork, self).__init__()
        # Spatial feature network 정의
        spatial_processor = make_sequential(in_channels=config['spatial_feature']['dim'],
                                            out_channels=config['spatial_feature']['dim'] // 2,
                                            kernel_size=(2, 2), stride=(1, 1))

        spatial_processor.append(make_sequential(in_channels=config['spatial_feature']['dim'] // 2,
                                                 out_channels=3,
                                                 kernel_size=(2, 2), stride=(1, 1)))
        backbone = getattr(models, config['spatial_feature']['backbone'])(weights=None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, config['neck_input'])
        spatial_processor.append(backbone)

        # non-spatial feature network 정의
        vector_processor = nn.Sequential(
            nn.Conv1d(in_channels=config['non_spatial_feature']['dim'],
                      out_channels=config['neck_input'] // 2, kernel_size=(1,)),
            nn.Conv1d(in_channels=config['neck_input'] // 2,
                      out_channels=config['neck_input'], kernel_size=(1,))
        )
        # neck 부분
        neck = nn.Sequential(
            nn.Linear(config['neck_input'] * 2, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
        )
        networks = {
            'spatial_feature': spatial_processor,
            'non_spatial_feature': vector_processor,
            'neck': neck,
        }

        # action 부분
        self.action_mask = []
        for index, action_dim in enumerate(config['n_of_actions']):
            key = "head" + str(index)
            self.action_mask.append(np.ones(action_dim))
            networks[key] = nn.Sequential(
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1)
            )
        self.n_of_heads = len(config['n_of_actions'])
        self.networks = nn.ModuleDict(networks)

    def pre_forward(self, x1, x2):
        x1 = self.networks['spatial_feature'](x1)
        x2 = self.networks['non_spatial_feature'](x2)
        x2 = x2.squeeze(dim=2)
        state = torch.cat([x1, x2], dim=1)
        return state

    def forward(self, x):
        state = self.networks['neck'](x)
        outputs = []

        for index in range(self.n_of_heads):
            key = "head" + str(index)
            outputs.append(self.networks[key](state))

        return outputs

    def act(self, state):
        rtn_actions = []
        rtn_action_logprob = []
        outputs = self.forward(x=state)
        for idx, action_probs in enumerate(outputs):
            action_probs *= self.action_mask[idx]
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            rtn_actions.append(action.detach())
            rtn_action_logprob.append(action_logprob)

        return rtn_actions, rtn_action_logprob

    def evaluate(self, state, action):
        rtn_evaluations = []
        outputs = self.forward(x1=state[0], x2=state[1])
        for action_probs in outputs:
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            rtn_evaluations.append((action_logprobs, dist_entropy))

        return rtn_evaluations

    def set_mask(self, mask):
        if mask is not None:
            self.action_mask = []
            for mask_value in mask:
                self.action_mask.append(mask_value)
