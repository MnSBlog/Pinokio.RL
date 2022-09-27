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
        self.outputs_dim = []
        for index, action_dim in enumerate(config['n_of_actions']):
            key = "head" + str(index)
            self.outputs_dim.append(action_dim)
            networks[key] = nn.Sequential(
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1)
            )
        self.n_of_heads = len(config['n_of_actions'])
        self.networks = nn.ModuleDict(networks)
        self.action_mask = []

    def pre_forward(self, x1, x2):
        x1 = self.networks['spatial_feature'](x1)
        x2 = self.networks['non_spatial_feature'](x2)
        x2 = x2.squeeze(dim=2)
        state = torch.cat([x1, x2], dim=1)
        return state

    def forward(self, x):
        state = self.networks['neck'](x)
        outputs = []
        dim = len(state.shape) - 1
        for index in range(self.n_of_heads):
            key = "head" + str(index)
            outputs.append(self.networks[key](state))

        return torch.cat(outputs, dim=dim)

    def act(self, state):
        rtn_action = []
        rtn_logprob = []
        outputs = self.forward(x=state)
        last = 0
        for idx, output_dim in enumerate(self.outputs_dim):
            outputs[:, last:last + output_dim] *= self.action_mask[idx]
            dist = Categorical(outputs[:, last:last + output_dim])
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            rtn_action.append(action.detach())
            rtn_logprob.append(action_logprob.detach())
            last = output_dim

        return torch.stack(rtn_action, dim=0), torch.stack(rtn_logprob, dim=0)

    def evaluate(self, state, actions):
        rtn_evaluations = []
        outputs = self.forward(x=state)
        last = 0
        for idx, output_dim in enumerate(self.outputs_dim):
            action = actions[:, idx, :].squeeze()
            dist = Categorical(outputs[:, :, last:last + output_dim])
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            rtn_evaluations.append((action_logprobs, dist_entropy))
            last = output_dim

        return rtn_evaluations

    def set_mask(self, mask):
        if mask is not None:
            self.action_mask = []
            last = 0
            for output_dim in self.outputs_dim:
                self.action_mask.append(mask[:, last:last + output_dim])
                last = output_dim

# 측정에서 모델 나오기까지의 시간
# AI 모델을 커스텀마이징 수준: 오픈포즈 스켈레톤, 디노이징 리컨스트럭션은