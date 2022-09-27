import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Categorical


def make_sequential(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


class CustomTorchNetwork(nn.Module):
    def __init__(self, config):
        super(CustomTorchNetwork, self).__init__()
        networks = dict()
        config['spatial_feature']['dim_in'] = config['spatial_feature']['dim_in'] * config['memory_q_len']
        config['non_spatial_feature']['dim_in'] = config['non_spatial_feature']['dim_in'] * config['memory_q_len']

        # Spatial feature network 정의
        if config['spatial_feature']['use']:
            spatial_processor = make_sequential(in_channels=config['spatial_feature']['dim_in'],
                                                out_channels=config['spatial_feature']['dim_in'] // 2,
                                                kernel_size=(2, 2), stride=(1, 1))

            spatial_processor.append(make_sequential(in_channels=config['spatial_feature']['dim_in'] // 2,
                                                     out_channels=3,
                                                     kernel_size=(2, 2), stride=(1, 1)))
            backbone = getattr(models, config['spatial_feature']['backbone'])(weights=None)
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, config['spatial_feature']['dim_out'])
            spatial_processor.append(backbone)
            networks['spatial_feature'] = spatial_processor

        # non-spatial feature network 정의
        if config['non_spatial_feature']['use']:
            if config['non_spatial_feature']['extension']:
                vector_processor = nn.Sequential(
                    nn.Conv1d(in_channels=config['non_spatial_feature']['dim_in'],
                              out_channels=config['non_spatial_feature']['dim_out'] // 2, kernel_size=(1,)),
                    nn.Conv1d(in_channels=config['non_spatial_feature']['dim_out'] // 2,
                              out_channels=config['non_spatial_feature']['dim_out'], kernel_size=(1,))
                )
                networks['non_spatial_feature'] = vector_processor
            else:
                config['non_spatial_feature']['dim_out'] = config['non_spatial_feature']['dim_in']

        # neck 부분
        config['neck_in'] = config['spatial_feature']['dim_out'] + config['non_spatial_feature']['dim_out']
        neck = nn.Sequential(
            nn.Linear(config['neck_in'], config['neck_out']),
            getattr(nn, config['neck_activation'])(),
            nn.Linear(config['neck_out'], config['neck_out'] // 2),
            getattr(nn, config['neck_activation'])(),
        )
        networks['neck'] = neck

        # action 부분
        self.outputs_dim = []
        for index, action_dim in enumerate(config['n_of_actions']):
            if isinstance(action_dim, int):
                key = "head" + str(index)
                self.outputs_dim.append(action_dim)
                networks[key] = nn.Sequential(
                    nn.Linear(config['neck_out'] // 2, action_dim),
                    nn.Softmax(dim=-1)
                )
            else:
                # 연속 액션은 아직 미구현
                raise NotImplementedError

        self.n_of_heads = len(config['n_of_actions'])
        self.networks = nn.ModuleDict(networks)
        self.action_mask = []

    def pre_forward(self, x1, x2):
        if 'spatial_feature' in self.networks:
            x1 = self.networks['spatial_feature'](x1)
        if 'non_spatial_feature' in self.networks:
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
            if len(self.action_mask) > 0:
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
            if len(self.outputs_dim) != 1:
                action = actions[:, idx, :].squeeze()
                dist = Categorical(outputs[:, :, last:last + output_dim])
            else:
                action = actions
                dist = Categorical(outputs)
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