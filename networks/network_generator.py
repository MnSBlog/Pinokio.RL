import torch
import torch.nn as nn
import torchvision.models as models


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
        backbone = getattr(models, config['spatial_feature']['backbone'])(pretrained=True)
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
        for index, action_dim in enumerate(config['n_of_actions']):
            key = "head" + str(index)
            networks[key] = nn.Sequential(
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1)
            )
        self.n_of_heads = len(config['n_of_actions'])
        self.networks = nn.ModuleDict(networks)

    def forward(self, x):
        spatial_x = self.networks['spatial_feature'](x[0])
        non_spatial_x = self.networks['non_spatial_feature'](x[1])
        state = torch.cat([spatial_x, non_spatial_x], dim=1)
        state = self.networks['neck'](state)
        outputs = []

        for index in range(self.n_of_heads):
            key = "head" + str(index)
            outputs.append(self.networks[key](state))

        return outputs
