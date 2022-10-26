import torch
import torch.nn as nn
import torchvision.models as models


def make_sequential(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


def make_lin_sequential(in_channel, out_channel, activation):
    temp_out = in_channel // 2
    body = nn.Sequential(
        nn.Linear(in_channel, temp_out),
        getattr(nn, activation)())
    while temp_out // 2 > out_channel:
        body.append(nn.Linear(temp_out, temp_out // 2))
        body.append(getattr(nn, activation)())
        temp_out = temp_out // 2
    body.append(nn.Linear(temp_out, out_channel))
    body.append(getattr(nn, activation)())
    return body


class CustomTorchNetwork(nn.Module):
    def __init__(self, config):
        # critic은 자동으로 yaml을 만들어줄것
        super(CustomTorchNetwork, self).__init__()
        networks = dict()

        # Spatial feature network 정의
        if config['spatial_feature']['use']:
            config['spatial_feature']['dim_in'] = config['spatial_feature']['dim_in'] * config['memory_q_len']
            if config['spatial_feature']['backbone'] != '':
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
            else:
                networks['spatial_feature'] = nn.Sequential(
                    make_sequential(in_channels=config['spatial_feature']['dim_in'],
                                    out_channels=32,
                                    kernel_size=(4, 4), stride=(2, 2)),
                    make_sequential(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(2, 2), stride=(1, 1)),
                    make_sequential(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(2, 2), stride=(1, 1)),
                    nn.Flatten(),
                    nn.Linear(64 * 2209, config['spatial_feature']['dim_out']),
                    nn.ReLU()
                )

        # non-spatial feature network 정의
        if config['non_spatial_feature']['use']:
            config['non_spatial_feature']['dim_in'] = config['non_spatial_feature']['dim_in'] * config['memory_q_len']
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
        spatial_flag = 0
        non_spatial_flag = 0
        if config['spatial_feature']['use']:
            spatial_flag = 1
        if config['non_spatial_feature']['use']:
            non_spatial_flag = 1

        config['neck_in'] = config['spatial_feature']['dim_out'] * spatial_flag + config['non_spatial_feature']['dim_out'] * non_spatial_flag
        config['neck_out'] = 16
        if config['use_memory_layer'] == "Raw":
            input_layer = nn.Sequential(
                nn.Linear(config['neck_in'], config['neck_in'] // 2),
                getattr(nn, config['neck_activation'])()
            )
            self.init_h_state = None
            self.recurrent = False
        else:
            input_layer = getattr(nn, config['use_memory_layer'])(config['neck_in'], config['neck_in'] // 2,
                                                                  config['memory_layer_len'], batch_first=True)
            self.init_h_state = self.get_initial_h_state(input_layer.num_layers,
                                                         input_layer.hidden_size)
            self.recurrent = True
        networks['input_layer'] = input_layer
        neck = make_lin_sequential(in_channel=config['neck_in'] // 2,
                                   out_channel=config['neck_out'],
                                   activation=config['neck_activation'])
        networks['neck'] = neck
        # 민구 추가

        # action 부분
        self.outputs_dim = []
        for index, action_dim in enumerate(config['n_of_actions']):
            if isinstance(action_dim, int):
                key = "head" + str(index)
                self.outputs_dim.append(action_dim)
                if config['action_mode'] == "Discrete":
                    networks[key] = nn.Sequential(
                        nn.Linear(config['neck_out'], action_dim),
                        nn.Softmax(dim=-1)
                    )
                else:
                    networks[key] = nn.Sequential(
                        nn.Linear(config['neck_out'], action_dim),
                    )
            else:
                # 연속 액션은 아직 미구현
                raise NotImplementedError

        self.n_of_heads = len(config['n_of_actions'])
        self.networks = nn.ModuleDict(networks)
        self.state_dim = config['neck_in']
        self.action_mask = []

    def pre_forward(self, x1, x2):
        cat_alter = []
        if 'spatial_feature' in self.networks:
            x1 = self.networks['spatial_feature'](x1)
            cat_alter.append(x1)
        if 'non_spatial_feature' in self.networks:
            x2 = self.networks['non_spatial_feature'](x2)
        x2 = x2.squeeze(dim=2)
        if 0 not in x2.shape:
            cat_alter.append(x2)
        if len(cat_alter) == 2:
            state = torch.cat(cat_alter, dim=1)
        else:
            state = cat_alter.pop()
        return state

    def forward(self, x, h=None):
        spatial_x = x['matrix']
        non_spatial_x = x['vector']

        x = self.pre_forward(x1=spatial_x, x2=non_spatial_x)
        if self.recurrent:
            self.networks['input_layer'].flatten_parameters()
            x = x.unsqueeze(dim=1)
            x, h = self.networks['input_layer'](x, h)
            x = x.squeeze(dim=1)
        else:
            x = self.networks['input_layer'](x)
        x = self.networks['neck'](x.data)
        outputs = []
        dim = len(x.shape) - 1
        for index in range(self.n_of_heads):
            key = "head" + str(index)
            outputs.append(self.networks[key](x))

        return torch.cat(outputs, dim=dim), h

    @staticmethod
    def get_initial_h_state(num_layers, hidden_size):
        h_0 = torch.zeros((
            num_layers,
            1,
            hidden_size),
            dtype=torch.float)

        # c_0 = torch.zeros((
        #     num_layers,
        #     hidden_size),
        #     dtype=torch.float)
        return h_0
