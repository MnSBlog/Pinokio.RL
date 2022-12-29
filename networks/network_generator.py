import copy
import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Categorical


def make_sequential(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


def make_lin_sequential(in_channel, out_channel, activation, num_layer):
    sep = int((out_channel - in_channel) / num_layer)
    sub_out_channel = in_channel

    body = nn.Sequential()
    for _ in range(num_layer - 1):
        body.append(nn.Linear(sub_out_channel, sub_out_channel + sep))
        body.append(getattr(nn, activation)())
        sub_out_channel += sep
    body.append(nn.Linear(sub_out_channel, out_channel))
    return body


def make_conv1d_sequential(in_channel, out_channel, num_layer):
    sep = int((out_channel - in_channel) / num_layer)
    sub_out_channel = in_channel

    body = nn.Sequential()
    for _ in range(num_layer - 1):
        body.append(nn.Conv1d(in_channels=sub_out_channel, out_channels=sub_out_channel + sep, kernel_size=(1,)))
        sub_out_channel += sep
    body.append(nn.Conv1d(in_channels=sub_out_channel, out_channels=out_channel, kernel_size=(1,)))
    return body


class CustomTorchNetwork(nn.Module):
    def __init__(self, config):
        # critic은 자동으로 yaml을 만들어줄것
        super(CustomTorchNetwork, self).__init__()
        networks = dict()
        self.local_len = config['memory_q_len']
        # Spatial feature network 정의
        if config['spatial_feature']['use']:
            in_channel = config['spatial_feature']['dim_in'][0] * self.local_len
            if config['spatial_feature']['backbone'] != '':
                spatial_processor = make_sequential(in_channels=in_channel,
                                                    out_channels=in_channel // 2,
                                                    kernel_size=(2, 2), stride=(1, 1))

                spatial_processor.append(make_sequential(in_channels=in_channel // 2,
                                                         out_channels=3,
                                                         kernel_size=(2, 2), stride=(1, 1)))
                backbone = getattr(models, config['spatial_feature']['backbone'])(weights=None)
                num_ftrs = backbone.fc.in_features
                backbone.fc = nn.Linear(num_ftrs, config['spatial_feature']['dim_out'])
                spatial_processor.append(backbone)
                networks['spatial_feature'] = spatial_processor
            else:
                shape = config['spatial_feature']['dim_in'][1:]
                networks['spatial_feature'] = nn.Sequential(
                    make_sequential(in_channels=in_channel,
                                    out_channels=32,
                                    kernel_size=(4, 4), stride=(2, 2)),
                    make_sequential(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(2, 2), stride=(1, 1)),
                    make_sequential(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(2, 2), stride=(1, 1)),
                    nn.Flatten(),
                    nn.Linear(64 * (shape[0] // 2 - 3) * (shape[1] // 2 - 3), config['spatial_feature']['dim_out']),
                    nn.ReLU()
                )
        else:
            config['spatial_feature']['dim_out'] = 0
        # non-spatial feature network 정의
        self.use_cnn = False
        if config['non_spatial_feature']['use']:
            if config['non_spatial_feature']['extension']:
                if config['non_spatial_feature']['use_cnn']:
                    self.use_cnn = True
                    channel = config['non_spatial_feature']['dim_out'] // config['non_spatial_feature']['dim_in']
                    config['non_spatial_feature']['dim_out'] = channel * config['non_spatial_feature']['dim_in']
                    vector_processor = make_conv1d_sequential(in_channel=self.local_len,
                                                              out_channel=channel,
                                                              num_layer=config['non_spatial_feature']['num_layer'])
                else:
                    in_node = config['non_spatial_feature']['dim_in'] * self.local_len
                    vector_processor = make_lin_sequential(in_channel=in_node,
                                                           out_channel=config['non_spatial_feature']['dim_out'],
                                                           activation=config['neck_activation'],
                                                           num_layer=config['non_spatial_feature']['num_layer'])
                networks['non_spatial_feature'] = vector_processor
            else:
                config['non_spatial_feature']['dim_out'] = config['non_spatial_feature']['dim_in'] * self.local_len
        else:
            config['non_spatial_feature']['dim_out'] = 0

        # neck 부분
        config['neck_in'] = config['spatial_feature']['dim_out'] + config['non_spatial_feature']['dim_out']
        sep = int((config['neck_out'] - config['neck_in']) / (config['neck_num_layer']) + 1)
        sub_out_channel = config['neck_in']

        if config['num_memory_layer'] == 0:
            input_layer = nn.Sequential(
                nn.Linear(sub_out_channel, sub_out_channel + sep),
                getattr(nn, config['neck_activation'])()
            )
            self.init_h_state = None
            self.recurrent = False
            sub_out_channel += sep
        else:
            while True:
                if sub_out_channel % config['memory_rnn_len'] == 0:
                    break
                else:
                    config['memory_rnn_len'] -= 1
            in_channel = sub_out_channel // config['memory_rnn_len']
            out_channel = (sub_out_channel + sep) // config['memory_rnn_len']
            input_layer = getattr(nn, "GRU")(in_channel, out_channel, config['num_memory_layer'])
            self.init_h_state = self.get_initial_h_state(input_layer.num_layers,
                                                         config['memory_rnn_len'],
                                                         input_layer.hidden_size)
            self.recurrent = True
            sub_out_channel = out_channel * config['memory_rnn_len']
        self.rnn_len = config['memory_rnn_len']
        networks['input_layer'] = input_layer

        neck = make_lin_sequential(in_channel=sub_out_channel,
                                   out_channel=config['neck_out'],
                                   activation=config['neck_activation'],
                                   num_layer=config['neck_num_layer'])
        networks['neck'] = neck

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
            if self.use_cnn is False:
                x2 = x2.view(x2.shape[0], -1)
            x2 = self.networks['non_spatial_feature'](x2)

        if 0 not in x2.shape:
            x2 = x2.view(x2.shape[0], -1)
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
            x = x.view(x.shape[0], self.rnn_len, -1)
            x, h = self.networks['input_layer'](x, h)
            x = x.view(x.shape[0], -1)
        else:
            x = self.networks['input_layer'](x)
        if 'neck' in self.networks:
            x = self.networks['neck'](x)
        outputs = []
        dim = len(x.shape) - 1
        for index in range(self.n_of_heads):
            key = "head" + str(index)
            outputs.append(self.networks[key](x))

        return torch.cat(outputs, dim=dim), h

    @staticmethod
    def get_initial_h_state(num_layers, mem_q, hidden_size):
        h_0 = torch.zeros((
            num_layers,
            mem_q,
            hidden_size),
            dtype=torch.float)

        # c_0 = torch.zeros((
        #     num_layers,
        #     hidden_size),
        #     dtype=torch.float)
        return h_0


class SimpleActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleActorNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        self.init_h_state = None
        self.action_mask = []
        self.networks = []
        self.recurrent = False
        self.outputs_dim = [output_dim]
        self.input_dim = input_dim

    def forward(self, x, h=None):
        x = x['vector'].view(-1, self.input_dim)
        return self.model(x), h


class SimpleCriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimpleCriticNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.init_h_state = None
        self.action_mask = []
        self.outputs_dim = []
        self.input_dim = input_dim

    def forward(self, x, h=None):
        x = x['vector'].view(-1, self.input_dim)
        return self.model(x), h
