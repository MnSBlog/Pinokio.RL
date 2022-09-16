import torch
import torch.nn as nn
import torchvision.models as models


class MakeSubModule(nn.Module):
    def __init__(self):
        super(MakeSubModule, self).__init__()
        self.spatial_feature_root = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=17, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(17),
            nn.ReLU(),
            nn.Conv2d(in_channels=17, out_channels=3, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 128)

        self.fully_feature_root = nn.Sequential(
            nn.Linear(20, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128)
        )
        self.neck = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )

        self.head_1 = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        self.head_2 = nn.Sequential(
            nn.Linear(32, 4),
            nn.Softmax(dim=-1)
        )
        self.head_3 = nn.Sequential(
            nn.Linear(32, 12),
            nn.Softmax(dim=-1)
        )

    def act(self, spatial_x, non_spatial_x):
        spatial_x = self.spatial_feature_root(spatial_x)
        spatial_x = self.backbone(spatial_x)

        non_spatial_x = self.fully_feature_root(non_spatial_x)
        state = torch.cat([spatial_x, non_spatial_x], dim=1)
        state = self.neck(state)
        output1 = self.head_1(state)
        output2 = self.head_2(state)
        output3 = self.head_3(state)

        outputs = [output1, output2, output3]
        return outputs



