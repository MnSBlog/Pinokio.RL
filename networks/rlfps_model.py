import torch
import torch.nn as nn


class RLModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model_list = []
        # Actor & Critic
        for _ in range(args['n_of_actor_networks']):
            test = 1

        for _ in range(args['n_of_critic_networks']):
            test = 2

        # Optimizer
        if args['optimizer'] == 'adam'
            self.optimizer = torch.optim.adam()
