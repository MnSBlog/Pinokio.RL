import copy
import os
from utils.yaml_config import YamlConfig
from envs import REGISTRY as ENV_REGISTRY
from main import load_config


def load_optim_config():
    config_handler = YamlConfig(root='./config')

    solver_root = './config/yaml/solvers/'
    parameters = config_handler.get_dict(os.path.join(solver_root, 'hyperparameters.yaml'))
    # parameter masking

    optimizer = 'HarmonySearch'
    opt_config = config_handler.get_dict(os.path.join(solver_root, optimizer + ".yaml"))
    args = dict(parameters, **opt_config)
    return args


def main():
    test = load_config()
    test2 = load_optim_config()

if __name__ == '__main__':
    main()