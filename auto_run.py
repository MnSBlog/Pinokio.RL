import copy
import os
import gym
from utils.yaml_config import YamlConfig
from runners.auto_rl_runner import AutoRLRunner
from utils.metaheuristics import HarmonySearch
from main import load_config


def load_optim_config():
    config_handler = YamlConfig(root='./config')

    solver_root = './config/yaml/solvers/'
    parameters = config_handler.get_dict(os.path.join(solver_root, 'hyperparameters.yaml'))
    optimizer = 'HarmonySearch'
    opt_config = config_handler.get_dict(os.path.join(solver_root, optimizer + ".yaml"))
    args = dict(parameters, **opt_config)
    return args


def test_function(memory):
    run_args = load_config()
    env = gym.make(run_args['env_name'], render_mode='human')
    run_args = update_config(run_args, memory)
    runner = AutoRLRunner(config=run_args, env=env)
    output = runner.run()
    return output


def update_config(old_config, update_note):
    print(update_note)
    new_config = copy.deepcopy(old_config)
    network_config = new_config['network']['actor']
    network_config['obs_stack'] = True
    network_config['memory_q_len'] = 'local'

    for key, value in update_note.items():
        sep = key.split('-')
        sub = new_config
        sub_dicts = []
        for level in sep[:-1]:
            sub = sub[level]
            sub_dicts.append(sub)
        sub[sep[-1]] = value
        for idx in reversed(range(1, len(sub_dicts))):
            sub_dicts[idx - 1][sep[idx]] = sub_dicts[idx]
        new_config[sep[0]] = sub_dicts[0]
    print(new_config)
    return new_config


def main():
    run_args = load_config()
    optim_args = load_optim_config()
    network_config = run_args['network']['actor']
    if network_config['spatial_feature']['use'] is False:
        optim_args.pop('network-actor-non_spatial_feature-memory_q_len', None)
        optim_args.pop('network-actor-non_spatial_feature-num_layer', None)
        optim_args.pop('network-actor-non_spatial_feature-dim_out', None)
    if network_config['non_spatial_feature']['use'] is False:
        optim_args.pop('network-actor-non_spatial_feature-memory_q_len', None)
        optim_args.pop('network-actor-non_spatial_feature-extension', None)
        optim_args.pop('network-actor-non_spatial_feature-use_cnn', None)
        optim_args.pop('network-actor-non_spatial_feature-num_layer', None)
        optim_args.pop('network-actor-non_spatial_feature-dim_out', None)

    optimizer = HarmonySearch(parameters=optim_args, test_function=test_function)
    optimizer.start()
    output_config, output = optimizer.close()
    print(output_config)
    print(output)


if __name__ == '__main__':
    main()
