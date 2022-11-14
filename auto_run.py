import copy
import os
import gym
from utils.yaml_config import YamlConfig
from runners.auto_rl_runner import AutoRLRunner
from main import load_config


def load_optim_config():
    config_handler = YamlConfig(root='./config')

    solver_root = './config/yaml/solvers/'
    parameters = config_handler.get_dict(os.path.join(solver_root, 'hyperparameters.yaml'))
    optimizer = 'HarmonySearch'
    opt_config = config_handler.get_dict(os.path.join(solver_root, optimizer + ".yaml"))
    args = dict(parameters, **opt_config)
    return args


def main():
    run_args = load_config()
    optim_args = load_optim_config()
    network_config = run_args['network']['actor']
    if network_config['spatial_feature']['use'] is False:
        optim_args.pop('matrix-stacking', None)
        optim_args.pop('matrix-layer_num', None)
        optim_args.pop('matrix-output_node', None)
    if network_config['non_spatial_feature']['use'] is False:
        optim_args.pop('vector-stacking', None)
        optim_args.pop('vector-layer_num', None)
        optim_args.pop('vector-use_cnn', None)
        optim_args.pop('vector-output_node', None)


def test_function(memory):
    run_args = load_config()
    env = gym.make(run_args['env_name'], render_mode='human')
    runner = AutoRLRunner(config=run_args, env=env)
    output = runner.run()
    return output


def update_config(old_config, update_note):
    new_config = copy.deepcopy(old_config)
    network_config = new_config['network']['actor']
    algo_config = new_config['agent']
    runner_config = new_config['runner']
    network_config['obs_stack'] = True
    network_config['use_memory_layer'] = "GRU" if update_note['neck-use_rnn'] else "Raw"
    if network_config['non_spatial_feature']['use']:
        network_config['non_spatial_feature']['memory_layer_len'] = update_note['vector-stacking']
        network_config['non_spatial_feature']['dim_out'] = update_note['vector-output_node']

        update_note['vector-use_cnn']
        update_note['vector-layer_num']

    if network_config['spatial_feature']['use']:
        network_config['spatial_feature']['memory_layer_len'] = update_note['matrix-stacking']
        network_config['spatial_feature']['dim_out'] = update_note['matrix-output_node']
        update_note['matrix-layer_num']

    update_note['neck-layer_num']
    update_note['neck-use_rnn']
    update_note['neck-output_node']

    update_note['output-loss_function']
    update_note['output-optimizer']
    update_note['output-batch_size']


if __name__ == '__main__':
    main()