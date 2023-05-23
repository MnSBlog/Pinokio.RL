import copy
import os

import numpy as np
import pandas as pd
import gym
from envs import REGISTRY as ENV_REGISTRY
import matplotlib.pyplot as plt
from utils.yaml_config import YamlConfig
from runners.auto_rl_runner import AutoRLRunner
import outer.solver as solver
from main import load_config
from datetime import datetime


def load_optim_config(optimizer: str):
    config_handler = YamlConfig(root='./config')

    solver_root = './config/yaml/solvers/'
    parameters = config_handler.get_dict(os.path.join(solver_root, 'hyperparameters.yaml'))
    opt_config = config_handler.get_dict(os.path.join(solver_root, optimizer + ".yaml"))
    args = dict(parameters, **opt_config)
    return args


def save_config(config, path):
    import yaml
    if '.yaml' not in path:
        path += '.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)


def save_outputs(args, metric, path):
    now = datetime.now()
    prefix = now.strftime("%Y-%m-%d-%H-%M-%S.%f")

    if os.path.isdir(path) is False:
        os.mkdir(path)
    save_config(args, os.path.join(path, prefix + "run_args.yaml"))
    if metric is not None:
        data = pd.DataFrame.from_dict(metric)
        data.to_csv(os.path.join(path, prefix + 'metric.csv'))


def test_function(memory, run_args):
    try:
        from gym import envs
        check = envs.registry

        if run_args['env_name'] in check:
            env = gym.make(run_args['env_name'], render_mode=run_args['runner']['render'])
        else:
            env = ENV_REGISTRY[run_args['env_name']](**run_args['envs'])

        run_args = update_config(run_args, memory)

        runner = AutoRLRunner(config=run_args, env=env)
        output, metric = runner.run()

        figure_path = './figures/AutoRL'
        env_path = os.path.join(figure_path, run_args['env_name'])
        folder_list = os.listdir(env_path)
        full_list = [os.path.join(env_path, folder) for folder in folder_list]
        time_sorted_list = sorted(full_list, key=os.path.getmtime)
        last_folder = time_sorted_list[-1]
        count = len(os.listdir(last_folder))
        env_path = os.path.join(last_folder, str(count) + '-Gen')

        save_outputs(args=run_args, metric=metric, path=os.path.join(env_path, str(output)))
    except ValueError:
        output = np.NINF
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


def draw_result(path):
    generations = os.listdir(path)
    generations = [folder for folder in generations if 'Gen' in folder]
    info = {'mu': [0.0], 'max': [0.0], 'min': [0.0], 'iteration': [0]}
    for gen in range(1, len(generations) + 1):
        output_path = os.path.join(path, generations[gen - 1])
        outputs = os.listdir(output_path)
        outputs = [float(i) for i in outputs if '.json' not in i]
        min_val = min(outputs)
        max_val = max(outputs)
        mu = np.mean(outputs).item()

        info['mu'].append(mu)
        info['max'].append(max_val)
        info['min'].append(min_val)
        info['iteration'].append(gen)

    plt.plot(info['iteration'], info['max'], '-')
    plt.fill_between(info['iteration'], info['mu'], info['max'], alpha=0.2)
    plt.savefig(os.path.join(path, "progress.jpg"))
    plt.clf()


def main(opt: str):
    run_args = load_config()
    optim_args = load_optim_config(opt)
    network_config = run_args['network']['actor']
    if network_config['spatial_feature']['use'] is False:
        optim_args.pop('network-actor-spatial_feature-memory_q_len', None)
        optim_args.pop('network-actor-spatial_feature-num_layer', None)
        optim_args.pop('network-actor-spatial_feature-dim_out', None)
    if network_config['non_spatial_feature']['use'] is False:
        optim_args.pop('network-actor-non_spatial_feature-memory_q_len', None)
        optim_args.pop('network-actor-non_spatial_feature-extension', None)
        optim_args.pop('network-actor-non_spatial_feature-use_cnn', None)
        optim_args.pop('network-actor-non_spatial_feature-num_layer', None)
        optim_args.pop('network-actor-non_spatial_feature-dim_out', None)

    # 저장할 폴더 생성
    figure_path = './figures/AutoRL'
    if os.path.isdir(figure_path) is False:
        os.mkdir(figure_path)
    env_path = os.path.join(figure_path, run_args['env_name'])
    if os.path.isdir(env_path) is False:
        os.mkdir(env_path)
    now = datetime.now()
    start_date = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(os.path.join(env_path, start_date))

    optimizer = getattr(solver, opt)(parameters=optim_args, test_function=test_function,
                                     init_points=5, n_iter=20, verbose=2, random_state=1)
    optimizer.start(root=os.path.join(env_path, start_date))
    output_config, output = optimizer.close()
    save_outputs(args=output_config, metric=None, path=os.path.join(env_path, start_date, "Best"))
    draw_result(path=os.path.join(env_path, start_date))
    print(output_config)
    print(output)


if __name__ == '__main__':
    # HarmonySearch / Bayesian
    solver_name = "HarmonySearch"
    main(solver_name)
