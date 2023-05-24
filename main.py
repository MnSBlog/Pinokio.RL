import copy
import gymnasium as gym
import os
import runners.standard_runner as std_runner
import runners.gym_runner as gym_runner
import runners.auto_rl_runner as opt_runner
from utils.yaml_config import YamlConfig
from envs import REGISTRY as ENV_REGISTRY


def load_config(form="yaml"):
    if form == "yaml":
        config_loader = YamlConfig(root='./config')
    else:
        raise NotImplementedError

    config = config_loader.final_config_dict
    module_list = []
    for key, value in config.items():
        if isinstance(value, str):
            key = key.replace('_name', 's')
            module_path = os.path.join(key, value)
            module_list.append(module_path)
    config["config_root"] = "./config"

    config = config_loader.config_copy(
        config_loader.get_config(filenames=module_list)
    )

    return config


def update_config(config, key, name):
    root = os.path.join("./config/yaml/", key)
    name = name + '.yaml'
    sub_dict = YamlConfig.get_dict(os.path.join(root, name))
    config[key] = sub_dict[key]
    return copy.deepcopy(config)


def save_folder_check(config):
    root = config['runner']['history_path']
    if os.path.exists(os.path.join(root, config['env_name'])) is False:
        os.mkdir(os.path.join(root, config['env_name']))
    config['runner']['history_path'] = os.path.join(root, config['env_name'])

    root = config['runner']['history_path']
    if os.path.exists(os.path.join(root, config['agent_name'])) is False:
        os.mkdir(os.path.join(root, config['agent_name']))
    if os.path.exists(os.path.join(root, 'best')) is False:
        os.mkdir(os.path.join(root, 'best'))
    config['runner']['history_path'] = os.path.join(root, config['agent_name'])


def check_parallel(config):
    parallel = False
    if isinstance(config['env_name'], list):
        parallel = True
    if isinstance(config['network']['actor']['memory_q_len'], list):
        parallel = True
    return parallel


def main(args, parallel):
    if parallel:
        args['runner_name'] = "Parallel" + args['runner_name']
        if "Gym" in args['runner_name']:
            runner = getattr(gym_runner,
                             args['runner_name'])(config=args)
        else:
            runner = getattr(std_runner,
                             args['runner_name'])(config=args)
    else:
        save_folder_check(args)
        if args['env_name'] in ENV_REGISTRY:
            env = ENV_REGISTRY[args['env_name']](**args['envs'])
        else:
            env = gym.make(args['env_name'], render_mode='human', autoreset=True)

        if "Gym" in args['runner_name']:
            from gym import envs
            check = envs.registry
            runner = gym_runner.GymRunner(config=copy.deepcopy(args), env=env)
        else:
            standard_attributes = dir(std_runner)
            if args['runner_name'] in standard_attributes:
                runner = getattr(std_runner, args['runner_name'])(config=args, env=env)
            else:
                runner = getattr(opt_runner, args['runner_name'])(config=args, env=env)

    runner.run()


if __name__ == '__main__':
    args = load_config()
    args['agent']['batch_size'] = args['runner']['batch_size']
    main(args=args, parallel=check_parallel(args))
