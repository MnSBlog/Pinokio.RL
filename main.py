import gym
import os
import runners.standard_runner as runner_instance
from utils.yaml_config import YamlConfig
from envs import REGISTRY as env_registry
from runners.gym_runner import GymRunner
from ray.tune.registry import register_env


def load_config(form="yaml"):
    if form == "yaml":
        config_loader = YamlConfig(root='./config')
    else:
        raise NotImplementedError

    config = config_loader.final_config_dict
    module_list = []
    for key, value in config.items():
        if isinstance(key, str):
            key = key.replace('_name', 's')
            module_path = os.path.join(key, value)
            module_list.append(module_path)
    config["config_root"] = "./config"

    config = config_loader.config_copy(
        config_loader.get_config(filenames=module_list)
    )

    config['network']['model_path'] = os.path.join(config['runner']['history_path'],
                                                   config['envs']['name'],
                                                   config['network']['model_path'])

    return config


def save_folder_check(config):
    root = config['runner']['history_path']
    if os.path.exists(os.path.join(root, config['env_name'])) is False:
        os.mkdir(os.path.join(root, config['env_name']))
    config['runner']['history_path'] = os.path.join(root, config['env_name'])

    root = config['runner']['history_path']
    if os.path.exists(os.path.join(root, config['agent_name'])) is False:
        os.mkdir(os.path.join(root, config['agent_name']))
        os.mkdir(os.path.join(root, 'best'))
    config['runner']['history_path'] = os.path.join(root, config['agent_name'])


def main(args):
    if args['runner_name'] == 'ray_tune':
        register_env(args['env_name'], env_registry[args['env_name']](**args['envs']))
        raise NotImplementedError
    elif args['runner_name'] == 'gym':
        runner = GymRunner(config=args, env=gym.make(args['env_name']))
    else:
        runner = getattr(runner_instance,
                         args['runner_name'])(config=args,
                                              env=None)
    runner.run()
    runner.plot_result()


if __name__ == '__main__':
    from gym import envs
    print(envs.registry.values())
    arguments = load_config()
    save_folder_check(arguments)
    main(args=arguments)
