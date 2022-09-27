import gym
import os
import runners.standard_runner as runner_instance
from utils.yaml_config import YamlConfig
from envs import REGISTRY as ENV_REGISTRY
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
        register_env(args['env_name'], ENV_REGISTRY[args['env_name']](**args['envs']))
        raise NotImplementedError
    elif args['runner_name'] == 'gym':
        from gym import envs
        check = envs.registry
        env = gym.make(args['env_name'], render_mode='human')
        runner = GymRunner(config=args, env=env)
    else:
        env = ENV_REGISTRY[args['env_name']](**args['envs'])
        runner = getattr(runner_instance,
                         args['runner_name'])(config=args,
                                              env=env)
    runner.run()
    runner.plot_result()


if __name__ == '__main__':
    arguments = load_config()
    save_folder_check(arguments)
    main(args=arguments)
