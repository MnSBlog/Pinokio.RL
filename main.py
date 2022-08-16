import gym
import os
from utils.yaml_config import YamlConfig
from envs import REGISTRY as env_registry
from runners.episode_runner import EpisodeRunner
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

    config['network']['model_path'] = os.path.join(config['runners']['history_path'],
                                                   config['envs']['name'],
                                                   config['network']['model_path'])

    return config


def main(args):
    # Env setting
    env: gym.Env = env_registry[args['env_name']](**args['envs'])

    if args['runner_name'] == 'ray_tune':
        register_env(args['env_name'], env)
        raise NotImplementedError
    else:
        runner = EpisodeRunner(config=args['runners'], env=env)

    if args['network_name'] == args['env_name']:

    else:
        raise NotImplementedError
    runner.run()

    env.close()

    return model


if __name__ == '__main__':
    arguments = load_config()
    main(args=arguments)
