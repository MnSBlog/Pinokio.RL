import copy
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
            if key == "envs":
                continue
            module_path = os.path.join(key, value)
            module_list.append(module_path)
    config["config_root"] = "./config"

    config = config_loader.config_copy(
        config_loader.get_config(filenames=module_list)
    )

    # config['network']['model_path'] = os.path.join(config['runner']['history_path'],
    #                                                config['envs']['name'],
    #                                                config['network']['model_path'])

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
        os.mkdir(os.path.join(root, 'best'))
    config['runner']['history_path'] = os.path.join(root, config['agent_name'])


def main(args):
    exp_cond = copy.deepcopy(args)
    for env_name in exp_cond['env_name']:
        for mem_len in exp_cond['network']['memory_q_len']:
            for layer_type in exp_cond['network']['use_memory_layer']:
                args['env_name'] = env_name
                args = update_config(config=args, key='envs', name=env_name)
                args['network']['memory_q_len'] = mem_len
                args['network']['use_memory_layer'] = layer_type
                save_folder_check(args)

                if args['runner_name'] == 'ray_tune':
                    register_env(args['env_name'], ENV_REGISTRY[args['env_name']](**args['envs']))
                    raise NotImplementedError
                elif args['runner_name'] == 'gym':
                    from gym import envs
                    check = envs.registry
                    env = gym.make(args['env_name'], render_mode='human')
                    runner = GymRunner(config=copy.deepcopy(args), env=env)
                else:
                    env = ENV_REGISTRY[args['env_name']](**args['envs'])
                    runner = getattr(runner_instance,
                                     args['runner_name'])(config=args,
                                                          env=env)
                runner.run()
                # runner.plot_result()


if __name__ == '__main__':
    arguments = load_config()
    main(args=arguments)
