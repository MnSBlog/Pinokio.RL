import pandas as pd
import os
from utils.yaml_config import YamlConfig


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

    return config
    # 이 조건은 Runners로
    # if config["self_play"]:
    #     algo_condition = pd.read_excel(config["condition_path"], engine='openpyxl')
    #     algo_condition = algo_condition.query('Select.str.contains("' + 'Use' + '")')
    #     algo_condition = algo_condition.query('`' + config["env_config"]["actions"] + ' Actions`.str.contains("Yes")')
    #     algo_condition = algo_condition.query('Frameworks.str.contains("' + config["framework"] + '")')
    #     if config["env_config"]["multi_agent"]:
    #         algo_condition = algo_condition.query('Multi-Agent.str.contains("Yes")')
    #
    #     config["algorithm"] = algo_condition['Algorithm'].to_list()
    # # Save Path Dir
    # if os.path.exists(config["history_path"]) is False:
    #     os.mkdir(config["history_path"])
    # if os.path.exists(config["history_path"] + "/" + config["env"]) is False:
    #     os.mkdir(config["history_path"] + "/" + config["env"])
    # if os.path.exists(config["history_path"] + "/" + config["env"] + "/" + "Best") is False:
    #     os.mkdir(config["history_path"] + "/" + config["env"] + "/" + "Best")
    # for algorithm in config["algorithm"]:
    #     algorithm_path = config["history_path"] + "/" + config["env"] + "/" + algorithm
    #     if os.path.exists(algorithm_path) is False:
    #         os.mkdir(algorithm_path)
    # config["history_path"] = config["history_path"] + "/" + config["env"]


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model


if __name__ == '__main__':
    main(sys.argv)
