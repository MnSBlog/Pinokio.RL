import copy
import os
import gym
import torch
import matplotlib.pyplot as plt
from runners.general_runner import GeneralRunner


class GymRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        config['agent']['mid_gamma'] = config['agent']['gamma'] ** int(config['runner']['batch_size'] / 2)
        super(GymRunner, self).__init__(config, env)

        if os.path.exists(os.path.join('./figures', config['env_name'])) is False:
            os.mkdir(os.path.join('./figures', config['env_name']))

    def run(self):
        # 에피소드마다 다음을 반복
        for ep in range(1, self._config['runner']['max_iteration_num'] + 1):
            # 에피소드 초기화
            state = self._env_init(reset_env=True)
            while not self.done:
                action = self._select_action(state)
                state = self._interaction(action)
                self._update_agent(next_state=state)

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep, 'Steps: ', self.count, 'Reward: ', self.batch_reward)
            self.save_batch_reward.append(self.batch_reward)
            self._sweep_cycle(ep)


class ParallelGymRunner:
    def __init__(self, config: dict):
        env_names = self.get_element(config['env_name'])
        q_cases = self.get_element(config['network']['actor']['memory_q_len'])
        layer_cases = self.get_element(config['network']['actor']['use_memory_layer'])

        self.worker_count = len(env_names) * len(q_cases) * len(layer_cases)
        self.env_names = env_names
        self.q_cases = q_cases
        self.layer_cases = layer_cases
        self.config = config

    def run(self):
        from multiprocessing import Pool
        pool = Pool(self.worker_count)
        args = []
        for env_name in self.env_names:
            sub_config = copy.deepcopy(self.config)
            sub_config['runner']['history_path'] = os.path.join("./history", env_name)
            if os.path.exists(sub_config['runner']['history_path']) is False:
                os.mkdir(sub_config['runner']['history_path'])

            sub_config['runner']['history_path'] = os.path.join(sub_config['runner']['history_path'],
                                                                sub_config['agent_name'])
            if os.path.exists(sub_config['runner']['history_path']) is False:
                os.mkdir(sub_config['runner']['history_path'])

            for q_len in self.q_cases:
                for layer_len in self.layer_cases:
                    sub_config['env_name'] = env_name
                    sub_config = self.update_config(config=sub_config, key='envs', name=env_name)
                    sub_config['network']['actor']['memory_q_len'] = q_len
                    sub_config['network']['actor']['use_memory_layer'] = layer_len
                    args.append(sub_config)

        pool.map(self.sub_runner_start, args)

    @staticmethod
    def update_config(config, key, name):
        from utils.yaml_config import YamlConfig
        root = os.path.join("./config/yaml/", key)
        name = name + '.yaml'
        sub_dict = YamlConfig.get_dict(os.path.join(root, name))
        config[key] = sub_dict[key]
        return copy.deepcopy(config)

    @staticmethod
    def sub_runner_start(config: dict):
        if config['runner']['render']:
            env = gym.make(config['env_name'], render_mode='human')
        else:
            env = gym.make(config['env_name'])
        sub_runner = GymRunner(config=config, env=env)
        sub_runner.run()

    @staticmethod
    def get_element(target):
        rtn_list = []
        if isinstance(target, list):
            return target
        else:
            rtn_list.append(target)
            return rtn_list
