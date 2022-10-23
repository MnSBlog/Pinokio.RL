import os
import random
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from networks.network_generator import CustomTorchNetwork
from agents import REGISTRY as AGENT_REGISTRY
from agents.general_agent import GeneralAgent
from utils.yaml_config import YamlConfig



def select_algorithm(args):
    config = args['runner']
    if config["self_play"]:
        algo_condition = pd.read_excel(config["condition_path"], engine='openpyxl')
        algo_condition = algo_condition.query('Select.str.contains("' + 'Use' + '")')
        algo_condition = algo_condition.query(
            '`' + config["env_config"]["actions"] + ' Actions`.str.contains("Yes")')
        algo_condition = algo_condition.query('Frameworks.str.contains("' + config["framework"] + '")')
        if config["env_config"]["multi_agent"]:
            algo_condition = algo_condition.query('Multi-Agent.str.contains("Yes")')

        config["agents"] = algo_condition['Algorithm'].to_list()

        for algorithm in config["agents"]:
            algorithm_path = config["history_path"].replace(args['agent_name'], algorithm)
            if os.path.exists(algorithm_path) is False:
                os.mkdir(algorithm_path)

    args['runner'] = config
    return args


class EpisodeRunner:
    def __init__(self, config: dict, env: gym.Env):
        config = select_algorithm(config)

        self.config = config
        self.net = net
        self.env = env

        self.env.reset()

    def run(self):
        iteration = self.config["iteration"]
        episode_count = 0
        config_loader = YamlConfig(root='./config')
        while iteration > 0:
            episode_count += 1
            agent_name = random.choice(self.config['agents'])
            config = config_loader.config_copy(
                config_loader.get_config(filenames='agents//' + agent_name))
            agent_config = config['agent']
            #agent: GeneralAgent = agent_registry[agent_name](**agent_config)
            network: nn.Module = self.net
            while True:
                agent_config


class StepRunner:
    def __init__(self, config: dict, env: gym.Env):
        self.config = select_algorithm(config)
        # Networks
        actor = CustomTorchNetwork(config['network']['actor'])
        critic = CustomTorchNetwork(config['network']['critic'])
        algo_name = config['agent']['framework'] + config['agent_name']
        # Algorithm(Agent)
        self.agent: GeneralAgent = AGENT_REGISTRY[algo_name](parameters=self.config['agent'],
                                                             actor=actor, critic=critic)
        if self.config['runner']['pretrain']:
            if os.listdir(self.config['runner']['history_path']):
                name_list = os.listdir(self.config['runner']['history_path'])
                full_list = [os.path.join(self.config['runner']['history_path'], name) for name in name_list]
                time_sorted_list = sorted(full_list, key=os.path.getmtime)
                last_file = time_sorted_list[-1]
                self.agent.load(checkpoint_path=last_file)
        # Environment
        self.env = env
        # Calculate
        self.save_epi_reward = []
        self.rew_min = self.config['envs']['reward_range'][0]
        self.rew_max = self.config['envs']['reward_range'][1]
        self.rew_gap = (self.rew_max - self.rew_min) / 2
        self.rew_numerator = (self.rew_max + self.rew_min) / 2
        self.reward_info = {'mu': [0], 'max': [0], 'min': [0], 'episode': [0], 'memory': []}
        self.memory_q = {'matrix': [], 'vector': []}

    def run(self):
        memory_len = self.config['network']['actor']['memory_q_len']
        network_type = '-' + self.config['network']['actor']['use_memory_layer']

        runner_config = self.config["runner"]
        max_step_num = runner_config['max_step_num']
        steps = 0
        updates = 1
        update_reward = 0
        update_fit_reward = 0
        fig = plt.figure()
        network_type = '-' + self.config['network']['actor']['use_memory_layer']
        init_state = self.env.reset()
        for _ in range(memory_len):
            self.__insert_q(init_state)
        state = self.__update_memory()

        while max_step_num >= 0:
            steps += 1
            max_step_num -= 1
            actions = self.agent.select_action(state)
            next_state, reward, done = self.env.step(actions)
            state = self.__update_memory(next_state)

            train_reward = self.__fit_reward(reward)
            self.agent.batch_reward.append(train_reward)
            self.agent.batch_done.append(done)
            update_fit_reward += train_reward
            update_reward += reward

            # Step runner는 self-play 아직 적용안함
            if steps >= runner_config['update_step']:
                # 업데이트마다 결과 보상값 출력
                print('Update: ', updates, 'Steps: ', runner_config['max_step_num'] - max_step_num,
                      'Reward: ', update_reward,  'fit_Reward: ', update_fit_reward)
                self.reward_info['memory'].append(float(update_reward))
                update_reward = 0
                update_fit_reward = 0
                steps = 0
                updates += 1
                self.agent.update()
                if runner_config['self_play']:
                    # agent_name = random.choice(self.config['agents'])
                    # config = config_loader.config_copy(
                    #     config_loader.get_config(filenames='agents//' + agent_name))
                    # agent_config = config['agent']
                    raise NotImplementedError

            # 업데이트 100번마다 신경망 파라미터를 파일에 저장
            if updates % 50 == 0:

                checkpoint_path = os.path.join(self.config['runner']['history_path'],
                                               time.strftime('%Y-%m-%d-%H-%M-%S'))
                self.agent.save(checkpoint_path=checkpoint_path)

                mu = np.mean(self.reward_info['memory'])
                sigma = np.std(self.reward_info['memory'])

                self.reward_info['mu'].append(mu)
                self.reward_info['max'].append(mu + (0.5 * sigma))
                self.reward_info['min'].append(mu - (0.5 * sigma))
                self.reward_info['episode'].append(updates)
                self.reward_info['memory'].clear()

                plt.clf()
                plt.plot(self.reward_info['episode'], self.reward_info['mu'], '-')
                plt.fill_between(self.reward_info['episode'],
                                 self.reward_info['min'], self.reward_info['max'], alpha=0.2)
                title = self.config['env_name'] + network_type + "-mem_len-" + ".png"
                plt.savefig("figures/" + title)

    def __fit_reward(self, rew):
        if self.rew_min > rew:
            print('reward min is updated: ', rew)
            self.rew_min = rew
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        elif self.rew_max < rew:
            print('reward max is updated: ', rew)
            self.rew_max = rew
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        # 학습용 보상 [-1, 1]로 fitting
        rew = np.reshape(rew, [1, 1])
        train_reward = (rew - self.rew_numerator) / self.rew_gap

        return train_reward

