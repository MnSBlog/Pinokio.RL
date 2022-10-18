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
        self.reward_info = {'mu': [0], 'max': [0], 'min': [0], 'episode': [0], 'memory': []}

    def run(self):
        runner_config = self.config["runner"]
        max_step_num = runner_config['max_step_num']
        steps = 0
        ep = 0
        episode_reward = 0
        fig = plt.figure()
        network_type = '-' + self.config['network']['actor']['use_memory_layer']
        state = self.env.reset()

        while max_step_num >= 0:
            # Step runner는 self-play 아직 적용안함
            if steps >= runner_config['update_step']:
                steps = 0
                ep = 0
                episode_reward = 0
                self.agent.update()
                self.agent.save(checkpoint_path=os.path.join(runner_config['history_path'],
                                                             time.strftime('%Y-%m-%d_%H-%M-%S') + '.pth'))
                if runner_config['self_play']:
                    # agent_name = random.choice(self.config['agents'])
                    # config = config_loader.config_copy(
                    #     config_loader.get_config(filenames='agents//' + agent_name))
                    # agent_config = config['agent']
                    raise NotImplementedError

            steps += 1
            ep += 1
            max_step_num -= 1
            actions = self.agent.select_action(state)
            state, reward, done = self.env.step(actions)
            self.agent.batch_reward.append(reward)
            self.agent.batch_done.append(done)

            self.save_epi_reward.append(torch.mean(reward).item())
        np.savetxt(os.path.join(self.config['runner']['history_path'], '_epi_reward.txt'), self.save_epi_reward)
        # 에피소드마다 결과 보상값 출력
        print('Episode: ', ep, 'Steps: ', steps, 'Reward: ', episode_reward)
        self.reward_info['memory'].append(episode_reward)
        self.save_epi_reward.append(episode_reward)

        # 에피소드 100번마다 신경망 파라미터를 파일에 저장
        if ep % 100 == 0:
            checkpoint_path = os.path.join(self.config['runner']['history_path'],
                                           time.strftime('%Y-%m-%d-%H-%M-%S'))
            self.agent.save(checkpoint_path=checkpoint_path)

            mu = np.mean(self.reward_info['memory'])
            sigma = np.std(self.reward_info['memory'])

            self.reward_info['mu'].append(mu)
            self.reward_info['max'].append(mu + (0.5 * sigma))
            self.reward_info['min'].append(mu - (0.5 * sigma))
            self.reward_info['episode'].append(ep)
            self.reward_info['memory'].clear()

            plt.clf()
            plt.plot(self.reward_info['episode'], self.reward_info['mu'], '-')
            plt.fill_between(self.reward_info['episode'],
                             self.reward_info['min'], self.reward_info['max'], alpha=0.2)
            plt.ylim([0, 550])
            title = self.config['env_name'] + network_type + "-mem_len-" + ".png"
            plt.savefig("figures/" + title)

        # 학습이 끝난 후, 누적 보상값 저장
        filename = self.config['runner']['history_path'] + self.config['envs']['name']
        filename += 'stack-' + str(self.config['network']['memory_q_len']) + '_epi_reward.txt'
        np.savetxt(filename, self.save_epi_reward)
        self.env.close()
        print(self.save_epi_reward)


def plot_result(self):
    plt.plot(self.save_epi_reward)
    plt.show()
    plt.clf()


    def plot_result(self):
        import matplotlib.pyplot as plt
        plt.plot(self.save_epi_reward)
        plt.show()


