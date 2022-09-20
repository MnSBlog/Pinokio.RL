import os
import random
import time

import gym
import pandas as pd
import torch.nn as nn
from networks.network_generator import CustomTorchNetwork
from agents import REGISTRY as AGENT_REGISTRY
from agents.general_agent import GeneralAgent
from utils.yaml_config import YamlConfig

from torchsummary import summary


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
        actor = CustomTorchNetwork(config['network'])
        critic = nn.Sequential(
            nn.Linear(config['network']['neck_input'] * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        algo_name = config['agent']['framework'] + config['agent_name']
        # Algorithm(Agent)
        self.agent: GeneralAgent = AGENT_REGISTRY[algo_name](parameters=self.config['agent'], actor=actor, critic=critic)
        # Environment
        self.env = env
        # Calculate
        self.save_epi_reward = []

    def run(self):
        runner_config = self.config["runner"]
        max_step_num = runner_config['max_step_num']
        steps = 0
        config_loader = YamlConfig(root='./config')
        state = self.env.reset()

        while max_step_num >= 0:
            # Step runner는 self-play 아직 적용안함
            if steps > runner_config['update_step']:
                steps = 0
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
            max_step_num -= 1
            actions = self.agent.select_action(state)
            state, reward, done, _ = self.env.step(actions)
            self.agent.batch_reward.append(reward)
            self.agent.batch_done(done)

            self.save_epi_reward.append(reward)

    def plot_result(self):
        import matplotlib as plt
        plt.plot(self.save_epi_reward)
        plt.show()


