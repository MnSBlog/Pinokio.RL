import os
import random
import gym
import pandas as pd
import torch.nn as nn
from utils.yaml_config import YamlConfig
from agents import REGISTRY as agent_registry
from agents.general_agent import GeneralAgent

class EpisodeRunner:
    def __init__(self, config: dict, env: gym.Env, net: nn.Module):
        if config["self_play"]:
            algo_condition = pd.read_excel(config["condition_path"], engine='openpyxl')
            algo_condition = algo_condition.query('Select.str.contains("' + 'Use' + '")')
            algo_condition = algo_condition.query(
                '`' + config["env_config"]["actions"] + ' Actions`.str.contains("Yes")')
            algo_condition = algo_condition.query('Frameworks.str.contains("' + config["framework"] + '")')
            if config["env_config"]["multi_agent"]:
                algo_condition = algo_condition.query('Multi-Agent.str.contains("Yes")')

            config["agents"] = algo_condition['Algorithm'].to_list()

        # Save Path Dir
        if os.path.exists(config["history_path"]) is False:
            os.mkdir(config["history_path"])
        if os.path.exists(config["history_path"] + "/" + config["env"]) is False:
            os.mkdir(config["history_path"] + "/" + config["env"])
        if os.path.exists(config["history_path"] + "/" + config["env"] + "/" + "Best") is False:
            os.mkdir(config["history_path"] + "/" + config["env"] + "/" + "Best")

        for algorithm in config["agents"]:
            algorithm_path = config["history_path"] + "/" + config["env"] + "/" + algorithm
            if os.path.exists(algorithm_path) is False:
                os.mkdir(algorithm_path)
        config["history_path"] = config["history_path"] + "/" + config["env"]

        self.config = config
        env.reset()

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
            agent: GeneralAgent = agent_registry[agent_name](**agent_config)
            network: nn.Module =
            while True:
                agent_config

