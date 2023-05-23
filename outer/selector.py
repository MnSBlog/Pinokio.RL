import os
import numpy as np
from utils.yaml_config import YamlConfig


class AlgorithmComparator:
    def __init__(self, config):
        self.Pool = dict()
        self.Scores = dict()
        action_mode = config['network']['actor']['action_mode']
        root_path = r'./config/yaml/agents'
        algorithm_list = os.listdir(root_path)
        for algorithm in algorithm_list:
            algo_config = YamlConfig.get_dict(os.path.join(root_path, algorithm))
            action_config = algo_config['agent']['action']
            if action_mode in action_config:
                self.Pool[algorithm.replace('.yaml', '')] = algo_config

        for name in self.Pool.keys():
            self.Scores[name] = np.NINF

    def update_score(self, key, value):
        score = self.Scores[key]
        self.Scores[key] = (score + value) / 2.0

    def __quicksort_dict(self, dictionary):
        if len(dictionary) <= 1:
            return dictionary

        pivot = next(iter(dictionary))
        pivot_value = dictionary[pivot]
        less = {key: value for key, value in dictionary.items() if value < pivot_value}
        equal = {key: value for key, value in dictionary.items() if value == pivot_value}
        greater = {key: value for key, value in dictionary.items() if value > pivot_value}

        return self.__quicksort_dict(greater) + sorted(equal.items(),
                                                       key=lambda x: x[0], reverse=True) + self.__quicksort_dict(less)

    def __getitem__(self, item):
        return self.Pool[item]

    def get_ranker(self, rank):
        sorted_list = self.__quicksort_dict(self.Scores)
        key, config = sorted_list[rank]
        return config, self.Scores[key]

    def get_highest(self):
        return self.get_ranker(0)
