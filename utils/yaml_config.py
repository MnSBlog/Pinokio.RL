import collections.abc
from copy import deepcopy
import os
import yaml


class YamlConfig:
    def __init__(self, root: str):
        self.root_path = root
        self.root_path = os.path.join(self.root_path, 'yaml')

        config_dir = '{0}/{1}'
        default_name = 'default'

        with open(config_dir.format(self.root_path, "{}.yaml".format(default_name)), "r") as f:
            try:
                default_config = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)

        self.final_config_dict = default_config

    def config_copy(self, config):
        if isinstance(config, dict):
            return {k: self.config_copy(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self.config_copy(v) for v in config]
        else:
            return deepcopy(config)

    def recursive_dict_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self.recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get_config(self, filenames: list):
        for file_path in filenames:
            sub_dict = self.get_dict(os.path.join(self.root_path, file_path + '.yaml'))
            self.final_config_dict = self.recursive_dict_update(self.final_config_dict, sub_dict)

        return self.final_config_dict

    @staticmethod
    def get_dict(path):
        with open(path, 'r') as f:
            try:
                sub_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format('sc2', exc)
        return sub_dict
