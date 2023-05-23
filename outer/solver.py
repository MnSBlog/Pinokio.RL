import os
import copy
import random
import shutil
from collections import defaultdict
import statistics
import numpy as np
from main import load_config
from utils.yaml_config import YamlConfig
from multiprocessing import Pool
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class Solver:
    def __init__(self, **kwargs):
        inputs = ['parameters', 'test_function']
        for k in inputs:
            if k not in kwargs:
                raise "Please set " + k

        self._Parameters = kwargs['parameters']
        self._TestFunc = kwargs['test_function']
        self._BestParameter = None
        if self._Parameters['maximize']:
            self._BestOutput = np.NINF
        else:
            self._BestOutput = np.Inf

    def __getitem__(self, key):
        return self._Parameters[key]

    def check_improvement(self, source, target):
        if self._Parameters['maximize']:
            if source < target:
                return True
            else:
                return False
        else:
            if source > target:
                return True
            else:
                return False

    def start(self, root):
        pass

    def close(self, forced=False):
        pass

    def is_terminal(self):
        pass


class WrappedMatern(Matern):
    def __init__(self, param_types, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        self.param_types = param_types
        super(WrappedMatern, self).__init__(length_scale, length_scale_bounds, nu)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = X.copy()  # copy X, just to be sure

        # we collect the positions of the categorical variables in this dict
        categorical_group_indices = defaultdict(list)

        for i, ptype in enumerate(self.param_types):
            if ptype == 'continuous':
                pass  # do nothing
            elif ptype == 'discrete':
                X[:, i] = np.round(X[:, i])
            else:
                categorical_group_indices[ptype].append(i)

        # set binary max for categorical groups
        for indices in categorical_group_indices.values():
            max_col = np.argmax(X[:, indices], axis=1)
            X[:, indices] = 0
            X[range(X.shape[0]), max_col] = 1

        return super(WrappedMatern, self).__call__(X, Y, eval_gradient)


class Bayesian(Solver):
    def __init__(self, **kwargs):
        super(Bayesian, self).__init__(**kwargs)

        self.__init_points = self._Parameters['init_points']
        self.__n_iter = self._Parameters['n_iter']
        self.__verbose = self._Parameters['verbose']
        self.__random_state = self._Parameters['random_state']
        self.__pbound = self.make_bounds()
        self.optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.__pbound,
            verbose=self.__verbose,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=self.__random_state,
        )

    def make_bounds(self):
        bounds = dict()
        for key, value in self._Parameters.items():
            if isinstance(value, list):
                bound = (0, len(value))
                bounds[key] = bound
            if isinstance(value, dict):
                bound = (value['min'], value['max'])
                bounds[key] = bound
        return copy.deepcopy(bounds)

    def objective_function(self, **kwargs):
        param = self.convert_param(kwargs)
        configs = []
        for _ in range(8):
            configs.append(param)

        with Pool(8) as p:
            new_output = p.map(self._TestFunc, configs)
        mean = statistics.mean(new_output)
        return mean

    def convert_param(self, kwargs):
        param = dict()
        for key, value in self._Parameters.items():
            if isinstance(value, list):
                index = int(kwargs[key])
                if len(value) >= index:
                    index = len(value) - 1
                param[key] = value[index]
            if isinstance(value, dict):
                param[key] = int(kwargs[key])
        return copy.deepcopy(param)

    def get_types(self):
        types = []
        for key, value in self.__pbound.items():
            if isinstance(value[0], int):
                types.append('discrete')
            else:
                types.append(0)
        return tuple(types)

    def start(self, root):
        count = len(os.listdir(root)) + 1
        os.mkdir(os.path.join(root, str(count) + '-Gen'))

        logger = JSONLogger(path=os.path.join(root, str(count) + '-Gen', "logs.json"))
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        param_type = self.get_types()
        modified_kernel = WrappedMatern(param_types=param_type, nu=2.5)
        self.optimizer._gp = GaussianProcessRegressor(kernel=modified_kernel,
                                                      alpha=1e-2,  # alpha=1e-6,
                                                      normalize_y=True,
                                                      n_restarts_optimizer=5,
                                                      random_state=self.optimizer._random_state)
        self.optimizer.maximize(
            init_points=self.__init_points,
            n_iter=self.__n_iter,
        )

    def close(self, forced=False):
        result = self.optimizer.max
        param = self.convert_param(result['params'])
        output = result['target']
        return param, output


class HarmonySearch(Solver):
    def __init__(self, **kwargs):
        super(HarmonySearch, self).__init__(**kwargs)
        # 0.7 ~ 0.95
        self.__hmcr = self._Parameters['hmcr']
        # 0.01 ~ 0.3
        self.__par = self._Parameters['par']
        self.__hms = self._Parameters['hms']
        self.__hm = {'output': [], 'value': []}
        self.not_update_memory = 0
        self._BestOutput = np.NINF
        if self._Parameters['load_last']:
            run_args = load_config()
            log_path = './figures/AutoRL/' + run_args['env_name']
            folder_list = os.listdir(log_path)
            load_path = os.path.join(log_path, folder_list[-2])
            gene_folders = os.listdir(load_path)
            # update 및 Best 추출
            for gene in gene_folders:
                self.not_update_memory += 1
                gene_path = os.path.join(load_path, gene)
                outputs = os.listdir(gene_path)
                for output in outputs:
                    if self._BestOutput < float(output):
                        self._BestOutput = float(output)
                        contents = os.listdir(os.path.join(gene_path, output))
                        for content in contents:
                            if 'yaml' in content:
                                self._BestParameter = self.__encode_memory(YamlConfig.get_dict(os.path.join(gene_path, output, content)))
                        self.not_update_memory = 0

            # 살아있는 Gen은 이어서 붙이기
            gene_path = os.path.join(load_path, gene_folders[-1])
            outputs = os.listdir(gene_path)
            for output in outputs:
                contents = os.listdir(os.path.join(gene_path, output))
                for content in contents:
                    if 'yaml' in content:
                        value = YamlConfig.get_dict(os.path.join(gene_path, output, content))
                        self.__hm['output'].append(float(output))
                        self.__hm['value'].append(self.__encode_memory(value))
                if len(self.__hm['output']) == self.__hms:
                    break
            # 초기 샘플마저 추출
            if len(self.__hm['output']) < self.__hms:
                for _ in range(self.__hms - len(self.__hm['output'])):
                    self.__hm['output'].append(np.NINF)
                    self.__hm['value'].append(self.__gen_memory())
            # 최신 폴더로 모두 이동
            for gene in gene_folders:
                old_path = os.path.join(load_path, gene)
                new_path = os.path.join(log_path, folder_list[-1], gene)
                shutil.move(old_path, new_path)
        else:
            for _ in range(self.__hms):
                self.__hm['output'].append(self._BestOutput)
                self.__hm['value'].append(self.__gen_memory())

            self._BestParameter = copy.deepcopy(self.__hm['value'][0])

    def start(self, root):
        short_term_memory = copy.deepcopy(self.__hm['value'])
        while self.is_terminal() is False:
            new_output = []
            count = len(os.listdir(root)) + 1
            os.mkdir(os.path.join(root, str(count) + '-Gen'))
            short_term_memory.append(self._BestParameter)
            run_args = load_config()
            run_args_list = []
            for i in range(self.__hms + 1):
                temp = copy.deepcopy(run_args)
                if 'port' in temp['envs']:
                    temp['envs']['port'] = 49999 - i
                run_args_list.append(temp)

            if self._Parameters['parallel']:
                with Pool(self.__hms + 1) as p:
                    new_output = p.starmap(self._TestFunc, zip(short_term_memory, run_args_list))
            else:
                for idx in range(self.__hms):
                    out = self._TestFunc(short_term_memory[idx])
                    new_output.append(copy.deepcopy(out))

            worst_output = copy.deepcopy(self.__hm['output'][-1])
            outputs = copy.deepcopy(self.__hm['output']) + new_output
            short_term_memory = self.__hm['value'] + short_term_memory

            index = range(len(outputs))
            sorted_index = sorted(index, key=lambda k: outputs[k], reverse=self._Parameters['maximize'])
            self.__hm['value'] = []
            self.__hm['output'] = []
            for winner in sorted_index[:self.__hms]:
                self.__hm['value'].append(copy.deepcopy(short_term_memory[winner]))
                self.__hm['output'].append(copy.deepcopy(outputs[winner]))

            # if self.check_improvement(worst_output, self.__hm['output'][-1]):
            #     self.not_update_memory = 0
            # else:
            #     self.not_update_memory += 1

            if self.check_improvement(self._BestOutput, self.__hm['output'][0]):
                self.not_update_memory = 0
                self._BestOutput = copy.deepcopy(self.__hm['output'][0])
                self._BestParameter = copy.deepcopy(self.__hm['value'][0])
            else:
                self.not_update_memory += 1
            short_term_memory = self.__next_harmony()

    def close(self, forced=False):
        if forced or self.is_terminal():
            return self._BestParameter, self._BestOutput
        else:
            return None, None

    def is_terminal(self):
        return self.not_update_memory >= 10

    def __encode_memory(self, args):
        memory = dict()
        for key, values in self._Parameters.items():
            if isinstance(values, list) or isinstance(values, dict):
                sep = key.split('-')
                sub = copy.deepcopy(args)
                for level in sep:
                    sub = sub[level]
                memory[key] = sub
        return copy.deepcopy(memory)

    def __gen_memory(self):
        value = dict()
        for key, values in self._Parameters.items():
            if isinstance(values, list):
                choice = random.choice(range(len(values)))
                value[key] = values[choice % len(values)]
            if isinstance(values, dict):
                temp_value = random.randrange(values['min'], values['max'] + 1)
                value[key] = temp_value
        return copy.deepcopy(value)

    def __next_harmony(self):
        new_harmony = []
        for _ in range(self.__hms):
            if self.__hmcr < random.random():
                new_harmony.append(self.__gen_memory())
            else:
                harmony = random.choice(self.__hm['value'][:self.__hms//2])
                for key, values in self._Parameters.items():
                    if isinstance(values, list):
                        choice = values.index(harmony[key])
                        choice, _ = self.__adjust_pitch(choice=choice)
                        harmony[key] = values[choice % len(values)]
                    if isinstance(values, dict):
                        value = harmony[key]
                        _, value = self.__adjust_pitch(value=value)
                        if value < values['min']:
                            value = values['min']
                        if value > values['max']:
                            value = values['max']
                        harmony[key] = value
                new_harmony.append(harmony)
        return new_harmony

    def __adjust_pitch(self, choice=None, value=None):
        if random.random() < self.__par:
            if choice is not None:
                choice += 1
                if random.random() < 0.5:
                    choice -= 2
            if value is not None:
                min_val = int(value * (1 - self.__par))
                max_val = int(value * (1 + self.__par))
                value = random.randrange(min_val, max_val + 1)
        return choice, value
