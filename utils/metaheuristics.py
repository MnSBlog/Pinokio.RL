from multiprocessing import Pool
import copy
import random


class Solver:
    def __init__(self, **kwargs):
        inputs = ['parameters', 'terminal_function', 'test_function', 'operation']
        for k in inputs:
            if k not in kwargs:
                raise "Please set " + k

        self._Parameters = kwargs['parameters']
        self._IsTerminal = kwargs['terminal_function']
        self._TestFunc = kwargs['test_function']
        self._Operation = kwargs['operation']

    def __getitem__(self, key):
        return self._Parameters[key]

    def start(self):
        pass

    def close(self, forced=False):
        pass


class HarmonySearch(Solver):
    def __init__(self, **kwargs):
        super(HarmonySearch, self).__init__(**kwargs)
        # 0.7 ~ 0.95
        self.__hmcr = self._Parameters['hmcr']
        # 0.01 ~ 0.3
        self.__par = self._Parameters['par']
        self.__hms = self._Parameters['hms']
        self.__hm = []
        for _ in range(self.__hms):
            memory = {"output": 0.0, "value": self.__gen_memory()}
            self.__hm.append(memory)
        self.__best_memory = {"output": 0.0, "value": None}

    def start(self):
        worst_memory = {"output": 0.0, "value": None}
        with Pool(5) as p:
            new_memory = p.map(self._TestFunc, [1, 2, 3])
        new_memory['output']

    def __gen_memory(self):
        value = dict()
        for key, values in self._Parameters.item():
            if isinstance(values, list):
                value[key] = random.choice(values)
            if isinstance(values, dict):
                value[key] = random.randrange(values['min'], values['max'] + 1)
        return copy.deepcopy(value)
