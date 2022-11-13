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
        self.__hm = {'output': [], 'value': []}
        for _ in range(self.__hms):
            self.__hm['output'].append(0.0)
            self.__hm['value'].append(self.__gen_memory())

    def start(self):
        short_term_memory = copy.deepcopy(self.__hm['value'])

        while self._IsTerminal() is False:
            with Pool(self.__hms) as p:
                new_output = p.map(self._TestFunc, short_term_memory)
            outputs = self.__hm['output'] + new_output
            short_term_memory = self.__hm['value'] + short_term_memory

            index = range(len(outputs))
            sorted_index = sorted(index, key=lambda k: outputs[k], reverse=True)
            self.__hm['value'] = short_term_memory[:sorted_index[:self.__hms]]
            self.__hm['output'] = outputs[:sorted_index[:self.__hms]]
            short_term_memory = self.__next_harmony()

    def close(self, forced=False):
        if forced or self._IsTerminal():
            return self.__hm['value'][0], self.__hm['output'][0]
        else:
            return None, None

    def __gen_memory(self):
        value = dict()
        for key, values in self._Parameters.item():
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
                harmony = random.choice(self.__hm['value'])
                for key, values in self._Parameters.item():
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
        if choice is not None and random.random() <= self.__par:
            choice += 1
            if random.random() < 0.5:
                choice -= 2
        if value is not None and random.random() <= self.__par:
            min_val = int(value * (1 - self.__par))
            max_val = int(value * (1 + self.__par))
            value = random.randrange(min_val, max_val + 1)
        return choice, value
