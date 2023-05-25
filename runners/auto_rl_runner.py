import copy
import numpy as np
import pandas as pd
import gymnasium as gym
from agents import GeneralAgent
from agents import REGISTRY as AGENT_REGISTRY
from outer.selector import AlgorithmComparator
from runners.general_runner import GeneralRunner


class AutoRLRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        super(AutoRLRunner, self).__init__(config, env)

    def run(self):
        reward_sum = 0.0
        for ep in range(1, self._config['runner']['max_iteration_num'] + 1):
            # 에피소드 초기화
            state = self._env_init(reset_env=True)
            while not self.done:
                action = self._select_action(state)
                state = self._interaction(action)
                self._update_agent(next_state=state)

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep, 'Steps: ', self.count, 'Reward: ', self.batch_reward)
            self._sweep_cycle(ep)
            reward_sum += copy.deepcopy(self.batch_reward.item())
        self._save_reward_log()
        self._close_env()
        return reward_sum, self._agent.metric


class AlgorithmFinder(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        self.comparator = AlgorithmComparator(config=config)
        self.reward_dataframe = dict()
        super(AlgorithmFinder, self).__init__(config, env)

    def run(self):
        while True:
            trajectory = self.__loop_inner()
            self.reward_dataframe[self._config['agent_name']] = trajectory
            self.comparator.update_score(self._config['agent_name'], sum(trajectory))
            config = self.__update_next()
            if config is None:
                dataframe = pd.DataFrame.from_dict(self.reward_dataframe)
                dataframe.to_csv('algorithm_comparison.csv', index=False)
                return
            else:
                self._config['agent'] = config['agent']
                self._config['agent_name'] = self._config['agent']['name']
                self.__change_algorithm()

    def __loop_inner(self):
        trajectory = []
        state = self._env_init(reset_env=True)
        for update in range(1, self._config['runner']['max_iteration_num'] + 1):
            self._env_init()
            steps = 0
            while self._update_agent(next_state=state, steps=steps) is False:
                steps += 1
                action = self._select_action(state)
                state = self._interaction(action)
            print('Update: ', update, 'Steps: ', self.count, 'Reward: ', self.batch_reward)
            trajectory.append(self.batch_reward)
            self._sweep_cycle(update)
        return trajectory

    def __update_next(self):
        lowest, score = self.comparator.get_ranker(-1)
        lowest['agent']['batch_size'] = self._config['runner']['batch_size']
        if score > np.NINF:
            return None
        else:
            return lowest

    def __change_algorithm(self):
        # 액터 신경망 및 크리틱 신경망 생성
        actor, critic = self._load_networks(self._config['agent']['framework'])

        # RL algorithm 생성
        algo_name = self._config['agent']['framework'] + self._config['agent_name']
        self._agent: GeneralAgent = AGENT_REGISTRY[algo_name](parameters=self._config['agent'],
                                                              actor=actor, critic=critic)

