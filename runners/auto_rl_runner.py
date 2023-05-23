import copy
import os
import gym
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
        super(AlgorithmFinder, self).__init__(config, env)
        self.comparator = AlgorithmComparator(self._config)

