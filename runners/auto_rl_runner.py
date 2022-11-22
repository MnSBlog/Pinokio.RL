import os
import gym
from runners.general_runner import GeneralRunner


class AutoRLRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        super(AutoRLRunner, self).__init__(config, env)

        if os.path.exists(os.path.join(self._config['runner']['figure_path'], config['env_name'])) is False:
            os.mkdir(os.path.join(self._config['runner']['figure_path'], config['env_name']))

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
            reward_sum += self.batch_reward
            self.save_batch_reward.append(self.batch_reward)
            self._sweep_cycle(ep)
        return reward_sum, self._agent.metric
