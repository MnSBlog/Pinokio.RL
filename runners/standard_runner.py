import os
import gym
import pandas as pd
from runners.general_runner import GeneralRunner


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


class StepRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        super(StepRunner, self).__init__(config, env)

        if os.path.exists(os.path.join('./figures', config['env_name'])) is False:
            os.mkdir(os.path.join('./figures', config['env_name']))

    def run(self):
        runner_config = self._config["runner"]
        max_step_num = runner_config['max_step_num']
        steps = 0
        updates = 1
        update_reward = 0
        update_fit_reward = 0
        
        init_state = self._env.reset()
        for _ in range(self.memory_len):
            self._insert_q(init_state)
        state = self._update_memory()

        while max_step_num >= 0:
            steps += 1
            max_step_num -= 1
            actions = self._agent.select_action(state)
            next_state, reward, done = self._env.step(actions)
            state = self._update_memory(next_state)

            train_reward = self._fit_reward(reward)
            self._agent.batch_reward.append(train_reward)
            self._agent.batch_done.append(done)
            update_fit_reward += train_reward
            update_reward += reward

            # Step runner는 self-play 아직 적용안함
            if steps >= runner_config['update_step']:
                # 업데이트마다 결과 보상값 출력
                print('Update: ', updates, 'Steps: ', runner_config['max_step_num'] - max_step_num,
                      'Reward: ', update_reward,  'fit_Reward: ', update_fit_reward)
                self.reward_info['memory'].append(float(update_reward))
                update_reward = 0
                update_fit_reward = 0
                steps = 0
                updates += 1
                self._agent.update()
                if runner_config['self_play']:
                    # agent_name = random.choice(self.config['agents'])
                    # config = config_loader.config_copy(
                    #     config_loader.get_config(filenames='agents//' + agent_name))
                    # agent_config = config['agent']
                    raise NotImplementedError

            # 업데이트 특정값 신경망 파라미터를 파일에 저장
            if updates % 50 == 0:
                self._save_agent()
                self._draw_reward_plot(now_ep=updates, y_lim=[])

