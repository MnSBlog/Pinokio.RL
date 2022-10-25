import os
import gym
import numpy as np
import pandas as pd
import torch

from runners.general_runner import GeneralRunner


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
                      'Reward: ', update_reward, 'fit_Reward: ', update_fit_reward)
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


class IntrinsicParallelRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        super(IntrinsicParallelRunner, self).__init__(config, env)

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
        # 1, 5, 180
        # 1, 5, 8, 30, 30
        while max_step_num >= 0:
            steps += 1
            max_step_num -= 1
            map_count = state['matrix'].shape[0]
            actions: torch.tensor = None
            for idx in range(map_count):
                temp_matrix = state['matrix'][idx]
                temp_vector = state['vector'][idx]
                temp_action_mask = None
                if state['action_mask'] is not None:
                    temp_action_mask = state['action_mask'][idx]
                temp_state = {'matrix': temp_matrix,
                              'vector': temp_vector,
                              'action_mask': temp_action_mask}
                temp = self._agent.select_action(temp_state)
                action = torch.stack([temp[0], temp[1], temp[2]], dim=1)
                action = self.__add_index(idx, action)
                if actions is None:
                    actions = action.unsqueeze(dim=0)
                else:
                    actions = torch.cat([actions, action.unsqueeze(dim=0)], dim=0)
            next_state, reward, done, _ = self._env.step(actions)
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
                      'Reward: ', update_reward, 'fit_Reward: ', update_fit_reward)
                self.reward_info['memory'].append(update_reward)
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

    def __add_index(self, env_index, action: torch.tensor):
        temp_numpy = action.numpy()
        agent_index = list(range(action.size(0)))
        temp_numpy = np.c_[agent_index, temp_numpy]
        env_index = [env_index for _ in range(action.size(0))]
        temp_numpy = np.c_[env_index, temp_numpy]

        return torch.tensor(temp_numpy, dtype=torch.float)

    def _fit_reward(self, rew):
        if True in (self.rew_min > rew[:]):
            print('reward min is updated: ', rew)
            self.rew_min = rew.min().item()
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        elif True in (self.rew_max < rew[:]):
            print('reward max is updated: ', rew)
            self.rew_max = rew.max().item()
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        # 학습용 보상 [-1, 1]로 fitting
        train_reward = (rew - self.rew_numerator) / self.rew_gap

        return train_reward
