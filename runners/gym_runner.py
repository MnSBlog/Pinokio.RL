import os
import gym
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
from agents.tf.actorcritic import Actor, Critic
from agents import REGISTRY as AGENT_REGISTRY
from agents.general_agent import GeneralAgent
from networks.network_generator import CustomTorchNetwork


class GymRunner:
    def __init__(self, config: dict, env: gym.Env):
        self.config = config
        algo_name = config['agent']['framework'] + config['agent_name']

        # 상태변수 차원
        # state_dim = config['network']['actor']['non_spatial_feature']['dim_in']
        state_dim = 17
        self.config['agent']['state_dim'] = state_dim

        # 액터 신경망 및 크리틱 신경망 생성
        if "tf" in config['agent']['framework']:
            # 행동 차원
            self.action_dim = env.action_space
            actor = Actor(self.action_dim.shape[0], self.action_dim.high[0])
            critic = Critic()
            actor.build(input_shape=(None, state_dim))
            critic.build(input_shape=(None, state_dim))
        else:
            actor = CustomTorchNetwork(config['network']['actor'])
            critic = CustomTorchNetwork(config['network']['critic'])

        # *****
        self.config['agent']['mid_gamma'] \
            = self.config['agent']['gamma'] ** int(self.config['runner']['batch_size'] / 2)

        self.agent: GeneralAgent = AGENT_REGISTRY[algo_name](parameters=self.config['agent'],
                                                             actor=actor, critic=critic)

        if self.config['runner']['pretrain']:
            try:
                if os.listdir(self.config['runner']['history_path']):
                    name_list = os.listdir(self.config['runner']['history_path'])
                    full_list = [os.path.join(self.config['runner']['history_path'], name) for name in name_list]
                    time_sorted_list = sorted(full_list, key=os.path.getmtime)
                    last_file = time_sorted_list[-1]
                    self.agent.load(checkpoint_path=last_file)
            finally:
                pass

        self.env = env
        self.save_epi_reward = []
        self.reward_info = {'mu': [0], 'max': [0], 'min': [0], 'episode': [0], 'memory': []}

        self.action = None
        self.reward = 0
        self.memory_q = {'matrix': [], 'vector': []}

        self.rew_min = self.config['envs']['reward_range'][0]
        self.rew_max = self.config['envs']['reward_range'][1]
        self.rew_gap = (self.rew_max - self.rew_min) / 2
        self.rew_numerator = (self.rew_max + self.rew_min) / 2

    def run(self):
        memory_len = self.config['network']['actor']['memory_q_len']
        network_type = '-' + self.config['network']['actor']['use_memory_layer']
        fig = plt.figure()
        # 에피소드마다 다음을 반복
        for ep in range(1, self.config['runner']['max_episode_num'] + 1):
            # 에피소드 초기화
            step, episode_reward, done = 0, 0, False
            self.memory_q = {'matrix': [], 'vector': []}
            # 환경 초기화 및 초기 상태 관측 및 큐
            ret = self.env.reset()
            state = ret[0]
            for _ in range(0, memory_len):
                self.__insert_q(state)

            state = self.__update_memory()
            while not done:
                if self.config['runner']['render']:
                    self.env.render()
                self.action = self.agent.select_action(state)
                if torch.is_tensor(self.action):
                    self.action = self.action.squeeze()
                    if self.action.shape[0] == 1:
                        self.action = self.action.item()
                # 다음 상태, 보상 관측
                state, reward, done, trunc, info = self.env.step(self.action)
                done |= trunc
                state = self.__update_memory(state[0])

                train_reward = self.__fit_reward(reward)
                self.agent.batch_reward.append(train_reward)
                self.agent.batch_done.append(done)
                step += 1
                episode_reward += reward

                if len(self.agent.batch_reward) >= self.config['runner']['batch_size']:
                    # 학습 추출
                    self.agent.update(next_state=state, done=done)
            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep, 'Steps: ', step, 'Reward: ', episode_reward)
            self.reward_info['memory'].append(episode_reward)
            self.save_epi_reward.append(episode_reward)

            # 에피소드 10번마다 신경망 파라미터를 파일에 저장
            if ep % 10 == 0:
                import time
                checkpoint_path = os.path.join(self.config['runner']['history_path'],
                                               time.strftime('%Y-%m-%d-%H-%M-%S'))
                self.agent.save(checkpoint_path=checkpoint_path)

                mu = np.mean(self.reward_info['memory'])
                sigma = np.std(self.reward_info['memory'])

                self.reward_info['mu'].append(mu)
                self.reward_info['max'].append(mu + (1 * sigma))
                self.reward_info['min'].append(mu - (1 * sigma))
                self.reward_info['episode'].append(ep)
                self.reward_info['memory'].clear()

                plt.clf()
                plt.plot(self.reward_info['episode'], self.reward_info['mu'], '-')
                plt.fill_between(self.reward_info['episode'],
                                 self.reward_info['min'], self.reward_info['max'], alpha=0.2)
                plt.ylim([0, 550])
                title = self.config['env_name'] + network_type + "-mem_len-" + str(memory_len) + ".png"
                plt.savefig("figures/" + title)

        # 학습이 끝난 후, 누적 보상값 저장
        filename = self.config['runner']['history_path'] + self.config['envs']['name']
        filename += 'stack-' + str(self.config['network']['memory_q_len']) + '_epi_reward.txt'
        np.savetxt(filename, self.save_epi_reward)
        self.env.close()
        print(self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
        plt.clf()

    def __insert_q(self, state):
        if isinstance(state, dict):
            pass
        else:
            if len(state.shape) > 1:
                self.memory_q['matrix'].append(state)
            else:
                self.memory_q['vector'].append(state)

    def __update_memory(self, state=None):
        matrix_obs, vector_obs = [], []

        if state is not None:
            self.__insert_q(state)

        if len(self.memory_q['matrix']) > 0:
            matrix_obs = np.concatenate(self.memory_q['matrix'], axis=0)
            self.memory_q['matrix'].pop(0)

        if len(self.memory_q['vector']) > 0:
            vector_obs = np.concatenate(self.memory_q['vector'], axis=0)
            self.memory_q['vector'].pop(0)

        state = {'matrix': matrix_obs, 'vector': vector_obs, 'action_mask': None}
        return state

    def __fit_reward(self, rew):
        if self.rew_min > rew:
            print('reward min is updated: ', rew)
            self.rew_min = rew
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        elif self.rew_max < rew:
            print('reward max is updated: ', rew)
            self.rew_max = rew
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        # 학습용 보상 [-1, 1]로 fitting
        rew = np.reshape(rew, [1, 1])
        train_reward = (rew - self.rew_numerator) / self.rew_gap

        return train_reward
