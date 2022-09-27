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
        state_dim = env.observation_space.shape[0]
        self.config['agent']['state_dim'] = state_dim
        # 행동 차원
        self.action_dim = env.action_space

        # 액터 신경망 및 크리틱 신경망 생성
        if "tf" in config['agent']['framework']:
            actor = Actor(self.action_dim.shape[0], self.action_dim.high[0])
            critic = Critic()
            actor.build(input_shape=(None, state_dim))
            critic.build(input_shape=(None, state_dim))
        else:
            actor = CustomTorchNetwork(config['network'])
            critic = nn.Sequential(
                nn.Linear(config['network']['neck_in'], 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

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

        self.action = None
        self.reward = 0

    def run(self):
        rew_min = self.config['envs']['reward_range'][0]
        rew_max = self.config['envs']['reward_range'][1]
        rew_gap = (rew_max - rew_min) / 2
        rew_numerator = (rew_max + rew_min) / 2
        memory_len = self.config['network']['memory_q_len']
        # 에피소드마다 다음을 반복
        for ep in range(self.config['runner']['max_episode_num']):
            done_checker = -1
            done_lives = 0
            # 에피소드 초기화
            step, episode_reward, done = 0, 0, False

            # 환경 초기화 및 초기 상태 관측 및 큐
            memory_q = {'matrix': [], 'vector': []}
            for _ in range(0, memory_len):
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    spatial_obs = obs[0]
                    spatial_obs = spatial_obs / 255
                    spatial_obs = spatial_obs.transpose((2, 0, 1))
                    memory_q['matrix'].append(spatial_obs)
                    non_spatial_obs = np.array(list(obs[1].values()))
                else:
                    non_spatial_obs = obs
                memory_q['vector'].append(non_spatial_obs)
            matrix_obs = np.concatenate(memory_q['matrix'], axis=0)
            vector_obs = np.concatenate(memory_q['vector'], axis=0)
            state = {'matrix': matrix_obs, 'vector': vector_obs, 'action_mask': None}

            if len(memory_q['matrix']) > 0:
                memory_q['matrix'].pop(0)
            if len(memory_q['vector']) > 0:
                memory_q['vector'].pop(0)

            while not done:
                if self.config['runner']['render']:
                    self.env.render()
                self.action = self.agent.select_action(state)
                if torch.is_tensor(self.action):
                    self.action = self.action.squeeze(dim=0)
                    self.action = self.action.item()
                # 다음 상태, 보상 관측
                spatial_obs, reward, done, trunc, info = self.env.step(self.action)
                if trunc:
                    test = 1
                done |= trunc
                if rew_min > reward:
                    print('reward min is updated: ', reward)
                    rew_min = reward
                    rew_gap = (rew_max - rew_min) / 2
                    rew_numerator = (rew_max + rew_min) / 2
                elif rew_max < reward:
                    print('reward max is updated: ', reward)
                    rew_max = reward
                    rew_gap = (rew_max - rew_min) / 2
                    rew_numerator = (rew_max + rew_min) / 2
                # 학습용 보상 [-1, 1]로 fitting
                reward = np.reshape(reward, [1, 1])
                train_reward = (reward - rew_numerator) / rew_gap

                # shape에 따라 잘라야하네
                spatial_obs = spatial_obs / 255
                spatial_obs = spatial_obs.transpose((2, 0, 1))

                memory_q['matrix'].append(spatial_obs)
                non_spatial_obs = np.array(list(info.values()))
                memory_q['vector'].append(non_spatial_obs)
                
                matrix_obs = np.concatenate(memory_q['matrix'], axis=0)
                vector_obs = np.concatenate(memory_q['vector'], axis=0)
                next_state = {'matrix': matrix_obs, 'vector': vector_obs, 'action_mask': None}

                if len(memory_q['matrix']) > 0:
                    memory_q['matrix'].pop(0)
                if len(memory_q['vector']) > 0:
                    memory_q['vector'].pop(0)

                # Bug로 인한 done check 및 mask 작업
                dead_line = spatial_obs[0, 189:196, :]
                left_lim = dead_line[:, 8:13]
                right_lim = dead_line[:, 146:151]
                action_mask = None
                if len(left_lim[left_lim != 0]) >= 20:
                    action_mask = np.array([1, 1, 1, 0])
                if len(right_lim[right_lim != 0]) >= 20:
                    action_mask = np.array([1, 1, 0, 1])
                next_state['action_mask'] = action_mask

                countable = len(dead_line[dead_line != 0])
                if countable > 170:
                    if done_checker < 0:
                        done_checker = 20
                        done_lives = info['lives']

                if done_checker > 0:
                    if done is True or done_lives != info['lives']:
                        done_checker = 0
                    done_checker -= 1
                    if done_checker == 0:
                        done = True

                self.agent.batch_reward.append(train_reward)
                self.agent.batch_done.append(done)
                step += 1
                episode_reward += reward[0]
                state = next_state

                if len(self.agent.batch_state) >= self.config['runner']['batch_size']:
                    # 학습 추출
                    self.agent.update(next_state=next_state, done=done)
            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep + 1, 'Steps: ', step, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)

            # 에피소드 10번마다 신경망 파라미터를 파일에 저장
            if ep % 10 == 0:
                import time
                checkpoint_path = os.path.join(self.config['runner']['history_path'],
                                               time.strftime('%Y-%m-%d-%H-%M-%S'))
                self.agent.save(checkpoint_path=checkpoint_path)

        # 학습이 끝난 후, 누적 보상값 저장
        filename = self.config['runner']['history_path'] + self.config['envs']['name']
        filename += 'stack-' + str(self.config['network']['memory_q_len']) + '_epi_reward.txt'
        np.savetxt(filename, self.save_epi_reward)
        print(self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
