import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from agents.tf.actorcritic import Actor, Critic
from agents.tf import REGISTRY as AgentRegistry
from agents.general_agent import GeneralAgent


class GymRunner:
    def __init__(self, config: dict, env: gym.Env):
        self.config = config
        algo_key = self.config['agent']['framework'] + '_' + self.config['agent']['name']

        # 상태변수 차원
        state_dim = env.observation_space.shape[0]
        self.config['agent']['state_dim'] = state_dim
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        # 행동의 최대 크기
        action_bound = env.action_space.high[0]

        # 액터 신경망 및 크리틱 신경망 생성
        actor = Actor(self.action_dim, action_bound)
        critic = Critic()
        actor.build(input_shape=(None, state_dim))
        critic.build(input_shape=(None, state_dim))

        # *****
        self.config['agent']['mid_gamma']\
            = self.config['agent']['gamma'] ** int(self.config['runner']['batch_size'] / 2)

        self.agent: GeneralAgent = AgentRegistry[algo_key](parameters=self.config['agent'],
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

        self.obs = self.env.reset()
        self.action = None
        self.reward = 0

    def run(self):
        rew_min = self.config['envs']['reward_range'][0]
        rew_max = self.config['envs']['reward_range'][1]
        rew_gap = (rew_max - rew_min) / 2
        rew_numerator = (rew_max + rew_min) / 2
        # 에피소드마다 다음을 반복
        for ep in range(self.config['runner']['max_episode_num']):
            # 에피소드 초기화
            step, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            self.obs = self.env.reset()
            while not done:
                if self.config['runner']['render']:
                    self.env.render()
                self.action = self.agent.select_action(tf.convert_to_tensor([self.obs], dtype=tf.float32))
                # 다음 상태, 보상 관측
                next_state, reward, done, _ = self.env.step(self.action)
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
                self.agent.batch_reward.append(train_reward)

                step += 1
                episode_reward += reward[0]
                self.obs = next_state
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
        np.savetxt(self.config['runner']['history_path'] + self.config['envs']['name'] + '_epi_reward.txt',
                   self.save_epi_reward)
        print(self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
