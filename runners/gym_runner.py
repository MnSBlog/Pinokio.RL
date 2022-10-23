import copy
import os
import gym
import numpy as np
import torch
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

        if os.path.exists(os.path.join('./figures', config['env_name'])) is False:
            os.mkdir(os.path.join('./figures', config['env_name']))

    def run(self):
        memory_len = self.config['network']['actor']['memory_q_len']
        layer_len = self.config['network']['actor']['memory_layer_len']
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
            for _ in range(memory_len):
                self.__insert_q(state)

            state = self.__update_memory()
            while not done:
                if self.config['runner']['render']:
                    self.env.render()
                self.action = self.agent.select_action(state)
                if torch.is_tensor(self.action):
                    self.action = self.action.squeeze()
                    if len(self.action.shape) == 0:
                        self.action = self.action.item()
                # 다음 상태, 보상 관측
                state, reward, done, trunc, info = self.env.step(self.action)
                done |= trunc
                state = self.__update_memory(state)

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
            if ep % 100 == 0:
                import time
                save_name = network_type + "-mem_len-" + str(memory_len) + "-layer_len-" + str(
                    layer_len) + time.strftime('%Y-%m-%d-%H-%M-%S')
                checkpoint_path = os.path.join(self.config['runner']['history_path'], save_name)
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
                title = network_type + "-mem_len-" + str(memory_len) + "-layer_len-" + str(layer_len) + ".png"
                plt.savefig(os.path.join('./figures', self.config['env_name'], title))

        # 학습이 끝난 후, 누적 보상값 저장
        filename = self.config['runner']['history_path'] + self.config['envs']['name']
        filename += network_type + "-mem_len-" + str(memory_len) + "-layer_len-" + str(layer_len) + '_epi_reward.txt'
        np.savetxt(filename, self.save_epi_reward)
        self.env.close()

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
        plt.clf()

    def __insert_q(self, state):
        if isinstance(state, dict):
            pass
        else:
            if len(state.shape) > 1:
                state = np.expand_dims(state[:, :, 0], axis=0)
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


class ParallelGymRunner:
    def __init__(self, config: dict):
        env_names = self.get_element(config['env_name'])
        q_cases = self.get_element(config['network']['actor']['memory_q_len'])
        layer_cases = self.get_element(config['network']['actor']['use_memory_layer'])

        self.worker_count = len(env_names) * len(q_cases) * len(layer_cases)
        self.env_names = env_names
        self.q_cases = q_cases
        self.layer_cases = layer_cases
        self.config = config

    def run(self):
        from multiprocessing import Pool
        pool = Pool(self.worker_count)
        args = []
        for env_name in self.env_names:
            sub_config = copy.deepcopy(self.config)
            sub_config['runner']['history_path'] = os.path.join("./history", env_name)
            if os.path.exists(sub_config['runner']['history_path']) is False:
                os.mkdir(sub_config['runner']['history_path'])

            sub_config['runner']['history_path'] = os.path.join(sub_config['runner']['history_path'],
                                                                sub_config['agent_name'])
            if os.path.exists(sub_config['runner']['history_path']) is False:
                os.mkdir(sub_config['runner']['history_path'])

            for q_len in self.q_cases:
                for layer_len in self.layer_cases:
                    sub_config['env_name'] = env_name
                    sub_config = self.update_config(config=sub_config, key='envs', name=env_name)
                    sub_config['network']['actor']['memory_q_len'] = q_len
                    sub_config['network']['critic']['memory_q_len'] = q_len
                    sub_config['network']['actor']['use_memory_layer'] = layer_len
                    args.append(sub_config)

        pool.map(self.sub_runner_start, args)

    @staticmethod
    def update_config(config, key, name):
        from utils.yaml_config import YamlConfig
        root = os.path.join("./config/yaml/", key)
        name = name + '.yaml'
        sub_dict = YamlConfig.get_dict(os.path.join(root, name))
        config[key] = sub_dict[key]
        return copy.deepcopy(config)

    @staticmethod
    def sub_runner_start(config: dict):
        if config['runner']['render']:
            env = gym.make(config['env_name'], render_mode='human')
        else:
            env = gym.make(config['env_name'])
        sub_runner = GymRunner(config=config, env=env)
        sub_runner.run()

    @staticmethod
    def get_element(target):
        rtn_list = []
        if isinstance(target, list):
            return target
        else:
            rtn_list.append(target)
            return rtn_list
