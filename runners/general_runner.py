import copy
import os
import gym
import time

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from agents.tf.actorcritic import Actor, Critic
from agents import REGISTRY as AGENT_REGISTRY
from agents.general_agent import GeneralAgent
from agents.pytorch.copy_ppo import PPO
from networks.network_generator import CustomTorchNetwork, SimpleActorNetwork, SimpleCriticNetwork


class GeneralRunner:
    def __init__(self, config: dict, env: gym.Env):
        self._config = config
        self._env = env
        # 액터 신경망 및 크리틱 신경망 생성
        actor, critic = self.__load_networks(config['agent']['framework'])

        # RL algorithm 생성
        algo_name = config['agent']['framework'] + config['agent_name']
        # state_dim = config['network']['actor']['non_spatial_feature']['dim_in'] * config['network']['actor']['memory_q_len']
        # self._agent = PPO(state_dim, 2, 0.0001, 0.001, 0.99, 20, 0.05, False)
        # self._agent.instance_networks((actor, critic))
        self._agent: GeneralAgent = AGENT_REGISTRY[algo_name](parameters=self._config['agent'],
                                                              actor=actor, critic=critic)

        # state
        self.memory_q = {'matrix': [], 'vector': [], 'action_mask': []}

        # Calculate information
        self.save_batch_reward = []
        self.reward_info = {'mu': [0], 'max': [0], 'min': [0], 'episode': [0], 'memory': []}
        self.rew_min = self._config['envs']['reward_range'][0]
        self.rew_max = self._config['envs']['reward_range'][1]
        self.rew_gap = (self.rew_max - self.rew_min) / 2
        self.rew_numerator = (self.rew_max + self.rew_min) / 2

        self.torch_state = False

        # Public information
        self.memory_len = self._config['network']['actor']['memory_q_len']
        self.layer_len = self._config['network']['actor']['num_memory_layer']
        self.network_type = "GRU" if self._config['network']['actor']['num_memory_layer'] > 0 else "Raw"

        # Public variables
        self.count, self.batch_reward, self.done = 0, 0, False

        # Pretrain(이어하기 조건)
        if self._config['runner']['pretrain']:
            self._load_pretrain_network()

    def run(self):
        pass

    def plot_result(self):
        plt.plot(self.save_batch_reward)
        plt.show()
        plt.clf()

    def _env_init(self, reset_env=False):
        self.count, self.batch_reward = 0, 0
        state = None
        if reset_env:
            self.done = False
            self._clear_state_memory()
            ret = self._env.reset()
            state = ret[0]
            for _ in range(self.memory_len):
                self._insert_q(state)

            state = self._update_memory()
        return state

    def _close_env(self):
        self._env.close()
        
    def _interaction(self, action):
        # 다음 상태, 보상 관측
        state, reward, done, trunc, info = self._env.step(action)
        done |= trunc
        state = self._update_memory(state)
        self.count += 1
        self.done = done

        if torch.is_tensor(reward) is False:
            reward = float(reward)

        train_reward = self._fit_reward(reward)
        if isinstance(done, bool):
            done = np.reshape(done, -1)
            train_reward = np.reshape(train_reward, -1)

        if self._config['envs']['trust_result']:
            if self.done:
                self.batch_reward += train_reward
        else:
            self.batch_reward += train_reward
        self._agent.batch_reward.append(train_reward)
        self._agent.batch_done.append(done)
        return state

    def _select_action(self, state):
        action = self._agent.select_action(state)

        if torch.is_tensor(action):
            action = action.squeeze()
            if len(action.shape) == 0:
                action = action.item()
        return action

    def _update_agent(self, next_state):
        if len(self._agent.batch_reward) >= self._config['runner']['batch_size']:
            # 학습 추출
            self._agent.update(next_state=next_state, done=self.done)
            # self._agent.update()
            return True
        else:
            return False

    def _sweep_cycle(self, itr):
        if isinstance(self.batch_reward, float):
            self.reward_info['memory'].append(self.batch_reward)
        else:
            self.reward_info['memory'].append(self.batch_reward.mean().item())

        self.save_batch_reward.append(self.reward_info['memory'][-1])
        # 업데이트 특정값 신경망 파라미터를 파일에 저장
        if itr % self._config['runner']['draw_interval'] == 0:
            self._save_agent()
            self._draw_reward_plot(now_ep=itr)
            self._save_metric(itr // self._config['runner']['draw_interval'] - 1)

    def _fit_reward(self, rew):
        if isinstance(rew, float):
            min_under = self.rew_min > rew
            max_over = self.rew_max < rew
        else:
            min_under = True in (self.rew_min > rew[:])
            max_over = True in (self.rew_max < rew[:])

        # 이젠 클리핑으로 대체
        if min_under:
            print('reward min is updated: ', rew)
            rew = self.rew_min
            # self.rew_min = rew.min().item()
            # self.rew_gap = (self.rew_max - self.rew_min) / 2
            # self.rew_numerator = (self.rew_max + self.rew_min) / 2
        if max_over:
            print('reward max is updated: ', rew)
            rew = self.rew_max
            # self.rew_max = rew.max().item()
            # self.rew_gap = (self.rew_max - self.rew_min) / 2
            # self.rew_numerator = (self.rew_max + self.rew_min) / 2
        # 학습용 보상 [-1, 1]로 fitting
        train_reward = (rew - self.rew_numerator) / self.rew_gap

        return train_reward

    def _insert_q(self, state):
        mem_lim = self._config['network']['actor']['memory_q_len']
        if isinstance(state, dict):
            # Custom environment, 이미 형식이 맞춰져 있다고 판단
            self.torch_state = True
            if len(state['matrix']) > 0:
                if mem_lim > len(self.memory_q['matrix']):
                    self.memory_q['matrix'].append(state['matrix'])
            if len(state['vector']) > 0:
                if mem_lim > len(self.memory_q['vector']):
                    self.memory_q['vector'].append(state['vector'])
            if len(state['action_mask']) > 0:
                self.memory_q['action_mask'].append(state['action_mask'])
        else:
            if isinstance(state, int):
                state = np.array(state)
            # Open AI gym에서 받아온 State
            if len(state.shape) > 1:
                # (b, c, w, h)로 변경
                if mem_lim > len(self.memory_q['matrix']):
                    state = np.expand_dims(state[:, :, 0], axis=0)
                    state = np.expand_dims(state, axis=0)
                    self.memory_q['matrix'].append(state)
            else:
                # (b, c, f)로 변경
                if mem_lim > len(self.memory_q['vector']):
                    self.memory_q['vector'].append(state.reshape(1, 1, -1))

    def _clear_state_memory(self):
        self.memory_q = {'matrix': [], 'vector': [], 'action_mask': []}

    def _update_memory(self, state=None):
        matrix_obs, vector_obs, mask_obs = [], [], []

        if state is not None:
            self._insert_q(state)

        if self.torch_state:
            if len(self.memory_q['matrix']) > 0:
                matrix_obs = torch.cat(self.memory_q['matrix'], dim=2).detach()
                shape = matrix_obs.shape
                matrix_obs = matrix_obs.view(shape[0], -1, shape[-2], shape[-1])
                self.memory_q['matrix'].pop(0)

            if len(self.memory_q['vector']) > 0:
                vector_obs = torch.cat(self.memory_q['vector'], dim=1).detach()
                shape = vector_obs.shape
                vector_obs = vector_obs.view(shape[0], -1, shape[-1])
                self.memory_q['vector'].pop(0)

            if len(self.memory_q['action_mask']) > 0:
                mask_obs = self.memory_q['action_mask'][-1]
                self.memory_q['action_mask'].pop(0)
        else:
            if len(self.memory_q['matrix']) > 0:
                matrix_obs = np.concatenate(self.memory_q['matrix'], axis=1)
                self.memory_q['matrix'].pop(0)

            if len(self.memory_q['vector']) > 0:
                vector_obs = np.concatenate(self.memory_q['vector'], axis=1)
                self.memory_q['vector'].pop(0)

            if len(self.memory_q['action_mask']) > 0:
                mask_obs = self.memory_q['action_mask'][-1]
                self.memory_q['action_mask'].pop(0)

        state = {'matrix': matrix_obs, 'vector': vector_obs, 'action_mask': mask_obs}
        return state

    def _save_agent(self, prefix_option=True):
        prefix = 'agent'
        if prefix_option:
            prefix = self.network_type + "-mem_len-" + str(self.memory_len) + "-layer_len-" + str(self.layer_len)

        save_name = prefix + time.strftime('%Y-%m-%d-%H-%M-%S')
        checkpoint_path = os.path.join(self._config['runner']['history_path'], save_name)
        self._agent.save(checkpoint_path=checkpoint_path)

    def _draw_reward_plot(self, now_ep, y_lim=None, prefix_option=True):
        prefix = 'reward'
        if prefix_option:
            prefix = self.network_type + "-mem_len-" + str(self.memory_len) + "-layer_len-" + str(self.layer_len)

        mu = np.max(self.reward_info['memory'])
        sigma = np.std(self.reward_info['memory'])
        self.reward_info['mu'].append(mu)
        self.reward_info['max'].append(mu + (0.5 * sigma))
        self.reward_info['min'].append(mu - (0.5 * sigma))
        self.reward_info['episode'].append(now_ep)
        self.reward_info['memory'].clear()

        plt.clf()
        plt.plot(self.reward_info['episode'], self.reward_info['mu'], '-')
        plt.fill_between(self.reward_info['episode'],
                         self.reward_info['min'], self.reward_info['max'], alpha=0.2)
        if y_lim is not None:
            plt.ylim(y_lim)
        title = prefix + ".png"
        plt.savefig(os.path.join("figures/", self._config['env_name'], title))

    def _save_metric(self, count):
        data = pd.DataFrame.from_dict(self._agent.metric)
        data.to_csv(os.path.join(self._config['runner']['history_path'], 'Metric' + str(count) + '.csv'))
        self._agent.metric = self._agent.make_metrics()

    def _save_reward_log(self, prefix_option=True):
        prefix = 'reward_log'
        if prefix_option:
            prefix = self.network_type + "-mem_len-" + str(self.memory_len) + "-layer_len-" + str(self.layer_len)
        filename = os.path.join("./history", self._config['env_name'], prefix + '_epi_reward.txt')
        np.savetxt(filename, self.save_batch_reward)

    def _load_pretrain_network(self, prefix_option=True):
        try:
            if os.listdir(self._config['runner']['history_path']):
                name_list = os.listdir(self._config['runner']['history_path'])
                if prefix_option:
                    prefix = self.network_type + "-mem_len-" \
                             + str(self.memory_len) \
                             + "-layer_len-" + str(self.layer_len)
                    name_list = [file for file in name_list if prefix in file]

                if len(name_list) > 0:
                    full_list = [os.path.join(self._config['runner']['history_path'], name) for name in name_list]
                    time_sorted_list = sorted(full_list, key=os.path.getmtime)
                    last_file = time_sorted_list[-1]
                    self._agent.load(checkpoint_path=last_file)
        finally:
            pass

    def __load_tf_models(self):
        ns_f_dim = self._config['network']['actor']['non_spatial_feature']['dim_in']
        s_f_dim = self._config['network']['actor']['spatial_feature']['dim_in']
        state_dim = ns_f_dim + s_f_dim
        self._config['agent']['state_dim'] = state_dim
        # 행동 차원
        self.action_dim = self._env.action_space

        actor = Actor(self.action_dim.shape[0], self.action_dim.high[0])
        critic = Critic()
        actor.build(input_shape=(None, state_dim))
        critic.build(input_shape=(None, state_dim))
        return actor, critic

    def __load_torch_models(self):
        self._config['network']['critic'] = self.__make_critic_config(self._config['network']['actor'])
        dim_in = self._config['network']['actor']['memory_q_len'] * self._config['network']['actor']['non_spatial_feature']['dim_in']
        actor = CustomTorchNetwork(self._config['network']['actor'])
        critic = CustomTorchNetwork(self._config['network']['critic'])
        # actor = SimpleActorNetwork(input_dim=dim_in, output_dim=2)
        # critic = SimpleCriticNetwork(input_dim=dim_in)
        return actor, critic

    def __load_networks(self, framework='tf'):
        if "tf" in framework:
            return self.__load_tf_models()
        else:
            return self.__load_torch_models()

    @staticmethod
    def __make_critic_config(actor_config):
        neck_in = 64
        critic_config = copy.deepcopy(actor_config)
        if critic_config['spatial_feature']['use'] and critic_config['non_spatial_feature']['use']:
            critic_config['spatial_feature']['dim_out'] = neck_in // 2
            critic_config['non_spatial_feature']['dim_out'] = neck_in // 2
        else:
            if critic_config['spatial_feature']['use']:
                critic_config['spatial_feature']['dim_out'] = neck_in
            else:
                critic_config['non_spatial_feature']['dim_out'] = neck_in
        critic_config['non_spatial_feature']['use_cnn'] = False
        critic_config['n_of_actions'] = [1]
        critic_config['action_mode'] = "Continuous"
        return critic_config
