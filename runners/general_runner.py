import copy
import os
import gym
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from agents.tf.actorcritic import Actor, Critic
from agents import REGISTRY as AGENT_REGISTRY
from agents.general_agent import GeneralAgent
from networks.network_generator import CustomTorchNetwork


## Design patter을 다시 생각해서 할 것
class GeneralRunner:
    def __init__(self, config: dict, env: gym.Env):
        self._config = config
        self._env = env
        # 액터 신경망 및 크리틱 신경망 생성
        if "tf" in config['agent']['framework']:
            actor, critic = self.__load_tf_models()
        else:
            actor, critic = self.__load_torch_models()

        # RL algorithm 생성
        algo_name = config['agent']['framework'] + config['agent_name']
        self._agent: GeneralAgent = AGENT_REGISTRY[algo_name](parameters=self._config['agent'],
                                                              actor=actor, critic=critic)
        # Public information
        self.memory_len = self._config['network']['actor']['memory_q_len']
        self.layer_len = self._config['network']['actor']['memory_layer_len']
        self.network_type = self._config['network']['actor']['use_memory_layer']

        # Pretrain(이어하기 조건)
        if self._config['runner']['pretrain']:
            self._load_pretrain_network()

        # state
        self.memory_q = {'matrix': [], 'vector': [], 'action_mask': []}

        # Calculate information
        self.save_epi_reward = []
        self.reward_info = {'mu': [0], 'max': [0], 'min': [0], 'episode': [0], 'memory': []}
        self.rew_min = self._config['envs']['reward_range'][0]
        self.rew_max = self._config['envs']['reward_range'][1]
        self.rew_gap = (self.rew_max - self.rew_min) / 2
        self.rew_numerator = (self.rew_max + self.rew_min) / 2

        fig = plt.figure()
        self.torch_state = False

    def run(self):
        pass

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
        plt.clf()

    def _fit_reward(self, rew):
        if self.rew_max < rew:
            rew = torch.tensor(self.rew_max, dtype=torch.float)
        if self.rew_min > rew:
            rew = torch.tensor(self.rew_min, dtype=torch.float)

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

    def _insert_q(self, state):
        if isinstance(state, dict):
            self.torch_state = True
            if len(state['matrix']) > 0:
                self.memory_q['matrix'].append(state['matrix'])
            if len(state['vector']) > 0:
                self.memory_q['vector'].append(state['vector'])
            if len(state['action_mask']) > 0:
                self.memory_q['action_mask'].append(state['action_mask'])
        else:
            if len(state.shape) > 1:
                state = np.expand_dims(state[:, :, 0], axis=0)
                self.memory_q['matrix'].append(state)
            else:
                self.memory_q['vector'].append(state)

    def _clear_memory(self):
        self.memory_q = {'matrix': [], 'vector': []}

    def _update_memory(self, state=None):
        matrix_obs, vector_obs, mask_obs = [], [], []

        if state is not None:
            self._insert_q(state)

        if self.torch_state:
            if len(self.memory_q['matrix']) > 0:
                matrix_obs = torch.cat(self.memory_q['matrix'], dim=1).detach()
                self.memory_q['matrix'].pop(0)

            if len(self.memory_q['vector']) > 0:
                vector_obs = torch.cat(self.memory_q['vector'], dim=1).detach()
                self.memory_q['vector'].pop(0)

            if len(self.memory_q['action_mask']) > 0:
                mask_obs = self.memory_q['action_mask'][-1]
                self.memory_q['action_mask'].pop(0)
        else:
            if len(self.memory_q['matrix']) > 0:
                matrix_obs = np.concatenate(self.memory_q['matrix'], axis=0)
                self.memory_q['matrix'].pop(0)

            if len(self.memory_q['vector']) > 0:
                vector_obs = np.concatenate(self.memory_q['vector'], axis=0)
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

        mu = np.mean(self.reward_info['memory'])
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
            plt.ylim(0, y_lim)
        title = prefix + ".png"
        plt.savefig(os.path.join("figures/", self._config['env_name'], title))

    def _save_reward_log(self, cnt, prefix_option=True):
        prefix = 'reward_log'
        if prefix_option:
            prefix = self.network_type + "-mem_len-" + str(self.memory_len) + "-layer_len-" + str(self.layer_len)
        filename = "./history" + self._config['envs']['name'] + prefix + '_epi_reward.txt'
        np.savetxt(filename, self.save_epi_reward)
        self._draw_reward_plot(now_ep=cnt, y_lim=None)

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
        actor = CustomTorchNetwork(self._config['network']['actor'])
        critic = CustomTorchNetwork(self._config['network']['critic'])
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
        critic_config['n_of_actions'] = [1]
        critic_config['action_mode'] = "Continuous"
        return critic_config
