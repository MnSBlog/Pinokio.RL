import time
import gym
import torch
import numpy as np
from typing import Optional, Union, Tuple

from matplotlib import pyplot as plt
from utils.comm_manger import CommunicationManager
from utils.zero_mq import ZmqClient


class RLFPSv4(gym.Env):
    def __init__(self, env_config):
        self.__version = 4
        self.__envConfig = env_config
        self.__pause_mode = self.__envConfig['mode'] == "Pause"
        self.__period = self.__envConfig['period']
        self.__zmq_client = ZmqClient(self.__envConfig['port'])
        self.__self_play = self.__envConfig['self_play']
        self.__debug = self.__envConfig['debug']
        self.__preset = self.__envConfig['play_mode_preset'][self.__envConfig['play_mode']]
        self.__agents_of_map = (self.__preset['agent_mode'] + self.__preset['enemy_mode']).count(1)
        self.__character_of_map = len(self.__preset['agent_mode'] + self.__preset['enemy_mode'])
        self.__total_agent_count = self.__agents_of_map * len(self.__envConfig['map_type'])
        self.__envCount = 0
        if self.__debug:
            self.__initialize_feature_display(self.__total_agent_count, self.__envConfig['spatial_dim'])

        self.initialize()

    def initialize(self):
        self.__zmq_client.send(
            'Initialize:MapType"' + str(','.join(str(e) for e in self.__envConfig['map_type']))
            + '"Pause"' + str(self.__pause_mode)
            + '"Period"' + str(self.__envConfig['period'])
            + '"Acceleration"' + str(self.__envConfig['acceleration'])
            + '"')

    def step(self, action: Optional[list] = None):
        # Send action list
        action_buffer = CommunicationManager.serialize_action("Action", action.cpu())
        self.__zmq_client.send("Action")

        self.__zmq_client.send(action_buffer)
        return self.__get_observation()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None):

        play_mode = self.__envConfig['play_mode']
        preset = self.__envConfig['play_mode_preset']
        msg = 'Rebuild:"Selfplay"' + str(play_mode) + \
              '"MaxStep"' + str(self.__envConfig['max_steps']) + \
              '"AgentMode"' + str(','.join(str(e) for e in (preset[play_mode]['agent_mode']
                                                            + preset[play_mode]['enemy_mode']))) + \
              '"AgentType"' + str(','.join(str(e) for e in (preset[play_mode]['agent_type']
                                                            + preset[play_mode]['enemy_type']))) + \
              '"FirstWeaponType"' + str(','.join(str(e) for e in (preset[play_mode]['agent_first_weapon_type']
                                                            + preset[play_mode]['enemy_first_weapon_type']))) + \
              '"SecondWeaponType"' + str(','.join(str(e) for e in (preset[play_mode]['agent_second_weapon_type']
                                                            + preset[play_mode]['enemy_second_weapon_type']))) + \
              '"GlRe"' + str(self.__envConfig['global_resolution']) + \
              '"LcRe"' + str(self.__envConfig['local_resolution']) + \
              '"InitHeight"' + str(self.__envConfig['init_height']) + \
              '"AlphaCount"' + str(len(preset[play_mode]['agent_mode'])) + \
              '"Physics"' + str(','.join(str(e) for e in self.__envConfig['kinematics'].values())) + '"'
        self.__zmq_client.send(msg)
        # To gathering spatial-feature
        state, _, _, _ = self.__get_observation()
        return state

    def __get_observation(self):
        character_info_list = self.__zmq_client.send('CharacterInfo')
        field_info_list = self.__zmq_client.send('FieldInfo')
        step_results = self.__zmq_client.send('Step')
        game_results = self.__zmq_client.send('GameResult')

        name, deserialized_char_info_list, deserialized_action_mask = CommunicationManager.deserialize_info(
            character_info_list)

        _, deserialized_field_info_list, _ = CommunicationManager.deserialize_info(field_info_list)
        # feature 시각화 함수 실행
        if self.__debug:
            self.__display_multi_features(deserialized_field_info_list)
        _, deserialized_step_info_list, _ = CommunicationManager.deserialize_info(step_results)

        done = torch.tensor(deserialized_step_info_list[:, :, 0], dtype=torch.bool)
        reward = self.__calculate_reward(deserialized_step_info_list)
        vector_shape = deserialized_char_info_list.shape
        state = {'matrix': deserialized_field_info_list,
                 'vector': deserialized_char_info_list.reshape([vector_shape[0], vector_shape[1], vector_shape[2] * vector_shape[3]]),
                 'action_mask': deserialized_action_mask}
        return state, reward, done, None

    def __initialize_feature_display(self, batch_size, feature_size):
        figure, ax = plt.subplots(batch_size, feature_size, figsize=(15, 15))
        self.__plots = (figure, ax)
        for i in range(batch_size):
            for j in range(feature_size):
                if len(ax.shape) > 1:
                    ax[i, j].get_xaxis().set_visible(False)
                    ax[i, j].get_yaxis().set_visible(False)
                else:
                    ax[j].get_xaxis().set_visible(False)
                    ax[j].get_yaxis().set_visible(False)
        subplot_num = 8
        subplot_title_list = ['char_index', 'visual_field_of_map', 'movable', 'obj_id', 'treat_points',
                              'grenade_damage', 'visible_enemy_location', 'enemy_location']
        for i in range(subplot_num):
            if len(ax.shape) > 1:
                ax[0, i].set_title(subplot_title_list[i], fontsize=12)
            else:
                ax[i].set_title(subplot_title_list[i], fontsize=12)
        plt.tight_layout()
        plt.show(block=False)
        figure.canvas.flush_events()
        self.__plots = (figure, ax)

    def __display_multi_features(self, spatial_data):
        (fig, ax) = self.__plots
        map_count = len(spatial_data)
        batch_size = len(spatial_data[0])
        feature_size = len(spatial_data[0][0])
        for i in range(map_count):
            for j in range(batch_size):
                for k in range(feature_size):
                    if len(ax.shape) > 1:
                        ax[j, k].imshow(np.array(spatial_data[i, j, k, :, :]))
                    else:
                        ax[k].imshow(np.array(spatial_data[i, j, k, :, :]))

        plt.show(block=False)
        plt.pause(0.000000001)
        self.__plots = (fig, ax)

    def __calculate_reward(self, step_result):
        shape = step_result.shape
        reward = torch.zeros([shape[0], shape[1]])
        reward[:, :] += 1 * step_result[:, :, 1]  # team win
        reward[:, :] += 1 * step_result[:, :, 2]  # kill score
        reward[:, :] -= 1 * step_result[:, :, 3]  # dead score
        # reward += 1 * step_result[:, 4]  # damage score
        # reward -= 1 * step_result[:, 5]  # hitted score
        reward[:, :] += 1 * step_result[:, :, 6]  # healthy ratio
        reward_min = self.__envConfig["reward_range"][0]
        reward_max = self.__envConfig["reward_range"][1]
        reward = torch.clamp(reward, reward_min, reward_max)
        return reward

    def render(self, mode="human", **kwargs):
        return

    def seed(self, seed=None):
        return

    def close(self):
        pass
